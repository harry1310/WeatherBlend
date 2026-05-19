using System.Globalization;
using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3g — direct dry-window probability via Monte Carlo over Phase 3a's
/// per-hour P(wet) outputs under independence. No LightGBM, no learned
/// parameters: the prediction is literally the fraction of MC samples
/// (Bernoulli per hour using 3a's q values) where the longest dry run within
/// the daytime window is at least N hours long.
///
/// Why this works (per the 2026-05-03 bake-off): 3a is well-calibrated at the
/// hourly level (no PAV needed), and although the independence assumption is
/// wrong (real rain clusters), the structural rule "longer windows are rarer"
/// averages over the dependency structure across many sample days. Net Brier
/// improvement vs Phase 3b: −6.4% on a 27-cell historical test slice.
///
/// Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) is GUARANTEED by
/// computing all requested window probabilities in a single MC pass per row
/// (one Bernoulli sequence per sample, three indicators read off it). This is
/// the structural fix the user requested for the 3b "P(4h) > P(3h)" bug.
///
/// Predict-time inputs are read from the live Phase 3a prediction parquet
/// (<c>data/predictions/precipitation/{station}/model_version={3a-version}/
/// date={anchor}/predictions.parquet</c>) — the same artefact the rain-skill
/// page already serves. Train-time the same shape comes from the replay
/// parquet under <c>precipitation_replay/</c>, produced by
/// <see cref="WeatherBlend.Commands.PrecipReplayCommand"/>.
/// </summary>
public static class DryWindow3gPredictor
{
    public const string Phase3g = "3g";

    /// <summary>Default MC sample count. 10,000 keeps Brier estimation noise
    /// well below the per-cell ±5% scale while staying cheap.</summary>
    public const int DefaultMcSamples = 10000;

    /// <summary>
    /// Single MC pass yielding P(longest dry run ≥ L) for every requested
    /// window length <paramref name="windowHours"/>. All windows share the
    /// same Bernoulli draws per sample, so monotonicity P(L=3) ≥ P(L=4) ≥ ...
    /// is preserved exactly (not just in expectation). Returns a dictionary
    /// keyed by window length; missing windows in the input array are not
    /// in the output.
    /// </summary>
    public static Dictionary<int, double> ProbDryWindow(
        double[] qHourly,
        IReadOnlyList<int> windowHours,
        Random rng,
        int nSamples = DefaultMcSamples)
    {
        var result = new Dictionary<int, double>(windowHours.Count);
        if (qHourly.Length == 0 || windowHours.Count == 0)
        {
            foreach (var w in windowHours) result[w] = 0.0;
            return result;
        }

        var sortedWindows = windowHours.Distinct().OrderBy(w => w).ToArray();
        var hits = new int[sortedWindows.Length];

        for (int s = 0; s < nSamples; s++)
        {
            int run = 0, longest = 0;
            for (int h = 0; h < qHourly.Length; h++)
            {
                bool dry = rng.NextDouble() >= qHourly[h];
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            for (int i = 0; i < sortedWindows.Length; i++)
                if (longest >= sortedWindows[i]) hits[i]++;
        }

        for (int i = 0; i < sortedWindows.Length; i++)
            result[sortedWindows[i]] = (double)hits[i] / nSamples;
        return result;
    }

    /// <summary>
    /// Start-hour curve from one MC pass over <paramref name="qHourly"/>.
    /// Returned <see cref="StartHourCurve.ProbAtLeast"/> matches what
    /// <see cref="ProbDryWindow"/> would return for the same inputs (it's
    /// a free byproduct of the same loop). <see cref="StartHourCurve.MarginalByStart"/>
    /// is <c>P(hours s..s+windowLength-1 are all dry)</c> per candidate
    /// start hour s — the MC estimate of the analytical product
    /// <c>(1-q_s)(1-q_{s+1})...(1-q_{s+windowLength-1})</c>. Under independence
    /// the two are mathematically equivalent; MC's value is that the curve
    /// is computed from the SAME samples that produce ProbAtLeast, so the
    /// two numbers shipped on the same site row stay numerically consistent
    /// (no risk of a drift between "P(any block today)" and "Σ start-hour
    /// probs").
    ///
    /// MarginalByStart sums to E[# valid starts per day], not 1 and not
    /// ProbAtLeast — caller normalises to π_s = marginal_s / Σ marginal
    /// for the conditional shape (which IS what the existing
    /// StartHourPredictionRow.ConditionalProb field stores) and then
    /// CalibratedProb_s = π_s × ProbAtLeast for the magnitude-scaled
    /// quantity that sums to ProbAtLeast.
    ///
    /// Returns null if qHourly.Length &lt; windowLength (no candidate
    /// starts at all, structurally impossible day).
    /// </summary>
    public readonly record struct StartHourCurve(
        double ProbAtLeast,
        double[] MarginalByStart);

    public static StartHourCurve? SampleStartHourCurve(
        double[] qHourly, int windowLength, Random rng, int nSamples = DefaultMcSamples)
    {
        if (qHourly.Length < windowLength) return null;
        var nStarts = qHourly.Length - windowLength + 1;
        var counts = new long[nStarts];
        long anyHits = 0;

        // Allocate once, reuse across samples — qHourly.Length is small
        // (≤24) but stackalloc inside a 10k-iter loop trips CA2014.
        Span<bool> dry = stackalloc bool[qHourly.Length];
        for (int s = 0; s < nSamples; s++)
        {
            for (int h = 0; h < qHourly.Length; h++)
                dry[h] = rng.NextDouble() >= qHourly[h];

            // Track whether ANY length-N dry block exists this sample —
            // ProbAtLeast = (any-hits / nSamples). Same expectation
            // ProbDryWindow / SampleStats compute, but free since we're
            // already iterating candidate starts below.
            bool anyValidStart = false;
            for (int i = 0; i < nStarts; i++)
            {
                bool allDry = true;
                for (int j = 0; j < windowLength; j++)
                {
                    if (!dry[i + j]) { allDry = false; break; }
                }
                if (allDry)
                {
                    counts[i]++;
                    anyValidStart = true;
                }
            }
            if (anyValidStart) anyHits++;
        }

        var marginal = new double[nStarts];
        for (int i = 0; i < nStarts; i++) marginal[i] = (double)counts[i] / nSamples;
        return new StartHourCurve(
            ProbAtLeast: (double)anyHits / nSamples,
            MarginalByStart: marginal);
    }

    /// <summary>
    /// Single-window predict path. Returns just the probability — for the
    /// richer summary including longest-dry-run quantiles, use
    /// <see cref="SampleStats"/>.
    /// </summary>
    public static double ProbDryWindow(
        double[] qHourly, int windowLength, Random rng, int nSamples = DefaultMcSamples)
        => SampleStats(qHourly, windowLength, rng, nSamples).ProbAtLeast;

    /// <summary>
    /// AR(1) Gaussian copula variant of <see cref="ProbDryWindow(double[],int,Random,int)"/>.
    /// Replaces independent Bernoulli draws with hour-to-hour correlated draws
    /// (correlation parameter <paramref name="rho"/> ∈ [0, 1)). Marginals are
    /// preserved: each hour's wet probability stays at <c>qHourly[h]</c>.
    /// Joint structure becomes AR(1): Cov(Z_i, Z_j) = ρ^|i-j| in the latent
    /// Gaussian space, so adjacent hours cluster (rho>0 → more bimodal
    /// "all-wet day" / "all-dry day" outcomes than independence).
    ///
    /// Mechanism per MC sample:
    ///   Z_0 ~ N(0,1)
    ///   Z_h = ρ·Z_{h-1} + sqrt(1-ρ²)·ε_h    (AR(1) recursion)
    ///   wet_h = Z_h &lt; Φ⁻¹(q_h)            (probit threshold)
    /// Equivalent to drawing from the Gaussian copula but avoids the
    /// per-sample Cholesky cost — 2 normals per hour vs O(n²) for full MVN.
    ///
    /// rho=0 reduces to independent draws (mathematically equivalent to the
    /// existing uniform-threshold sampler, though not bit-for-bit since the
    /// random-number consumption pattern differs).
    /// </summary>
    public static double ProbDryWindow(
        double[] qHourly, int windowLength, double rho, Random rng, int nSamples = DefaultMcSamples)
    {
        if (rho < 0 || rho >= 1)
            throw new ArgumentOutOfRangeException(nameof(rho), $"rho must be in [0, 1), got {rho}");
        if (qHourly.Length == 0) return 0.0;

        var thresholds = new double[qHourly.Length];
        for (int h = 0; h < qHourly.Length; h++)
            thresholds[h] = NormalQuantile(Math.Clamp(qHourly[h], 1e-9, 1.0 - 1e-9));

        var sqrtOneMinusRhoSq = Math.Sqrt(1.0 - rho * rho);
        var normal = new BoxMullerSampler();
        int hits = 0;

        for (int s = 0; s < nSamples; s++)
        {
            int run = 0, longest = 0;
            double prevZ = 0.0;
            for (int h = 0; h < qHourly.Length; h++)
            {
                var z = h == 0
                    ? normal.Sample(rng)
                    : rho * prevZ + sqrtOneMinusRhoSq * normal.Sample(rng);
                var dry = z >= thresholds[h];   // wet = Z < t, so dry = Z >= t
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
                prevZ = z;
            }
            if (windowLength <= qHourly.Length && longest >= windowLength) hits++;
        }
        return (double)hits / nSamples;
    }

    /// <summary>Inverse standard-normal CDF (probit / quantile function) via
    /// Acklam's rational approximation — accurate to ~1e-9 on (0, 1).
    /// Public so the rho-bakeoff harness and any downstream caller wanting
    /// to threshold their own latent draws at calibrated marginals can reuse
    /// it without pulling in MathNet.</summary>
    public static double NormalQuantile(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        const double a1 = -3.969683028665376e+01;
        const double a2 =  2.209460984245205e+02;
        const double a3 = -2.759285104469687e+02;
        const double a4 =  1.383577518672690e+02;
        const double a5 = -3.066479806614716e+01;
        const double a6 =  2.506628277459239e+00;
        const double b1 = -5.447609879822406e+01;
        const double b2 =  1.615858368580409e+02;
        const double b3 = -1.556989798598866e+02;
        const double b4 =  6.680131188771972e+01;
        const double b5 = -1.328068155288572e+01;
        const double c1 = -7.784894002430293e-03;
        const double c2 = -3.223964580411365e-01;
        const double c3 = -2.400758277161838e+00;
        const double c4 = -2.549732539343734e+00;
        const double c5 =  4.374664141464968e+00;
        const double c6 =  2.938163982698783e+00;
        const double d1 =  7.784695709041462e-03;
        const double d2 =  3.224671290700398e-01;
        const double d3 =  2.445134137142996e+00;
        const double d4 =  3.754408661907416e+00;
        const double pLow = 0.02425;
        const double pHigh = 1.0 - pLow;

        double q, r;
        if (p < pLow)
        {
            q = Math.Sqrt(-2.0 * Math.Log(p));
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                   ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        }
        if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
        }
        q = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    }

    /// <summary>Box-Muller standard-normal sampler with single-element cache
    /// (returns one draw, stashes the orthogonal pair for the next call).
    /// Halves the per-draw cost vs throwing away the second component.</summary>
    private sealed class BoxMullerSampler
    {
        private double _cached;
        private bool _hasCached;

        public double Sample(Random rng)
        {
            if (_hasCached) { _hasCached = false; return _cached; }
            // Math.Log(0) is -Inf; clamp just above zero to keep the radius finite.
            var u1 = Math.Max(rng.NextDouble(), 1e-300);
            var u2 = rng.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            var theta = 2.0 * Math.PI * u2;
            _cached = radius * Math.Sin(theta);
            _hasCached = true;
            return radius * Math.Cos(theta);
        }
    }

    /// <summary>
    /// Aleatoric-uncertainty summary from one MC pass. ProbAtLeast is the
    /// "would the day satisfy a length-N dry block?" headline; the four
    /// LongestRun stats characterise the per-sample distribution of the
    /// longest contiguous dry stretch (in hours) under independence with
    /// 3a's per-hour q. Useful as a confidence signal: a narrow P10–P90
    /// band (e.g. 4–6h) means even unlikely realisations land near the
    /// mean; a wide band (e.g. 1–8h) means the headline number is more
    /// fragile.
    ///
    /// Longest-run distribution is independent of windowLength (it's a
    /// property of the sample sequences, not the threshold check), so a
    /// caller scoring multiple windows on the same q can call this once
    /// per (station, lead, target_date) and reuse the LongestRun fields
    /// across rows. ProbAtLeast varies per windowLength.
    /// </summary>
    public readonly record struct DryWindowMcStats(
        double ProbAtLeast,
        double MeanLongestRun,
        double P10LongestRun,
        double P50LongestRun,
        double P90LongestRun);

    public static DryWindowMcStats SampleStats(
        double[] qHourly, int windowLength, Random rng, int nSamples = DefaultMcSamples)
    {
        if (qHourly.Length == 0)
            return new DryWindowMcStats(0.0, 0.0, 0.0, 0.0, 0.0);

        var longestRuns = new int[nSamples];
        int hits = 0;
        for (int s = 0; s < nSamples; s++)
        {
            int run = 0, longest = 0;
            for (int h = 0; h < qHourly.Length; h++)
            {
                bool dry = rng.NextDouble() >= qHourly[h];
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            longestRuns[s] = longest;
            if (windowLength <= qHourly.Length && longest >= windowLength) hits++;
        }

        Array.Sort(longestRuns);
        return new DryWindowMcStats(
            ProbAtLeast: (double)hits / nSamples,
            MeanLongestRun: longestRuns.Average(),
            P10LongestRun: Percentile(longestRuns, 0.10),
            P50LongestRun: Percentile(longestRuns, 0.50),
            P90LongestRun: Percentile(longestRuns, 0.90));
    }

    /// <summary>
    /// Phase 3a-uncertainty extension: SampleStats wrapped in an epistemic
    /// outer loop that perturbs the per-hour q vector by a shared shift
    /// δ ~ N(0, sigmaEpistemic) on each outer draw. The mean of the outer
    /// ProbAtLeast values reproduces SampleStats.ProbAtLeast in expectation
    /// when sigmaEpistemic = 0; the (q10, q90) of those same outer
    /// ProbAtLeasts is a Bayesian-flavoured envelope on the dry-window
    /// probability — "if 3a's hourly q is uncertain to ±σ globally for
    /// today, here's how much P(dry block) wobbles".
    ///
    /// The shift is shared across hours (perfectly correlated perturbation)
    /// because the WeatherProbabilistic Bayesian model emits at most ~2
    /// valid_times per day at lead 24 — its CI carries day-level uncertainty,
    /// not hour-level. When the lead-as-feature Bayesian model lands
    /// (Phase 3b in this thread), the σ source becomes per-hour and the
    /// inner sampler can be upgraded without changing the outer-loop API.
    ///
    /// sigmaEpistemic = 0 → outer loop is a no-op identity, results match
    /// SampleStats exactly modulo MC noise. Caller passes σ=0 when no
    /// Bayesian CI is available for this (station, lead, target_date) cell
    /// so 3g degrades cleanly to its existing behaviour.
    /// </summary>
    public readonly record struct DryWindowEpistemicStats(
        double ProbAtLeastMean,
        double ProbAtLeastQ10,
        double ProbAtLeastQ90,
        double SigmaEpistemic,
        DryWindowMcStats InnerStats);

    public static DryWindowEpistemicStats SampleStatsWithEpistemic(
        double[] qHourly,
        int windowLength,
        double sigmaEpistemic,
        Random rng,
        int innerSamples = DefaultMcSamples,
        int outerSamples = 200)
    {
        if (sigmaEpistemic < 0)
            throw new ArgumentOutOfRangeException(nameof(sigmaEpistemic), $"σ must be ≥ 0, got {sigmaEpistemic}");

        // σ = 0 short-circuit: the outer loop adds no information, so collapse
        // to a single inner pass. ProbAtLeastMean = inner ProbAtLeast and the
        // q10/q90 band degenerates to that point. Saves outerSamples × inner
        // work in the no-Bayesian-CI fallback path.
        if (sigmaEpistemic == 0.0)
        {
            var inner = SampleStats(qHourly, windowLength, rng, innerSamples);
            return new DryWindowEpistemicStats(
                ProbAtLeastMean: inner.ProbAtLeast,
                ProbAtLeastQ10:  inner.ProbAtLeast,
                ProbAtLeastQ90:  inner.ProbAtLeast,
                SigmaEpistemic:  0.0,
                InnerStats:      inner);
        }

        if (qHourly.Length == 0)
            return new DryWindowEpistemicStats(0, 0, 0, sigmaEpistemic,
                new DryWindowMcStats(0, 0, 0, 0, 0));

        var outerProbs = new double[outerSamples];
        var perturbed = new double[qHourly.Length];
        var normal = new BoxMullerSampler();

        // Single inner pass at δ=0 to populate the InnerStats payload — gives
        // callers the longest-dry-run distribution at the unperturbed q for
        // free, identical to what SampleStats would return if called directly.
        // Done first so the outer loop's RNG draws don't shift this baseline.
        var baselineStats = SampleStats(qHourly, windowLength, rng, innerSamples);

        for (int k = 0; k < outerSamples; k++)
        {
            var delta = sigmaEpistemic * normal.Sample(rng);
            for (int h = 0; h < qHourly.Length; h++)
                perturbed[h] = Math.Clamp(qHourly[h] + delta, 1e-6, 1.0 - 1e-6);

            // Inner MC pass on the perturbed q. We only need ProbAtLeast here;
            // longest-run quantiles are baseline-only (the epistemic envelope
            // is on the headline probability, not the run distribution — that
            // would be a future cross-product if needed).
            int hits = 0;
            for (int s = 0; s < innerSamples; s++)
            {
                int run = 0, longest = 0;
                for (int h = 0; h < qHourly.Length; h++)
                {
                    bool dry = rng.NextDouble() >= perturbed[h];
                    run = dry ? run + 1 : 0;
                    if (run > longest) longest = run;
                }
                if (windowLength <= qHourly.Length && longest >= windowLength) hits++;
            }
            outerProbs[k] = (double)hits / innerSamples;
        }

        Array.Sort(outerProbs);
        return new DryWindowEpistemicStats(
            ProbAtLeastMean: outerProbs.Average(),
            ProbAtLeastQ10:  PercentileD(outerProbs, 0.10),
            ProbAtLeastQ90:  PercentileD(outerProbs, 0.90),
            SigmaEpistemic:  sigmaEpistemic,
            InnerStats:      baselineStats);
    }

    /// <summary>Convert a Bayesian 80% CI width on logit-P(wet) (or P(wet)
    /// directly — the script writes both) into a global-σ for the epistemic
    /// shift. Width = q90 − q10; under a normal approximation the
    /// corresponding σ = width / (2 · Φ⁻¹(0.9)) = width / 2.5631.
    /// Returns 0 (= no perturbation) for negative or NaN widths so the caller
    /// can use it as a guard alongside the σ=0 short-circuit above.</summary>
    public static double SigmaFromCi80Width(double width)
    {
        if (double.IsNaN(width) || width <= 0) return 0.0;
        return width / 2.5631;  // 2 × Φ⁻¹(0.9) under N(0,1)
    }

    /// <summary>Linear-interpolated percentile of a pre-sorted ascending int
    /// array, returned as a double so half-integer percentiles read
    /// naturally on a continuous "expected hours" scale.</summary>
    private static double Percentile(int[] sortedAsc, double p)
    {
        if (sortedAsc.Length == 0) return 0.0;
        var rank = p * (sortedAsc.Length - 1);
        int lo = (int)Math.Floor(rank), hi = (int)Math.Ceiling(rank);
        if (lo == hi) return sortedAsc[lo];
        return sortedAsc[lo] + (rank - lo) * (sortedAsc[hi] - sortedAsc[lo]);
    }

    /// <summary>Linear-interpolated percentile of a pre-sorted ascending
    /// double array. Same shape as <see cref="Percentile(int[], double)"/>
    /// but for the epistemic outer-loop ProbAtLeast samples.</summary>
    private static double PercentileD(double[] sortedAsc, double p)
    {
        if (sortedAsc.Length == 0) return 0.0;
        var rank = p * (sortedAsc.Length - 1);
        int lo = (int)Math.Floor(rank), hi = (int)Math.Ceiling(rank);
        if (lo == hi) return sortedAsc[lo];
        return sortedAsc[lo] + (rank - lo) * (sortedAsc[hi] - sortedAsc[lo]);
    }

    /// <summary>
    /// Read (ValidTimeUtc → ProbWet) for a specific lead bucket out of the
    /// Phase 3a replay parquet. Used at training time only — the historical
    /// per-hour q for every (station, lead) combination is needed to score
    /// the chronological test split.
    /// </summary>
    public static Dictionary<DateTime, double> LoadReplayHourly(
        string predictionsRoot, string stationSlug, string precip3aVersion, int leadHours)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation_replay", stationSlug, precip3aVersion, $"lead_{leadHours}h.parquet")
            .Replace('\\', '/');
        return ReadValidTimeProbWet($"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path}')");
    }

    /// <summary>
    /// Read (ValidTimeUtc → ProbWet) from the live 3a prediction parquet for
    /// one anchor cycle. Predict-time path: 3a writes one parquet per
    /// (station, model_version, anchor_date) covering ~120h of hourly
    /// forecasts; we read the rows whose ValidTimeUtc falls within the
    /// requested target_date's daytime window (caller filters).
    /// </summary>
    public static Dictionary<DateTime, double> LoadLivePredictionsHourly(
        string predictionsRoot, string stationSlug, string precip3aVersion, DateTime anchorDate)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation", stationSlug,
            $"model_version={precip3aVersion}",
            $"date={anchorDate:yyyy-MM-dd}",
            "predictions.parquet").Replace('\\', '/');
        return ReadValidTimeProbWet($"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path}')");
    }

    /// <summary>
    /// Pull the daytime hourly q vector for a single target_date from the
    /// supplied (ValidTimeUtc → ProbWet) dictionary. Returns null if any
    /// daytime hour is missing — under independence we can't honestly
    /// produce a probability with a gap (and the 3b training pipeline drops
    /// such days from its rows anyway, so this stays consistent).
    /// </summary>
    public static double[]? ExtractDaytimeQ(
        IReadOnlyDictionary<DateTime, double> hourly,
        DateTime targetDateUtc, int startUtcHour, int endUtcHourExclusive)
    {
        var n = endUtcHourExclusive - startUtcHour;
        if (n <= 0) return null;
        var q = new double[n];
        var midnight = new DateTime(
            targetDateUtc.Year, targetDateUtc.Month, targetDateUtc.Day, 0, 0, 0, DateTimeKind.Utc);
        for (int h = startUtcHour; h < endUtcHourExclusive; h++)
        {
            var t = midnight.AddHours(h);
            if (!hourly.TryGetValue(t, out var p)) return null;
            q[h - startUtcHour] = Math.Clamp(p, 0.0, 1.0);
        }
        return q;
    }

    /// <summary>
    /// Phase 3a-uncertainty support: read the WeatherProbabilistic Phase 5a
    /// Bayesian CI80 width for a (station, target_date) and return the widest
    /// value across that UTC day's hourly 5a rows. Drives the epistemic σ —
    /// see <see cref="SigmaFromCi80Width"/>.
    ///
    /// NOT lead-filtered, deliberately. Phase 5a stores an hourly P(wet) curve
    /// whose per-row LeadHours is measured from the NWP cycle; the dry-window
    /// lead bucket (24/48/72) is a different quantity — a day-offset — and the
    /// two do not align. An exact <c>LeadHours = bucket</c> match silently
    /// misses: it only lands when a fresh-enough cycle puts "tomorrow" exactly
    /// at lead 24h, which the early-cycle (03:xx) predict runs never do, so
    /// the band would vanish for whole predict cycles. Taking the day-wide
    /// max instead — the widest CI any 5a row reports for the target day —
    /// is robust and matches the "over-estimate rather than under-" intent.
    /// Returns 0 when no 5a parquet covers the day; the caller's
    /// <see cref="SampleStatsWithEpistemic"/> short-circuits on σ=0 and
    /// produces results identical to <see cref="SampleStats"/>, so cells with
    /// no Bayesian coverage degrade cleanly.
    /// </summary>
    public static double TryLoadBayesianCi80Width(
        string predictionsRoot, string stationSlug, DateTime targetDateUtc)
    {
        // Phase 5a CI lives in the standard predictions tree under
        // model_version=*phase5a* (renamed 2026-05-09 from the legacy
        // precipitation_bayesian_ci/widths.parquet hive). Path glob:
        //   data/predictions/precipitation/{slug}/model_version=*phase5a*/date=*/predictions.parquet
        var stationDir = Path.Combine(
            predictionsRoot, "precipitation", stationSlug);
        if (!Directory.Exists(stationDir)) return 0.0;

        var glob = Path.Combine(stationDir, "model_version=*phase5a*", "date=*", "predictions.parquet")
            .Replace('\\', '/').Replace("'", "''");
        var dayStart = new DateTime(
            targetDateUtc.Year, targetDateUtc.Month, targetDateUtc.Day, 0, 0, 0, DateTimeKind.Utc);
        var dayEnd = dayStart.AddDays(1);

        // max() so the wider end of the day's CI dominates — better to over-
        // estimate epistemic uncertainty than under-. No lead filter (see the
        // method summary): every hourly 5a row valid in the target day is
        // scanned. Returns 0 → no perturbation when nothing matches.
        var sql = $@"
SELECT max(Ci80Width) AS w
FROM read_parquet('{glob}')
WHERE ValidTimeUtc >= TIMESTAMP '{dayStart:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <  TIMESTAMP '{dayEnd:yyyy-MM-dd HH:mm:ss}'";

        try
        {
            using var conn = new DuckDBConnection("DataSource=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = sql;
            using var rdr = cmd.ExecuteReader();
            if (!rdr.Read()) return 0.0;
            return rdr.IsDBNull(0) ? 0.0 : rdr.GetDouble(0);
        }
        catch
        {
            // Glob mismatch / empty parquet tree / DuckDB hiccup: treat as
            // "no Bayesian signal for this cell" rather than failing the
            // whole predict run. Caller short-circuits on σ=0.
            return 0.0;
        }
    }

    private static Dictionary<DateTime, double> ReadValidTimeProbWet(string sql)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var dict = new Dictionary<DateTime, double>();
        using var rdr = cmd.ExecuteReader();
        while (rdr.Read())
        {
            var t = DateTime.SpecifyKind(rdr.GetDateTime(0), DateTimeKind.Utc);
            dict[t] = rdr.GetDouble(1);
        }
        return dict;
    }
}
