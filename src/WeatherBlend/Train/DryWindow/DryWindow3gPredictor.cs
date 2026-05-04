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
