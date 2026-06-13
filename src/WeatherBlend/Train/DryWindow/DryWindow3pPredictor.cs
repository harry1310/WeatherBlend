using System.Globalization;
using System.Text.Json;
using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3p — Gaussian copula Monte Carlo over Phase 3o's hourly P(wet) marginals.
///
/// Stage 1: Phase 3o supplies daytime hourly P(wet) marginals q_h. 3o is the
/// 3c-oro 4-station Bonehill pool (rich features + 9 terrain features), so
/// these marginals carry the orographic-uplift signal absent from the lean
/// 3a baseline.
///
/// Stage 2: a Gaussian copula joins those marginals into correlated daytime
/// wet/dry sequences. The copula uses a SINGLE empirical correlation matrix
/// Σ per station, fit on train-split observed daytime wet/dry binary
/// sequences. Pooling across leads is correct because the structure being
/// estimated — daytime-shape autocorrelation in OBSERVED labels — is
/// lead-independent by construction: truth doesn't move with the forecast
/// horizon.
///
/// Sampling per draw:
///   1. Draw ε ∈ R⁹, ε_h ~ iid N(0,1) (Box-Muller).
///   2. Z = L · ε where L L^T = Σ (Cholesky), so Z ~ N(0, Σ).
///   3. U_h = Φ(Z_h), so U_h ~ Uniform(0,1) marginally with the rank-
///      correlation structure of Σ preserved.
///   4. X_h = 1[U_h &lt; q_h] for q_h from 3o's hourly P(wet), giving correlated
///      Bernoullis with marginal P(X_h = 1) = q_h (the copula property).
///   5. dry_h = NOT X_h; count longest contiguous dry run; threshold by N.
///
/// Cross-window monotonicity P(L=2) ≥ P(L=3) ≥ … ≥ P(L=6) is guaranteed by
/// computing all windows from one shared correlated-Bernoulli sequence per
/// sample. (Windows 2..6 since 2026-06-10 — any length ≤ the daytime span
/// works; the per-window manifest composites are minted by
/// DryWindowTrainCommand.RunPhase3pAsync.)
///
/// Bake-off Brier 0.1064 aggregate across 9 (station, lead) Bonehill cells
/// on post-JMA-backfill data (see project_phase2_dry_window_bakeoff_2026-05-25).
/// </summary>
public static class DryWindow3pPredictor
{
    public const string Phase3p = "3p";

    /// <summary>3q — the SAME copula-MC engine bound to Phase 3c's marginals
    /// instead of 3o's. For locations without a 3o (orographic) model — Sennen,
    /// where the flat coastal terrain makes 3o pointless — 3c is the rich precip
    /// champion, so 3q is the natural dry-window MC challenger there. A distinct
    /// phase (not a silent "3o-else-3c" fallback) so verify scores it as its own
    /// line and the stage-1 source stays part of the model's identity
    /// (reference_stage1_source_pattern_mcsources).</summary>
    public const string Phase3q = "3q";

    /// <summary>Hyperparameter key under <c>training_metadata.Hyperparameters</c>
    /// where the bound 3o per-station champion version is persisted at train
    /// time. Predict reads this back so the live 3o version the MC samples
    /// over is provenance-stamped on the bundle, not silently re-resolved.</summary>
    public const string Precip3oVersionKey = "precip_3o_version";

    /// <summary>3q's analogue of <see cref="Precip3oVersionKey"/> — the bound
    /// 3c champion version.</summary>
    public const string Precip3cVersionKey = "precip_3c_version";

    /// <summary>The stage-1 precip source for a copula-MC dry-window phase:
    /// which precipitation phase supplies the hourly P(wet) marginals, the
    /// metadata key its bound version is stamped under, and the start-hour
    /// curve's model_version. The marginals are read from the same
    /// <c>predictions/precipitation/{station}/model_version=…</c> tree
    /// regardless of source phase (3o/3c/3a all write ProbWet there), so only
    /// the version string + provenance key differ.</summary>
    public readonly record struct CopulaSource(string PrecipPhase, string VersionKey, string StartHourCurveVersion);

    /// <summary>Map a dry-window copula-MC phase to its stage-1 source.
    /// Throws for non-copula phases — callers gate on <see cref="IsCopulaMcPhase"/>.</summary>
    public static CopulaSource SourceFor(string dryWindowPhase) => dryWindowPhase switch
    {
        Phase3p => new CopulaSource("3o", Precip3oVersionKey, "v3-3p"),
        Phase3q => new CopulaSource("3c", Precip3cVersionKey, "v3-3q"),
        _ => throw new ArgumentException($"'{dryWindowPhase}' is not a copula-MC dry-window phase.", nameof(dryWindowPhase)),
    };

    /// <summary>Map the train CLI <c>--feature-set</c> token to its copula-MC
    /// phase: <c>copula-mc</c>→3p (3o), <c>copula-mc-3c</c>→3q (3c). Null for
    /// non-copula feature sets.</summary>
    public static string? PhaseForFeatureSet(string featureSet) => featureSet switch
    {
        "copula-mc"    => Phase3p,
        "copula-mc-3c" => Phase3q,
        _ => null,
    };

    /// <summary>True for the copula-MC dry-window phases (3p, 3q) — the
    /// parameter-free MC phases that ship no model.zip and read a precip
    /// phase's live predictions at inference time.</summary>
    public static bool IsCopulaMcPhase(string phase)
        => phase is Phase3p or Phase3q;

    /// <summary>Default MC sample count. 20,000 trades a small predict-time
    /// hit for low MC noise on the Brier estimate; the copula path runs one
    /// Φ evaluation + a 9-dim matrix-vector product per draw.</summary>
    public const int DefaultMcSamples = 20_000;

    /// <summary>
    /// Single copula-MC pass yielding P(longest dry run ≥ L) for every window
    /// in <paramref name="windowHours"/>. All windows share the same correlated-
    /// Bernoulli draws per sample, preserving exact cross-window monotonicity
    /// (shorter window ⇒ ≥ probability).
    /// </summary>
    /// <param name="qHourly">Daytime hourly P(wet) from 3o, length matches the
    /// fitted Σ dimension (typically 9 — the daytime window). Caller is
    /// responsible for slicing to daytime hours and clamping to (0,1).</param>
    /// <param name="choleskyL">Lower-triangular Cholesky factor of Σ (L L^T = Σ).
    /// Must be square with side length equal to <paramref name="qHourly"/>.
    /// Loaded once per bundle via <see cref="LoadCholesky"/>.</param>
    public static Dictionary<int, double> ProbDryWindow(
        double[] qHourly,
        double[,] choleskyL,
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

        int n = qHourly.Length;
        if (choleskyL.GetLength(0) != n || choleskyL.GetLength(1) != n)
            throw new ArgumentException(
                $"Cholesky factor dimensions ({choleskyL.GetLength(0)}×{choleskyL.GetLength(1)}) " +
                $"do not match qHourly length ({n}).", nameof(choleskyL));

        var sortedWindows = windowHours.Distinct().OrderBy(w => w).ToArray();
        var hits = new int[sortedWindows.Length];

        var sampler = new BoxMullerSampler();
        var eps = new double[n];
        for (int s = 0; s < nSamples; s++)
        {
            for (int h = 0; h < n; h++) eps[h] = sampler.Sample(rng);

            int run = 0, longest = 0;
            for (int h = 0; h < n; h++)
            {
                // z_h = (L · eps)_h. L is lower triangular so only k ≤ h contribute.
                double z = 0.0;
                for (int k = 0; k <= h; k++) z += choleskyL[h, k] * eps[k];
                double u = StandardNormalCdf(z);
                bool dry = u >= qHourly[h];
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

    /// <summary>Convenience: single window length.</summary>
    public static double ProbDryWindow(
        double[] qHourly,
        double[,] choleskyL,
        int windowLength,
        Random rng,
        int nSamples = DefaultMcSamples)
        => ProbDryWindow(qHourly, choleskyL, new[] { windowLength }, rng, nSamples)[windowLength];

    /// <summary>
    /// Summary of the per-MC-sample longest-dry-run distribution, written into
    /// the McMean/P10/P50/P90LongestDryRunHours fields on
    /// <see cref="Models.DryWindowPredictionRow"/>. Site uses this as a
    /// confidence signal: narrow P10–P90 band → headline P(dry-window) is
    /// robust across MC realisations; wide band → fragile. 3p has no fitted
    /// conformal calibrator (the "fit on val slice" pattern 3b uses doesn't
    /// carry over to MC-over-a-separate-model without a 3o val replay tree),
    /// so the MC band is the only confidence signal shipped on 3p rows.
    /// </summary>
    public readonly record struct DryWindowMcStats(
        double ProbWindow,
        double MeanLongestDryRunHours,
        int P10LongestDryRunHours,
        int P50LongestDryRunHours,
        int P90LongestDryRunHours);

    /// <summary>
    /// Single copula-MC pass yielding both P(window) and the per-MC-sample
    /// longest-dry-run distribution stats. Same draws, same RNG seed → same
    /// P(window) as the plain <see cref="ProbDryWindow(double[], double[,], int, Random, int)"/>
    /// overload for the same caller (verified by
    /// <c>ProbDryWindowWithStats_matches_ProbDryWindow_for_same_seed</c>).
    /// One extra <c>int[nSamples]</c> alloc + a per-sample comparator vs the
    /// plain overload — negligible at the 20k default.
    /// </summary>
    public static DryWindowMcStats ProbDryWindowWithStats(
        double[] qHourly,
        double[,] choleskyL,
        int windowLength,
        Random rng,
        int nSamples = DefaultMcSamples)
    {
        if (qHourly.Length == 0)
            return new DryWindowMcStats(0.0, 0.0, 0, 0, 0);

        int n = qHourly.Length;
        if (choleskyL.GetLength(0) != n || choleskyL.GetLength(1) != n)
            throw new ArgumentException(
                $"Cholesky factor dimensions ({choleskyL.GetLength(0)}×{choleskyL.GetLength(1)}) " +
                $"do not match qHourly length ({n}).", nameof(choleskyL));

        var sampler = new BoxMullerSampler();
        var eps = new double[n];
        var longestRuns = new int[nSamples];
        int hits = 0;
        long longestSum = 0;
        for (int s = 0; s < nSamples; s++)
        {
            for (int h = 0; h < n; h++) eps[h] = sampler.Sample(rng);

            int run = 0, longest = 0;
            for (int h = 0; h < n; h++)
            {
                double z = 0.0;
                for (int k = 0; k <= h; k++) z += choleskyL[h, k] * eps[k];
                double u = StandardNormalCdf(z);
                bool dry = u >= qHourly[h];
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            longestRuns[s] = longest;
            longestSum += longest;
            if (longest >= windowLength) hits++;
        }

        // Sort once for the quantiles. nSamples = 20k by default → ~200 µs on
        // a laptop, dominated by the MC pass itself.
        Array.Sort(longestRuns);
        int Quantile(double q) =>
            longestRuns[Math.Clamp((int)Math.Floor(q * (nSamples - 1)), 0, nSamples - 1)];

        return new DryWindowMcStats(
            ProbWindow: (double)hits / nSamples,
            MeanLongestDryRunHours: (double)longestSum / nSamples,
            P10LongestDryRunHours: Quantile(0.10),
            P50LongestDryRunHours: Quantile(0.50),
            P90LongestDryRunHours: Quantile(0.90));
    }

    /// <summary>
    /// Single copula-MC pass returning both the daily stats (as
    /// <see cref="ProbDryWindowWithStats"/> would) AND a per-start-hour
    /// curve: <c>StartHourProbs[h]</c> = MC fraction of samples in which
    /// hours <c>[h, h + windowLength)</c> were all dry.
    ///
    /// Curve length is <c>max(0, n - windowLength + 1)</c> where
    /// <c>n = qHourly.Length</c>. <see cref="DryWindowMcStats.ProbWindow"/>
    /// equals "P(any start hour wins)" — the union over the start-hour
    /// curve, which under the same draws is the longest-dry-run-≥-N
    /// indicator, so this method's <c>ProbWindow</c> matches
    /// <see cref="ProbDryWindowWithStats"/> for the same RNG seed.
    /// </summary>
    public static (DryWindowMcStats Stats, double[] StartHourProbs) ProbDryWindowWithStartHours(
        double[] qHourly,
        double[,] choleskyL,
        int windowLength,
        Random rng,
        int nSamples = DefaultMcSamples)
    {
        if (qHourly.Length == 0 || windowLength <= 0 || windowLength > qHourly.Length)
            return (new DryWindowMcStats(0.0, 0.0, 0, 0, 0), Array.Empty<double>());

        int n = qHourly.Length;
        if (choleskyL.GetLength(0) != n || choleskyL.GetLength(1) != n)
            throw new ArgumentException(
                $"Cholesky factor dimensions ({choleskyL.GetLength(0)}×{choleskyL.GetLength(1)}) " +
                $"do not match qHourly length ({n}).", nameof(choleskyL));

        int nStarts = n - windowLength + 1;
        var startHits = new long[nStarts];

        var sampler = new BoxMullerSampler();
        var eps = new double[n];
        var dry = new bool[n];
        var longestRuns = new int[nSamples];
        int hits = 0;
        long longestSum = 0;

        for (int s = 0; s < nSamples; s++)
        {
            for (int h = 0; h < n; h++) eps[h] = sampler.Sample(rng);

            int run = 0, longest = 0;
            for (int h = 0; h < n; h++)
            {
                double z = 0.0;
                for (int k = 0; k <= h; k++) z += choleskyL[h, k] * eps[k];
                double u = StandardNormalCdf(z);
                bool isDry = u >= qHourly[h];
                dry[h] = isDry;
                run = isDry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            longestRuns[s] = longest;
            longestSum += longest;
            if (longest >= windowLength) hits++;

            // Per-start-hour pass over the same dry[] vector. A block at
            // start h exists iff dry[h..h+W) are all true. Using the running
            // 'run' values we could shortcut, but the explicit AND loop is
            // simpler and still ~1 ns/hour at n=9 — dwarfed by the MC math.
            for (int start = 0; start < nStarts; start++)
            {
                bool allDry = true;
                for (int h = start; h < start + windowLength; h++)
                {
                    if (!dry[h]) { allDry = false; break; }
                }
                if (allDry) startHits[start]++;
            }
        }

        Array.Sort(longestRuns);
        int Quantile(double q) =>
            longestRuns[Math.Clamp((int)Math.Floor(q * (nSamples - 1)), 0, nSamples - 1)];

        var startProbs = new double[nStarts];
        for (int s = 0; s < nStarts; s++) startProbs[s] = (double)startHits[s] / nSamples;

        var stats = new DryWindowMcStats(
            ProbWindow: (double)hits / nSamples,
            MeanLongestDryRunHours: (double)longestSum / nSamples,
            P10LongestDryRunHours: Quantile(0.10),
            P50LongestDryRunHours: Quantile(0.50),
            P90LongestDryRunHours: Quantile(0.90));
        return (stats, startProbs);
    }

    /// <summary>
    /// Standard normal CDF Φ(z) via Abramowitz-Stegun 7.1.26 (max abs error
    /// ≈ 1.5e-7). Bernoulli quantisation downstream wipes out any sub-ε
    /// difference.
    /// </summary>
    public static double StandardNormalCdf(double z)
    {
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;

        double x = z / Math.Sqrt(2.0);
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + p * x);
        double erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * erf);
    }

    /// <summary>
    /// Cholesky decompose Σ into the lower-triangular L (L L^T = Σ). Direct
    /// in-place implementation — too small to pull in MathNet. Throws if Σ
    /// isn't strictly positive definite.
    /// </summary>
    public static double[,] CholeskyDecompose(double[,] sigma)
    {
        int n = sigma.GetLength(0);
        if (sigma.GetLength(1) != n)
            throw new ArgumentException("Sigma must be square.", nameof(sigma));
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = sigma[i, j];
                for (int k = 0; k < j; k++) sum -= L[i, k] * L[j, k];
                if (i == j)
                {
                    if (sum <= 0)
                        throw new InvalidOperationException(
                            $"Cholesky failed: diagonal {i} non-positive ({sum:G6}). " +
                            $"Σ is not strictly positive-definite — likely degenerate train slice.");
                    L[i, i] = Math.Sqrt(sum);
                }
                else
                {
                    L[i, j] = sum / L[j, j];
                }
            }
        }
        return L;
    }

    /// <summary>
    /// Fit a single Pearson correlation matrix from observed daytime
    /// wet/dry binary sequences. Each row of <paramref name="binarySequences"/>
    /// is one day's daytime indicators (length = number of daytime hours).
    /// Diagonal entries are forced to 1.0 to guard against floating-point drift
    /// that would otherwise cause Cholesky to fail.
    /// </summary>
    public static double[,] FitCorrelation(byte[][] binarySequences)
    {
        if (binarySequences.Length == 0)
            throw new ArgumentException("Cannot fit correlation from zero sequences.", nameof(binarySequences));
        int n = binarySequences[0].Length;
        int m = binarySequences.Length;

        var mean = new double[n];
        for (int i = 0; i < m; i++)
        {
            if (binarySequences[i].Length != n)
                throw new ArgumentException(
                    $"Sequence {i} length {binarySequences[i].Length} ≠ expected {n}.",
                    nameof(binarySequences));
            for (int h = 0; h < n; h++) mean[h] += binarySequences[i][h];
        }
        for (int h = 0; h < n; h++) mean[h] /= m;

        var sigma = new double[n, n];
        for (int h1 = 0; h1 < n; h1++)
        for (int h2 = 0; h2 < n; h2++)
        {
            double cov = 0.0;
            for (int i = 0; i < m; i++)
                cov += (binarySequences[i][h1] - mean[h1]) * (binarySequences[i][h2] - mean[h2]);
            sigma[h1, h2] = cov / m;
        }

        // Normalise to correlation (diag = 1, off-diag = cov / (σ_i σ_j)).
        var stdev = new double[n];
        for (int h = 0; h < n; h++) stdev[h] = Math.Sqrt(Math.Max(sigma[h, h], 1e-12));
        var corr = new double[n, n];
        for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            corr[i, j] = i == j ? 1.0 : sigma[i, j] / (stdev[i] * stdev[j]);
        return corr;
    }

    /// <summary>
    /// Load the single per-station Σ from a 3p bundle's <c>correlation.json</c>
    /// and return its Cholesky factor L. Phase 3p uses ONE Σ per station:
    /// the daytime-shape autocorrelation in observed labels is lead-independent
    /// by construction (truth doesn't move with forecast horizon), so pooling
    /// the per-lead truth sequences gives a tighter Σ estimate.
    ///
    /// Expected JSON shape:
    /// <code>
    /// { "Sigma": [[...]] }
    /// </code>
    /// </summary>
    public static double[,] LoadCholesky(string bundleDir)
    {
        var path = Path.Combine(bundleDir, "correlation.json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"3p bundle missing correlation.json at {path}", path);
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        if (!doc.RootElement.TryGetProperty("Sigma", out var sigmaEl)
            || sigmaEl.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException(
                $"correlation.json at {path} missing 'Sigma' array.");
        int n = sigmaEl.GetArrayLength();
        var sigma = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            var row = sigmaEl[i];
            if (row.ValueKind != JsonValueKind.Array || row.GetArrayLength() != n)
                throw new InvalidDataException(
                    $"correlation.json 'Sigma' row {i} at {path} is not a length-{n} array.");
            for (int j = 0; j < n; j++) sigma[i, j] = row[j].GetDouble();
        }
        return CholeskyDecompose(sigma);
    }

    /// <summary>
    /// Write the per-station Σ matrix to <c>correlation.json</c>. Single
    /// matrix, no per-lead keying (see <see cref="LoadCholesky"/> for why).
    /// L is recomputed on load.
    /// </summary>
    public static void WriteCorrelationJson(string bundleDir, double[,] sigma)
    {
        Directory.CreateDirectory(bundleDir);
        var path = Path.Combine(bundleDir, "correlation.json");
        int n = sigma.GetLength(0);
        var rows = new double[n][];
        for (int i = 0; i < n; i++)
        {
            rows[i] = new double[n];
            for (int j = 0; j < n; j++) rows[i][j] = sigma[i, j];
        }
        var payload = new Dictionary<string, object> { ["Sigma"] = rows };
        File.WriteAllText(path,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }),
            System.Text.Encoding.UTF8);
    }

    /// <summary>
    /// Read (ValidTimeUtc → ProbWet) from the live 3o prediction parquet for
    /// one anchor cycle + station. 3o writes one parquet per
    /// (station, model_version, anchor_date) using the same per-station
    /// bundle (per-station orographic features differ even though the model
    /// weights are pooled).
    /// </summary>
    public static Dictionary<DateTime, double> LoadLivePredictionsHourly(
        string predictionsRoot, string stationSlug, string precip3oVersion, DateTime anchorDate)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation", stationSlug,
            $"model_version={precip3oVersion}",
            $"date={anchorDate:yyyy-MM-dd}",
            "predictions.parquet").Replace('\\', '/');
        return ReadValidTimeProbWet($"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path}')");
    }

    /// <summary>
    /// Pull the daytime hourly q vector for a single target_date from a
    /// (ValidTimeUtc → ProbWet) dictionary. Returns null if any daytime hour
    /// is missing — under the copula MC we can't honestly produce a
    /// probability with a gap.
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

    /// <summary>Local Box-Muller sampler, private to this predictor so 3p
    /// stays self-contained.</summary>
    private sealed class BoxMullerSampler
    {
        private double _cached;
        private bool _hasCached;

        public double Sample(Random rng)
        {
            if (_hasCached) { _hasCached = false; return _cached; }
            var u1 = Math.Max(rng.NextDouble(), 1e-300);
            var u2 = rng.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            var theta = 2.0 * Math.PI * u2;
            _cached = radius * Math.Sin(theta);
            _hasCached = true;
            return radius * Math.Cos(theta);
        }
    }
}
