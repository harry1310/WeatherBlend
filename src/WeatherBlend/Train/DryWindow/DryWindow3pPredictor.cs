using System.Globalization;
using System.Text.Json;
using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3p — Gaussian copula Monte Carlo over Phase 3o's hourly P(wet) marginals.
///
/// Built 2026-05-25 as the cleanup-Phase-2 dry-window challenger; structurally
/// the same as the retired 3j (deleted in cleanup Phase 1) but with two
/// deliberate changes the Phase 2 bake-off settled on:
///   * Marginal source: Phase 3o (3c-oro, 4-station Bonehill pool with rich +
///     9 terrain features) instead of Phase 3a. Source change shaves ~6%
///     aggregate Brier on its own (per project_phase2_dry_window_bakeoff_2026-05-25).
///   * Σ scope: a SINGLE empirical correlation matrix per station, fit on
///     train-split observed daytime wet/dry binary sequences. The retired 3j
///     fit one Σ per (station, lead); pooling across leads gives ~3× more
///     train sequences and produces a tighter, less-overfit estimate without
///     changing the algorithm (the structure being estimated — daytime-shape
///     autocorrelation in OBSERVED labels — is lead-independent by construction:
///     truth doesn't move with the forecast horizon).
///
/// Sampling (identical to 3j):
///   1. Draw ε ∈ R⁹, ε_h ~ iid N(0,1) (Box-Muller).
///   2. Z = L · ε where L L^T = Σ (Cholesky), so Z ~ N(0, Σ).
///   3. U_h = Φ(Z_h), so U_h ~ Uniform(0,1) marginally with the rank-
///      correlation structure of Σ preserved.
///   4. X_h = 1[U_h &lt; q_h] for q_h from 3o's hourly P(wet), giving correlated
///      Bernoullis with marginal P(X_h = 1) = q_h (the copula property).
///   5. dry_h = NOT X_h; count longest contiguous dry run; threshold by N.
///
/// Cross-window monotonicity P(L=3) ≥ P(L=4) ≥ P(L=6) is guaranteed by
/// computing all windows from one shared correlated-Bernoulli sequence per
/// sample — same structural guarantee as 3g.
///
/// Bake-off result: 0.1064 aggregate Brier on the 9 (station, lead) Bonehill
/// cells vs 3g's 0.1265 (−15.9%). Composition: ~6% from source swap
/// (3a→3o), ~11% from algorithm swap (iid→copula). The Phase 2 MC bake-off
/// validated this exact recipe on post-JMA-backfill data.
/// </summary>
public static class DryWindow3pPredictor
{
    public const string Phase3p = "3p";

    /// <summary>Hyperparameter key under <c>training_metadata.Hyperparameters</c>
    /// where the bound 3o per-station champion version is persisted at train
    /// time. Predict reads this back so the live 3o version the MC samples
    /// over is provenance-stamped on the bundle, not silently re-resolved.</summary>
    public const string Precip3oVersionKey = "precip_3o_version";

    /// <summary>Default MC sample count. 20,000 mirrors the retired 3j —
    /// the copula path is slightly noisier per sample than iid (one Φ
    /// evaluation + a 9-dim matrix-vector product per draw) so we trade
    /// a small predict-time hit for tighter Brier-estimate noise.</summary>
    public const int DefaultMcSamples = 20_000;

    /// <summary>
    /// Single copula-MC pass yielding P(longest dry run ≥ L) for every window
    /// in <paramref name="windowHours"/>. All windows share the same correlated-
    /// Bernoulli draws per sample, preserving exact cross-window monotonicity
    /// P(L=3) ≥ P(L=4) ≥ P(L=6).
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
    /// Summary of the per-MC-sample longest-dry-run distribution. Mirrors the
    /// retired 3g's aleatoric-uncertainty fields on <see cref="Models.DryWindowPredictionRow"/>
    /// — site uses this as a confidence signal (narrow P10–P90 band → headline
    /// P(dry-window) is robust; wide band → fragile). 3p has no fitted conformal
    /// calibrator (3p is parameter-free MC over a separate model, so the
    /// "fit on val slice" pattern 3b uses doesn't carry over without a 3o val
    /// replay tree), so the MC band is the only signal we can ship without a
    /// retrain dependency on 3o.
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
    /// and return its Cholesky factor L. Phase 3p uses ONE Σ per station —
    /// the daytime-shape autocorrelation in observed labels is lead-independent
    /// by construction (truth doesn't move with forecast horizon), so pooling
    /// across leads gives a tighter estimate than the retired 3j's per-lead Σ.
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
    /// one anchor cycle + station. Same path shape as the retired 3g's 3a
    /// lookup — 3o writes one parquet per (station, model_version, anchor_date)
    /// using the same per-station bundle (per-station orographic features
    /// differ even though the model weights are pooled).
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

    /// <summary>Local copy of the Box-Muller sampler. Keeping it private to
    /// this predictor matches the retired 3j's self-contained shape.</summary>
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
