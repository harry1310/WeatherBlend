using System.Globalization;
using System.Text.Json;
using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3j — dry-window probability via Gaussian copula MC over Phase 3a's
/// per-hour P(wet) outputs. Identical to 3g (<see cref="DryWindow3gPredictor"/>)
/// except the per-hour Bernoulli draws within a single sample day are correlated
/// according to a 9×9 Pearson correlation matrix Σ fit at training time on the
/// observed daytime wet/dry binary sequences.
///
/// Sampling:
///   1. Draw ε ∈ R⁹, ε_h ~ iid N(0,1) (Box-Muller).
///   2. Z = L · ε where L L^T = Σ (Cholesky), so Z ~ N(0, Σ).
///   3. U_h = Φ(Z_h), so U_h ~ Uniform(0,1) marginally with the rank-correlation
///      structure of Σ preserved.
///   4. X_h = 1[U_h &lt; q_h], giving correlated Bernoullis with marginal
///      P(X_h = 1) = q_h exactly (the copula property).
///   5. dry_h = NOT X_h; count longest contiguous dry run; threshold by N.
///
/// Why this beats 3g at 3h windows but loses at 6h (per the 2026-05-13 bake-off,
/// aggregate 0.1076 vs 3g 0.1129 at 3h, 0.1462 vs 0.1314 at 6h): wet hours
/// cluster within a day. The copula captures that positive autocorrelation,
/// which under iid sampling causes 3g to *underestimate* the prob of "at least
/// one 3-hour dry block" (because iid scatters wet hours uniformly, leaving
/// few long dry stretches). At 6h windows the constraint is on long dry runs
/// and the train-fit Σ doesn't extrapolate to those tails — the copula's
/// correlation skeleton fits the bulk of the joint distribution but not the
/// extreme-quantile structure.
///
/// Σ is window-independent. The train path fits one Σ per (station, lead) on
/// observed labels and writes the same correlation.json into all 3 window
/// bundles for that (station, lead).
///
/// Predict-time inputs read from 3a's live prediction parquet, same as 3g
/// (path documented in <see cref="DryWindow3gPredictor"/>).
/// </summary>
public static class DryWindow3jPredictor
{
    public const string Phase3j = "3j";

    /// <summary>Default MC sample count. 20,000 is 2× 3g's default — the
    /// copula path is slightly noisier per sample (one Φ evaluation + a 9-dim
    /// matrix-vector product per draw) so we trade a small predict-time hit
    /// for tighter Brier-estimate noise. At 20k samples per cell the Brier
    /// estimation noise is ≈ 0.0035, well below the ~0.005 absolute Brier
    /// lift 3j delivers at 3h windows over 3g.</summary>
    public const int DefaultMcSamples = 20_000;

    /// <summary>
    /// Single copula-MC pass yielding P(longest dry run ≥ L) for every window
    /// in <paramref name="windowHours"/>. All windows share the same correlated-
    /// Bernoulli draws per sample, preserving exact cross-window monotonicity
    /// P(L=3) ≥ P(L=4) ≥ P(L=6) — same structural guarantee as 3g.
    /// </summary>
    /// <param name="qHourly">Daytime hourly P(wet) from 3a, length matches the
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
                // P(U_h < q_h) = q_h marginally — so wet ↔ U_h < q_h.
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
    /// Standard normal CDF Φ(z) via the Abramowitz-Stegun 7.1.26 approximation
    /// for erf (max absolute error ≈ 1.5e-7). Sufficient precision for MC
    /// thresholding — the Bernoulli quantisation downstream wipes out any
    /// sub-ε-precision differences.
    /// </summary>
    public static double StandardNormalCdf(double z)
    {
        // Φ(z) = 0.5 * (1 + erf(z / √2))
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
    /// Cholesky decompose a real symmetric positive-definite matrix Σ into the
    /// lower-triangular L satisfying L L^T = Σ. Direct in-place 9×9 implementation
    /// — too small to justify pulling in MathNet. Throws if Σ isn't SPD (e.g.
    /// degenerate observed sequences where all 9 hours are perfectly correlated).
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
    /// Load per-lead Σ matrices from a 3j bundle's <c>correlation.json</c> and
    /// return their Cholesky factors L (recomputed at load time — cheap for
    /// 9×9, and keeps the file inspectable as plain symmetric matrices). One
    /// (station, lead) cell fits its own Σ on the slice of observed daytime
    /// binary sequences whose date range overlaps the lead-specific train
    /// slice; predict-time uses the lead's own Σ rather than a pooled one.
    ///
    /// Expected JSON shape:
    /// <code>
    /// {
    ///   "ByLead": {
    ///     "24": { "Sigma": [[...]] },
    ///     "48": { "Sigma": [[...]] },
    ///     "72": { "Sigma": [[...]] }
    ///   }
    /// }
    /// </code>
    /// </summary>
    public static Dictionary<int, double[,]> LoadCholeskyByLead(string bundleDir)
    {
        var path = Path.Combine(bundleDir, "correlation.json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"3j bundle missing correlation.json at {path}", path);
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        if (!doc.RootElement.TryGetProperty("ByLead", out var byLeadEl))
            throw new InvalidDataException(
                $"correlation.json at {path} has no 'ByLead' property.");
        if (byLeadEl.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException(
                $"correlation.json 'ByLead' at {path} is not an object.");

        var result = new Dictionary<int, double[,]>();
        foreach (var leadProp in byLeadEl.EnumerateObject())
        {
            if (!int.TryParse(leadProp.Name, NumberStyles.Integer, CultureInfo.InvariantCulture, out var lead))
                throw new InvalidDataException(
                    $"correlation.json 'ByLead' key '{leadProp.Name}' at {path} is not an integer.");
            if (!leadProp.Value.TryGetProperty("Sigma", out var sigmaEl)
                || sigmaEl.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException(
                    $"correlation.json 'ByLead.{lead}.Sigma' at {path} missing or not an array.");
            int n = sigmaEl.GetArrayLength();
            var sigma = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                var row = sigmaEl[i];
                if (row.ValueKind != JsonValueKind.Array || row.GetArrayLength() != n)
                    throw new InvalidDataException(
                        $"correlation.json 'ByLead.{lead}.Sigma' row {i} at {path} is not a length-{n} array.");
                for (int j = 0; j < n; j++) sigma[i, j] = row[j].GetDouble();
            }
            result[lead] = CholeskyDecompose(sigma);
        }
        return result;
    }

    /// <summary>
    /// Write per-lead Σ matrices to <c>correlation.json</c>. Sigma is the only
    /// persistent artefact for 3j; L is recomputed on load.
    /// </summary>
    public static void WriteCorrelationJson(string bundleDir, IReadOnlyDictionary<int, double[,]> sigmaByLead)
    {
        Directory.CreateDirectory(bundleDir);
        var path = Path.Combine(bundleDir, "correlation.json");
        var byLead = new Dictionary<string, object>();
        foreach (var (lead, sigma) in sigmaByLead.OrderBy(kv => kv.Key))
        {
            int n = sigma.GetLength(0);
            var rows = new double[n][];
            for (int i = 0; i < n; i++)
            {
                rows[i] = new double[n];
                for (int j = 0; j < n; j++) rows[i][j] = sigma[i, j];
            }
            byLead[lead.ToString(CultureInfo.InvariantCulture)] = new Dictionary<string, object> { ["Sigma"] = rows };
        }
        var payload = new Dictionary<string, object> { ["ByLead"] = byLead };
        File.WriteAllText(path,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }),
            System.Text.Encoding.UTF8);
    }

    /// <summary>
    /// Fit a 9×9 (or however many hours) Pearson correlation matrix from
    /// observed binary daytime sequences. Each row of <paramref name="binarySequences"/>
    /// is one day's daytime wet/dry indicators (length = number of daytime hours).
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
    /// Read the hourly observed wet/dry labels from a Phase 3a replay parquet.
    /// Sibling of <see cref="DryWindow3gPredictor.LoadReplayHourly"/> which
    /// pulls ProbWet; 3j needs the observed Label column to fit Σ on
    /// realised daytime binary sequences. The replay parquet stores
    /// hourly truth alongside the model's hourly prediction.
    /// </summary>
    public static Dictionary<DateTime, byte> LoadReplayLabelsHourly(
        string predictionsRoot, string stationSlug, string precip3aVersion, int leadHours)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation_replay", stationSlug, precip3aVersion, $"lead_{leadHours}h.parquet")
            .Replace('\\', '/');
        var sql = $"SELECT ValidTimeUtc, Label FROM read_parquet('{path}')";
        var result = new Dictionary<DateTime, byte>();
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var rdr = cmd.ExecuteReader();
        while (rdr.Read())
        {
            var t = rdr.GetDateTime(0);
            var t0 = DateTime.SpecifyKind(t, DateTimeKind.Utc);
            var rawLabel = rdr.GetValue(1);
            // Replay schema writes Label as bool (per PrecipReplayCommand);
            // tolerate int 0/1 from older bundles too.
            byte b = rawLabel switch
            {
                bool bv  => (byte)(bv ? 1 : 0),
                byte by  => by,
                short sh => (byte)(sh != 0 ? 1 : 0),
                int  iv  => (byte)(iv != 0 ? 1 : 0),
                long lv  => (byte)(lv != 0 ? 1 : 0),
                _ => throw new InvalidDataException(
                    $"Replay parquet Label column has unexpected type {rawLabel.GetType().Name}.")
            };
            result[t0] = b;
        }
        return result;
    }

    /// <summary>Local copy of 3g's Box-Muller sampler. Keeping it local rather
    /// than sharing across predictors makes 3j self-contained and removes the
    /// cross-class internal coupling that would otherwise be needed.</summary>
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
