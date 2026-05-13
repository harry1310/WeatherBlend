using System.Globalization;
using System.Text.Json;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3n — regime-conditioned Gaussian copula MC over Phase 3a's hourly
/// P(wet). Same per-hour MC mechanics as 3j (Z ~ N(0, Σ) → Φ → threshold
/// against q), but Σ is one of TWO matrices per (station, lead): Σ_settled
/// fit on train days where NWP-ensemble consensus was high, Σ_unsettled fit
/// on days where consensus was low. At predict time the day's NWP consensus
/// picks which Σ to use.
///
/// Pre-flight diagnostic (2026-05-13, Bellever lead 24) showed the two Σs
/// differ dramatically — pooled 3j Σ has mean off-diag ≈ 0.52, Σ_settled
/// ≈ 0.80 (atmosphere locked in, hours all correlated), Σ_unsettled ≈ 0.35
/// (showery/transitional, weaker block structure). Frobenius norm of the
/// difference is 4.04, max abs entry 0.70 — far past the "go" threshold.
///
/// Bundle layout adds ByLead.{N}.Sigma_settled / Sigma_unsettled / Threshold:
/// <code>
/// {
///   "ByLead": {
///     "24": {
///       "Sigma_settled":   [[..]],
///       "Sigma_unsettled": [[..]],
///       "Threshold":       0.889,
///       "DaysSettled":     150,
///       "DaysUnsettled":   146
///     }
///   }
/// }
/// </code>
///
/// Bake-off scores 3n by computing the day's agreement and routing each
/// test row through the correct Σ — same data flow as live predict.
/// </summary>
public static class DryWindow3nPredictor
{
    public const string Phase3n = "3n";

    /// <summary>Inherits 3j's 20k default since the math is identical
    /// modulo Σ selection.</summary>
    public const int DefaultMcSamples = DryWindow3jPredictor.DefaultMcSamples;

    /// <summary>Per-(station, lead) regime artefact: both Σs, their
    /// Cholesky factors, and the agreement threshold that splits them.</summary>
    public sealed record RegimeBundle(
        double[,] SigmaSettled,
        double[,] SigmaUnsettled,
        double[,] CholeskySettled,
        double[,] CholeskyUnsettled,
        double Threshold,
        int DaysSettled,
        int DaysUnsettled);

    /// <summary>
    /// Run copula MC with the regime-selected Σ for the given day's agreement.
    /// Falls back to Σ_settled if agreement is NaN (extremely rare —
    /// unclassifiable day — would only happen if forecast tree is too sparse).
    /// </summary>
    public static Dictionary<int, double> ProbDryWindow(
        double[] qHourly,
        RegimeBundle bundle,
        double dayAgreement,
        IReadOnlyList<int> windowHours,
        Random rng,
        int nSamples = DefaultMcSamples)
    {
        var L = SelectCholesky(bundle, dayAgreement);
        return DryWindow3jPredictor.ProbDryWindow(qHourly, L, windowHours, rng, nSamples);
    }

    /// <summary>Single-window convenience.</summary>
    public static double ProbDryWindow(
        double[] qHourly,
        RegimeBundle bundle,
        double dayAgreement,
        int windowLength,
        Random rng,
        int nSamples = DefaultMcSamples)
        => ProbDryWindow(qHourly, bundle, dayAgreement, new[] { windowLength }, rng, nSamples)[windowLength];

    /// <summary>
    /// Pick the Cholesky factor matching the day's regime. Agreement ≥
    /// threshold → settled; otherwise → unsettled. NaN agreement (degenerate
    /// case where the live forecast tree didn't have enough models) falls
    /// back to settled — the safer choice if we can't classify.
    /// </summary>
    public static double[,] SelectCholesky(RegimeBundle bundle, double dayAgreement)
    {
        if (double.IsNaN(dayAgreement) || dayAgreement >= bundle.Threshold)
            return bundle.CholeskySettled;
        return bundle.CholeskyUnsettled;
    }

    /// <summary>
    /// Load all per-lead regime bundles from a 3n bundle's
    /// <c>correlation.json</c>. Both Σs are Cholesky-decomposed at load time.
    /// </summary>
    public static Dictionary<int, RegimeBundle> LoadByLead(string bundleDir)
    {
        var path = Path.Combine(bundleDir, "correlation.json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"3n bundle missing correlation.json at {path}", path);
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        if (!doc.RootElement.TryGetProperty("ByLead", out var byLeadEl))
            throw new InvalidDataException(
                $"correlation.json at {path} has no 'ByLead' property.");

        var result = new Dictionary<int, RegimeBundle>();
        foreach (var leadProp in byLeadEl.EnumerateObject())
        {
            if (!int.TryParse(leadProp.Name, NumberStyles.Integer, CultureInfo.InvariantCulture, out var lead))
                throw new InvalidDataException(
                    $"correlation.json 'ByLead' key '{leadProp.Name}' at {path} is not an integer.");

            var sigmaSettled   = ParseMatrix(leadProp.Value, "Sigma_settled",   path, lead);
            var sigmaUnsettled = ParseMatrix(leadProp.Value, "Sigma_unsettled", path, lead);
            var threshold = leadProp.Value.GetProperty("Threshold").GetDouble();
            int daysSettled   = leadProp.Value.TryGetProperty("DaysSettled",   out var s) ? s.GetInt32() : 0;
            int daysUnsettled = leadProp.Value.TryGetProperty("DaysUnsettled", out var u) ? u.GetInt32() : 0;

            var lSettled   = DryWindow3jPredictor.CholeskyDecompose(sigmaSettled);
            var lUnsettled = DryWindow3jPredictor.CholeskyDecompose(sigmaUnsettled);
            result[lead] = new RegimeBundle(
                sigmaSettled, sigmaUnsettled, lSettled, lUnsettled,
                threshold, daysSettled, daysUnsettled);
        }
        return result;
    }

    private static double[,] ParseMatrix(JsonElement el, string name, string path, int lead)
    {
        if (!el.TryGetProperty(name, out var matEl) || matEl.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException(
                $"correlation.json 'ByLead.{lead}.{name}' at {path} missing or not an array.");
        int n = matEl.GetArrayLength();
        var m = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            var row = matEl[i];
            if (row.ValueKind != JsonValueKind.Array || row.GetArrayLength() != n)
                throw new InvalidDataException(
                    $"correlation.json 'ByLead.{lead}.{name}' row {i} at {path} is not a length-{n} array.");
            for (int j = 0; j < n; j++) m[i, j] = row[j].GetDouble();
        }
        return m;
    }

    /// <summary>
    /// Write per-lead regime bundles to <c>correlation.json</c>. Σs and
    /// threshold are the persistent artefacts; Choleskys are recomputed
    /// at load time.
    /// </summary>
    public static void WriteCorrelationJson(
        string bundleDir,
        IReadOnlyDictionary<int, (double[,] SigmaSettled, double[,] SigmaUnsettled, double Threshold, int DaysSettled, int DaysUnsettled)> byLead)
    {
        Directory.CreateDirectory(bundleDir);
        var path = Path.Combine(bundleDir, "correlation.json");

        var byLeadJson = new Dictionary<string, object>();
        foreach (var (lead, b) in byLead.OrderBy(kv => kv.Key))
        {
            byLeadJson[lead.ToString(CultureInfo.InvariantCulture)] = new Dictionary<string, object>
            {
                ["Sigma_settled"]   = MatrixToJaggedArray(b.SigmaSettled),
                ["Sigma_unsettled"] = MatrixToJaggedArray(b.SigmaUnsettled),
                ["Threshold"]       = b.Threshold,
                ["DaysSettled"]     = b.DaysSettled,
                ["DaysUnsettled"]   = b.DaysUnsettled,
            };
        }
        var payload = new Dictionary<string, object> { ["ByLead"] = byLeadJson };
        File.WriteAllText(path,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }),
            System.Text.Encoding.UTF8);
    }

    private static double[][] MatrixToJaggedArray(double[,] m)
    {
        int n = m.GetLength(0);
        var rows = new double[n][];
        for (int i = 0; i < n; i++)
        {
            rows[i] = new double[n];
            for (int j = 0; j < n; j++) rows[i][j] = m[i, j];
        }
        return rows;
    }
}
