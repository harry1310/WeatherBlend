using System.Globalization;
using System.Text;
using WeatherBlend.Train;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Evaluate.Element;

/// <summary>
/// Markdown rendering for element verify rows. Mirrors the temperature
/// <c>TempVerifyReporter</c> shape (drift section first, per-version × lead headline
/// table) plus per-element stratification (month / truth quintile).
/// </summary>
public static class ElementVerifyReporter
{
    public static string BuildMarkdown(
        ElementTarget target,
        DateTime asOfUtc,
        int windowDays,
        int era5LatencyDays,
        double driftThreshold,
        IReadOnlyList<ElementVerifier.VerifyRow> rows,
        IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata>? metadataByVersion = null)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;
        var u = target.Units;

        sb.AppendLine($"# Rolling verification report — {target.Display}");
        sb.AppendLine();
        sb.AppendLine($"- As-of (UTC): {asOfUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Window: last {windowDays} days, excluding the last {era5LatencyDays} days (ERA5 release latency)");
        sb.AppendLine($"- Drift threshold: {driftThreshold.ToString("0.00", ci)}× training-test MAE");
        sb.AppendLine($"- Element: `{target.CliName}` (truth = ERA5 at Bonehill grid cell)");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No predictions in window with matching ERA5 truth. Nothing to verify.");
            return sb.ToString();
        }

        var drifting = rows.Where(r => r.DriftFlag).ToList();
        sb.AppendLine("## Drift flags");
        sb.AppendLine();
        if (drifting.Count == 0)
        {
            sb.AppendLine("None — every (version, lead) bucket is within threshold.");
        }
        else
        {
            foreach (var r in drifting)
            {
                var ratio = r.ReferenceTestMae is > 0
                    ? (r.BlendMae / r.ReferenceTestMae.Value).ToString("0.00", ci)
                    : "—";
                sb.AppendLine(
                    $"- **`{r.ModelVersion}` lead {r.LeadHours}h:** rolling blend MAE " +
                    $"{r.BlendMae.ToString("0.000", ci)} {u} is {ratio}× training test MAE " +
                    $"({r.ReferenceTestMae!.Value.ToString("0.000", ci)} {u}), n={r.N}.");
            }
        }
        sb.AppendLine();

        sb.AppendLine("## Per-version × lead");
        sb.AppendLine();
        sb.AppendLine($"| Version | Lead | N | Blend MAE ({u}) | Blend bias | Blend RMSE | Mean-of-models MAE | Best single (window) | Best single MAE | Ref test MAE | Drift |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|");
        foreach (var r in rows)
        {
            sb.AppendLine(
                $"| `{r.ModelVersion}` | {r.LeadHours}h | {r.N} | " +
                $"{Fmt(r.BlendMae, ci)} | " +
                $"{FmtBias(r.BlendBias, ci)} | " +
                $"{Fmt(r.BlendRmse, ci)} | " +
                $"{Fmt(r.MeanMae, ci)} | " +
                $"{(string.IsNullOrEmpty(r.BestSingleName) ? "—" : $"`{r.BestSingleName}`")} | " +
                $"{Fmt(r.BestSingleMae, ci)} | " +
                $"{FmtNullable(r.ReferenceTestMae, ci)} | " +
                $"{(r.DriftFlag ? "**YES**" : "no")} |");
        }
        sb.AppendLine();

        sb.AppendLine("## Stratified MAE by month");
        sb.AppendLine();
        foreach (var r in rows)
        {
            sb.AppendLine($"### `{r.ModelVersion}` lead {r.LeadHours}h");
            sb.AppendLine();
            sb.AppendLine($"| Month | N | MAE ({u}) |");
            sb.AppendLine("|---:|---:|---:|");
            foreach (var (m, val) in r.MaeByMonth.OrderBy(kv => kv.Key))
                sb.AppendLine($"| {m:00} | {val.N} | {Fmt(val.Mae, ci)} |");
            sb.AppendLine();
        }

        sb.AppendLine("## Stratified MAE by truth-value quintile");
        sb.AppendLine();
        sb.AppendLine("Quintile 0 = lowest 20% of observed truth values; quintile 4 = highest 20%. Useful for spotting tail-bias issues.");
        sb.AppendLine();
        foreach (var r in rows)
        {
            sb.AppendLine($"### `{r.ModelVersion}` lead {r.LeadHours}h");
            sb.AppendLine();
            sb.AppendLine($"| Quintile | N | MAE ({u}) |");
            sb.AppendLine("|---:|---:|---:|");
            foreach (var (q, val) in r.MaeByTruthQuintile.OrderBy(kv => kv.Key))
                sb.AppendLine($"| Q{q} | {val.N} | {Fmt(val.Mae, ci)} |");
            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("METAR secondary check is deferred to a follow-up phase: it requires per-element handling " +
                      "(RH from EGTE T+Td via Magnus for humidity, wind speed direct for wind, none possible for " +
                      "radiation; cloud cover absent from the current ObservationRow schema). Headline metric remains " +
                      "ERA5 MAE per the brief; secondary will be additive when wired.");
        sb.AppendLine();

        return sb.ToString();
    }

    private static string Fmt(double v, CultureInfo ci) =>
        double.IsNaN(v) ? "—" : v.ToString("0.000", ci);

    private static string FmtBias(double v, CultureInfo ci) =>
        double.IsNaN(v) ? "—" : v.ToString("+0.000;-0.000;0.000", ci);

    private static string FmtNullable(double? v, CultureInfo ci) =>
        v.HasValue ? Fmt(v.Value, ci) : "—";
}
