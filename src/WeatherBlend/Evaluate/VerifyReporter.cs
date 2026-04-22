using System.Globalization;
using System.Text;

namespace WeatherBlend.Evaluate;

/// <summary>
/// Renders <see cref="Verifier.VerifyRow"/> output as a markdown report. Separate from
/// the phase-2b <see cref="Reporter"/> because the verify report is what an on-call
/// human reads weekly — its structure prioritises drift flags over training diagnostics.
/// </summary>
public static class VerifyReporter
{
    public static string BuildMarkdown(
        DateTime asOfUtc,
        int windowDays,
        int era5LatencyDays,
        double driftThreshold,
        IReadOnlyList<Verifier.VerifyRow> rows)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;

        sb.AppendLine("# Rolling verification report");
        sb.AppendLine();
        sb.AppendLine($"- As-of (UTC): {asOfUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Window: last {windowDays} days, excluding the last {era5LatencyDays} days (ERA5 release latency)");
        sb.AppendLine($"- Drift threshold: {driftThreshold.ToString("0.00", ci)}× training-test MAE");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No predictions in window with matching ERA5 truth. Nothing to verify.");
            return sb.ToString();
        }

        // Drift flags up front — an on-call human reading this weekly wants to see
        // "anything wrong?" before any other number.
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
                    $"- **`{r.ModelVersion}` lead {r.LeadHours}h:** " +
                    $"rolling blend MAE {r.BlendMae.ToString("0.000", ci)}°C is " +
                    $"{ratio}× training test MAE " +
                    $"({r.ReferenceTestMae!.Value.ToString("0.000", ci)}°C), n={r.N}.");
            }
        }
        sb.AppendLine();

        sb.AppendLine("## Per-version × lead");
        sb.AppendLine();
        sb.AppendLine("| Version | Lead | N | Blend MAE | Blend bias | Blend RMSE | Mean-of-models MAE | Best single (window) | Best single MAE | Persistence MAE | Ref test MAE | Drift |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|");
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
                $"{FmtNullable(r.PersistenceMae, ci)} | " +
                $"{FmtNullable(r.ReferenceTestMae, ci)} | " +
                $"{(r.DriftFlag ? "**YES**" : "no")} |");
        }
        sb.AppendLine();

        // Persistence drop accounting — a sudden spike would hint at ERA5 gaps that
        // should be investigated separately.
        var totalDropped = rows.Sum(r => r.PersistenceDropped);
        if (totalDropped > 0)
        {
            sb.AppendLine($"_Persistence lookups without a truth value at t−lead: {totalDropped} rows._");
            sb.AppendLine();
        }

        return sb.ToString();
    }

    private static string Fmt(double v, CultureInfo ci)
        => double.IsNaN(v) ? "—" : v.ToString("0.000", ci);

    private static string FmtBias(double v, CultureInfo ci)
        => double.IsNaN(v) ? "—" : v.ToString("+0.000;-0.000;0.000", ci);

    private static string FmtNullable(double? v, CultureInfo ci)
        => v.HasValue ? Fmt(v.Value, ci) : "—";
}
