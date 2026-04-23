using System.Globalization;
using System.Text;
using WeatherBlend.Train;

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
        IReadOnlyList<Verifier.VerifyRow> rows,
        IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata>? metadataByVersion = null)
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

        // Champion/challenger delta: when both a Phase 2b (lean, 13 features) and a
        // Phase 2c (rich, ~88 features) version have rows at the same lead, show the
        // MAE difference. Negative → rich wins. Only rendered when both phases are
        // present; otherwise the section is silently skipped.
        if (metadataByVersion is not null && metadataByVersion.Count > 0)
        {
            var phaseByVersion = metadataByVersion.ToDictionary(
                kv => kv.Key,
                kv => kv.Value.Phase ?? "",
                StringComparer.Ordinal);

            var byLead = rows
                .Where(r => phaseByVersion.TryGetValue(r.ModelVersion, out var p) && (p == "2b" || p == "2c"))
                .GroupBy(r => r.LeadHours)
                .OrderBy(g => g.Key)
                .Select(g => new
                {
                    Lead = g.Key,
                    Lean = g.FirstOrDefault(r => phaseByVersion[r.ModelVersion] == "2b"),
                    Rich = g.FirstOrDefault(r => phaseByVersion[r.ModelVersion] == "2c"),
                })
                .Where(x => x.Lean is not null && x.Rich is not null)
                .ToList();

            if (byLead.Count > 0)
            {
                sb.AppendLine("## Champion vs challenger (Phase 2b lean vs 2c rich)");
                sb.AppendLine();
                sb.AppendLine("| Lead | Lean version | Rich version | Lean MAE | Rich MAE | Δ MAE (rich−lean) | Lean bias | Rich bias | Winner |");
                sb.AppendLine("|---:|---|---|---:|---:|---:|---:|---:|---|");
                foreach (var row in byLead)
                {
                    var lean = row.Lean!;
                    var rich = row.Rich!;
                    var delta = rich.BlendMae - lean.BlendMae;
                    var winner = double.IsNaN(delta)
                        ? "—"
                        : delta < -0.001 ? "rich"
                        : delta >  0.001 ? "lean"
                        : "tie";
                    sb.AppendLine(
                        $"| {row.Lead}h | `{lean.ModelVersion}` | `{rich.ModelVersion}` | " +
                        $"{Fmt(lean.BlendMae, ci)} | {Fmt(rich.BlendMae, ci)} | " +
                        $"{FmtSignedDelta(delta, ci)} | " +
                        $"{FmtBias(lean.BlendBias, ci)} | {FmtBias(rich.BlendBias, ci)} | " +
                        $"{winner} |");
                }
                sb.AppendLine();
            }
        }

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

    private static string FmtSignedDelta(double v, CultureInfo ci)
        => double.IsNaN(v) ? "—" : v.ToString("+0.000;-0.000;0.000", ci);
}
