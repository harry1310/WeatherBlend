using System.Globalization;
using System.Text;

namespace WeatherBlend.Evaluate.RainfallAmount;

/// <summary>
/// Renders <see cref="RainfallAmountVerifier.VerifyRow"/> output as a markdown
/// report for the twice-weekly rainfall_amount verify run. Same shape as
/// <see cref="Evaluate.Precip.PrecipVerifyReporter"/> (drift block up front, then
/// detail table) so the GH App's <c>[ci-fail] verify</c> issue auto-filer
/// recognises it without target-specific parsing. Distributional-only columns
/// (Coverage80, PIT mean, exceedance Brier per threshold) live in a second
/// table below the headline so the row count of the primary table doesn't
/// explode horizontally.
/// </summary>
public static class RainfallAmountVerifyReporter
{
    public static string BuildMarkdown(
        DateTime asOfUtc,
        int windowDays,
        int latencyDays,
        double driftThreshold,
        IReadOnlyList<RainfallAmountVerifier.VerifyRow> rows)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;

        sb.AppendLine("# Rolling rainfall_amount verification report");
        sb.AppendLine();
        sb.AppendLine($"- As-of (UTC): {asOfUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Window: last {windowDays} days, excluding the last {latencyDays} days (EA provisional-reading buffer)");
        sb.AppendLine($"- Drift threshold: {driftThreshold.ToString("0.00", ci)}× training-test CRPS");
        sb.AppendLine($"- Wet threshold (MAE_wet stratification): ≥{RainfallAmountVerifier.WetThresholdMm.ToString("0.0", ci)} mm/hour");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No predictions in window with matching EA rainfall truth. Nothing to verify.");
            return sb.ToString();
        }

        // ----- Drift block -----
        var drifting = rows.Where(r => r.DriftFlag).ToList();
        sb.AppendLine("## Drift flags");
        sb.AppendLine();
        if (drifting.Count == 0)
        {
            sb.AppendLine("None — every (station, version, lead) bucket is within threshold.");
        }
        else
        {
            foreach (var r in drifting)
            {
                var ratio = r.ReferenceTestCrps is > 0
                    ? (r.BlendCrps / r.ReferenceTestCrps.Value).ToString("0.00", ci)
                    : "—";
                sb.AppendLine(
                    $"- **`{r.TruthStation}` / `{r.ModelVersion}` lead {r.LeadHours}h:** " +
                    $"rolling blend CRPS {Fmt(r.BlendCrps, ci)} is " +
                    $"{ratio}× training test CRPS " +
                    $"({(r.ReferenceTestCrps.HasValue ? Fmt(r.ReferenceTestCrps.Value, ci) : "—")}), n={r.N} (wet={r.WetN}).");
            }
        }
        sb.AppendLine();

        // ----- Primary table: CRPS + point-skill -----
        sb.AppendLine("## Per-station × version × lead — skill headline");
        sb.AppendLine();
        sb.AppendLine("| Station | Version | Lead | N | Wet | Wet rate | Blend CRPS | MAE_wet | Ref test CRPS | Drift |");
        sb.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---:|---|");
        foreach (var r in rows)
        {
            sb.AppendLine(
                $"| `{r.TruthStation}` | `{r.ModelVersion}` | {r.LeadHours}h | {r.N} | {r.WetN} | " +
                $"{FmtRate(r.WetRate, ci)} | " +
                $"{Fmt(r.BlendCrps, ci)} | " +
                $"{Fmt(r.MaeWet, ci)} | " +
                $"{FmtNullable(r.ReferenceTestCrps, ci)} | " +
                $"{(r.DriftFlag ? "**YES**" : "no")} |");
        }
        sb.AppendLine();

        // ----- Calibration table: coverage + PIT mean + per-threshold Brier -----
        sb.AppendLine("## Calibration + exceedance reliability");
        sb.AppendLine();
        var thresholds = RainfallAmountVerifier.ExceedanceThresholdsMm;
        var thresholdCols = string.Join(" | ",
            thresholds.Select(t => $"Brier ≥{RainfallAmountVerifier.FormatThresholdKey(t)}mm"));
        var thresholdDashes = string.Join(" | ", thresholds.Select(_ => "---:"));
        sb.AppendLine($"| Station | Version | Lead | Coverage 80% | PIT mean | {thresholdCols} |");
        sb.AppendLine($"|---|---|---:|---:|---:|{thresholdDashes}|");
        foreach (var r in rows)
        {
            var cells = thresholds.Select(t =>
                r.ExceedanceBriers.TryGetValue(RainfallAmountVerifier.FormatThresholdKey(t), out var b)
                    ? Fmt(b, ci) : "—");
            sb.AppendLine(
                $"| `{r.TruthStation}` | `{r.ModelVersion}` | {r.LeadHours}h | " +
                $"{FmtRate(r.Coverage80, ci)} | " +
                $"{Fmt(r.PitMean, ci)} | " +
                $"{string.Join(" | ", cells)} |");
        }
        sb.AppendLine();

        // ----- PIT histograms inline so on-call sees calibration shape -----
        sb.AppendLine("## PIT histograms (10 bins on [0,1] — flat = well-calibrated)");
        sb.AppendLine();
        foreach (var r in rows)
        {
            sb.AppendLine($"- **`{r.TruthStation}` / `{r.ModelVersion}` lead {r.LeadHours}h** " +
                $"(N={r.N}): `{string.Join(",", r.PitBins)}`");
        }
        sb.AppendLine();

        return sb.ToString();
    }

    private static string Fmt(double v, CultureInfo ci) =>
        double.IsNaN(v) || double.IsInfinity(v) ? "—" : v.ToString("0.0000", ci);

    private static string FmtNullable(double? v, CultureInfo ci) =>
        v.HasValue ? Fmt(v.Value, ci) : "—";

    private static string FmtRate(double v, CultureInfo ci) =>
        double.IsNaN(v) ? "—" : v.ToString("0.000", ci);
}
