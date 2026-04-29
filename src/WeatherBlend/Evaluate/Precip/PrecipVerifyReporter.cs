using System.Globalization;
using System.Text;

namespace WeatherBlend.Evaluate.Precip;

/// <summary>
/// Renders <see cref="PrecipVerifier.VerifyRow"/> output as a markdown report for
/// the weekly precipitation verify run. Structure parallels the temperature
/// <see cref="Evaluate.TempVerifyReporter"/>: drift flags up front, then a detail
/// table, then reliability bins per (station, version, lead) for on-call eyeballing.
/// </summary>
public static class PrecipVerifyReporter
{
    public static string BuildMarkdown(
        DateTime asOfUtc,
        int windowDays,
        int latencyDays,
        double driftThreshold,
        IReadOnlyList<PrecipVerifier.VerifyRow> rows)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;

        sb.AppendLine("# Rolling precipitation verification report");
        sb.AppendLine();
        sb.AppendLine($"- As-of (UTC): {asOfUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Window: last {windowDays} days, excluding the last {latencyDays} days (EA provisional-reading buffer)");
        sb.AppendLine($"- Drift threshold: {driftThreshold.ToString("0.00", ci)}× training-test Brier");
        sb.AppendLine($"- Wet threshold: ≥{PrecipVerifier.WetThresholdMm.ToString("0.0", ci)} mm/hour");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No predictions in window with matching EA rainfall truth. Nothing to verify.");
            return sb.ToString();
        }

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
                var ratio = r.ReferenceTestBrier is > 0
                    ? (r.BlendBrier / r.ReferenceTestBrier.Value).ToString("0.00", ci)
                    : "—";
                sb.AppendLine(
                    $"- **`{r.TruthStation}` / `{r.ModelVersion}` lead {r.LeadHours}h:** " +
                    $"rolling blend Brier {Fmt(r.BlendBrier, ci)} is " +
                    $"{ratio}× training test Brier " +
                    $"({Fmt(r.ReferenceTestBrier!.Value, ci)}), n={r.N} (wet={r.WetN}).");
            }
        }
        sb.AppendLine();

        sb.AppendLine("## Per-station × version × lead");
        sb.AppendLine();
        sb.AppendLine("| Station | Version | Lead | N | Wet | Wet rate | Blend Brier | Clim Brier | BSS | Mean-of-models Brier | Best single (window) | Best single Brier | Persistence Brier | Freq bias @0.5 | Ref test Brier | Drift |");
        sb.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|");
        foreach (var r in rows)
        {
            sb.AppendLine(
                $"| `{r.TruthStation}` | `{r.ModelVersion}` | {r.LeadHours}h | {r.N} | {r.WetN} | " +
                $"{FmtRate(r.WetRate, ci)} | " +
                $"{Fmt(r.BlendBrier, ci)} | " +
                $"{Fmt(r.ClimBrier, ci)} | " +
                $"{FmtBss(r.Bss, ci)} | " +
                $"{Fmt(r.MeanOfModelsBrier, ci)} | " +
                $"{(string.IsNullOrEmpty(r.BestSingleName) ? "—" : $"`{r.BestSingleName}`")} | " +
                $"{Fmt(r.BestSingleBrier, ci)} | " +
                $"{FmtNullable(r.PersistenceBrier, ci)} | " +
                $"{Fmt(r.FrequencyBiasAt05, ci)} | " +
                $"{FmtNullable(r.ReferenceTestBrier, ci)} | " +
                $"{(r.DriftFlag ? "**YES**" : "no")} |");
        }
        sb.AppendLine();

        // Reliability diagrams per bucket. Ten bins is noisy for small n but this is a
        // weekly spot-check, not a journal plot — the numbers call out obvious mis-calibration.
        sb.AppendLine("## Reliability bins");
        sb.AppendLine();
        foreach (var r in rows)
        {
            sb.AppendLine($"### `{r.TruthStation}` / `{r.ModelVersion}` lead {r.LeadHours}h");
            sb.AppendLine();
            sb.AppendLine("| Forecast bin | Mean forecast | Observed freq | N |");
            sb.AppendLine("|---|---:|---:|---:|");
            foreach (var bin in r.Reliability)
            {
                var range = $"[{bin.BinLower.ToString("0.0", ci)}–{bin.BinUpper.ToString("0.0", ci)})";
                sb.AppendLine(
                    $"| {range} | {Fmt(bin.MeanForecast, ci)} | {Fmt(bin.ObservedFrequency, ci)} | {bin.N} |");
            }
            sb.AppendLine();
        }

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

    private static string FmtRate(double v, CultureInfo ci)
        => double.IsNaN(v) ? "—" : v.ToString("0.00%", ci);

    private static string FmtBss(double v, CultureInfo ci)
        => double.IsNaN(v) ? "—" : v.ToString("+0.000;-0.000;0.000", ci);

    private static string FmtNullable(double? v, CultureInfo ci)
        => v.HasValue ? Fmt(v.Value, ci) : "—";
}
