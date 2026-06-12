using System.Globalization;
using System.Text;

namespace WeatherBlend.Evaluate.Waves;

/// <summary>
/// Markdown rendering for the wave verify rows. Mirrors the element reporter
/// (drift section first, per-version × lead-bucket headline table) with the
/// wave-specific band-coverage column and the buoy-distance caveat carried in
/// the header — every reader of an absolute MAE number needs the 28 km
/// location-mismatch context next to it.
/// </summary>
public static class WaveVerifyReporter
{
    public static string BuildMarkdown(
        string locationName,
        string buoySlug,
        string buoyDisplayName,
        DateTime asOfUtc,
        int windowDays,
        int truthLatencyDays,
        double coverageDriftFloor,
        IReadOnlyList<WaveVerifier.VerifyRow> rows,
        IReadOnlyDictionary<string, WaveVerifier.CalibrationReference> referenceByVersion)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;

        sb.AppendLine($"# Rolling verification report — significant wave height ({locationName})");
        sb.AppendLine();
        sb.AppendLine($"- As-of (UTC): {asOfUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Window: last {windowDays} days, excluding the last {truthLatencyDays} day(s)");
        sb.AppendLine($"- Truth: `{buoySlug}` ({buoyDisplayName}) hourly Hs — the buoy sits ~28 km W of the");
        sb.AppendLine("  pinned forecast cell, so absolute MAE carries a location mismatch by design");
        sb.AppendLine("  (~0.3 m of the level is the cell↔buoy gap alone — read the trend, not the level).");
        sb.AppendLine("  Realtime rows are non-QC'd and get silently upgraded by later archive re-pulls;");
        sb.AppendLine("  the rolling window re-scores them on the next run.");
        sb.AppendLine($"- Drift contract: COVERAGE-ONLY — flags when the empirical 90%-band coverage drops below {coverageDriftFloor.ToString("0.00", ci)}.");
        sb.AppendLine("  There is no MAE-threshold drift: the only training-time MAE reference (cal_mae, vs era5_ocean)");
        sb.AppendLine("  sits under the buoy's location-mismatch floor, so a multiplier threshold would alarm permanently.");
        sb.AppendLine("  cal_mae is shown for context only.");
        sb.AppendLine("- Lead buckets: 24h-wide ACTUAL-lead slices labelled by upper edge (24 = leads 0-23h, 48 = 24-47h, …).");
        sb.AppendLine("  The model is any-lead v1 — there are no trained per-lead buckets to stratify by.");
        sb.AppendLine("- era5_ocean is the TRAINING truth and is deliberately not verified against here — the buoy is the point.");
        sb.AppendLine();

        if (rows.Count == 0)
        {
            sb.AppendLine("No predictions in window with matching buoy truth. Nothing to verify.");
            return sb.ToString();
        }

        var drifting = rows.Where(r => r.DriftFlag).ToList();
        sb.AppendLine("## Drift flags");
        sb.AppendLine();
        if (drifting.Count == 0)
        {
            sb.AppendLine("None — every (version, lead-bucket) cell is within threshold.");
        }
        else
        {
            foreach (var r in drifting)
            {
                sb.AppendLine(
                    $"- **`{r.ModelVersion}` ≤{r.LeadBucketHours}h:** 90% band coverage " +
                    $"{Fmt(r.BandCoverage, ci)} is well below nominal 0.90 " +
                    $"(floor {coverageDriftFloor.ToString("0.00", ci)}), n={r.NBand}.");
            }
        }
        sb.AppendLine();

        sb.AppendLine("## Per-version × actual-lead bucket");
        sb.AppendLine();
        sb.AppendLine("| Version | Lead bucket | N | Hs MAE (m) | Bias | RMSE | Band coverage (n) | Ref cal MAE | Drift |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---|");
        foreach (var r in rows)
        {
            sb.AppendLine(
                $"| `{r.ModelVersion}` | ≤{r.LeadBucketHours}h | {r.N} | " +
                $"{Fmt(r.BlendMae, ci)} | " +
                $"{FmtBias(r.BlendBias, ci)} | " +
                $"{Fmt(r.BlendRmse, ci)} | " +
                $"{Fmt(r.BandCoverage, ci)} ({r.NBand}) | " +
                $"{FmtNullable(r.ReferenceCalMae, ci)} | " +
                $"{(r.DriftFlag ? "**YES**" : "no")} |");
        }
        sb.AppendLine();

        // Pooled all-lead line per version — the single headline number for
        // an any-lead model, next to its calibration-slice expectations.
        sb.AppendLine("## Pooled (all leads) per version");
        sb.AppendLine();
        sb.AppendLine("| Version | N | Hs MAE (m) | Bias | Band coverage (n) | Cal MAE | Cal coverage |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|");
        foreach (var g in rows.GroupBy(r => r.ModelVersion, StringComparer.Ordinal)
                     .OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var n = g.Sum(r => r.N);
            var mae = n == 0 ? double.NaN : g.Where(r => r.N > 0).Sum(r => r.BlendMae * r.N) / n;
            var bias = n == 0 ? double.NaN : g.Where(r => r.N > 0).Sum(r => r.BlendBias * r.N) / n;
            var nBand = g.Sum(r => r.NBand);
            var cov = nBand == 0
                ? double.NaN
                : g.Where(r => r.NBand > 0).Sum(r => r.BandCoverage * r.NBand) / nBand;
            referenceByVersion.TryGetValue(g.Key, out var reference);
            sb.AppendLine(
                $"| `{g.Key}` | {n} | {Fmt(mae, ci)} | {FmtBias(bias, ci)} | " +
                $"{Fmt(cov, ci)} ({nBand}) | {FmtNullable(reference?.CalMae, ci)} | " +
                $"{FmtNullable(reference?.CalBandCoverage, ci)} |");
        }
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
