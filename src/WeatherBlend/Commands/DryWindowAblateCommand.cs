using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3d diagnostic driver. Reads the latest 3b / 3d-shape / 3d-calibrated
/// training_metadata.json for every composite (station, window) entry in the
/// dry_window manifest and emits a side-by-side comparison report under
/// data/reports/phase3d_vs_3b_{ts}.md.
///
/// Read-only: no models are retrained, no manifests are touched. When a phase
/// is missing for a given (station, window) — most commonly because 3d-shape
/// or 3d-calibrated have not been trained yet — the corresponding column is
/// rendered as "—" so the table stays interpretable in partial-rollout states.
///
/// Shape-feature gain importance is read from the 3d-shape version dir's
/// feature_importance.json (already saved at training time). The seven 3d
/// shape features are highlighted so it is easy to see whether the new tier
/// is doing useful work versus the 53 base features.
/// </summary>
public sealed class DryWindowAblateCommand
{
    private readonly ILogger<DryWindowAblateCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly IReadOnlyList<string> ShapeFeatureNames = DryWindowFeatureBuilder.ShapeFeatureNames;

    public DryWindowAblateCommand(ILogger<DryWindowAblateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var manifestPath = Path.Combine(modelsRoot, "dry_window", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            _log.LogError("No dry_window manifest at {Path}. Train first.", manifestPath);
            return 2;
        }

        var compositeKeys = ModelArtifact.ListStations(modelsRoot, "dry_window")
            .OrderBy(k => k, StringComparer.Ordinal)
            .ToList();
        if (compositeKeys.Count == 0)
        {
            _log.LogError("Manifest contains no station entries.");
            return 2;
        }

        var comparisons = new List<CompositeComparison>();
        foreach (var key in compositeKeys)
        {
            ct.ThrowIfCancellationRequested();
            var (slug, windowHours) = ParseCompositeKey(key);
            if (slug is null) { _log.LogWarning("Could not parse composite key '{Key}'; skipping.", key); continue; }

            var phase3b           = TryLoadLatestMetadataForPhase(modelsRoot, key, DryWindowFeatureBuilder.Phase3b);
            var phase3dShape      = TryLoadLatestMetadataForPhase(modelsRoot, key, DryWindowFeatureBuilder.Phase3dShape);
            var phase3dCalibrated = TryLoadLatestMetadataForPhase(modelsRoot, key, DryWindowFeatureBuilder.Phase3dCalibrated);

            if (phase3b is null && phase3dShape is null && phase3dCalibrated is null)
            {
                _log.LogWarning("No metadata for any phase under {Key}; skipping.", key);
                continue;
            }

            var shapeImportance = phase3dShape is { } shape
                ? LoadShapeImportance(modelsRoot, key, shape.Version)
                : new Dictionary<int, IReadOnlyList<(string Name, double Gain)>>();

            comparisons.Add(new CompositeComparison(
                CompositeKey: key,
                StationSlug: slug,
                WindowHours: windowHours,
                Phase3b: phase3b,
                Phase3dShape: phase3dShape,
                Phase3dCalibrated: phase3dCalibrated,
                ShapeImportance: shapeImportance));
        }

        if (comparisons.Count == 0)
        {
            _log.LogError("No comparisons produced.");
            return 3;
        }

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(
            _cfg.Storage.ReportsPath,
            $"phase3d_vs_3b_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
        await File.WriteAllTextAsync(outPath, BuildReport(comparisons), ct);
        _log.LogInformation("Report → {Path}", outPath);
        return 0;
    }

    // ---- Composite-key + metadata loading -----------------------------------

    private sealed record CompositeComparison(
        string CompositeKey,
        string StationSlug,
        int WindowHours,
        ModelArtifact.TrainingMetadata? Phase3b,
        ModelArtifact.TrainingMetadata? Phase3dShape,
        ModelArtifact.TrainingMetadata? Phase3dCalibrated,
        IReadOnlyDictionary<int, IReadOnlyList<(string Name, double Gain)>> ShapeImportance);

    private (string? Slug, int WindowHours) ParseCompositeKey(string key)
    {
        var m = Regex.Match(key, @"^(?<slug>[^/]+)/window_(?<w>\d+)h$");
        if (!m.Success) return (null, 0);
        return (m.Groups["slug"].Value, int.Parse(m.Groups["w"].Value, CultureInfo.InvariantCulture));
    }

    private ModelArtifact.TrainingMetadata? TryLoadLatestMetadataForPhase(
        string modelsRoot, string compositeKey, string phase)
    {
        var stationDir = Path.Combine(modelsRoot, "dry_window", compositeKey).Replace('\\', '/');
        if (!Directory.Exists(stationDir)) return null;
        foreach (var versionDir in Directory.GetDirectories(stationDir).OrderByDescending(d => d, StringComparer.Ordinal))
        {
            var metaPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metaPath)) continue;
            try
            {
                var meta = ModelArtifact.LoadTrainingMetadata(versionDir);
                if (string.Equals(meta.Phase, phase, StringComparison.OrdinalIgnoreCase))
                    return meta;
            }
            catch (Exception ex)
            {
                _log.LogWarning("Failed to load {Path}: {Msg}", metaPath, ex.Message);
            }
        }
        return null;
    }

    private IReadOnlyDictionary<int, IReadOnlyList<(string Name, double Gain)>> LoadShapeImportance(
        string modelsRoot, string compositeKey, string version)
    {
        try
        {
            var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "dry_window", compositeKey, version);
            return ModelArtifact.LoadPerLeadFeatureImportance(versionDir);
        }
        catch (Exception ex)
        {
            _log.LogWarning("No feature importance for {Key} v{V}: {Msg}", compositeKey, version, ex.Message);
            return new Dictionary<int, IReadOnlyList<(string Name, double Gain)>>();
        }
    }

    // ---- Markdown report ----------------------------------------------------

    private static string BuildReport(List<CompositeComparison> comparisons)
    {
        var sb = new StringBuilder();
        var ts = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss'Z'", CultureInfo.InvariantCulture);
        sb.AppendLine("# Phase 3d vs Phase 3b — Dry-Window Blender");
        sb.AppendLine();
        sb.AppendLine($"Generated {ts}");
        sb.AppendLine();
        sb.AppendLine("Same hyperparameters, same chronological train/val/test split, same EA rainfall truth and 4-of-4 hourly gate. ");
        sb.AppendLine("Two changes are isolated against the 3b baseline:");
        sb.AppendLine();
        sb.AppendLine("- **3b** (baseline) — 53 features: per-model precip/dry-window aggregates, spread stats, ensemble-mean covariates, calendar encodings.");
        sb.AppendLine("- **3d-shape** — 3b's 53 features + 7 within-day shape features computed from ensemble-mean hourly forecasts (`first_wet_hour`, `last_wet_hour`, longest dry/wet run, `n_rain_events`, morning + afternoon precip sums).");
        sb.AppendLine("- **3d-calibrated** — 3b's saved model with post-hoc isotonic (PAV) regression fitted on the validation partition. Identical features and pre-calibration probabilities, only the mapping changes.");
        sb.AppendLine();
        sb.AppendLine("Numbers are read directly from each artefact's `training_metadata.json` (`PerLead.BlendTestMae` → Brier, `BlendTestRmse` → climatology Brier, `BlendTestBias` → frequency bias). Empty cells mean a phase has not been trained yet for that (station, window).");
        sb.AppendLine();

        foreach (var c in comparisons)
        {
            sb.AppendLine($"## {c.StationSlug} · window {c.WindowHours}h");
            sb.AppendLine();
            sb.AppendLine($"- 3b artefact: {ArtefactCell(c.Phase3b)}");
            sb.AppendLine($"- 3d-shape artefact: {ArtefactCell(c.Phase3dShape)}");
            sb.AppendLine($"- 3d-calibrated artefact: {ArtefactCell(c.Phase3dCalibrated)}");
            sb.AppendLine();

            sb.AppendLine("### Test-set Brier by lead");
            sb.AppendLine();
            sb.AppendLine("| Lead | 3b Brier | 3d-shape Brier | Δ shape − 3b | 3d-cal Brier | Δ cal − 3b | Climatology Brier |");
            sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|");
            foreach (var lead in Leads)
            {
                var bBaseline = Lead(c.Phase3b, lead);
                var bShape    = Lead(c.Phase3dShape, lead);
                var bCal      = Lead(c.Phase3dCalibrated, lead);
                var clim      = bBaseline?.BlendTestRmse ?? bShape?.BlendTestRmse ?? bCal?.BlendTestRmse;

                var deltaShape = (bShape?.BlendTestMae is double s && bBaseline?.BlendTestMae is double b1) ? s - b1 : (double?)null;
                var deltaCal   = (bCal?.BlendTestMae   is double k && bBaseline?.BlendTestMae is double b2) ? k - b2 : (double?)null;

                sb.Append("| ").Append(lead).Append("h | ");
                sb.Append(Fmt(bBaseline?.BlendTestMae, "0.0000")).Append(" | ");
                sb.Append(Fmt(bShape?.BlendTestMae,    "0.0000")).Append(" | ");
                sb.Append(Fmt(deltaShape,              "+0.0000;-0.0000;0.0000")).Append(" | ");
                sb.Append(Fmt(bCal?.BlendTestMae,      "0.0000")).Append(" | ");
                sb.Append(Fmt(deltaCal,                "+0.0000;-0.0000;0.0000")).Append(" | ");
                sb.Append(Fmt(clim,                    "0.0000"));
                sb.AppendLine(" |");
            }
            sb.AppendLine();

            sb.AppendLine("### BSS by lead (vs month-keyed climatology)");
            sb.AppendLine();
            sb.AppendLine("| Lead | 3b BSS | 3d-shape BSS | 3d-cal BSS |");
            sb.AppendLine("|---|---:|---:|---:|");
            foreach (var lead in Leads)
            {
                var bBaseline = Lead(c.Phase3b, lead);
                var bShape    = Lead(c.Phase3dShape, lead);
                var bCal      = Lead(c.Phase3dCalibrated, lead);
                sb.Append("| ").Append(lead).Append("h | ");
                sb.Append(Fmt(Bss(bBaseline), "+0.0000;-0.0000;0.0000")).Append(" | ");
                sb.Append(Fmt(Bss(bShape),    "+0.0000;-0.0000;0.0000")).Append(" | ");
                sb.Append(Fmt(Bss(bCal),      "+0.0000;-0.0000;0.0000"));
                sb.AppendLine(" |");
            }
            sb.AppendLine();

            sb.AppendLine("### Frequency bias at p≥0.5");
            sb.AppendLine();
            sb.AppendLine("| Lead | 3b | 3d-shape | 3d-cal |");
            sb.AppendLine("|---|---:|---:|---:|");
            foreach (var lead in Leads)
            {
                sb.Append("| ").Append(lead).Append("h | ");
                sb.Append(Fmt(Lead(c.Phase3b, lead)?.BlendTestBias,            "0.00")).Append(" | ");
                sb.Append(Fmt(Lead(c.Phase3dShape, lead)?.BlendTestBias,       "0.00")).Append(" | ");
                sb.Append(Fmt(Lead(c.Phase3dCalibrated, lead)?.BlendTestBias,  "0.00"));
                sb.AppendLine(" |");
            }
            sb.AppendLine();

            // Shape-feature gain importance — only meaningful when 3d-shape exists
            // and feature_importance.json was saved at training time.
            if (c.ShapeImportance.Count > 0)
            {
                sb.AppendLine("### Shape-feature gain importance (3d-shape, per lead)");
                sb.AppendLine();
                sb.AppendLine("Rank within the 60-feature gain-sorted list and absolute gain. Lower rank = the shape feature contributes more than most of the 53 baseline features.");
                sb.AppendLine();
                sb.AppendLine("| Lead | Shape feature | Rank | Gain | Top-N share |");
                sb.AppendLine("|---|---|---:|---:|---:|");
                foreach (var lead in Leads)
                {
                    if (!c.ShapeImportance.TryGetValue(lead, out var imp) || imp.Count == 0) continue;
                    var totalGain = imp.Sum(t => t.Gain);
                    var ranked = imp
                        .Select((t, i) => (Rank: i + 1, t.Name, t.Gain))
                        .Where(t => ShapeFeatureNames.Contains(t.Name, StringComparer.OrdinalIgnoreCase))
                        .OrderBy(t => t.Rank)
                        .ToList();
                    if (ranked.Count == 0)
                    {
                        sb.Append("| ").Append(lead).AppendLine("h | (no shape features in importance list) | — | — | — |");
                        continue;
                    }
                    var shapeShare = totalGain > 0 ? ranked.Sum(t => t.Gain) / totalGain : 0;
                    foreach (var (i, (rank, name, gain)) in ranked.Select((t, i) => (i, t)))
                    {
                        var shareCell = i == 0 ? $"{shapeShare:0.00%}" : "";
                        sb.Append("| ").Append(lead).Append("h | `").Append(name).Append("` | ")
                          .Append(rank).Append(" | ")
                          .Append(gain.ToString("0.0", CultureInfo.InvariantCulture)).Append(" | ")
                          .Append(shareCell).AppendLine(" |");
                    }
                }
                sb.AppendLine();
            }
        }

        sb.AppendLine("## Honest interpretation");
        sb.AppendLine();
        sb.AppendLine("The Phase 3a → 3a_isotonic experiment showed that PAV calibration alone moved Brier by ≈0 — the 3a probabilities were already well-calibrated on EA truth. The same headline applies to 3d-calibrated by construction: it is a strict reweighting of the 3b output, so any Brier delta has to come from miscalibration the validation partition was able to expose. Treat 3d-calibrated as risk insurance, not a skill increase.");
        sb.AppendLine();
        sb.AppendLine("3d-shape adds new information (within-day rain structure isn't recoverable from 3b's daily aggregates). If the 7 shape features land in the upper third of the gain-importance list and Δ Brier is meaningfully negative at 24h, the tier earns its keep. If shape features rank near the bottom and Δ ≈ 0, it's evidence that the 4-of-4 daily aggregation already captures the within-day signal — same lesson as 3a/3c/isotonic where extra signals didn't survive contact with the held-out test partition.");
        sb.AppendLine();
        sb.AppendLine("All rows share the same hyperparameters (`{iter:600, lr:0.04, leaves:31, minLeaf:40, L1:0.1, L2:0.1, esr:40, seed:42}`) and the same 70/15/15 chronological split.");
        return sb.ToString();
    }

    private static string ArtefactCell(ModelArtifact.TrainingMetadata? meta)
        => meta is null ? "**missing** — phase not yet trained for this composite" : $"`{meta.Version}` (trained {meta.TrainedAtUtc:yyyy-MM-dd HH:mm}Z)";

    private static ModelArtifact.PerLeadStats? Lead(ModelArtifact.TrainingMetadata? meta, int lead)
    {
        if (meta is null) return null;
        return meta.PerLead.TryGetValue(lead.ToString(CultureInfo.InvariantCulture), out var v) ? v : null;
    }

    private static double? Bss(ModelArtifact.PerLeadStats? p)
    {
        if (p is null || p.BlendTestRmse <= 0) return null;
        return 1.0 - p.BlendTestMae / p.BlendTestRmse;
    }

    private static string Fmt(double? v, string format)
    {
        if (!v.HasValue || double.IsNaN(v.Value)) return "—";
        return v.Value.ToString(format, CultureInfo.InvariantCulture);
    }
}
