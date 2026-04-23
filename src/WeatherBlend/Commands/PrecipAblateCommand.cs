using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3c diagnostic driver. Combines two jobs into one report:
///   1. Read training_metadata.json for the currently-active 3a and 3c artefacts
///      and tabulate per-lead test-set Brier + BSS side by side.
///   2. At lead 24h only, retrain 3c variants with each feature tier dropped
///      (humidity / pressure / ea-persistence) so the contribution of each tier
///      is isolated. Variants are trained in memory — no artefacts are saved,
///      no manifests touched. Same hyperparameters as 3c so the only variable
///      is the feature-column list.
/// The report is written to data/reports/phase3c_vs_3a_{ts}.md. Running the
/// command is idempotent: no state outside the report file changes.
/// </summary>
public sealed class PrecipAblateCommand
{
    private readonly ILogger<PrecipAblateCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] Leads = { 24, 48, 72 };

    public PrecipAblateCommand(ILogger<PrecipAblateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured.");
            return 2;
        }

        var modelsRoot = Path.Combine("data", "models");
        var comparisons = new List<StationComparison>();

        foreach (var station in _cfg.Location.Rainfall.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var slug = "ea_" + Slugify(station.Name);
            _log.LogInformation("=== {Station} (slug={Slug}) ===", station.Name, slug);

            var phase3a = TryLoadLatestMetadata(modelsRoot, slug, "3a");
            var phase3c = TryLoadLatestMetadata(modelsRoot, slug, "3c");
            if (phase3c is null)
            {
                _log.LogWarning("No 3c metadata for station {Slug}; skipping.", slug);
                continue;
            }

            var ablation = RunAblationAt24h(station.Name, ct);
            comparisons.Add(new StationComparison(
                StationName: station.Name,
                StationSlug: slug,
                Phase3a: phase3a,
                Phase3c: phase3c,
                Ablation: ablation));
        }

        if (comparisons.Count == 0)
        {
            _log.LogError("No comparisons produced.");
            return 3;
        }

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(
            _cfg.Storage.ReportsPath,
            $"phase3c_vs_3a_{DateTime.UtcNow:yyyy-MM-dd_HHmmss}.md");
        await File.WriteAllTextAsync(outPath, BuildReport(comparisons), ct);
        _log.LogInformation("Report → {Path}", outPath);
        return 0;
    }

    // ---- Tier ablation at 24h -------------------------------------------------

    private sealed record AblationResult(
        string VariantName,
        int FeatureCount,
        double TestBrier,
        double ClimBrier,
        double Bss,
        double FreqBias);

    private List<AblationResult> RunAblationAt24h(string stationName, CancellationToken ct)
    {
        _log.LogInformation("Lead 24h — building rich dataset for ablation...");
        var rows = PrecipRichFeatureBuilder.BuildForLead(
            _cfg.Storage.ForecastsPath,
            _cfg.Storage.RainfallPath,
            _cfg.Location.Name,
            stationName,
            leadHours: 24,
            ct);
        if (rows.Count < 500)
        {
            _log.LogWarning("Only {N} rows — skipping ablation.", rows.Count);
            return new List<AblationResult>();
        }

        var ds = PrecipDataset.Split(rows.Cast<PrecipTrainingRow>().ToList());
        var train = ds.Train.Cast<RichPrecipTrainingRow>().ToList();
        var val   = ds.Val.Cast<RichPrecipTrainingRow>().ToList();
        var test  = ds.Test.Cast<RichPrecipTrainingRow>().ToList();

        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var allRich = PrecipOccurrenceTrainer.RichFeatureColumnInternalNames();
        var climPred = PrecipBaselines.Climatology(ds.Train, ds.Test);
        var truth = test.Select(r => r.WetBinary ? 1.0 : 0.0).ToArray();
        var climBrier = PrecipMetrics.Brier(climPred, truth);

        var variants = new (string Name, string[] Cols)[]
        {
            ("full_rich (lean+humidity+pressure+persistence)", allRich),
            ("drop_humidity",    Filter(allRich, HumidityPrefixes)),
            ("drop_pressure",    Filter(allRich, PressurePrefixes)),
            ("drop_ea_persistence", Filter(allRich, PersistencePrefixes)),
        };

        var results = new List<AblationResult>();
        foreach (var (name, cols) in variants)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("  variant {Name} — {N} features", name, cols.Length);
            var trained = PrecipOccurrenceTrainer.Train(train, val, cols, hp);
            var prob = PrecipOccurrenceTrainer.PredictProbability(trained.Ml, trained.Model, test);
            var brier = PrecipMetrics.Brier(prob, truth);
            var bss = PrecipMetrics.BrierSkillScore(brier, climBrier);
            var binary = prob.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
            var fbias = PrecipMetrics.FrequencyBias(binary, truth);
            _log.LogInformation(
                "    Brier={Brier:0.000} BSS={Bss:+0.000;-0.000;0.000} fbias={Fb:0.00}",
                brier, bss, fbias);
            results.Add(new AblationResult(name, cols.Length, brier, climBrier, bss, fbias));
        }
        return results;
    }

    // Each prefix matches the internal column-name prefix on RichPrecipTrainingRow.
    // Matching is case-insensitive, anchored at start of name.
    private static readonly string[] HumidityPrefixes   = { "Dew", "Rh", "DewDepression" };
    private static readonly string[] PressurePrefixes   = { "Pressure" };
    private static readonly string[] PersistencePrefixes = { "Ea" };

    private static string[] Filter(string[] cols, string[] dropPrefixes)
    {
        // DewDepressionMean / RhMean are part of the LEAN tier — drop only the per-model
        // columns (those have a 3-letter model suffix like Gfs/Mf). Leaning on the
        // property-name convention: per-model cols always end in one of the model suffixes.
        var modelSuffixes = new[] { "Gfs", "Ecmwf", "Icon", "Mf", "Ukmo", "Gem" };
        return cols.Where(c =>
        {
            foreach (var dp in dropPrefixes)
            {
                if (!c.StartsWith(dp, StringComparison.OrdinalIgnoreCase)) continue;
                // For humidity/pressure prefixes: only drop per-model variants.
                // For EA persistence: there are no "mean" Ea* lean cols, so drop every match.
                if (dp == "Ea") return false;
                if (modelSuffixes.Any(s => c.EndsWith(s, StringComparison.OrdinalIgnoreCase)))
                    return false;
            }
            return true;
        }).ToArray();
    }

    // ---- Phase 3a/3c metadata loading ----------------------------------------

    private sealed record StationComparison(
        string StationName,
        string StationSlug,
        ModelArtifact.TrainingMetadata? Phase3a,
        ModelArtifact.TrainingMetadata Phase3c,
        List<AblationResult> Ablation);

    private ModelArtifact.TrainingMetadata? TryLoadLatestMetadata(string modelsRoot, string slug, string phase)
    {
        var stationDir = Path.Combine(modelsRoot, "precipitation", slug);
        if (!Directory.Exists(stationDir)) return null;
        foreach (var versionDir in Directory.GetDirectories(stationDir).OrderByDescending(d => d))
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

    // ---- Markdown report ------------------------------------------------------

    private string BuildReport(List<StationComparison> comparisons)
    {
        var sb = new StringBuilder();
        var ts = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss'Z'", CultureInfo.InvariantCulture);
        sb.AppendLine("# Phase 3c vs Phase 3a — Precipitation Occurrence Blender");
        sb.AppendLine();
        sb.AppendLine($"Generated {ts}");
        sb.AppendLine();
        sb.AppendLine("Same hyperparameters, same chronological train/val/test split, same EA rainfall truth. ");
        sb.AppendLine("Feature richness is the isolated variable:");
        sb.AppendLine();
        sb.AppendLine("- **Phase 3a** (lean) — 27 features: per-model precip + precip-prob, spread stats, covariate means, calendar encodings.");
        sb.AppendLine("- **Phase 3c** (rich) — 55 features: lean + 18 per-model humidity (dew/RH/depression) + 6 per-model surface pressure + 4 EA-observation trailing-rainfall persistence features.");
        sb.AppendLine();

        // Per-station headline table.
        foreach (var c in comparisons)
        {
            sb.AppendLine($"## {c.StationName} (`{c.StationSlug}`)");
            sb.AppendLine();
            if (c.Phase3a is not null)
            {
                sb.AppendLine($"- 3a artefact: `{c.Phase3a.Version}`");
            }
            else
            {
                sb.AppendLine("- 3a artefact: **missing** — 3c scores stand alone for this station.");
            }
            sb.AppendLine($"- 3c artefact: `{c.Phase3c.Version}`");
            sb.AppendLine();

            sb.AppendLine("### Test-set Brier by lead");
            sb.AppendLine();
            sb.AppendLine("| Lead | 3a Brier | 3c Brier | Δ (3c − 3a) | Climatology Brier | 3a BSS | 3c BSS |");
            sb.AppendLine("|------|---------:|---------:|------------:|------------------:|-------:|-------:|");
            foreach (var lead in Leads)
            {
                var a = c.Phase3a?.PerLead.TryGetValue(lead.ToString(), out var av) == true ? av : null;
                var b = c.Phase3c.PerLead.TryGetValue(lead.ToString(), out var bv) ? bv : null;

                var aBrier  = a?.BlendTestMae;
                var bBrier  = b?.BlendTestMae;
                var climB   = a?.BlendTestRmse ?? b?.BlendTestRmse;
                var aBss = (aBrier.HasValue && climB.HasValue && climB.Value > 0) ? 1.0 - aBrier.Value / climB.Value : (double?)null;
                var bBss = (bBrier.HasValue && climB.HasValue && climB.Value > 0) ? 1.0 - bBrier.Value / climB.Value : (double?)null;
                var delta = (aBrier.HasValue && bBrier.HasValue) ? bBrier.Value - aBrier.Value : (double?)null;

                sb.Append("| ").Append(lead).Append("h ");
                sb.Append(" | ").Append(Fmt(aBrier, "0.000"));
                sb.Append(" | ").Append(Fmt(bBrier, "0.000"));
                sb.Append(" | ").Append(Fmt(delta, "+0.000;-0.000;0.000"));
                sb.Append(" | ").Append(Fmt(climB, "0.000"));
                sb.Append(" | ").Append(Fmt(aBss, "+0.000;-0.000;0.000"));
                sb.Append(" | ").Append(Fmt(bBss, "+0.000;-0.000;0.000"));
                sb.AppendLine(" |");
            }
            sb.AppendLine();

            sb.AppendLine("### Frequency bias at p≥0.5");
            sb.AppendLine();
            sb.AppendLine("| Lead | 3a | 3c |");
            sb.AppendLine("|------|---:|---:|");
            foreach (var lead in Leads)
            {
                var a = c.Phase3a?.PerLead.TryGetValue(lead.ToString(), out var av) == true ? av : null;
                var b = c.Phase3c.PerLead.TryGetValue(lead.ToString(), out var bv) ? bv : null;
                sb.Append("| ").Append(lead).Append("h | ")
                  .Append(Fmt(a?.BlendTestBias, "0.00"))
                  .Append(" | ")
                  .Append(Fmt(b?.BlendTestBias, "0.00"))
                  .AppendLine(" |");
            }
            sb.AppendLine();

            if (c.Ablation.Count > 0)
            {
                sb.AppendLine("### Feature-tier ablation @ 24h lead (in-memory retrain, same split)");
                sb.AppendLine();
                sb.AppendLine("| Variant | Features | Test Brier | Δ vs full | BSS | Freq bias |");
                sb.AppendLine("|---------|--------:|----------:|---------:|----:|---------:|");
                var full = c.Ablation.FirstOrDefault(a => a.VariantName.StartsWith("full"));
                foreach (var a in c.Ablation)
                {
                    var delta = full is null ? (double?)null : a.TestBrier - full.TestBrier;
                    sb.Append("| ").Append(a.VariantName)
                      .Append(" | ").Append(a.FeatureCount)
                      .Append(" | ").Append(Fmt(a.TestBrier, "0.000"))
                      .Append(" | ").Append(Fmt(delta, "+0.000;-0.000;0.000"))
                      .Append(" | ").Append(Fmt(a.Bss, "+0.000;-0.000;0.000"))
                      .Append(" | ").Append(Fmt(a.FreqBias, "0.00"))
                      .AppendLine(" |");
                }
                sb.AppendLine();
                sb.AppendLine("Δ values are in Brier points — negative = variant _improves_ over the full rich model, positive = dropping that tier hurts the model.");
                sb.AppendLine();
            }
        }

        sb.AppendLine("## Reading the numbers");
        sb.AppendLine();
        sb.AppendLine("- Brier is a strictly-proper probabilistic score — lower is better. Range on hourly P(wet) is roughly 0 (perfect) → 0.25 (uniform-50% forecast on balanced classes).");
        sb.AppendLine("- BSS = 1 − Brier / climatology_Brier. Positive means beats the (month, hour) climatology baseline; 0 means ties it; negative means worse than climatology.");
        sb.AppendLine("- Frequency bias near 1.0 is desired — under 1.0 means the classifier under-predicts wet hours at the 0.5 threshold; over 1.0 means it cries wolf.");
        sb.AppendLine("- The ablation retrains with each tier stripped. If a drop variant has Δ ≈ 0, that tier contributed little. Δ > 0 means the tier is load-bearing.");
        sb.AppendLine();
        sb.AppendLine("All rows share the same hyperparameters (`{iter:600, lr:0.04, leaves:31, minLeaf:40, L1:0.1, L2:0.1, esr:40, seed:42}`) and the same 70/15/15 chronological split.");
        return sb.ToString();
    }

    private static string Fmt(double? v, string format)
    {
        if (!v.HasValue || double.IsNaN(v.Value)) return "—";
        return v.Value.ToString(format, CultureInfo.InvariantCulture);
    }

    private static string Slugify(string name)
    {
        var chars = name.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();
        var slug = new string(chars);
        while (slug.Contains("__")) slug = slug.Replace("__", "_");
        return slug.Trim('_');
    }
}
