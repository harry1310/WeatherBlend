using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.DryWindow;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3d-calibrated: post-hoc isotonic calibration of the Phase 3b dry-window
/// classifiers. For each (station, window-length) composite we rebuild the same
/// per-lead training rows the trainer saw, redo the chronological split, re-score
/// the validation partition with the saved 3b model, fit a PAV calibrator against
/// the val labels, and write a new version directory under the same composite key
/// that reuses the 3b model zips + climatology + feature schema while adding a
/// <c>calibration.json</c>.
///
/// The new version registers as a challenger: appended to the composite's Active
/// list (champion/challenger pattern shared with Phase 3a_isotonic) so predict +
/// verify score it alongside 3b on every cycle. Predict dispatches on
/// <c>metadata.Phase == "3d-calibrated"</c> and applies the lead-specific
/// calibrator after the raw LightGBM score.
///
/// The on-disk calibration container reuses <see cref="PrecipIsotonicCalibration"/>
/// — its shape (per-lead key → <see cref="IsotonicCalibrator"/> + provenance) is
/// identical for the dry-window case. Naming is slightly misleading but a
/// dedicated DryWindowIsotonicCalibration sibling would be a copy-paste alias
/// with no behaviour difference.
/// </summary>
public sealed class DryWindowCalibrateCommand
{
    private readonly ILogger<DryWindowCalibrateCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

    public DryWindowCalibrateCommand(ILogger<DryWindowCalibrateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string truthStation, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var composites = ResolveComposites(modelsRoot, truthStation);
        if (composites.Count == 0)
        {
            _log.LogError("No dry-window blender artefacts found under {Dir}. Train Phase 3b first.",
                Path.Combine(modelsRoot, "dry_window"));
            return 2;
        }

        var anyFail = false;
        foreach (var compositeKey in composites)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await CalibrateCompositeAsync(modelsRoot, compositeKey, ct);
            if (rc != 0) anyFail = true;
        }

        return anyFail ? 3 : 0;
    }

    private async Task<int> CalibrateCompositeAsync(string modelsRoot, string compositeKey, CancellationToken ct)
    {
        _log.LogInformation("=== Composite '{Key}' ===", compositeKey);

        if (!TryParseComposite(compositeKey, out var stationSlug, out var windowHours))
        {
            _log.LogError("Manifest entry '{Key}' is not in <station>/window_<N>h form — skipping.", compositeKey);
            return 2;
        }

        var sourceVersion = FindPhase3bVersion(modelsRoot, compositeKey);
        if (sourceVersion is null)
        {
            _log.LogError("Composite {Key}: no Phase 3b version found in history; nothing to calibrate.", compositeKey);
            return 2;
        }
        var sourceDir = Path.Combine(modelsRoot, "dry_window", compositeKey, sourceVersion);
        _log.LogInformation("Source 3b version: {V} (window={W}h)", sourceVersion, windowHours);

        var friendly = ResolveFriendlyStationName(stationSlug);
        if (friendly is null)
        {
            _log.LogError("Station slug '{Slug}' has no matching rainfall config entry. Known: [{Known}]",
                stationSlug, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
            return 2;
        }

        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "dry_window", compositeKey, now, suffix: "phase3d_calibrated");
        var versionName = Path.GetFileName(versionDir);
        Directory.CreateDirectory(versionDir);

        var ml = new MLContext(seed: 42);
        var calibration = new PrecipIsotonicCalibration
        {
            SourceVersion = sourceVersion,
            FitAtUtc = now,
        };

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var totalValRows = 0;
        var sourceMetadata = ModelArtifact.LoadTrainingMetadata(sourceDir);

        foreach (var lead in DefaultLeads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = DryWindowFeatureBuilder.BuildSpec(
                _cfg.Blenders, lead, DryWindowFeatureBuilder.Phase3b);
            var rows = DryWindowFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                friendly,
                spec,
                windowHours,
                ct);
            if (rows.Count < 100)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to calibrate.", rows.Count, lead);
                return 3;
            }

            var ds = DryWindowDataset.Split(rows);
            _log.LogInformation("Split → train {TN}, val {VN} (pos {VP} / {Vp:P1}), test {EN}",
                ds.Train.Count, ds.Val.Count, ds.ValPositives,
                ds.Val.Count == 0 ? 0 : (double)ds.ValPositives / ds.Val.Count,
                ds.Test.Count);

            var model = ModelArtifact.LoadLeadModel(ml, sourceDir, lead, out _);
            var rawVal = DryWindowTrainer.PredictVectorProbability(ml, model, spec, ds.Val);
            var truthVal = ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray();

            var calibrator = IsotonicCalibrator.Fit(rawVal, truthVal);
            calibration.ByLead[lead.ToString()] = calibrator;
            totalValRows += ds.Val.Count;

            // Calibrated test-set Brier so the metadata has a headline figure
            // comparable to the 3b BlendTestMae column.
            var rawTest = DryWindowTrainer.PredictVectorProbability(ml, model, spec, ds.Test);
            var calTest = rawTest.Select(calibrator.Apply).ToArray();
            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var rawBrier = PrecipMetrics.Brier(rawTest, truthTest);
            var calBrier = PrecipMetrics.Brier(calTest, truthTest);
            var climPred = DryWindowBaselines.Climatology(ds.Train, ds.Test, windowHours);
            var climBrier = PrecipMetrics.Brier(climPred, truthTest);
            var bss = PrecipMetrics.BrierSkillScore(calBrier, climBrier);

            var calBinary = calTest.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
            var fbias = PrecipMetrics.FrequencyBias(calBinary, truthTest);

            // Copy the source lead zip — model unchanged, only its output is remapped.
            File.Copy(
                Path.Combine(sourceDir, ModelArtifact.LeadModelFileName(lead)),
                Path.Combine(versionDir, ModelArtifact.LeadModelFileName(lead)),
                overwrite: true);

            var testMonths = ds.Test.Select(r => new DateTime(r.TargetDateUtc.Year, r.TargetDateUtc.Month, 1))
                                    .Distinct().Count();

            perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
            {
                LeadHours = lead,
                DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd}Z → {ds.TrainEnd:yyyy-MM-dd}Z",
                DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd}Z → {ds.ValEnd:yyyy-MM-dd}Z",
                DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd}Z → {ds.TestEnd:yyyy-MM-dd}Z",
                TrainRows = ds.Train.Count,
                ValRows   = ds.Val.Count,
                TestRows  = ds.Test.Count,
                TestCalendarMonths = testMonths,
                BestSingle = "",         // baselines unchanged from 3b — not re-run
                BestSingleValMae = 0.0,
                BlendTestMae = calBrier, // calibrated test Brier (column reuse matches 3b)
                BlendTestRmse = climBrier,
                BlendTestBias = fbias,
            };

            _log.LogInformation(
                "Lead {Lead}h — raw Brier={Raw:0.0000} → cal Brier={Cal:0.0000} (Δ {D:+0.0000;-0.0000;0.0000}), BSS={Bss:+0.0000;-0.0000;0.0000}, freq_bias={Fb:0.00}, knots={K}",
                lead, rawBrier, calBrier, calBrier - rawBrier, bss, fbias, calibrator.Knots.Count);
        }

        calibration.SourceRowCount = totalValRows;
        calibration.SaveTo(Path.Combine(versionDir, PrecipIsotonicCalibration.FileName));

        // Copy the rest of the 3b artefact alongside the new calibration so predict
        // can reach climatology + feature schema through the same resolver.
        CopyIfExists(sourceDir, versionDir, "dry_window_climatology.json");
        CopyIfExists(sourceDir, versionDir, ModelArtifact.FeatureSchemaFileName);
        CopyIfExists(sourceDir, versionDir, ModelArtifact.FeatureImportanceFileName);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "dry_window",
            Phase = DryWindowFeatureBuilder.Phase3dCalibrated,
            DataSource = sourceMetadata.DataSource,
            TrainedAtUtc = now,
            Hyperparameters = new Dictionary<string, object>
            {
                ["method"] = "pool-adjacent-violators (PAV) isotonic regression",
                ["source_version"] = sourceVersion,
                ["source_phase"] = sourceMetadata.Phase,
                ["window_hours"] = windowHours,
                ["fit_data"] = "validation slice (middle 15% of chronological 70/15/15 split)",
                ["interpolation"] = "linear between pool means, clamped at endpoints",
                ["fit_rows_per_lead"] = perLead.ToDictionary(kv => kv.Key, kv => (object)kv.Value.ValRows),
                ["knots_per_lead"] = calibration.ByLead.ToDictionary(kv => kv.Key, kv => (object)kv.Value.Knots.Count),
            },
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Not a retrained blender: reuses the Phase 3b LightGBM model zips unchanged and adds a monotone isotonic post-map per lead. The calibration.json holds the pool-adjacent-violators knots; predict applies the lead's map after scoring.",
                "Fit on the validation slice (ds.Val) — training rows would over-fit the map because the 3b model has already seen them. Test rows are held out for Brier reporting only.",
                "Calibration container reuses PrecipIsotonicCalibration — same shape, no separate dry-window class. Documented in DryWindowCalibrateCommand class header.",
                "PerLeadStats fields repurposed (matches 3b convention): BlendTestMae=calibrated-test Brier, BlendTestRmse=climatology Brier, BlendTestBias=frequency bias at p=0.5. BestSingle/BestSingleValMae left blank (3b's baselines apply unchanged).",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

        // Register as a challenger: append + extend Active. Drop any stale
        // 3d-calibrated entry so each calibrate run leaves at most one current.
        ModelArtifact.AppendStationVersion(modelsRoot, "dry_window", compositeKey, versionName);
        var existing = ModelArtifact.ResolveStationActive(modelsRoot, "dry_window", compositeKey);
        var nextActive = existing
            .Where(v => !v.Contains("phase3d_calibrated", StringComparison.Ordinal))
            .Append(versionName)
            .ToList();
        ModelArtifact.SetStationActive(modelsRoot, "dry_window", compositeKey, nextActive);

        _log.LogInformation("Phase 3d-calibrated artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for {Key} now: [{Active}]", compositeKey, string.Join(", ", nextActive));

        await Task.CompletedTask;
        return 0;
    }

    /// <summary>
    /// Look back through a composite's version history for the most recent entry
    /// whose training metadata declares Phase == "3b". We avoid picking 3d-shape
    /// or another calibrated version even if Current happens to point there.
    /// </summary>
    private static string? FindPhase3bVersion(string modelsRoot, string compositeKey)
    {
        var dir = Path.Combine(modelsRoot, "dry_window", compositeKey);
        if (!Directory.Exists(dir)) return null;

        var dirs = Directory.GetDirectories(dir)
            .Select(Path.GetFileName)
            .Where(n => n is not null)
            .Cast<string>()
            .OrderByDescending(n => n, StringComparer.Ordinal)
            .ToArray();

        foreach (var name in dirs)
        {
            var metaPath = Path.Combine(dir, name, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metaPath)) continue;
            try
            {
                var meta = ModelArtifact.LoadTrainingMetadata(Path.Combine(dir, name));
                if (string.Equals(meta.Phase, DryWindowFeatureBuilder.Phase3b, StringComparison.OrdinalIgnoreCase))
                    return name;
            }
            catch
            {
                // Malformed metadata — skip.
            }
        }
        return null;
    }

    /// <summary>Composite manifest key shape: <c>ea_&lt;slug&gt;/window_&lt;N&gt;h</c>.</summary>
    private static bool TryParseComposite(string key, out string stationSlug, out int windowHours)
    {
        stationSlug = ""; windowHours = 0;
        var parts = key.Split('/');
        if (parts.Length != 2) return false;
        if (!parts[1].StartsWith("window_") || !parts[1].EndsWith("h")) return false;
        var middle = parts[1]["window_".Length..^1];
        if (!int.TryParse(middle, out var w)) return false;
        stationSlug = parts[0];
        windowHours = w;
        return true;
    }

    private IReadOnlyList<string> ResolveComposites(string modelsRoot, string truthStation)
    {
        var all = ModelArtifact.ListStations(modelsRoot, "dry_window");
        if (string.Equals(truthStation, "all", StringComparison.OrdinalIgnoreCase))
            return all;

        var match = "ea_" + Slugify(truthStation);
        var picked = all.Where(k => k.StartsWith(match + "/", StringComparison.OrdinalIgnoreCase)
                                 || k.StartsWith(truthStation + "/", StringComparison.OrdinalIgnoreCase))
                        .ToArray();
        if (picked.Length == 0)
            _log.LogError("Truth-station '{S}' matches no manifest composite. Known: [{Known}]",
                truthStation, string.Join(", ", all));
        return picked;
    }

    private string? ResolveFriendlyStationName(string stationSlug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
        {
            var slug = "ea_" + Slugify(s.Name);
            if (string.Equals(slug, stationSlug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        }
        return null;
    }

    private static void CopyIfExists(string sourceDir, string destDir, string fileName)
    {
        var src = Path.Combine(sourceDir, fileName);
        if (File.Exists(src))
            File.Copy(src, Path.Combine(destDir, fileName), overwrite: true);
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
