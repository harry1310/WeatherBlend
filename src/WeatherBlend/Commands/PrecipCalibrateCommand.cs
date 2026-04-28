using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3a_isotonic: post-hoc isotonic calibration of the Phase 3a (lean)
/// occurrence classifier. For each rainfall station we rebuild the same per-lead
/// training rows the trainer saw, redo the chronological split, re-score the
/// validation partition with the saved 3a model, fit a PAV calibrator against
/// the val labels, and write a new version directory that reuses the 3a model
/// zips + climatology + feature schema while adding a <c>calibration.json</c>.
///
/// The new version registers as a challenger: it's appended to the station's
/// Active list (champion/challenger pattern used by Phase 3c) so predict + verify
/// score it alongside 3a and 3c on every cycle. Predict dispatches on
/// <c>metadata.Phase == "3a_isotonic"</c> and applies the lead-specific calibrator
/// after the raw LightGBM score.
/// </summary>
public sealed class PrecipCalibrateCommand
{
    private readonly ILogger<PrecipCalibrateCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] DefaultLeads = { 24, 48, 72 };

    public PrecipCalibrateCommand(ILogger<PrecipCalibrateCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(string truthStation, CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var stationsToRun = ResolveStations(modelsRoot, truthStation);
        if (stationsToRun.Count == 0)
        {
            _log.LogError("No precipitation blender artefacts found under {Dir}. Train Phase 3a first.",
                Path.Combine(modelsRoot, "precipitation"));
            return 2;
        }

        var anyFail = false;
        foreach (var station in stationsToRun)
        {
            ct.ThrowIfCancellationRequested();
            var rc = await CalibrateStationAsync(modelsRoot, station, ct);
            if (rc != 0) anyFail = true;
        }

        return anyFail ? 3 : 0;
    }

    private async Task<int> CalibrateStationAsync(string modelsRoot, string stationSlug, CancellationToken ct)
    {
        _log.LogInformation("=== Station '{Station}' ===", stationSlug);

        var sourceVersion = FindPhase3aVersion(modelsRoot, stationSlug);
        if (sourceVersion is null)
        {
            _log.LogError("Station {Station}: no Phase 3a version found under its history; nothing to calibrate.", stationSlug);
            return 2;
        }
        var sourceDir = Path.Combine(modelsRoot, "precipitation", stationSlug, sourceVersion);
        _log.LogInformation("Source 3a version: {V}", sourceVersion);

        var friendly = ResolveFriendlyStationName(stationSlug);
        if (friendly is null)
        {
            _log.LogError("Station {Station}: cannot resolve rainfall config name. Known: [{Known}]",
                stationSlug, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
            return 2;
        }

        var now = DateTime.UtcNow;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "precipitation", stationSlug, now, suffix: "phase3a_isotonic");
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

            var spec = PrecipFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var rows = PrecipFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.RainfallPath,
                _cfg.Location.Name,
                friendly,
                spec,
                ct);
            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to calibrate.", rows.Count, lead);
                return 3;
            }

            var ds = BinaryDataset.Split(rows);
            _log.LogInformation("Split → train {TN}, val {VN} (wet {VW} / {VP:P1}), test {EN}",
                ds.Train.Count, ds.Val.Count, ds.ValWet,
                ds.Val.Count == 0 ? 0 : (double)ds.ValWet / ds.Val.Count,
                ds.Test.Count);

            var model = ModelArtifact.LoadLeadModel(ml, sourceDir, lead, out _);
            var rawVal = PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, ds.Val);
            var truthVal = ds.Val.Select(r => r.Label ? 1.0 : 0.0).ToArray();

            var calibrator = IsotonicCalibrator.Fit(rawVal, truthVal);
            calibration.ByLead[lead.ToString()] = calibrator;
            totalValRows += ds.Val.Count;

            // Report calibrated test-set Brier so the metadata has a headline
            // figure comparable to 3a/3c's BlendTestMae field.
            var rawTest = PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, ds.Test);
            var calTest = rawTest.Select(calibrator.Apply).ToArray();
            var truthTest = ds.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
            var rawBrier = PrecipMetrics.Brier(rawTest, truthTest);
            var calBrier = PrecipMetrics.Brier(calTest, truthTest);
            var climPred = PrecipBaselines.Climatology(ds.Train, ds.Test);
            var climBrier = PrecipMetrics.Brier(climPred, truthTest);
            var bss = PrecipMetrics.BrierSkillScore(calBrier, climBrier);

            var calBinary = calTest.Select(p => p >= 0.5 ? 1.0 : 0.0).ToArray();
            var fbias = PrecipMetrics.FrequencyBias(calBinary, truthTest);

            // Copy the source lead zip — the model is unchanged, only its output is remapped.
            File.Copy(
                Path.Combine(sourceDir, ModelArtifact.LeadModelFileName(lead)),
                Path.Combine(versionDir, ModelArtifact.LeadModelFileName(lead)),
                overwrite: true);

            var testMonths = ds.Test.Select(r => new DateTime(r.ValidTimeUtc.Year, r.ValidTimeUtc.Month, 1))
                                    .Distinct().Count();

            perLead[lead.ToString()] = new ModelArtifact.PerLeadStats
            {
                LeadHours = lead,
                DataRangeTrain = $"{ds.TrainStart:yyyy-MM-dd HH:mm}Z → {ds.TrainEnd:yyyy-MM-dd HH:mm}Z",
                DataRangeVal   = $"{ds.ValStart:yyyy-MM-dd HH:mm}Z → {ds.ValEnd:yyyy-MM-dd HH:mm}Z",
                DataRangeTest  = $"{ds.TestStart:yyyy-MM-dd HH:mm}Z → {ds.TestEnd:yyyy-MM-dd HH:mm}Z",
                TrainRows = ds.Train.Count,
                ValRows   = ds.Val.Count,
                TestRows  = ds.Test.Count,
                TestCalendarMonths = testMonths,
                BestSingle = "",             // not computed here — same baselines as 3a apply
                BestSingleValMae = 0.0,      // unused (would be NaN conceptually; JSON doesn't allow NaN)
                BlendTestMae = calBrier,     // calibrated-test Brier, same column reuse as 3a
                BlendTestRmse = climBrier,
                BlendTestBias = fbias,
            };

            _log.LogInformation(
                "Lead {Lead}h — raw Brier={Raw:0.000} → calibrated Brier={Cal:0.000} (Δ {D:+0.000;-0.000;0.000}), BSS={Bss:+0.000;-0.000;0.000}, freq_bias={Fb:0.00}, knots={K}",
                lead, rawBrier, calBrier, calBrier - rawBrier, bss, fbias, calibrator.Knots.Count);
        }

        calibration.SourceRowCount = totalValRows;
        calibration.SaveTo(Path.Combine(versionDir, PrecipIsotonicCalibration.FileName));

        // Copy the rest of the 3a artefact alongside the new calibration so predict
        // can reach it through the same resolver as a native 3a version.
        CopyIfExists(sourceDir, versionDir, ModelArtifact.ClimatologyFileName);
        CopyIfExists(sourceDir, versionDir, ModelArtifact.FeatureSchemaFileName);
        CopyIfExists(sourceDir, versionDir, ModelArtifact.FeatureImportanceFileName);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "precipitation",
            Phase = "3a_isotonic",
            DataSource = sourceMetadata.DataSource,
            TrainedAtUtc = now,
            Hyperparameters = new Dictionary<string, object>
            {
                ["method"] = "pool-adjacent-violators (PAV) isotonic regression",
                ["source_version"] = sourceVersion,
                ["source_phase"] = sourceMetadata.Phase,
                ["fit_data"] = "validation slice (middle 15% of chronological 70/15/15 split)",
                ["interpolation"] = "linear between pool means, clamped at endpoints",
                ["fit_rows_per_lead"] = perLead.ToDictionary(kv => kv.Key, kv => (object)kv.Value.ValRows),
                ["knots_per_lead"] = calibration.ByLead.ToDictionary(kv => kv.Key, kv => (object)kv.Value.Knots.Count),
            },
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_brier", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Not a retrained blender: reuses the Phase 3a LightGBM model zips unchanged and adds a monotone isotonic post-map per lead. The calibration.json holds the pool-adjacent-violators knots; predict applies the lead's map after scoring.",
                "Fit on the validation slice (ds.Val). The training slice can't be used — the model has already seen those examples, so its predictions there are over-confident and the fitted map would under-correct. The test slice is held for Brier reporting only.",
                "PerLeadStats fields repurposed (same convention as Phase 3a): BlendTestMae=calibrated-test Brier, BlendTestRmse=climatology Brier, BlendTestBias=frequency bias at p=0.5. BestSingle/BestSingleValMae left blank — re-running the single-model baseline here would just duplicate 3a's numbers.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);

        // Register as a challenger: preserve existing Active list + append the new version.
        ModelArtifact.AppendStationVersion(modelsRoot, "precipitation", stationSlug, versionName);
        var existing = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);
        var newActive = existing.Concat(new[] { versionName }).Distinct().ToList();
        ModelArtifact.SetStationActive(modelsRoot, "precipitation", stationSlug, newActive);

        _log.LogInformation("Phase 3a_isotonic artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions for station {Station} now: [{Active}]", stationSlug, string.Join(", ", newActive));

        await Task.CompletedTask;
        return 0;
    }

    /// <summary>
    /// Look back through a station's version history for the first entry whose
    /// training metadata declares Phase == "3a". We prefer that over the Current
    /// pointer because Current may be overwritten by later trainings (see the
    /// Hexworthy incident).
    /// </summary>
    private string? FindPhase3aVersion(string modelsRoot, string stationSlug)
    {
        var stationDir = Path.Combine(modelsRoot, "precipitation", stationSlug);
        if (!Directory.Exists(stationDir)) return null;

        // Iterate directories in reverse chronological order so we pick the most
        // recent 3a version if multiple exist (rare but possible if the user
        // retrained).
        var dirs = Directory.GetDirectories(stationDir)
            .Select(Path.GetFileName)
            .Where(n => n is not null)
            .Cast<string>()
            .OrderByDescending(n => n, StringComparer.Ordinal)
            .ToArray();

        foreach (var name in dirs)
        {
            var metaPath = Path.Combine(stationDir, name, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metaPath)) continue;
            try
            {
                var meta = ModelArtifact.LoadTrainingMetadata(Path.Combine(stationDir, name));
                if (string.Equals(meta.Phase, "3a", StringComparison.OrdinalIgnoreCase))
                    return name;
            }
            catch
            {
                // Ignore malformed metadata — just skip.
            }
        }
        return null;
    }

    private IReadOnlyList<string> ResolveStations(string modelsRoot, string truthStation)
    {
        var manifestStations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (string.Equals(truthStation, "all", StringComparison.OrdinalIgnoreCase))
            return manifestStations;

        var explicitSlug = manifestStations.FirstOrDefault(s =>
            string.Equals(s, truthStation, StringComparison.OrdinalIgnoreCase));
        if (explicitSlug is not null)
            return new[] { explicitSlug };

        var derivedSlug = "ea_" + Slugify(truthStation);
        var match = manifestStations.FirstOrDefault(s =>
            string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        if (match is not null)
            return new[] { match };

        _log.LogError("Unknown truth station '{Station}'. Known: [{Known}]", truthStation, string.Join(", ", manifestStations));
        return Array.Empty<string>();
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
