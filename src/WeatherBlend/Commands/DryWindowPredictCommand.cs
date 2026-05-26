using System.Globalization;
using System.Text.RegularExpressions;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.DryWindow;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended P(dry window) forecasts for each (station, window, lead ∈
/// {24, 48, 72}) blender recorded in the dry_window manifest. Parallels
/// <see cref="PrecipPredictCommand"/> but at day granularity: each prediction
/// covers one UTC target day (anchor_date + 1/2/3 days).
///
/// Feature row is built via <see cref="DryWindowFeatureBuilder.ComposeRow"/> so
/// training and inference share a single composition path. The training-time
/// SQL pulls <c>RunTimeSource='offset_day'</c>; predict uses live-cycle rows via
/// <see cref="PredictForecastFilters.LiveCycleAsOf"/>. The feature-row shape is
/// identical — this is a known train/predict distribution difference documented
/// in the phase-3b audit.
/// </summary>
public sealed class DryWindowPredictCommand
{
    private readonly ILogger<DryWindowPredictCommand> _log;
    private readonly AppConfig _cfg;

    /// <summary>
    /// Phases this command's predict path knows how to dispatch (L2 of the
    /// two-layer phase gating in <see cref="RunCompositeVersionAsync"/>).
    /// Of the active dry_window phases in phases.yaml, these are the ones
    /// DryWindowPredictCommand serves. Today the set IS the full
    /// dry_window phase list (no Python-side or cross-command split as
    /// precipitation has), but the layer is symmetric with
    /// PrecipPredictCommand and ready for a future split.
    /// </summary>
    private static readonly HashSet<string> HandledDryWindowPhases =
        new(StringComparer.Ordinal) { "3b", "3p" };

    public DryWindowPredictCommand(ILogger<DryWindowPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public Task<int> RunAsync(string stationArg, string windowArg, string modelVersion, DateOnly? forDate, CancellationToken ct)
        => RunAsync(stationArg, windowArg, modelVersion, forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(string stationArg, string windowArg, string modelVersion, DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (location, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (location is null) return locRc;

        var modelsRoot = _cfg.Storage.ModelsPath;
        var manifestPath = Path.Combine(modelsRoot, "dry_window", ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            _log.LogError("No dry_window manifest at {Path}. Train first.", manifestPath);
            return 2;
        }
        var manifest = System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(
            File.ReadAllText(manifestPath))
            ?? throw new InvalidOperationException("Failed to parse dry_window manifest.");

        var entries = FilterEntries(manifest, stationArg, windowArg);
        if (entries.Count == 0)
        {
            _log.LogError("No manifest entries match station='{Station}' window='{Window}'.", stationArg, windowArg);
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var anchorDate = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
        var targets = new[]
        {
            (Lead: 24, Date: anchorDate.AddDays(1)),
            (Lead: 48, Date: anchorDate.AddDays(2)),
            (Lead: 72, Date: anchorDate.AddDays(3)),
        };

        _log.LogInformation(
            "Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}). Targets: {T}. Entries: {N}",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", targets.Select(t => $"{t.Lead}h→{t.Date:yyyy-MM-dd}")),
            entries.Count);

        var earliest = targets.Min(t => t.Date);
        var latest = targets.Max(t => t.Date).AddHours(23);
        var perDayPerModel = QueryForecastDaysByTarget(
            _cfg.Storage.ForecastsPath, location.Name, earliest, latest, anchor, ct);

        var anyWritten = false;
        foreach (var (compositeKey, station) in entries)
        {
            ct.ThrowIfCancellationRequested();
            var parsed = ParseCompositeKey(compositeKey);
            if (parsed is null)
            {
                _log.LogWarning("Skipping unparsable manifest key '{Key}'.", compositeKey);
                continue;
            }
            var (stationSlug, windowHours) = parsed.Value;

            // Active versions for this composite — 3b champion plus 3g challenger.
            // When the user pins --model-version we honour it and skip the manifest list.
            var activeVersions = string.Equals(modelVersion, "current", StringComparison.OrdinalIgnoreCase)
                ? ModelArtifact.ResolveStationActive(modelsRoot, "dry_window", compositeKey)
                : new[] { modelVersion };

            if (activeVersions.Count == 0)
            {
                _log.LogWarning("{Key}: no active versions in manifest; skipping.", compositeKey);
                continue;
            }

            foreach (var versionName in activeVersions)
            {
                ct.ThrowIfCancellationRequested();
                if (await RunCompositeVersionAsync(
                    modelsRoot, compositeKey, stationSlug, windowHours, versionName,
                    targets, perDayPerModel, anchorDate, predictionMadeAt, location, ct))
                {
                    anyWritten = true;
                }
            }
        }

        return anyWritten ? 0 : 3;
    }

    private async Task<bool> RunCompositeVersionAsync(
        string modelsRoot, string compositeKey, string stationSlug, int windowHours,
        string versionName,
        IReadOnlyList<(int Lead, DateTime Date)> targets,
        IReadOnlyDictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>> perDayPerModel,
        DateTime anchorDate, DateTime predictionMadeAt,
        LocationConfig location, CancellationToken ct)
    {
        var versionDir = Path.Combine(modelsRoot, "dry_window", compositeKey, versionName);
        if (!Directory.Exists(versionDir))
        {
            _log.LogWarning("{Key}: version dir missing → {Dir}; skipping.", compositeKey, versionDir);
            return false;
        }

        // Two-layer phase gating (introduced 2026-05-26). See
        // PrecipPredictCommand.RunStationAsync for the architecture note —
        // identical pattern.
        //   L1: phases.yaml membership (catches retired-phase orphans like
        //       3g/3j/3n/3s still living in the on-R2 manifest's Active
        //       list because PromoteStationVersion only replaces same-phase
        //       entries).
        //   L2: HandledDryWindowPhases set (catches phases active in
        //       phases.yaml but served by a different predict path — none
        //       today for dry_window, but the layer is symmetric with
        //       PrecipPredictCommand and ready for future phases).
        var phaseFromSuffix = ModelArtifact.ExtractPhaseFromVersionName(versionName);
        if (phaseFromSuffix is not null)
        {
            var registeredDryWindowPhases = PhaseRegistry.Default.AllPhases("dry_window");
            var inYaml = registeredDryWindowPhases
                .Any(p => string.Equals(p.Id, phaseFromSuffix, StringComparison.Ordinal));
            if (!inYaml)
            {
                _log.LogInformation(
                    "{Key}: skipping {V} — phase '{Phase}' not in phases.yaml " +
                    "(orphaned bundle in manifest Active; will leave on next promote sweep).",
                    compositeKey, versionName, phaseFromSuffix);
                return true;
            }
            if (!HandledDryWindowPhases.Contains(phaseFromSuffix))
            {
                _log.LogInformation(
                    "{Key}: skipping {V} — phase '{Phase}' active in phases.yaml " +
                    "but served by a different predict path. DryWindowPredictCommand handles only [{Handled}].",
                    compositeKey, versionName, phaseFromSuffix, string.Join(", ", HandledDryWindowPhases));
                return true;
            }
        }

        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        // Phase A multi-location safety: metadata.LocationName is
        // [JsonRequired] so a missing field already threw at deserialise.
        if (!string.Equals(metadata.LocationName, location.Name, StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError(
                "{Key} bundle {V} was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score.",
                compositeKey, versionName, metadata.LocationName, location.Name);
            return false;
        }
        _log.LogInformation("{Key}: version {V} (phase {P}), window {W}h",
            compositeKey, metadata.Version, metadata.Phase, windowHours);

        // Phases 3g / 3j / 3n / 3s retired 2026-05-25 in model-cleanup
        // Phase 1 — predict dispatch removed. Surviving R2 bundles with
        // those phase tags are filtered out by phases.yaml gating.
        //
        // Phase 3p — Gaussian copula MC over Phase 3o's hourly P(wet).
        // No LightGBM artefacts and no climatology by design (line ~448
        // emits ClimatologyProbHasDryWindow=0.0). Dispatch before the
        // climatology check so 3p bundles aren't rejected for missing
        // a file they're never meant to ship.
        if (string.Equals(metadata.Phase, DryWindow3pPredictor.Phase3p, StringComparison.Ordinal))
        {
            return await RunPhase3pAsync(
                versionDir, stationSlug, windowHours, versionName, metadata,
                targets, anchorDate, predictionMadeAt, ct);
        }

        // Belt-and-braces against the 2026-05-26 regression: the dispatch
        // above is the primary defence (3p returns before the check), but
        // also gate the File.Exists guard on PhaseRequiresClimatology so a
        // future refactor that re-orders the dispatch below the check
        // can't silently re-reject 3p (or any future climatology-free
        // phase) for missing a file it never ships. The named helper is
        // unit-tested in DryWindowPredictCommandTests.
        var climPath = Path.Combine(versionDir, "dry_window_climatology.json");
        if (PhaseRequiresClimatology(metadata.Phase) && !File.Exists(climPath))
        {
            _log.LogWarning("{Key} {V}: missing climatology at {P}; skipping.", compositeKey, versionName, climPath);
            return false;
        }
        var climatology = DryWindowClimatology.LoadFrom(climPath);

        // Phase 3d-calibrated handling removed 2026-04-29 — PAV calibration on
        // dry-window didn't move test Brier vs raw 3b. Old 3d-calibrated
        // artefacts on R2 are inert; if any persist in a manifest's Active
        // list they should be dropped.

        // Per-lead BlenderSpec lives in feature_schema.json.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder.ToList();

        var ml = new MLContext(seed: 42);
        var predictions = new List<DryWindowPredictionRow>();

        foreach (var (lead, targetDate) in targets)
        {
            ct.ThrowIfCancellationRequested();
            if (!perDayPerModel.TryGetValue(DateOnly.FromDateTime(targetDate), out var modelDayList))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: no forecast rows for {D:yyyy-MM-dd}; skipping.",
                    compositeKey, versionName, lead, targetDate);
                continue;
            }
            if (!modelDayList.Any(d => d is { AnyPresent: true }))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: forecast rows exist for {D:yyyy-MM-dd} but no model is populated; skipping.",
                    compositeKey, versionName, lead, targetDate);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h: no BlenderSpec in feature_schema.json; skipping.",
                    compositeKey, versionName, lead);
                continue;
            }

            // Filter the canonical 6-slot model-day list down to spec.Models, in spec order.
            var specModelDays = new List<DryWindowFeatureBuilder.ForecastDay?>(spec.Models.Count);
            foreach (var modelId in spec.Models)
            {
                var ci = canonOrder.IndexOf(modelId);
                specModelDays.Add(modelDayList[ci]);
            }

            var (startHour, endHour) = _cfg.DryWindow.BuildDaytimeWindow()
                .UtcHourRangeFor(DateOnly.FromDateTime(targetDate));
            var row = DryWindowFeatureBuilder.ComposeRow(
                spec,
                DateOnly.FromDateTime(targetDate),
                windowHours,
                specModelDays,
                label: false,
                truthMmDay: 0.0,
                startHour: startHour,
                endHour: endHour);

            double rawProb;
            {
                var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                var probs = DryWindowTrainer.PredictVectorProbability(ml, loadedModel, spec, new[] { row });
                rawProb = probs[0];
            }

            // Apply isotonic calibration if the artefact carries one — older
            // pre-calibration models simply return raw probs unchanged. For
            // 3e/4h the calibrator was fitted to the PRODUCT against 4h truth
            // at training time, so applying it here is the right end-to-end
            // calibration step.
            var calibrator = ModelArtifact.TryLoadLeadCalibrator(versionDir, lead);
            var prob = calibrator is null ? rawProb : calibrator.Predict(rawProb);
            var climProb = climatology.Predict(targetDate);

            // Build per-model output fields: populate only spec.Models, null elsewhere.
            // Sized from DryWindowPredictionRow.PerModelFieldCount (8: Gfs..Gem + Aifs + Jma).
            var perModelHasDry  = new double?[DryWindowPredictionRow.PerModelFieldCount];
            var perModelSum     = new double?[DryWindowPredictionRow.PerModelFieldCount];
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= DryWindowPredictionRow.PerModelFieldCount) continue;
                var hasDry = row.Features[spec.IndexOf($"has_dry_window_{WeatherBlend.Train.TempFeatureBuilder.ShortName(spec.Models[i])}")];
                var sum    = row.Features[spec.IndexOf($"precip_sum_{WeatherBlend.Train.TempFeatureBuilder.ShortName(spec.Models[i])}")];
                perModelHasDry[ci] = NanToNullDouble(hasDry);
                perModelSum[ci]    = NanToNullDouble(sum);
            }

            predictions.Add(new DryWindowPredictionRow
            {
                LocationName = location.Name,
                TruthStation = stationSlug,
                WindowHours = windowHours,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                TargetDateUtc = targetDate,
                LeadHours = lead,
                ProbHasDryWindow = prob,
                ClimatologyProbHasDryWindow = climProb,
                AgreementHasDryWindow = NanToNullDouble(row.Features[spec.IndexOf("agreement_has_dry_window")]),
                PrecipSumMean = NanToNullDouble(row.Features[spec.IndexOf("precip_sum_mean")]),
                LongestDryRunMean = NanToNullDouble(row.Features[spec.IndexOf("longest_dry_run_mean")]),
                WetHourCountMean = NanToNullDouble(row.Features[spec.IndexOf("wet_hour_count_mean")]),
                HasDryWindowGfs   = perModelHasDry[0], HasDryWindowEcmwf = perModelHasDry[1],
                HasDryWindowIcon  = perModelHasDry[2], HasDryWindowMf    = perModelHasDry[3],
                HasDryWindowUkmo  = perModelHasDry[4], HasDryWindowGem   = perModelHasDry[5],
                HasDryWindowAifs  = perModelHasDry[6], HasDryWindowJma   = perModelHasDry[7],
                PrecipSumGfs   = perModelSum[0], PrecipSumEcmwf = perModelSum[1],
                PrecipSumIcon  = perModelSum[2], PrecipSumMf    = perModelSum[3],
                PrecipSumUkmo  = perModelSum[4], PrecipSumGem   = perModelSum[5],
                PrecipSumAifs  = perModelSum[6], PrecipSumJma   = perModelSum[7],
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
                ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, prob),
            });

            _log.LogInformation(
                "  lead {Lead}h ({Date:yyyy-MM-dd}) → P(dry {W}h)={P:0.000} (clim {C:0.000}, agreement {A:0.00})",
                lead, targetDate, windowHours, prob, climProb,
                row.Features[spec.IndexOf("agreement_has_dry_window")]);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("{Key} {V}: no predictions produced.", compositeKey, versionName);
            return false;
        }

        await WritePredictionsAsync(predictions, stationSlug, windowHours, anchorDate, metadata.Version, ct);
        return true;
    }

    /// <summary>
    /// Phase 3p predict path. No model.zip files; read the bound 3o
    /// champion's hourly P(wet) parquet for the anchor cycle, extract the
    /// daytime hour vector for each target_date, run Gaussian copula MC
    /// against the bundle's single Σ, write a dry-window prediction row.
    /// </summary>
    private async Task<bool> RunPhase3pAsync(
        string versionDir, string stationSlug, int windowHours, string versionName,
        ModelArtifact.TrainingMetadata metadata,
        IReadOnlyList<(int Lead, DateTime Date)> targets,
        DateTime anchorDate, DateTime predictionMadeAt,
        CancellationToken ct)
    {
        // Bound 3o version is captured in metadata at train time. If it's
        // missing, the bundle is malformed — log + skip rather than crash.
        var v3o = metadata.Hyperparameters.HpString(DryWindow3pPredictor.Precip3oVersionKey);
        if (string.IsNullOrEmpty(v3o))
        {
            _log.LogWarning("{Key} {V}: 3p bundle is missing `{Key2}` metadata; skipping.",
                stationSlug, versionName, DryWindow3pPredictor.Precip3oVersionKey);
            return false;
        }

        // Σ + L loaded once per bundle.
        double[,] L;
        try
        {
            L = DryWindow3pPredictor.LoadCholesky(versionDir);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "{Key} {V}: failed to load Σ — bundle correlation.json is malformed.",
                stationSlug, versionName);
            return false;
        }

        // 3o's live predictions for the anchor cycle. ONE parquet covers
        // all hourly P(wet) the MC needs; load once, slice by target_date.
        Dictionary<DateTime, double> hourly;
        try
        {
            hourly = DryWindow3pPredictor.LoadLivePredictionsHourly(
                _cfg.Storage.PredictionsPath, stationSlug, v3o, anchorDate);
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "{Key} {V}: cannot load 3o live predictions ({V3o}) for anchor {A:yyyy-MM-dd} — skipping " +
                "(this is expected immediately after a fresh deploy until 3o has run at least once).",
                stationSlug, versionName, v3o, anchorDate);
            return false;
        }
        if (hourly.Count == 0)
        {
            _log.LogWarning("{Key} {V}: 3o live predictions parquet for anchor {A:yyyy-MM-dd} is empty — skipping.",
                stationSlug, versionName, anchorDate);
            return false;
        }

        var daytime = _cfg.DryWindow.BuildDaytimeWindow();
        var mcSamples = metadata.Hyperparameters.HpInt("mc_samples") ?? DryWindow3pPredictor.DefaultMcSamples;
        var rng = new Random(metadata.Hyperparameters.HpInt("mc_seed") ?? 42);
        var predictions = new List<DryWindowPredictionRow>();
        var primaryLocation = _cfg.Location.Name;

        foreach (var (lead, targetDate) in targets)
        {
            ct.ThrowIfCancellationRequested();
            var (startUtc, endUtcExclusive) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(targetDate));
            var qDaytime = DryWindow3pPredictor.ExtractDaytimeQ(hourly, targetDate, startUtc, endUtcExclusive);
            if (qDaytime is null)
            {
                _log.LogWarning("{Key} {V} lead {Lead}h ({Date:yyyy-MM-dd}): daytime q vector incomplete from 3o predictions; skipping.",
                    stationSlug, versionName, lead, targetDate);
                continue;
            }
            // If Σ was fit on a 9-hour daytime window but the target date
            // gives a different length (DST tail edge — extremely rare for
            // Europe/London at 9-18 local), refuse rather than re-fit.
            if (qDaytime.Length != L.GetLength(0))
            {
                _log.LogWarning("{Key} {V} lead {Lead}h ({Date:yyyy-MM-dd}): daytime q length {Qn} != Σ dim {Sn} — likely a DST edge day, skipping.",
                    stationSlug, versionName, lead, targetDate, qDaytime.Length, L.GetLength(0));
                continue;
            }
            // Stats variant runs the same MC + RNG seed; returns P(window)
            // alongside the per-sample longest-dry-run distribution. Populates
            // McMean/McP10/McP50/McP90 on the row — the site reads the
            // P10/P90 band as the 3p confidence inline detail
            // (3p has no fitted conformal calibrator on disk; the band is
            // the only confidence signal we ship without retrain-dependency
            // plumbing — see 2026-05-26 thread).
            var mc = DryWindow3pPredictor.ProbDryWindowWithStats(qDaytime, L, windowHours, rng, mcSamples);
            var prob = mc.ProbWindow;

            predictions.Add(new DryWindowPredictionRow
            {
                LocationName = primaryLocation,
                TruthStation = stationSlug,
                WindowHours = windowHours,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                TargetDateUtc = targetDate,
                LeadHours = lead,
                ProbHasDryWindow = prob,
                ClimatologyProbHasDryWindow = 0.0, // 3p has no climatology
                                                  // sidecar — verify uses the
                                                  // per-station EA-derived base
                                                  // rate from its own pipeline.
                AgreementHasDryWindow = null,
                PrecipSumMean = null,
                LongestDryRunMean = null,
                WetHourCountMean = null,
                McMeanLongestDryRunHours = mc.MeanLongestDryRunHours,
                McP10LongestDryRunHours  = mc.P10LongestDryRunHours,
                McP50LongestDryRunHours  = mc.P50LongestDryRunHours,
                McP90LongestDryRunHours  = mc.P90LongestDryRunHours,
                FeatureVectorHash = "",
                ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, prob),
            });
            _log.LogInformation(
                "  lead {Lead}h ({Date:yyyy-MM-dd}) → P(dry {W}h)={P:0.000} (3p, MC samples={Mc}, longest dry run P10–P90={P10}-{P90}h, bound 3o={V3o})",
                lead, targetDate, windowHours, prob, mcSamples,
                mc.P10LongestDryRunHours, mc.P90LongestDryRunHours, v3o);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("{Key} {V}: 3p produced no predictions — likely no overlap between 3o coverage and target dates.",
                stationSlug, versionName);
            return false;
        }
        await WritePredictionsAsync(predictions, stationSlug, windowHours, anchorDate, metadata.Version, ct);
        return true;
    }


    private List<KeyValuePair<string, ModelArtifact.StationEntry>> FilterEntries(
        ModelArtifact.Manifest manifest, string stationArg, string windowArg)
    {
        var result = new List<KeyValuePair<string, ModelArtifact.StationEntry>>();
        foreach (var kv in manifest.Stations)
        {
            var parsed = ParseCompositeKey(kv.Key);
            if (parsed is null) continue;
            var (slug, window) = parsed.Value;

            if (!string.Equals(stationArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                if (!SlugMatches(slug, stationArg)) continue;
            }
            if (!string.Equals(windowArg, "all", StringComparison.OrdinalIgnoreCase))
            {
                if (!int.TryParse(windowArg, out var w) || w != window) continue;
            }
            if (kv.Value.Active.Count == 0) continue;
            result.Add(kv);
        }
        return result;
    }

    /// <summary>
    /// Returns true iff <paramref name="phase"/>'s bundle is expected to ship
    /// a <c>dry_window_climatology.json</c> artefact. The climatology-free
    /// phases are dispatched earlier in <c>RunAsync</c>; this helper also
    /// gates the <c>File.Exists</c> guard so a future refactor that
    /// re-orders the dispatch below the check can't silently re-reject
    /// climatology-free bundles for missing a file they're not meant to
    /// ship — the regression of 2026-05-26 (3p bundles getting rejected
    /// with "missing climatology" across two retrain sweeps, 0 3p
    /// predictions on R2). Add a new phase to the exempt list AND give
    /// it a dispatch branch above; both are mandatory.
    /// </summary>
    internal static bool PhaseRequiresClimatology(string phase) =>
        !string.Equals(phase, DryWindow3pPredictor.Phase3p, StringComparison.Ordinal);

    internal static bool SlugMatches(string slug, string arg)
    {
        if (slug.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var slugWithoutPrefix = slug.StartsWith("ea_") ? slug[3..] : slug;
        if (slugWithoutPrefix.Equals(arg, StringComparison.OrdinalIgnoreCase)) return true;
        var derived = StationSlug.Of(arg);
        return slug.Equals("ea_" + derived, StringComparison.OrdinalIgnoreCase)
            || slugWithoutPrefix.Equals(derived, StringComparison.OrdinalIgnoreCase);
    }

    internal static (string StationSlug, int WindowHours)? ParseCompositeKey(string key)
    {
        var m = Regex.Match(key, @"^(?<slug>[^/]+)/window_(?<w>\d+)h$");
        if (!m.Success) return null;
        return (m.Groups["slug"].Value, int.Parse(m.Groups["w"].Value, CultureInfo.InvariantCulture));
    }

    private Dictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>> QueryForecastDaysByTarget(
        string forecastsPath,
        string locationName,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime anchorAsOfRunTime,
        CancellationToken ct)
    {
        var glob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var filter = PredictForecastFilters.LiveCycleAsOf(locationName, anchorAsOfRunTime, earliestValid, latestValid);

        // Latest live-cycle row per (valid_time, model). Mirrors PrecipPredictCommand.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model,
           Precipitation, PrecipitationProbability,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE {filter}
)
SELECT ValidTimeUtc, Model,
       Precipitation, PrecipitationProbability,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m
FROM latest WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        // Slot index by canonical model order so the indices line up with
        // TempFeatureBuilder.CanonicalModelOrder used at predict time below — when AIFS
        // (slot 6) is in the trained spec, modelDayList[6] needs to exist. The legacy
        // DryWindowFeatureBuilder.ModelIds is only 6 wide; using it here was the bug.
        var canonOrder = WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder;
        var slotByModel = canonOrder
            .Select((id, i) => (id, i))
            .ToDictionary(x => x.id, x => x.i);
        var slotCount = canonOrder.Count;

        var byDate = new Dictionary<DateOnly, List<DryWindowFeatureBuilder.ForecastDay?>>();

        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = DateTime.SpecifyKind(r.GetDateTime(0), DateTimeKind.Utc);
            var model = r.GetString(1);
            if (!slotByModel.TryGetValue(model, out var slot)) continue;

            var date = DateOnly.FromDateTime(valid);
            if (!byDate.TryGetValue(date, out var list))
            {
                list = new List<DryWindowFeatureBuilder.ForecastDay?>(slotCount);
                for (int s = 0; s < slotCount; s++) list.Add(null);
                byDate[date] = list;
            }
            if (list[slot] is null) list[slot] = new DryWindowFeatureBuilder.ForecastDay();

            var fr = new DryWindowFeatureBuilder.ForecastRow(
                valid, model,
                r.IsDBNull(2) ? null : r.GetDouble(2),
                r.IsDBNull(3) ? null : r.GetDouble(3),
                r.IsDBNull(4) ? null : r.GetDouble(4),
                r.IsDBNull(5) ? null : r.GetDouble(5),
                r.IsDBNull(6) ? null : r.GetDouble(6),
                r.IsDBNull(7) ? null : r.GetDouble(7),
                r.IsDBNull(8) ? null : r.GetDouble(8),
                r.IsDBNull(9) ? null : r.GetDouble(9),
                r.IsDBNull(10) ? null : r.GetDouble(10),
                r.IsDBNull(11) ? null : r.GetDouble(11));
            list[slot]!.SetHour(valid.Hour, fr);
        }

        return byDate;
    }

    private async Task WritePredictionsAsync(
        IReadOnlyList<DryWindowPredictionRow> predictions,
        string stationSlug,
        int windowHours,
        DateTime anchorDate,
        string modelVersion,
        CancellationToken ct)
    {
        var dateStr = anchorDate.ToString("yyyy-MM-dd");
        var outPath = Path.Combine(_cfg.Storage.PredictionsPath,
            "dry_window",
            stationSlug,
            $"window_{windowHours}h",
            $"model_version={modelVersion}",
            $"date={dateStr}",
            "predictions.parquet");

        // Dry-window is day-granular: one row per (PMT, lead), so the dedup
        // key omits ValidTimeUtc (unlike the hourly temp/precip/element writers).
        var total = await PredictionParquetWriter.WriteAsync(
            outPath, predictions,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.TargetDateUtc).ThenBy(r => r.LeadHours),
            ct);
        _log.LogInformation("Wrote {N} new predictions (file now holds {T}) → {Path}",
            predictions.Count, total, outPath);
    }

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;
    private static double? NanToNullDouble(float v) => float.IsNaN(v) ? null : (double)v;
}
