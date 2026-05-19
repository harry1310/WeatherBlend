using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Site;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;
using WeatherBlend.Train.PrecipExact;

namespace WeatherBlend.Commands;

/// <summary>
/// Produces blended P(wet) forecasts for leads {24, 48, 72} using the per-station
/// Phase 3a occurrence classifier. Runs one pass per truth station so the output
/// tree mirrors the model tree: one parquet folder per
/// <c>data/predictions/precipitation/{truth_station}/</c>.
///
/// "Latest available" forecast semantics match <see cref="TempPredictCommand"/> —
/// for each (valid_time, model) pick the most recent run that covers it,
/// excluding the historical-forecast (<c>RunTimeSource='offset_day'</c>) rows.
/// The six per-model covariates (RH, dew depression, clouds, CAPE, wind) are
/// averaged across whichever models are present, matching how the feature row
/// is assembled at training time.
/// </summary>
public sealed class PrecipPredictCommand
{
    private readonly ILogger<PrecipPredictCommand> _log;
    private readonly AppConfig _cfg;
    // `--location` resolves into _activeLocation at RunAsync entry; everything
    // downstream reads from here. Defaults to the primary location for the
    // existing call paths (predict-and-render, manual `predict` without
    // --location), which continue to behave exactly as before. Mutable for
    // the lifetime of one RunAsync — fine because PrecipPredictCommand is
    // registered as Transient in DI so we get a fresh instance per call.
    private LocationConfig _activeLocation;

    private static readonly int[] DefaultLeads = Leads.Full;

    public PrecipPredictCommand(ILogger<PrecipPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
        _activeLocation = cfg.Location;
    }

    /// <summary>
    /// For each lead bucket L in <paramref name="leads"/>, emit one (Lead, ValidTime)
    /// pair per UTC hour of the target day <c>(anchor.Date + L/24)</c>. So one cycle
    /// at the standard {24, 48, 72, 96, 120}h leads produces 5 × 24 = 120 (Lead,
    /// ValidTime) targets. Each target's row will be picked from the forecast tree
    /// using the lead-bucket-aware filter (<c>RunTime ≤ Valid − L</c>) so the actual
    /// lead lands in the [L, L+24h) band the model was trained on.
    /// </summary>
    /// <remarks>
    /// Emitting all 24 hours of each target day (rather than just the wall-clock
    /// daytime window) keeps this method scope-neutral: the dry-window start-hour
    /// curve consumes the daytime subset, but anything else that wants nighttime
    /// hourly P(wet) (e.g. a future "is it raining at midnight" widget) gets it for
    /// free at zero extra storage cost relative to the same-cycle load.
    /// </remarks>
    internal static (int Lead, DateTime Valid)[] BuildHourlyTargets(
        DateTime anchor, IReadOnlyList<int> leads)
    {
        var anchorDate = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc);
        var result = new List<(int Lead, DateTime Valid)>(leads.Count * 24);
        foreach (var lead in leads)
        {
            var dayStart = anchorDate.AddDays(lead / 24);
            for (int h = 0; h < 24; h++)
                result.Add((lead, dayStart.AddHours(h)));
        }
        return result.ToArray();
    }

    /// <summary>
    /// <paramref name="truthStation"/> is a slug (<c>ea_bellever_dartmoor</c>), or
    /// <c>all</c> to run every station in the manifest, or a config rainfall
    /// station name (<c>Bellever Dartmoor</c>) for ergonomic CLI use.
    /// </summary>
    public Task<int> RunAsync(string truthStation, string modelVersion, DateOnly? forDate, CancellationToken ct)
        => RunAsync(truthStation, modelVersion, forDate, locationOverride: null, ct);

    /// <summary>
    /// <inheritdoc cref="RunAsync(string, string, DateOnly?, CancellationToken)"/>
    /// <paramref name="locationOverride"/> selects which configured location's
    /// forecast tree + rainfall config to use; when set, stationsToRun is also
    /// filtered to that location's rainfall stations (so `--truth-station all
    /// --location membury_devon` only predicts the 3 Membury stations, not all
    /// 7 stations in the precipitation tree).
    /// </summary>
    public async Task<int> RunAsync(string truthStation, string modelVersion, DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        var (resolvedLocation, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (resolvedLocation is null) return locRc;
        _activeLocation = resolvedLocation;

        var modelsRoot = _cfg.Storage.ModelsPath;

        var stationsToRun = ResolveStations(modelsRoot, truthStation);
        // Always intersect with the active location's rainfall config — a
        // station's predict MUST use NWP from its own location's grid cell,
        // never from a sibling location's. Pre-2026-05-12 this filter only
        // ran when --location was passed, so step 1 of predict-and-render
        // (no --location → primary) silently ran Membury stations against
        // Bonehill NWP. The render-side LocationName filter dropped the
        // resulting wrong-location rows but the predict job still wasted
        // the cycles AND tagged the parquet with bonehill_rocks for a
        // membury_devon station, which is a category error. Filtering at
        // the source means step 1 only ever predicts its own location's
        // stations, regardless of whether --truth-station was "all" or a
        // specific slug. A Membury slug requested without --location now
        // returns 0 with a clear log message instead of silently producing
        // wrong-NWP predictions.
        var locationSlugs = _activeLocation.Rainfall.Stations
            .Select(s => StationSlug.WithEaPrefix(s.Name))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var beforeCount = stationsToRun.Count;
        var dropped = stationsToRun.Where(s => !locationSlugs.Contains(s)).ToList();
        stationsToRun = stationsToRun.Where(s => locationSlugs.Contains(s)).ToList();
        if (dropped.Count > 0)
        {
            _log.LogInformation(
                "Dropping {Count}/{Before} station(s) not in location '{Loc}' rainfall config: [{Dropped}]. " +
                "Predict each station against its own location's NWP — re-run with --location <name> to score the rest.",
                dropped.Count, beforeCount, _activeLocation.Name, string.Join(", ", dropped));
        }
        if (stationsToRun.Count == 0)
        {
            _log.LogError(
                "No precipitation blender artefacts under {Dir} match location '{Loc}' rainfall stations [{Configured}]. " +
                "Either add the station to that location's config or pass --location <name>.",
                Path.Combine(modelsRoot, "precipitation"),
                _activeLocation.Name,
                string.Join(", ", _activeLocation.Rainfall.Stations.Select(s => s.Name)));
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var targets = BuildHourlyTargets(anchor, DefaultLeads);

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — stations=[{Stations}] — {Count} targets across leads {Leads}",
            anchor,
            forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", stationsToRun),
            targets.Length,
            string.Join(",", DefaultLeads));

        // One forecast pivot per lead bucket — each call enforces the lead-bucket
        // training constraint (RunTime ≤ Valid - L) so every row passed to the
        // lead-L model sits in the [L, L+24h) actual-lead band the model was
        // trained on (Open-Meteo's previous_day_(L/24) aggregation). Without this
        // filter, "latest cycle wins" would feed the lead-24 model rows at
        // actual lead < 24h once we predict for valid times within 24h of anchor.
        var perLeadValid = new Dictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>>();
        foreach (var lead in DefaultLeads)
        {
            var leadTargets = targets.Where(t => t.Lead == lead).ToArray();
            if (leadTargets.Length == 0) continue;
            var pivot = QueryLatestForecastRows(
                _cfg.Storage.ForecastsPath,
                _activeLocation.Name,
                leadTargets.Min(t => t.Valid),
                leadTargets.Max(t => t.Valid),
                asOfRunTime: anchor,
                leadHoursLowerBound: lead,
                ct);
            perLeadValid[lead] = pivot;
        }

        var anyWritten = false;
        foreach (var station in stationsToRun)
        {
            ct.ThrowIfCancellationRequested();
            foreach (var version in ResolveRequestedVersions(modelsRoot, station, modelVersion))
            {
                var wrote = await RunStationAsync(
                    modelsRoot, station, version, predictionMadeAt, anchor, targets, perLeadValid, ct);
                anyWritten |= wrote;
            }
        }

        return anyWritten ? 0 : 3;
    }

    private IReadOnlyList<string> ResolveRequestedVersions(string modelsRoot, string station, string modelVersion)
    {
        // Mirrors TempPredictCommand.ResolveRequestedVersions for the per-station layout:
        // "current"/"all" → iterate every active version for this station (Phase 3c
        // champion/challenger). Anything else is an explicit version dir name.
        var v = modelVersion?.ToLowerInvariant() ?? "current";
        if (v is "current" or "all")
        {
            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
            return active.Count == 0 ? new[] { "current" } : active.ToList();
        }
        return new[] { modelVersion! };
    }

    private async Task<bool> RunStationAsync(
        string modelsRoot,
        string station,
        string modelVersion,
        DateTime predictionMadeAt,
        DateTime anchor,
        (int Lead, DateTime Valid)[] targets,
        IReadOnlyDictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>> perLeadValid,
        CancellationToken ct)
    {
        // Python-trained phases (4a per-cell BART, 5a INLA Bayesian) live
        // in WeatherProbabilistic and have their own dedicated predict
        // workflows (predict-4a.yml + predict-5a.yml). Their bundles end up
        // in the same data/models/precipitation/{station}/ tree + their
        // versions are auto-promoted into MANIFEST.json's Active (since
        // 2026-05-12), so once-around the Active loop in this .NET
        // PrecipPredictCommand they'd hit `metadata.Phase == "4a"` or
        // "5a" and try to load a LightGBM blender that isn't there.
        // Worse — train_4a.py writes BestSingleTestMae=null which the .NET
        // PerLeadStats deserialiser rejects (System.Double can't accept
        // null), so we crash BEFORE reaching the dispatch. Cheapest fix:
        // sniff the version-name suffix and skip without trying to load.
        // Caught 2026-05-12 03:22 UTC by predict-and-render run 25711115753
        // shortly after the manifest fix promoted 4a back into Active.
        if (modelVersion.EndsWith("_phase4a", StringComparison.Ordinal)
            || modelVersion.EndsWith("_phase4b", StringComparison.Ordinal)
            || modelVersion.EndsWith("_phase5a", StringComparison.Ordinal))
        {
            var which = modelVersion.EndsWith("_phase4a", StringComparison.Ordinal) ? "4a"
                      : modelVersion.EndsWith("_phase4b", StringComparison.Ordinal) ? "4b"
                      : "5a";
            _log.LogInformation(
                "Station {Station}: skipping {V} — Python-{Workflow} phase, served by its own predict step.",
                station, modelVersion, which);
            return true;   // not a failure — just not this command's job
        }

        var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, modelVersion);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Station {Station} model version {V} has no per-lead blenders.", station, metadata.Version);
            return false;
        }

        // Phase A multi-location safety: refuse to score a bundle against
        // any NWP source other than the one it was trained on. metadata.
        // LocationName is [JsonRequired] post-backfill so a missing field
        // already threw at deserialise; we only need the mismatch check.
        if (!string.Equals(metadata.LocationName, _activeLocation.Name, StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError(
                "Station {Station} bundle {V} was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score. " +
                "Pass --location {TrainedRetry} or fix the manifest entry.",
                station, modelVersion, metadata.LocationName, _activeLocation.Name, metadata.LocationName);
            return false;
        }

        var climPath = Path.Combine(versionDir, ModelArtifact.ClimatologyFileName);
        if (!File.Exists(climPath))
        {
            _log.LogError("Station {Station} version {V} is missing {File} — retrain to persist it.",
                station, metadata.Version, ModelArtifact.ClimatologyFileName);
            return false;
        }
        var climatology = PrecipClimatology.LoadFrom(climPath);

        _log.LogInformation("Station {Station}: using blender version {V} (phase={Phase})",
            station, metadata.Version, metadata.Phase);

        // Phase 3e (TorchSharp MLP) takes a separate predict path because
        // the bundle file format is different (mlp_lead_NNh.pt + preprocess.json
        // vs ML.NET's per-lead .zip). Same input vector layout as 3c (both
        // built from PrecipRichFeatureBuilder), so perLeadValid pivots feed
        // unchanged. Dispatched first so an MLP version routed past the 3a/3c
        // ML.NET load path can never silently fail at "missing lead_NNh.zip".
        var isMlp = string.Equals(metadata.Phase, "3e", StringComparison.Ordinal);
        if (isMlp)
        {
            return await RunStationAsMlpAsync(
                modelsRoot, station, versionDir, metadata, climatology,
                predictionMadeAt, anchor, targets, perLeadValid, ct);
        }

        // Phase 3d (exact-runtime) takes a separate predict path: different
        // forecast tree filter (RunTimeSource='exact'), different lead set
        // ({12, 24}), different ValidTime grid ({0, 6, 12, 18}), per-V-hour
        // UKV pull. perLeadValid (Open-Meteo offset_day pivot) is irrelevant
        // for 3d and we build our own pivot inside.
        var isExact = string.Equals(metadata.Phase, "3d", StringComparison.Ordinal);
        if (isExact)
        {
            return await RunStationAsExactRuntimeAsync(
                modelsRoot, station, versionDir, metadata, climatology,
                predictionMadeAt, anchor, ct);
        }

        var isRich = PrecipPhases.IsRich(metadata.Phase);
        // Isotonic-calibration handling (Phase 3a_isotonic) removed 2026-04-29 —
        // the bake-off found PAV calibration didn't move test Brier vs raw 3a.
        // Old 3a_isotonic artefacts on R2 will fail to load against a current
        // active list; if any persist, drop them via a manifest edit + R2 purge.

        // Phase 3c needs EA observation persistence anchored at run_time = valid - lead;
        // load the whole hourly series once (small) and reuse across the three leads.
        Dictionary<DateTime, double>? hourlyRain = null;
        if (isRich)
        {
            var friendly = ResolveFriendlyStationName(station);
            if (friendly is null)
            {
                _log.LogError("Station {Station}: cannot resolve rainfall config name for phase-3c persistence lookup. Known: [{Known}]",
                    station, string.Join(", ", _activeLocation.Rainfall.Stations.Select(s => s.Name)));
                return false;
            }
            hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
                _cfg.Storage.RainfallPath, _activeLocation.Name, friendly, ct);
            _log.LogInformation("Station {Station}: loaded {N} hourly rainfall rows for persistence features (friendly='{Friendly}')",
                station, hourlyRain.Count, friendly);
        }

        var ml = new MLContext(seed: 42);
        var predictions = new List<PrecipPredictionRow>();

        // Both lean (3a) and rich (3c) artefacts use the per-lead BlenderSpec layout.
        // Rich's spec just has a longer feature vector. Schema is read from feature_schema.json.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();

        foreach (var (lead, valid) in targets)
        {
            if (!perLeadValid.TryGetValue(lead, out var perValid)
                || !perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no forecast rows for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }
            if (!pivot.Precip.Any(p => p.HasValue))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: all six per-model precip values null for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }
            if (!specs.TryGetValue(lead, out var spec))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no BlenderSpec in feature_schema.json for this lead; skipping.",
                    station, lead);
                continue;
            }

            // Pull spec.Models precip from the canonical pivot, in spec order.
            // prob_* removed 2026-04-28 (zero-gain, see PrecipFeatureBuilder).
            int N = spec.Models.Count;
            var specPrecip = new double[N];
            var missingRequired = new List<string>();
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                var pv = pivot.Precip[ci];
                specPrecip[i] = pv ?? double.NaN;
                if (!pv.HasValue && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0)
            {
                _log.LogWarning("Station {Station} lead {Lead}h: missing required per-model precip [{Missing}] for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, string.Join(",", missingRequired), valid);
                continue;
            }

            // Compose the row — lean or rich depending on spec.FeatureSet.
            BinaryTrainingRow row;
            if (isRich)
            {
                var dew    = new double[N];
                var rh     = new double[N];
                var dewDep = new double[N];
                var pres   = new double[N];
                for (int i = 0; i < N; i++)
                {
                    var ci = canonOrder.IndexOf(spec.Models[i]);
                    var t  = pivot.Temp2m[ci];
                    var d  = pivot.Dew[ci];
                    dew[i]    = d ?? double.NaN;
                    rh[i]     = pivot.Rh[ci] ?? double.NaN;
                    dewDep[i] = (t.HasValue && d.HasValue) ? t.Value - d.Value : double.NaN;
                    pres[i]   = pivot.Pressure[ci] ?? double.NaN;
                }
                var runTime = valid.AddHours(-lead);
                var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain!, runTime);
                row = PrecipRichFeatureBuilder.ComposeRow(
                    spec, valid, specPrecip, dew, rh, dewDep, pres,
                    rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                    cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                    cloudHighMean: pivot.CloudHighMean,
                    capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                    eaRainPrev24hMm: persist.Prev24hMm,
                    eaRainPrev72hMm: persist.Prev72hMm,
                    eaWetHoursLast24h: persist.WetHoursLast24h,
                    eaDryHoursTrailing: persist.DryHoursTrailing,
                    truthMmHour: 0.0);
            }
            else
            {
                row = PrecipFeatureBuilder.ComposeRow(
                    spec, valid, specPrecip,
                    rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                    cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                    cloudHighMean: pivot.CloudHighMean,
                    capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                    truthMmHour: 0.0);
            }

            var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var pWet = PrecipOccurrenceTrainer.PredictVectorProbability(ml, loadedModel, spec, new[] { row });

            var climPWet = climatology.Predict(valid);

            // Build per-model output fields: populate only spec.Models, null elsewhere.
            // Sized from PrecipPredictionRow.PerModelFieldCount.
            var perModelPrecip = new double?[PrecipPredictionRow.PerModelFieldCount];
            var perModelRun    = new DateTime?[PrecipPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= PrecipPredictionRow.PerModelFieldCount) continue;
                perModelPrecip[ci] = specPrecip[i];
                perModelRun[ci]    = pivot.RunTime[ci];
            }

            // Spread features live at offset 2N in both lean and rich layouts
            // (precip per model, then prob per model, then mean/std/max/agreement).
            // Spread features at offset N now that prob_* is gone (was 2*N pre-cleanup).
            var spreadStart = N;
            predictions.Add(new PrecipPredictionRow
            {
                LocationName = _activeLocation.Name,
                TruthStation = station,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                ProbWet = pWet[0],
                ClimatologyPWet = climPWet,
                PrecipGfs   = perModelPrecip[0], PrecipEcmwf = perModelPrecip[1],
                PrecipIcon  = perModelPrecip[2], PrecipMf    = perModelPrecip[3],
                PrecipUkmo  = perModelPrecip[4], PrecipGem   = perModelPrecip[5],
                PrecipAifs  = perModelPrecip[6], PrecipJma   = perModelPrecip[7],
                RunTimeGfs   = perModelRun[0], RunTimeEcmwf = perModelRun[1],
                RunTimeIcon  = perModelRun[2], RunTimeMf    = perModelRun[3],
                RunTimeUkmo  = perModelRun[4], RunTimeGem   = perModelRun[5],
                RunTimeAifs  = perModelRun[6], RunTimeJma   = perModelRun[7],
                PrecipMean = NanToNull(row.Features[spreadStart + 0]),
                PrecipStd  = NanToNull(row.Features[spreadStart + 1]),
                PrecipMax  = NanToNull(row.Features[spreadStart + 2]),
                PrecipAgreementWet01 = NanToNull(row.Features[spreadStart + 3]),
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
                ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, pWet[0]),
            });

            _log.LogInformation(
                "Station {Station} lead {Lead}h → P(wet) {P:0.000} (clim {Clim:0.000}, valid {Valid:yyyy-MM-dd HH:mm}Z, agreement {Ag:0.00})",
                station, lead, pWet[0], climPWet, valid, row.Features[spreadStart + 3]);
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Station {Station}: no predictions produced — likely missing forecast data.", station);
            return false;
        }

        await WritePredictionsAsync(predictions, station, anchor, metadata.Version, ct);
        return true;
    }

    /// <summary>
    /// Phase 3e MLP predict path. Mirrors the rich (3c) feature row build —
    /// SAME 59-feat input vector, head-to-head with 3a/3c on the same
    /// (valid, lead) targets — but loads MLP weights via MlpArtifact +
    /// scores via MlpTrainer.PredictVectorProbability instead of ML.NET.
    /// Output schema is the same PrecipPredictionRow so verify + site read
    /// 3e rows transparently alongside 3a/3c.
    /// </summary>
    private async Task<bool> RunStationAsMlpAsync(
        string modelsRoot,
        string station,
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        PrecipClimatology climatology,
        DateTime predictionMadeAt,
        DateTime anchor,
        (int Lead, DateTime Valid)[] targets,
        IReadOnlyDictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>> perLeadValid,
        CancellationToken ct)
    {
        // Hourly EA rainfall lookup for the rich-feature persistence block.
        // SAME source as the 3c predict path; reused verbatim.
        var friendly = ResolveFriendlyStationName(station);
        if (friendly is null)
        {
            _log.LogError("Station {Station}: cannot resolve rainfall config name for phase-3e persistence lookup. Known: [{Known}]",
                station, string.Join(", ", _activeLocation.Rainfall.Stations.Select(s => s.Name)));
            return false;
        }
        var hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
            _cfg.Storage.RainfallPath, _activeLocation.Name, friendly, ct);
        _log.LogInformation("Station {Station}: loaded {N} hourly rainfall rows for 3e persistence features (friendly='{Friendly}')",
            station, hourlyRain.Count, friendly);

        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();

        // Cache loaded MLP modules per lead so we only deserialise each
        // bundle file once per cron tick (not once per (lead, valid) pair).
        var mlpByLead = new Dictionary<int, (Train.Mlp.MlpTrainer.TrainedMlp Trained, BlenderSpec Spec)>();
        Train.Mlp.MlpTrainer.TrainedMlp LoadOrGetMlp(int lead, BlenderSpec spec)
        {
            if (mlpByLead.TryGetValue(lead, out var cached)) return cached.Trained;
            var (module, cfg) = Train.Mlp.MlpArtifact.LoadLeadModel(versionDir, lead);
            var trained = new Train.Mlp.MlpTrainer.TrainedMlp(
                Module: module,
                ScalerMean: cfg.ScalerMean.ToArray(),
                ScalerScale: cfg.ScalerScale.ToArray(),
                Hyperparameters: new Train.Mlp.MlpTrainer.Hyperparameters(
                    HiddenSizes: cfg.HiddenSizes.ToArray(),
                    Dropout: cfg.Dropout, LearningRate: cfg.LearningRate,
                    BatchSize: cfg.BatchSize, MaxEpochs: cfg.MaxEpochs,
                    EarlyStoppingPatience: cfg.EarlyStoppingPatience, Seed: cfg.Seed),
                FeatureNames: cfg.FeatureNames,
                EpochsRun: cfg.EpochsRun,
                BestValBrier: cfg.BestValBrier);
            mlpByLead[lead] = (trained, spec);
            return trained;
        }

        var predictions = new List<PrecipPredictionRow>();

        foreach (var (lead, valid) in targets)
        {
            if (!perLeadValid.TryGetValue(lead, out var perValid)
                || !perValid.TryGetValue(valid, out var pivot))
                continue;
            if (!pivot.Precip.Any(p => p.HasValue))
                continue;
            if (!specs.TryGetValue(lead, out var spec))
                continue;

            int N = spec.Models.Count;
            var specPrecip = new double[N];
            var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
            var missingRequired = new List<string>();
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                var pv = pivot.Precip[ci];
                specPrecip[i] = pv ?? double.NaN;
                if (!pv.HasValue && requiredSet.Contains(spec.Models[i]))
                    missingRequired.Add(spec.Models[i]);
            }
            if (missingRequired.Count > 0) continue;

            var dew = new double[N]; var rh = new double[N];
            var dewDep = new double[N]; var pres = new double[N];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                var t  = pivot.Temp2m[ci];
                var d  = pivot.Dew[ci];
                dew[i]    = d ?? double.NaN;
                rh[i]     = pivot.Rh[ci] ?? double.NaN;
                dewDep[i] = (t.HasValue && d.HasValue) ? t.Value - d.Value : double.NaN;
                pres[i]   = pivot.Pressure[ci] ?? double.NaN;
            }
            var runTime = valid.AddHours(-lead);
            var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain, runTime);
            var row = PrecipRichFeatureBuilder.ComposeRow(
                spec, valid, specPrecip, dew, rh, dewDep, pres,
                rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                cloudHighMean: pivot.CloudHighMean,
                capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                eaRainPrev24hMm: persist.Prev24hMm,
                eaRainPrev72hMm: persist.Prev72hMm,
                eaWetHoursLast24h: persist.WetHoursLast24h,
                eaDryHoursTrailing: persist.DryHoursTrailing,
                truthMmHour: 0.0);

            var trained = LoadOrGetMlp(lead, spec);
            var pWet = Train.Mlp.MlpTrainer.PredictVectorProbability(trained, new[] { row });
            var climPWet = climatology.Predict(valid);

            var perModelPrecip = new double?[PrecipPredictionRow.PerModelFieldCount];
            var perModelRun    = new DateTime?[PrecipPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= PrecipPredictionRow.PerModelFieldCount) continue;
                perModelPrecip[ci] = specPrecip[i];
                perModelRun[ci]    = pivot.RunTime[ci];
            }
            var spreadStart = N;
            predictions.Add(new PrecipPredictionRow
            {
                LocationName = _activeLocation.Name,
                TruthStation = station,
                ModelVersion = metadata.Version,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = valid,
                LeadHours = lead,
                ProbWet = pWet[0],
                ClimatologyPWet = climPWet,
                PrecipGfs   = perModelPrecip[0], PrecipEcmwf = perModelPrecip[1],
                PrecipIcon  = perModelPrecip[2], PrecipMf    = perModelPrecip[3],
                PrecipUkmo  = perModelPrecip[4], PrecipGem   = perModelPrecip[5],
                PrecipAifs  = perModelPrecip[6], PrecipJma   = perModelPrecip[7],
                RunTimeGfs   = perModelRun[0], RunTimeEcmwf = perModelRun[1],
                RunTimeIcon  = perModelRun[2], RunTimeMf    = perModelRun[3],
                RunTimeUkmo  = perModelRun[4], RunTimeGem   = perModelRun[5],
                RunTimeAifs  = perModelRun[6], RunTimeJma   = perModelRun[7],
                PrecipMean = NanToNull(row.Features[spreadStart + 0]),
                PrecipStd  = NanToNull(row.Features[spreadStart + 1]),
                PrecipMax  = NanToNull(row.Features[spreadStart + 2]),
                PrecipAgreementWet01 = NanToNull(row.Features[spreadStart + 3]),
                FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
                ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, pWet[0]),
            });
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Station {Station}: no 3e predictions produced — likely missing forecast data.", station);
            return false;
        }

        await WritePredictionsAsync(predictions, station, anchor, metadata.Version, ct);
        return true;
    }

    private IReadOnlyList<string> ResolveStations(string modelsRoot, string truthStation)
    {
        var manifestStations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (string.Equals(truthStation, "all", StringComparison.OrdinalIgnoreCase))
            return manifestStations;

        // Accept either the slug (ea_bellever_dartmoor) or the config station name
        // (Bellever Dartmoor). The config name is the human-facing input on the CLI,
        // but the blender tree is keyed by slug.
        var explicitSlug = manifestStations.FirstOrDefault(s =>
            string.Equals(s, truthStation, StringComparison.OrdinalIgnoreCase));
        if (explicitSlug is not null)
            return new[] { explicitSlug };

        var derivedSlug = StationSlug.WithEaPrefix(truthStation);
        var match = manifestStations.FirstOrDefault(s =>
            string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        if (match is not null)
            return new[] { match };

        _log.LogError("Unknown truth station '{Station}'. Known: [{Known}]", truthStation, string.Join(", ", manifestStations));
        return Array.Empty<string>();
    }

    private async Task WritePredictionsAsync(
        IReadOnlyList<PrecipPredictionRow> predictions,
        string station,
        DateTime anchor,
        string modelVersion,
        CancellationToken ct)
    {
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outPath = Path.Combine(_cfg.Storage.PredictionsPath,
            "precipitation",
            station,
            $"model_version={modelVersion}",
            $"date={dateStr}",
            "predictions.parquet");

        var total = await PredictionParquetWriter.WriteAsync(outPath, predictions, MergeRows, ct);
        _log.LogInformation("Wrote {New} new {Station} predictions (file now holds {Total}) → {Path}",
            predictions.Count, station, total, outPath);
    }

    /// <summary>
    /// Concat existing + new precip prediction rows, dedup, return in
    /// (ValidTime, Lead) order. Dedup key is
    /// <c>(PredictionMadeAtUtc, LeadHours, ValidTimeUtc)</c>: today every cycle
    /// emits one row per (PMT, lead) — so the ValidTime in the key is a no-op
    /// and the function behaves identically to the prior 2-tuple key. The
    /// 3-tuple is load-bearing the moment the predict pipeline starts emitting
    /// multiple ValidTimes per (PMT, lead) (the upcoming hourly extension);
    /// keying on (PMT, lead) alone would silently collapse those siblings.
    /// Shared merge algorithm lives in <see cref="PredictionParquetWriter.Merge"/>;
    /// this method just pins the precip-specific key + order.
    /// </summary>
    internal static List<PrecipPredictionRow> MergeRows(
        IEnumerable<PrecipPredictionRow> existing,
        IEnumerable<PrecipPredictionRow> incoming)
        => PredictionParquetWriter.Merge(existing, incoming,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours));

    // Per-valid-time pivot mirrors TempPredictCommand.PivotedRow but carries the wider
    // precip feature set required by the occurrence blender. Per-model arrays
    // (Dew/Rh/Temp/Pressure) feed the Phase 3c rich composer; the *Mean fields stay
    // for Phase 3a's lean composer and keep the prediction-row covariate summary.
    private sealed record PivotedRow(
        double?[] Precip,
        DateTime?[] RunTime,
        double?[] Dew,
        double?[] Rh,
        double?[] Temp2m,
        double?[] Pressure,
        double RhMean,
        double DewDepressionMean,
        double CloudLowMean,
        double CloudMidMean,
        double CloudHighMean,
        double CapeMean,
        double WindSpeedMean);

    private Dictionary<DateTime, PivotedRow> QueryLatestForecastRows(
        string forecastsPath,
        string locationName,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime asOfRunTime,
        int? leadHoursLowerBound,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var liveCycleFilter = PredictForecastFilters.LiveCycleAsOf(
            locationName, asOfRunTime, earliestValid, latestValid, leadHoursLowerBound);

        // Mirrors PrecipFeatureBuilder's SQL skeleton but with the live-cycle filter
        // in place of RunTimeSource='offset_day' + LeadHours=. Pivot in .NET so we
        // can emit each model's RunTimeUtc into the prediction row for provenance.
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           Precipitation,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           Cape, WindSpeed10m, SurfacePressure,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {liveCycleFilter}
)
SELECT ValidTimeUtc, Model, RunTimeUtc,
       Precipitation,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       Cape, WindSpeed10m, SurfacePressure
FROM latest
WHERE rn = 1
ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var modelSlot = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, Index: i))
            .ToDictionary(x => x.id, x => x.Index);

        // Scratch accumulators per valid-time — the covariate means are computed
        // after the read loop so a missing model row doesn't skew the average.
        var scratch = new Dictionary<DateTime, Scratch>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = reader.GetDateTime(0);
            var model = reader.GetString(1);
            var runTime = reader.GetDateTime(2);
            var precip = reader.IsDBNull(3) ? (double?)null : reader.GetDouble(3);
            var rh     = reader.IsDBNull(4) ? (double?)null : reader.GetDouble(4);
            var t2m    = reader.IsDBNull(5) ? (double?)null : reader.GetDouble(5);
            var td     = reader.IsDBNull(6) ? (double?)null : reader.GetDouble(6);
            var cL     = reader.IsDBNull(7) ? (double?)null : reader.GetDouble(7);
            var cM     = reader.IsDBNull(8) ? (double?)null : reader.GetDouble(8);
            var cH     = reader.IsDBNull(9) ? (double?)null : reader.GetDouble(9);
            var cape   = reader.IsDBNull(10) ? (double?)null : reader.GetDouble(10);
            var wind   = reader.IsDBNull(11) ? (double?)null : reader.GetDouble(11);
            var pres   = reader.IsDBNull(12) ? (double?)null : reader.GetDouble(12);

            if (!modelSlot.TryGetValue(model, out var slot))
                continue;

            if (!scratch.TryGetValue(valid, out var s))
            {
                s = new Scratch();
                scratch[valid] = s;
            }
            s.Precip[slot]   = precip;
            s.RunTime[slot]  = runTime;
            s.Dew[slot]      = td;
            s.Rh[slot]       = rh;
            s.Temp2m[slot]   = t2m;
            s.CloudLow[slot] = cL;
            s.CloudMid[slot] = cM;
            s.CloudHigh[slot]= cH;
            s.Cape[slot]     = cape;
            s.WindSpeed[slot]= wind;
            s.Pressure[slot] = pres;
        }

        return scratch.ToDictionary(
            kv => kv.Key,
            kv => new PivotedRow(
                Precip: kv.Value.Precip,
                RunTime: kv.Value.RunTime,
                Dew: kv.Value.Dew,
                Rh: kv.Value.Rh,
                Temp2m: kv.Value.Temp2m,
                Pressure: kv.Value.Pressure,
                RhMean:            MeanOfSlots(kv.Value.Rh),
                DewDepressionMean: MeanOfDepressions(kv.Value.Temp2m, kv.Value.Dew),
                CloudLowMean:      MeanOfSlots(kv.Value.CloudLow),
                CloudMidMean:      MeanOfSlots(kv.Value.CloudMid),
                CloudHighMean:     MeanOfSlots(kv.Value.CloudHigh),
                CapeMean:          MeanOfSlots(kv.Value.Cape),
                WindSpeedMean:     MeanOfSlots(kv.Value.WindSpeed)));
    }

    private sealed class Scratch
    {
        // Indexed by canon-order position (TempFeatureBuilder.CanonicalModelOrder.IndexOf(model)).
        // Sourcing the size from CanonicalModelOrder.Count means a new NWP added to
        // the canonical order auto-grows these arrays — without it, we'd get the
        // same IndexOutOfRange that bit DryWindowPredictCommand at AIFS-add time.
        // Note: distinct from PrecipPredictionRow.PerModelFieldCount (output slots).
        private static int N => TempFeatureBuilder.CanonicalModelOrder.Count;
        public double?[] Precip { get; } = new double?[N];
        public DateTime?[] RunTime { get; } = new DateTime?[N];
        public double?[] Dew { get; } = new double?[N];
        public double?[] Rh { get; } = new double?[N];
        public double?[] Temp2m { get; } = new double?[N];
        public double?[] CloudLow { get; } = new double?[N];
        public double?[] CloudMid { get; } = new double?[N];
        public double?[] CloudHigh { get; } = new double?[N];
        public double?[] Cape { get; } = new double?[N];
        public double?[] WindSpeed { get; } = new double?[N];
        public double?[] Pressure { get; } = new double?[N];
    }

    internal static double MeanOfSlots(double?[] slots)
    {
        double sum = 0; int n = 0;
        foreach (var v in slots) if (v.HasValue) { sum += v.Value; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    internal static double MeanOfDepressions(double?[] temps, double?[] dews)
    {
        double sum = 0; int n = 0;
        for (int i = 0; i < temps.Length; i++)
        {
            if (temps[i].HasValue && dews[i].HasValue)
            {
                sum += temps[i]!.Value - dews[i]!.Value;
                n++;
            }
        }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double? NanToNull(float v) => float.IsNaN(v) ? null : v;

    /// <summary>
    /// Reverse-lookup the config rainfall-station friendly name from an "ea_..." slug
    /// so predict time can reach for the same hourly rainfall series the trainer used.
    /// </summary>
    private string? ResolveFriendlyStationName(string stationSlug)
    {
        foreach (var s in _activeLocation.Rainfall.Stations)
        {
            var slug = StationSlug.WithEaPrefix(s.Name);
            if (string.Equals(slug, stationSlug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Phase 3d (exact-runtime) predict path
    // ------------------------------------------------------------------
    //
    // Mirrors TempPredictCommand.PredictExactForVersionAsync: separate SQL
    // pivot from the exact-runtime tree (RunTimeSource='exact'), 4-cycle
    // ValidTime grid {0,6,12,18}, lead set from specs.Keys (typically {12,
    // 24}), per-V-hour UKV LEFT JOIN with target-lead-aware tuples.

    /// <summary>The 4-cycle ValidTime grid 3d trains on. Same grid as
    /// temperature 2d — UKV-conditional + IFS/MO Global cycle structure
    /// align here.</summary>
    private static readonly int[] Phase3dValidHoursUtc = { 0, 6, 12, 18 };

    private async Task<bool> RunStationAsExactRuntimeAsync(
        string modelsRoot,
        string station,
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        PrecipClimatology climatology,
        DateTime predictionMadeAt,
        DateTime anchor,
        CancellationToken ct)
    {
        var ml = new MLContext(seed: 42);
        var predictions = new List<PrecipPredictionRow>();
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        if (specs.Count == 0)
        {
            _log.LogWarning("Station {Station} version {V}: no per-lead specs in feature_schema.json", station, metadata.Version);
            return false;
        }

        var maxLead = specs.Keys.DefaultIfEmpty(0).Max();
        var windowStart = anchor;
        var windowEnd = anchor.AddHours(maxLead + 24);

        var pivotByLeadValid = QueryExactForecastRows(
            _cfg.Storage.ForecastsPath,
            _activeLocation.Name,
            specs: specs,
            earliestValid: windowStart,
            latestValid: windowEnd,
            asOfRunTime: anchor,
            ct);

        foreach (var (lead, spec) in specs.OrderBy(kv => kv.Key))
        {
            // Generate target valid times for TODAY's date + (lead/24) days
            // AND the day after, so a late-evening predict still produces
            // future-of-now lead-12 slots from tomorrow once today's are
            // all in the past. Without the +1-day extension, lead-12 at
            // 21:15 had zero future targets (all of today's {0,6,12,18}
            // were past-of-anchor → filtered out → no predictions
            // written), which surfaced as "zero 12h data on the site"
            // for the entire evening.
            var dayStart = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc).AddDays(lead / 24);
            var leadTargets = Enumerable.Range(0, 48)
                .Select(h => dayStart.AddHours(h))
                .Where(v => Phase3dValidHoursUtc.Contains(v.Hour))
                .ToList();
            var includeUkv = spec.FeatureNames.Contains("precip_ukv");

            foreach (var valid in leadTargets)
            {
                if (!pivotByLeadValid.TryGetValue((lead, valid), out var pivot))
                {
                    _log.LogDebug("Station {Station} lead {Lead}h: no exact-runtime row for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                        station, lead, valid);
                    continue;
                }

                var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
                var modelPrecip = new double[spec.Models.Count];
                var modelRunTimes = new DateTime?[spec.Models.Count];
                var missingRequired = new List<string>();
                for (int i = 0; i < spec.Models.Count; i++)
                {
                    var m = spec.Models[i];
                    if (pivot.PerModelPrecip.TryGetValue(m, out var v) && v.HasValue)
                    {
                        modelPrecip[i] = v.Value;
                        if (pivot.PerModelRunTime.TryGetValue(m, out var rt)) modelRunTimes[i] = rt;
                    }
                    else
                    {
                        modelPrecip[i] = double.NaN;
                        if (requiredSet.Contains(m)) missingRequired.Add(m);
                    }
                }
                if (missingRequired.Count > 0)
                {
                    _log.LogDebug("Station {Station} lead {Lead}h: missing required exact-runtime models [{M}] at valid={V:yyyy-MM-dd HH:mm}Z; skipping.",
                        station, lead, string.Join(",", missingRequired), valid);
                    continue;
                }

                var ukvPrecip = pivot.UkvPrecip;
                var row = PrecipExactFeatureBuilder.ComposeRow(
                    spec, valid, modelPrecip, truthMmHour: 0.0, ukvPrecip: ukvPrecip);

                var loadedModel = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
                var pWet = PrecipOccurrenceTrainer.PredictVectorProbability(ml, loadedModel, spec, new[] { row });
                var climPWet = climatology.Predict(valid);

                // Map spec.Models → exact column slots by model id (not slot
                // index) so reordered specs still populate the right columns.
                double? gfs = null, ifs = null, aifs = null, mog = null;
                DateTime? gfsR = null, ifsR = null, aifsR = null, mogR = null;
                for (int i = 0; i < spec.Models.Count; i++)
                {
                    var v = double.IsNaN(modelPrecip[i]) ? (double?)null : modelPrecip[i];
                    switch (spec.Models[i])
                    {
                        case "gfs_ncep":          gfs = v;  gfsR  = modelRunTimes[i]; break;
                        case "ecmwf_ifs_oper":    ifs = v;  ifsR  = modelRunTimes[i]; break;
                        case "ecmwf_aifs_oper":   aifs = v; aifsR = modelRunTimes[i]; break;
                        case "met_office_global": mog = v;  mogR  = modelRunTimes[i]; break;
                    }
                }
                var ukvCol = includeUkv && !double.IsNaN(ukvPrecip) ? (double?)ukvPrecip : null;

                // FeatureNames layout: per-model precip, [optional ukv], mean,
                // std, range, 4 calendar floats. Spread block at index
                // spec.Models.Count + (ukv ? 1 : 0).
                var spreadStart = spec.Models.Count + (includeUkv ? 1 : 0);
                predictions.Add(new PrecipPredictionRow
                {
                    LocationName = _activeLocation.Name,
                    TruthStation = station,
                    ModelVersion = metadata.Version,
                    PredictionMadeAtUtc = predictionMadeAt,
                    ValidTimeUtc = valid,
                    LeadHours = lead,
                    ProbWet = pWet[0],
                    ClimatologyPWet = climPWet,
                    // Offset_day per-model slots stay null on 3d rows.
                    PrecipGfsExact      = gfs,  RunTimeGfsExact      = gfsR,
                    PrecipIfsOperExact  = ifs,  RunTimeIfsOperExact  = ifsR,
                    PrecipAifsOperExact = aifs, RunTimeAifsOperExact = aifsR,
                    PrecipMoGlobalExact = mog,  RunTimeMoGlobalExact = mogR,
                    PrecipUkvExact      = ukvCol, RunTimeUkvExact    = pivot.UkvRunTime,
                    PrecipMean = NanToNull(row.Features[spreadStart + 0]),
                    PrecipStd  = NanToNull(row.Features[spreadStart + 1]),
                    PrecipMax  = NanToNull(row.Features[spreadStart + 2]),
                    // Range, not max — exact builder uses range; the offset_day
                    // builder used max. Reuse the parquet column anyway since
                    // it's a "spread of per-model precip" semantically. Note
                    // for any reader: PrecipMax on 3d rows is range, on 3a/3c
                    // it's max. Inspect Phase before interpreting.
                    PrecipAgreementWet01 = null,
                    FeatureVectorHash = FeatureHashing.HashFloats(row.Features),
                    ConformalSetTag = ModelArtifact.PredictConformalIfPresent(versionDir, lead, pWet[0]),
                });

                _log.LogInformation(
                    "Station {Station} lead {Lead}h → P(wet) {P:0.000} (clim {Clim:0.000}, valid {Valid:yyyy-MM-dd HH:mm}Z, exact-runtime mean-of-{N} {Mean:0.000})",
                    station, lead, pWet[0], climPWet, valid, spec.Models.Count, row.Features[spreadStart + 0]);
            }
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Station {Station}: no exact-runtime predictions produced — likely missing forecast data.", station);
            return false;
        }

        await WritePredictionsAsync(predictions, station, anchor, metadata.Version, ct);
        return true;
    }

    /// <summary>Per-(lead, ValidTime) pivot from the exact-runtime tree.
    /// Per-model dictionaries keyed by canonical model id rather than slot
    /// index — keeps the dispatch above tolerant to spec.Models reordering
    /// across phases. UKV pulled per blender lead via UNION ALL with the
    /// per-targetLead picks from Exact12hFeatureBuilder.UkvPerVOrClause.</summary>
    private sealed record ExactPivotRow(
        Dictionary<string, double?> PerModelPrecip,
        Dictionary<string, DateTime> PerModelRunTime,
        double UkvPrecip,
        DateTime? UkvRunTime);

    private Dictionary<(int Lead, DateTime ValidTime), ExactPivotRow> QueryExactForecastRows(
        string forecastsPath,
        string locationName,
        IReadOnlyDictionary<int, BlenderSpec> specs,
        DateTime earliestValid,
        DateTime latestValid,
        DateTime asOfRunTime,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (specs.Count == 0) return new();

        var specLeads = specs.Keys.OrderBy(l => l).ToArray();
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var loc = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",",
            PrecipExactFeatureBuilder.CanonicalModelOrder.Select(m => $"'{m}'")) + ")";
        var leadInClause = "(" + string.Join(",", specLeads) + ")";
        var hourInClause = "(" + string.Join(",", Phase3dValidHoursUtc) + ")";

        // Same UKV-leads gating as TempPredictCommand.QueryExactForecastRows
        // — see the docstring there for the bug + fix history (2026-05-08
        // production crash on no-UKV long-lead 3d versions).
        var ukvLeads = specs
            .Where(kv => kv.Value.FeatureNames.Contains("precip_ukv", StringComparer.Ordinal))
            .Select(kv => kv.Key)
            .OrderBy(l => l)
            .ToArray();

        var latestCte = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, LeadHours, RunTimeUtc, Precipitation,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model, LeadHours
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{loc}'
      AND RunTimeSource = 'exact'
      AND LeadHours IN {leadInClause}
      AND HOUR(ValidTimeUtc) IN {hourInClause}
      AND Precipitation IS NOT NULL
      AND Model IN {modelInClause}
      AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)";

        string sql;
        if (ukvLeads.Length > 0)
        {
            var ukvUnionLegs = string.Join("\n        UNION ALL\n",
                ukvLeads.Select(blenderLead => $@"        SELECT
            ValidTimeUtc, {blenderLead} AS BlenderLead,
            Precipitation AS ukv_precip, RunTimeUtc AS ukv_run_time,
            ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn2
        FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{loc}'
          AND Model = 'met_office_ukv'
          AND RunTimeSource = 'exact'
          AND Precipitation IS NOT NULL
          AND HOUR(ValidTimeUtc) IN {hourInClause}
          AND RunTimeUtc <= TIMESTAMP '{asOfRunTime:yyyy-MM-dd HH:mm:ss}'
          AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                               AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
          AND ({Exact12hFeatureBuilder.UkvPerVOrClause(blenderLead, Exact12hFeatureBuilder.UkvPickStrategy.Averaging)})"));

            sql = $@"{latestCte},
ukv AS (
    SELECT ValidTimeUtc, BlenderLead, ukv_precip, ukv_run_time
    FROM (
{ukvUnionLegs}
    )
    WHERE rn2 = 1
)
SELECT l.ValidTimeUtc, l.Model, l.LeadHours, l.RunTimeUtc, l.Precipitation,
       u.ukv_precip, u.ukv_run_time
FROM latest l
LEFT JOIN ukv u
       ON l.ValidTimeUtc = u.ValidTimeUtc
      AND l.LeadHours    = u.BlenderLead
WHERE l.rn = 1
  AND l.RunTimeUtc <= l.ValidTimeUtc - (l.LeadHours * INTERVAL 1 HOUR)
ORDER BY l.ValidTimeUtc, l.LeadHours, l.Model;";
        }
        else
        {
            sql = $@"{latestCte}
SELECT l.ValidTimeUtc, l.Model, l.LeadHours, l.RunTimeUtc, l.Precipitation,
       CAST(NULL AS DOUBLE) AS ukv_precip, CAST(NULL AS TIMESTAMP) AS ukv_run_time
FROM latest l
WHERE l.rn = 1
  AND l.RunTimeUtc <= l.ValidTimeUtc - (l.LeadHours * INTERVAL 1 HOUR)
ORDER BY l.ValidTimeUtc, l.LeadHours, l.Model;";
        }

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var reader = cmd.ExecuteReader();

        var result = new Dictionary<(int Lead, DateTime ValidTime), ExactPivotRow>();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid    = reader.GetDateTime(0);
            var model    = reader.GetString(1);
            var lead     = reader.GetInt32(2);
            var runTime  = reader.GetDateTime(3);
            var precip   = NullableDouble(reader, 4);
            var ukvPrecip = NullableDouble(reader, 5) ?? double.NaN;
            var ukvRunTime = reader.IsDBNull(6) ? (DateTime?)null : reader.GetDateTime(6);

            var key = (lead, valid);
            if (!result.TryGetValue(key, out var existing))
            {
                existing = new ExactPivotRow(
                    PerModelPrecip: new Dictionary<string, double?>(StringComparer.Ordinal),
                    PerModelRunTime: new Dictionary<string, DateTime>(StringComparer.Ordinal),
                    UkvPrecip: ukvPrecip,
                    UkvRunTime: ukvRunTime);
                result[key] = existing;
            }
            existing.PerModelPrecip[model] = precip;
            if (precip.HasValue) existing.PerModelRunTime[model] = runTime;
        }
        return result;
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);
}
