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
using WeatherBlend.Train.Oro;
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

    private static readonly int[] DefaultLeads = Leads.Full;

    /// <summary>
    /// Lead buckets actually targeted per cycle: the standard day buckets
    /// PLUS the lead-12 bucket (TODAY's hours, 2026-06-10 — the short-lead
    /// cell of docs/PRECIP_LEAD_POLICY_PLAN.md). Lead 12 is served by the
    /// 3c/3o paths ONLY (3a filters it out): no bundle carries a lead-12
    /// model — training data is day-granular previous_day_N — so lead-12
    /// targets run the lead-24 model (or the policy's pick) on input the
    /// lead-12 pivot cycle-selects (RunTime ≤ Valid − 12, i.e. fresher than
    /// the model's nominal lead; the cross-lead freshness finding).
    /// </summary>
    private static readonly int[] TargetLeads = new[] { 12 }.Concat(Leads.Full).ToArray();

    /// <summary>
    /// Phases this command's predict path knows how to dispatch (L2 of the
    /// two-layer phase gating in <see cref="RunStationAsync"/>). Of the
    /// active precipitation phases in phases.yaml, these are the ones
    /// PrecipPredictCommand serves — the remainder are skipped silently
    /// because they're handled elsewhere:
    ///   * <c>4a</c>            — impl=python, served by predict-4a.yml
    ///                              in WeatherProbabilistic.
    ///   * <c>4b</c>           — impl=dotnet but synthesised by
    ///                            <see cref="Phase4bPredictCommand"/> (its
    ///                            own CLI subcommand, invoked as a separate
    ///                            step inside predict-and-render.yml).
    /// Adding a new precipitation phase to phases.yaml that THIS command
    /// should handle = add its id here too. The L1 gate (phases.yaml
    /// membership) catches the retired-phase case automatically — no
    /// hardcoded suffix-blocklist needed.
    /// </summary>
    // internal (not private) so PhaseWiringConsistencyTests can assert the
    // set stays a subset of phases.yaml's precipitation lineup.
    internal static readonly HashSet<string> HandledPrecipPhases =
        new(StringComparer.Ordinal) { "3a", "3c", "3d", "3o", "3oni" };

    public PrecipPredictCommand(ILogger<PrecipPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
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
        var (location, locRc) = PredictLocationResolver.Resolve(_cfg, locationOverride, _log);
        if (location is null) return locRc;

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
        var locationSlugs = location.Rainfall.Stations
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
                dropped.Count, beforeCount, location.Name, string.Join(", ", dropped));
        }
        if (stationsToRun.Count == 0)
        {
            _log.LogError(
                "No precipitation blender artefacts under {Dir} match location '{Loc}' rainfall stations [{Configured}]. " +
                "Either add the station to that location's config or pass --location <name>.",
                Path.Combine(modelsRoot, "precipitation"),
                location.Name,
                string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var targets = BuildHourlyTargets(anchor, TargetLeads);

        // LEAD_POLICY.json (docs/PRECIP_LEAD_POLICY_PLAN.md): optional per-6h-
        // band model deviations for 3c/3o. Null (absent / unreadable) means
        // "production bucket models" everywhere — predict must never break on
        // a policy problem, so any load failure degrades to today's behaviour.
        // Per-location policy (falls back to the legacy global file until the
        // per-location fit has run for this location).
        var leadPolicy = PrecipLeadPolicy.TryLoad(modelsRoot, "precipitation", location.Name);
        if (leadPolicy is not null)
            _log.LogInformation(
                "Lead policy loaded (fitted {Fit:yyyy-MM-dd}, {N} deviation band(s)); absent bands use bucket models.",
                leadPolicy.FittedAtUtc, leadPolicy.Phases.Sum(p => p.Value.Count));

        _log.LogInformation("Anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate}) — stations=[{Stations}] — {Count} targets across leads {Leads}",
            anchor,
            forDate?.ToString("yyyy-MM-dd") ?? "live",
            string.Join(", ", stationsToRun),
            targets.Length,
            string.Join(",", TargetLeads));

        // One forecast pivot per lead bucket — each call enforces the lead-bucket
        // training constraint (RunTime ≤ Valid - L) so every row passed to the
        // lead-L model sits in the [L, L+24h) actual-lead band the model was
        // trained on (Open-Meteo's previous_day_(L/24) aggregation). Without this
        // filter, "latest cycle wins" would feed the lead-24 model rows at
        // actual lead < 24h once we predict for valid times within 24h of anchor.
        var perLeadValid = new Dictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>>();
        foreach (var lead in TargetLeads)
        {
            var leadTargets = targets.Where(t => t.Lead == lead).ToArray();
            if (leadTargets.Length == 0) continue;
            var pivot = QueryLatestForecastRows(
                _cfg.Storage.ForecastsPath,
                location.Name,
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
                    modelsRoot, station, version, predictionMadeAt, anchor, targets, perLeadValid, leadPolicy, location, ct);
                anyWritten |= wrote;
            }
        }

        // Ungauged Bonehill-tor rain line (3oni). The pooled no-station-id
        // model can extrapolate to a terrain it never saw at train time, so we
        // evaluate it at the tor's own orographic vector. There is no tor gauge
        // — this borrows the nearest gauge's (Bellever) bundle + NWP forecasts
        // + rainfall-persistence and only swaps the terrain block + output key.
        // Experimental / not gauge-verified; runs only for the Bonehill site.
        if (string.Equals(location.Name, TorPoiLocation, StringComparison.Ordinal))
            anyWritten |= await RunTorPointOfInterestAsync(
                modelsRoot, predictionMadeAt, anchor, targets, perLeadValid, leadPolicy, location, ct);

        return anyWritten ? 0 : 3;
    }

    // The Bonehill tor point-of-interest line: location name, the reference
    // gauge whose bundle/forecasts/persistence we borrow, and the station key
    // (== the orographic slug data/static/orographic/bonehill_rocks.json) the
    // tor predictions are written under.
    private const string TorPoiLocation = "bonehill_rocks";
    private const string TorReferenceGauge = "Bellever Dartmoor";
    private const string TorStationSlug = "bonehill_rocks";

    /// <summary>
    /// Predict the experimental ungauged-tor 3oni line. Resolves the reference
    /// gauge's newest 3oni bundle, then runs the oro predict path with the
    /// terrain read from <c>bonehill_rocks.json</c> and the rows written under
    /// the tor station key. No-op (returns false) if 3oni hasn't been trained
    /// yet — expected until the first Sunday retrain produces the bundle.
    /// </summary>
    private async Task<bool> RunTorPointOfInterestAsync(
        string modelsRoot,
        DateTime predictionMadeAt,
        DateTime anchor,
        (int Lead, DateTime Valid)[] targets,
        IReadOnlyDictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>> perLeadValid,
        PrecipLeadPolicy? leadPolicy,
        LocationConfig location,
        CancellationToken ct)
    {
        var refGauge = StationSlug.WithEaPrefix(TorReferenceGauge);
        var version = ModelArtifact.ResolveStationPhaseVersion(modelsRoot, "precipitation", refGauge, "3oni");
        if (string.IsNullOrEmpty(version))
        {
            _log.LogInformation(
                "Tor 3oni line: no 3oni bundle for reference gauge {Gauge} yet — skipping (expected until the first retrain trains 3oni).",
                refGauge);
            return false;
        }

        var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", refGauge, version);
        var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
        if (metadata.PerLead.Count == 0)
        {
            _log.LogError("Tor 3oni line: reference bundle {V} has no per-lead blenders.", version);
            return false;
        }
        if (!string.Equals(metadata.Phase, "3oni", StringComparison.Ordinal))
        {
            _log.LogError("Tor 3oni line: resolved bundle {V} has phase '{Phase}', expected 3oni.", version, metadata.Phase);
            return false;
        }

        var climPath = Path.Combine(versionDir, ModelArtifact.ClimatologyFileName);
        if (!File.Exists(climPath))
        {
            _log.LogError("Tor 3oni line: reference bundle {V} missing {File}.", version, ModelArtifact.ClimatologyFileName);
            return false;
        }
        var climatology = PrecipClimatology.LoadFrom(climPath);

        _log.LogInformation(
            "Tor 3oni line: borrowing gauge {Gauge} bundle {V} (phase=3oni); terrain={Terrain}.json, writing under station '{Out}'.",
            refGauge, version, TorStationSlug, TorStationSlug);

        return await RunStationAsOroAsync(
            modelsRoot, refGauge, versionDir, metadata, climatology,
            predictionMadeAt, anchor, targets, perLeadValid, leadPolicy, location, ct,
            oroSlugOverride: TorStationSlug, outputStation: TorStationSlug);
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
        PrecipLeadPolicy? leadPolicy,
        LocationConfig location,
        CancellationToken ct)
    {
        // Python-trained phases (4a per-cell BART) live in
        // WeatherProbabilistic and have their own dedicated predict
        // workflow (predict-4a.yml). Their bundles end up in the same
        // data/models/precipitation/{station}/ tree + their versions are
        // auto-promoted into MANIFEST.json's Active (since 2026-05-12),
        // so once-around the Active loop in this .NET PrecipPredictCommand
        // they'd hit `metadata.Phase == "4a"` and try to load a LightGBM
        // blender that isn't there. Worse — train_4a.py writes
        // BestSingleTestMae=null which the .NET PerLeadStats deserialiser
        // rejects (System.Double can't accept null), so we crash BEFORE
        // reaching the dispatch. Cheapest fix: sniff the version-name
        // suffix and skip without trying to load.
        // Two-layer phase gating (introduced 2026-05-26 to replace the
        // older hardcoded-suffix guards). Source of truth:
        //   L1 (project-wide): phases.yaml — "is this phase active in our
        //                      project AT ALL?" — answered by
        //                      PhaseRegistry.AllPhases(target).
        //   L2 (per-command):  HandledPhases — "of the active phases, which
        //                      does THIS command serve?" — explicit set
        //                      below so adding a new phase requires an
        //                      intentional code change matching the YAML.
        //
        // L1 catches orphaned bundles from retired phases still living in
        // the manifest's Active list. PromoteStationVersion only replaces
        // same-phase entries — retired bundles never age out on their own.
        // (Caught 2026-05-26 03:18 UTC by predict-and-render run
        // 26430107996 — a stale retired-phase bundle hit the default
        // lean/rich row build and threw "Feature pack mismatch".)
        //
        // L2 catches valid-but-not-mine phases: 4a (impl=python, served by
        // predict-4a.yml), 4b (impl=dotnet but its own CLI command
        // Phase4bPredictCommand handles it inside predict-and-render's
        // "Synthesise Phase 4b" step).
        var phaseFromSuffix = ModelArtifact.ExtractPhaseFromVersionName(modelVersion);
        if (phaseFromSuffix is not null)
        {
            var registeredPrecipPhases = PhaseRegistry.Default.AllPhases("precipitation");
            var inYaml = registeredPrecipPhases
                .Any(p => string.Equals(p.Id, phaseFromSuffix, StringComparison.Ordinal));
            if (!inYaml)
            {
                _log.LogInformation(
                    "Station {Station}: skipping {V} — phase '{Phase}' not in phases.yaml " +
                    "(orphaned bundle in manifest Active; will leave on next promote sweep).",
                    station, modelVersion, phaseFromSuffix);
                return true;
            }
            if (!HandledPrecipPhases.Contains(phaseFromSuffix))
            {
                _log.LogInformation(
                    "Station {Station}: skipping {V} — phase '{Phase}' active in phases.yaml " +
                    "but served by a different predict path. PrecipPredictCommand handles only [{Handled}].",
                    station, modelVersion, phaseFromSuffix, string.Join(", ", HandledPrecipPhases));
                return true;
            }
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
        if (!string.Equals(metadata.LocationName, location.Name, StringComparison.OrdinalIgnoreCase))
        {
            _log.LogError(
                "Station {Station} bundle {V} was trained on location '{Trained}' but predict is using NWP from '{Active}' — refusing to score. " +
                "Pass --location {TrainedRetry} or fix the manifest entry.",
                station, modelVersion, metadata.LocationName, location.Name, metadata.LocationName);
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

        // Phase 3o (pooled rich-oro) takes a separate predict path: same
        // 55-feat rich row as 3c PLUS a 9-feat terrain block whose dynamic
        // entries come from NWP-ensemble-mean wind / temp / dew / pressure
        // at each target valid_time. Predict reads the station's static
        // orographic JSON + computes the terrain block via the SAME
        // ComposeTerrainBlock helper the trainer used.
        // 3o and 3oni (no-station-id tor variant) share the oro predict path.
        var isOro = metadata.Phase is "3o" or "3oni";
        if (isOro)
        {
            return await RunStationAsOroAsync(
                modelsRoot, station, versionDir, metadata, climatology,
                predictionMadeAt, anchor, targets, perLeadValid, leadPolicy, location, ct);
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
                predictionMadeAt, anchor, location, ct);
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
            var friendly = ResolveFriendlyStationName(station, location);
            if (friendly is null)
            {
                _log.LogError("Station {Station}: cannot resolve rainfall config name for phase-3c persistence lookup. Known: [{Known}]",
                    station, string.Join(", ", location.Rainfall.Stations.Select(s => s.Name)));
                return false;
            }
            hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
                _cfg.Storage.RainfallPath, location.Name, friendly, minValidTime: null, ct);
            _log.LogInformation("Station {Station}: loaded {N} hourly rainfall rows for persistence features (friendly='{Friendly}')",
                station, hourlyRain.Count, friendly);
        }

        var ml = new MLContext(seed: 42);
        var predictions = new List<PrecipPredictionRow>();

        // Both lean (3a) and rich (3c) artefacts use the per-lead BlenderSpec layout.
        // Rich's spec just has a longer feature vector. Schema is read from feature_schema.json.
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();

        // Lead-12 targets (today's hours) are a 3c-only product on this path —
        // 3a's short-lead skill is unstudied, 3o/3d have their own paths. See
        // TargetLeads. Policy lookups below also no-op for 3a (no "3a" key in
        // LEAD_POLICY.json by construction).
        var is3c = string.Equals(metadata.Phase, "3c", StringComparison.Ordinal);
        var phaseTargets = is3c ? targets : targets.Where(t => t.Lead != 12).ToArray();

        // Per-lead model cache — policy blends run two bundle models per
        // target, and the old load-per-target pattern would double that cost.
        var modelCache = new Dictionary<int, ITransformer>();
        ITransformer ModelFor(int modelLead)
        {
            if (!modelCache.TryGetValue(modelLead, out var m))
                modelCache[modelLead] = m = ModelArtifact.LoadLeadModel(ml, versionDir, modelLead, out _);
            return m;
        }

        // Pre-load live upper-air per lead for any lead whose schema carries the
        // UA block (3c-with-UA, 2026-06-02). Mirrors the training exact_ua ASOF:
        // freshest exact lead-L pressure ≤ target valid. One DuckDB scan per UA
        // lead; rows ASOF-matched per target at compose time below. Empty until
        // the live collector has accumulated exact pressure — then UA slots are
        // NaN (missing), which the model tolerates.
        var uaByLead = new Dictionary<int, IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)>>();
        if (isRich)
        {
            foreach (var lead in phaseTargets.Select(t => t.Lead).Distinct())
            {
                // No bundle carries a lead-12 spec (no native m12 model) — use
                // the m24 spec to decide whether the UA block is on, but pull
                // the exact-pressure ASOF rows AT the target lead so the input
                // stays τ-appropriate. An empty lead-12 UA tree just yields
                // the NaN UA block the model tolerates.
                var specLead = lead == 12 ? 24 : lead;
                if (!specs.TryGetValue(specLead, out var s) || !s.FeatureNames.Contains("t850_mean")) continue;
                var lv = phaseTargets.Where(t => t.Lead == lead).Select(t => t.Valid).ToList();
                uaByLead[lead] = PrecipFeatureBuilder.LoadUpperAirLive(
                    _cfg.Storage.ForecastsPath, location.Name, lead, lv.Min().AddDays(-2), lv.Max(), ct);
                _log.LogInformation("Station {Station} lead {Lead}h: loaded {N} live upper-air rows (exact pressure ASOF).",
                    station, lead, uaByLead[lead].Count);
            }
        }

        foreach (var (lead, valid) in phaseTargets)
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
            // Which bundle model lead(s) serve this target: the lead policy's
            // pick for (phase, τ) when one exists (3c — though 3c is locked
            // singles-only at fit time), else the bucket default (lead 12 →
            // m24; every other bucket → its own model). Missing model files
            // degrade to the default — policy can never break predict.
            var modelLeads = ResolveModelLeads(leadPolicy, metadata.Phase, lead, valid, anchor, specs);
            if (!specs.TryGetValue(modelLeads[0], out var spec))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no BlenderSpec for model lead {M}h in feature_schema.json; skipping.",
                    station, lead, modelLeads[0]);
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
            // uaIncluded: null = phase never uses UA (lean 3a / legacy 3c); set
            // true/false in the rich branch per whether real UA values were in
            // hand for this valid-time (drives the site's UA-backed shading).
            BinaryTrainingRow row;
            bool? uaIncluded = null;
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
                // Upper-air block (3c-with-UA) — freshest exact lead-L pressure
                // ≤ this valid, in the same model-major order the trainer used.
                // Null when the schema has no UA (legacy 3c bundles) so ComposeRow
                // emits the legacy-length vector unchanged.
                double[]? uaValues = spec.FeatureNames.Contains("t850_mean")
                    ? PrecipFeatureBuilder.UpperAirValuesFor(
                        uaByLead.TryGetValue(lead, out var asof) ? asof : System.Array.Empty<(DateTime, double[])>(), valid)
                    : null;
                // UA-capable bundle (t850_mean in schema): record whether real
                // (non-NaN) UA values were actually present for this valid-time.
                if (spec.FeatureNames.Contains("t850_mean"))
                    uaIncluded = uaValues is not null && uaValues.Any(v => !double.IsNaN(v));
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
                    truthMmHour: 0.0,
                    upperAir: uaValues);
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

            // Primary member predicts on the composed row; extra members (an
            // equal-weight policy blend) reuse the SAME row — admissible only
            // when their spec has identical model membership + feature layout,
            // which per-lead 3c specs share. A layout mismatch drops the extra
            // member (single-model fallback) rather than mis-composing.
            var pVals = new List<double>(modelLeads.Length)
            {
                PrecipOccurrenceTrainer.PredictVectorProbability(ml, ModelFor(modelLeads[0]), spec, new[] { row })[0],
            };
            foreach (var extra in modelLeads.Skip(1))
            {
                if (specs.TryGetValue(extra, out var specX)
                    && specX.FeatureCount == spec.FeatureCount
                    && specX.Models.SequenceEqual(spec.Models, StringComparer.OrdinalIgnoreCase))
                    pVals.Add(PrecipOccurrenceTrainer.PredictVectorProbability(ml, ModelFor(extra), specX, new[] { row })[0]);
                else
                    _log.LogWarning(
                        "Station {Station} lead {Lead}h: policy blend member m{Extra}h has an incompatible spec layout — using the remaining member(s) only.",
                        station, lead, extra);
            }
            var pWet = new[] { pVals.Average() };

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
                LocationName = location.Name,
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
                // Conformal tags are calibrated per (bundle, lead) on that
                // lead's val distribution — only attach when this row ran the
                // plain bucket model. Policy blends / lead-12 m24 rows get no
                // tag rather than a falsely-calibrated one.
                ConformalSetTag = modelLeads.Length == 1 && modelLeads[0] == lead
                    ? ModelArtifact.PredictConformalIfPresent(versionDir, lead, pWet[0])
                    : null,
                UpperAirIncluded = uaIncluded,
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

    // ---- Phase 3o (pooled rich-oro) predict path ----------------------------------
    //
    // Same shape as RunStationAsync's rich (3c) path BUT with two changes:
    //   1. The feature row gets an additional 9-feature terrain block
    //      (PrecipRichOroFeatureBuilder.ComposeTerrainBlock) appended after
    //      the 55-feat rich vector. Total feature dim 64.
    //   2. The terrain block needs NWP-ensemble-mean wind / temp / dew /
    //      pressure per target valid_time. Predict pulls these via
    //      PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLIVE — the live
    //      forecast tree (RunTimeSource != 'offset_day'), bounded by the
    //      target valid_time window. The train-time LoadAuxNwpMeans reads
    //      the historical offset_day backfill which has no rows for
    //      future valid_times — caught 2026-05-26 when every (station,
    //      lead, valid) skipped with "no NWP-mean aux row for terrain
    //      block" and the live cycle wrote zero 3o predictions.
    //
    // Station-index mapping (the 9th terrain feature, oro_station_id) MUST
    // match the train-time mapping in PrecipTrainCommand.BonehillStationOrder3o:
    //   Bellever=0, Bovey=1, Hexworthy=2, Princetown=3.
    // The mapping is also persisted in metadata.Hyperparameters["phase3o_station_index"]
    // for cross-check.

    /// <summary>
    /// Which bundle model lead(s) serve a (bucket, valid) target. Order of
    /// precedence:
    ///   1. LEAD_POLICY.json deviation for (phase, τ) — τ is the target's
    ///      ACTUAL lead (valid − anchor), so a band boundary can fall inside
    ///      a day bucket (each hourly target resolves its own band);
    ///   2. the bucket default: the bucket's own model, except lead 12 which
    ///      has no native model and defaults to m24 (the short-lead cell —
    ///      m24 fed the lead-12 pivot's fresher input).
    /// A policy entry referencing a model lead this bundle doesn't carry
    /// degrades to the default — predict never breaks on a policy problem.
    /// </summary>
    private static int[] ResolveModelLeads(
        PrecipLeadPolicy? policy, string phase, int bucketLead, DateTime valid, DateTime anchor,
        IReadOnlyDictionary<int, BlenderSpec> specs)
    {
        var fallback = new[] { bucketLead == 12 ? 24 : bucketLead };
        if (policy is null) return fallback;
        var tau = (valid - anchor).TotalHours;
        var entry = policy.Lookup(phase, tau);
        if (entry is null || entry.Leads.Count == 0) return fallback;
        return entry.Leads.All(specs.ContainsKey) ? entry.Leads.ToArray() : fallback;
    }

    /// <summary>Train-time canonical station-index mapping for Phase 3o.
    /// Must stay in lockstep with <c>PrecipTrainCommand.BonehillStationOrder3o</c>.
    /// Keyed by station slug (with ea_ prefix).</summary>
    private static readonly IReadOnlyDictionary<string, int> Phase3oStationIndex =
        new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            ["ea_bellever_dartmoor"]      = 0,
            ["ea_bovey_tracey"]           = 1,
            ["ea_dartmoor_nr_hexworthy"]  = 2,
            ["ea_princetown"]             = 3,
        };

    private async Task<bool> RunStationAsOroAsync(
        string modelsRoot,
        string station,
        string versionDir,
        ModelArtifact.TrainingMetadata metadata,
        PrecipClimatology climatology,
        DateTime predictionMadeAt,
        DateTime anchor,
        (int Lead, DateTime Valid)[] targets,
        IReadOnlyDictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>> perLeadValid,
        PrecipLeadPolicy? leadPolicy,
        LocationConfig location,
        CancellationToken ct,
        // Ungauged-tor line (3oni): borrow this station's BUNDLE + forecasts +
        // rainfall-persistence, but read terrain from oroSlugOverride and write
        // under outputStation. Both null = normal per-gauge predict.
        string? oroSlugOverride = null,
        string? outputStation = null)
    {
        var oroSlug = oroSlugOverride ?? station;
        var outStation = outputStation ?? station;

        // 3oni = the no-station-id variant; its bundle schema carries 8 terrain
        // features (no oro_station_id) vs 3o's 9. Drive the terrain count +
        // compose off the bundle's phase so the predict vector matches the
        // trained schema exactly.
        var includeStationId = !string.Equals(metadata.Phase, "3oni", StringComparison.Ordinal);

        // Resolve the station's terrain index. Bundle metadata carries it
        // too — guard against a slug typo in either source.
        if (!Phase3oStationIndex.TryGetValue(station, out var stationIndex))
        {
            _log.LogWarning("Station {Station}: Phase 3o bundle present but slug not in BonehillStationOrder3o map — skipping.",
                station);
            return true;
        }
        // HpInt unwraps JsonElement → int — Convert.ToInt32 can't handle
        // JsonElement (InvalidCastException, caught 2026-05-26 in retrain
        // rerun 26437976508). HpInt returns null on missing keys / wrong
        // types so the != cross-check is silently skipped on legacy
        // bundles that don't carry the station_index stamp.
        var metaIdx = metadata.Hyperparameters.HpInt("phase3o_station_index");
        if (metaIdx is not null)
        {
            if (metaIdx.Value != stationIndex)
            {
                _log.LogError(
                    "Station {Station} Phase 3o bundle {V} metadata station_index={Meta} disagrees with predict-time map={Live} — refusing to predict (train/predict feature drift).",
                    station, metadata.Version, metaIdx, stationIndex);
                return false;
            }
        }

        // Load static orographic features for this station. Disk path
        // derived from cfg.Storage.ForecastsPath's parent — matches
        // PrecipTrainCommand's resolution so train + predict read from
        // the same place (and stays config-driven rather than CWD-relative
        // so parallel test runs don't collide on the process-global CWD).
        var oroPath = Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!,
            "static", "orographic", $"{oroSlug}.json");
        if (!File.Exists(oroPath))
        {
            _log.LogError("Station {Station}: orographic record missing at {Path} — cannot Phase 3o predict.",
                oroSlug, oroPath);
            return false;
        }
        var oroBySlug = OroStaticFeatures.LoadAll(Path.GetDirectoryName(oroPath)!);
        if (!oroBySlug.TryGetValue(oroSlug, out var oro))
        {
            _log.LogError("Station {Station}: orographic record at {Path} parsed but no entry for slug.",
                oroSlug, oroPath);
            return false;
        }

        var ml = new MLContext(seed: 42);
        var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();

        // Live-tree NWP-mean aux for the target valid-time window. Loaded
        // ONCE for the whole predict run (not per lead) — the live tree's
        // freshest-per-(valid, model) pick is lead-independent, so a single
        // bounded query covers every (lead, valid) tuple we'll ask about.
        // Cheaper than the per-lead caching the buggy v1 used, and avoids
        // re-issuing the same SQL across leads.
        var auxValidWindow = targets.Length == 0
            ? (Earliest: DateTime.MinValue, Latest: DateTime.MaxValue)
            : (Earliest: targets.Min(t => t.Valid), Latest: targets.Max(t => t.Valid));
        // Use any lead's spec for the Model IN list — the spec's NWP
        // membership doesn't vary by lead in the rich-oro spec. Pick the
        // first spec we have so we don't have to special-case empty.
        var anySpec = specs.Values.FirstOrDefault();
        Dictionary<DateTime, PrecipRichOroFeatureBuilder.NwpMeanRow> nwpMeanByValid;
        if (anySpec is not null)
        {
            nwpMeanByValid = PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(
                _cfg.Storage.ForecastsPath, location.Name, anySpec,
                auxValidWindow.Earliest, auxValidWindow.Latest, ct);
            _log.LogInformation(
                "Station {Station}: loaded {N} NWP-mean aux rows from LIVE tree for valid window {S:yyyy-MM-dd HH:mm}Z..{E:yyyy-MM-dd HH:mm}Z (3o predict).",
                station, nwpMeanByValid.Count, auxValidWindow.Earliest, auxValidWindow.Latest);
        }
        else
        {
            nwpMeanByValid = new();
        }

        // Live upper-air per lead (3o-with-UA, 2026-06-02) — freshest exact
        // lead-L pressure ASOF, same loader the 3c predict path uses. Composed
        // into the rich half of the row below (layout [rich || UA || terrain]).
        var uaByLead = new Dictionary<int, IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)>>();
        foreach (var lead in targets.Select(t => t.Lead).Distinct())
        {
            // Lead 12 has no bundle spec — gate the UA load on the m24 spec
            // but pull the exact-pressure ASOF rows at the target lead (an
            // empty lead-12 UA tree yields the tolerated NaN block).
            var specLead = lead == 12 ? 24 : lead;
            if (!specs.TryGetValue(specLead, out var s) || !s.FeatureNames.Contains("t850_mean")) continue;
            var lv = targets.Where(t => t.Lead == lead).Select(t => t.Valid).ToList();
            uaByLead[lead] = PrecipFeatureBuilder.LoadUpperAirLive(
                _cfg.Storage.ForecastsPath, location.Name, lead, lv.Min().AddDays(-2), lv.Max(), ct);
        }

        // EA hourly rainfall for the rich row's persistence features —
        // exact same load path 3c uses.
        var friendly = ResolveFriendlyStationName(station, location);
        if (friendly is null)
        {
            _log.LogError("Station {Station}: cannot resolve config name for 3o persistence lookup.", station);
            return false;
        }
        var hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
            _cfg.Storage.RainfallPath, location.Name, friendly, minValidTime: null, ct);

        // Per-lead model cache — policy blends run two bundle models per
        // target (e.g. 3o's blend(24,48) zone); load each at most once.
        var modelCache = new Dictionary<int, ITransformer>();
        ITransformer ModelFor(int modelLead)
        {
            if (!modelCache.TryGetValue(modelLead, out var m))
                modelCache[modelLead] = m = ModelArtifact.LoadLeadModel(ml, versionDir, modelLead, out _);
            return m;
        }

        var predictions = new List<PrecipPredictionRow>();
        foreach (var (lead, valid) in targets)
        {
            ct.ThrowIfCancellationRequested();
            // Policy / bucket-default model resolution — see ResolveModelLeads.
            // 3o serves the lead-12 targets (default m24 on the lead-12 pivot).
            var modelLeads = ResolveModelLeads(leadPolicy, metadata.Phase, lead, valid, anchor, specs);
            if (!specs.TryGetValue(modelLeads[0], out var spec))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no BlenderSpec for model lead {M}h in 3o bundle; skipping.",
                    station, lead, modelLeads[0]);
                continue;
            }
            if (!perLeadValid.TryGetValue(lead, out var perValid)
                || !perValid.TryGetValue(valid, out var pivot))
            {
                _log.LogWarning("Station {Station} lead {Lead}h: no forecast pivot for valid={Valid:yyyy-MM-dd HH:mm}Z; skipping.",
                    station, lead, valid);
                continue;
            }
            if (!pivot.Precip.Any(p => p.HasValue)) continue;

            // Build the rich half of the spec (same as 3c). The spec passed
            // to LightGBM is the FULL rich-oro spec; we just need the rich
            // dim for ComposeRow + slot ordering.
            int richDim = spec.FeatureCount - PrecipRichOroFeatureBuilder.TerrainCountFor(includeStationId);
            // We compose the rich row using the rich sub-spec carved out of
            // the rich-oro spec — matches the trainer's BuildForLead inner.
            var richSpec = new BlenderSpec
            {
                Target = spec.Target,
                FeatureSet = PrecipRichFeatureBuilder.SpecFeatureSet,
                LeadHours = spec.LeadHours,
                RequiredModels = spec.RequiredModels,
                OptionalModels = spec.OptionalModels,
                Models = spec.Models,
                FeatureNames = spec.FeatureNames.Take(richDim).ToList(),
                DataSource = spec.DataSource,
                Tier = PrecipRichFeatureBuilder.SpecFeatureSet,
                UkvStrategy = spec.UkvStrategy,
            };

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
                _log.LogWarning("Station {Station} lead {Lead}h: missing required precip [{M}]; skipping.",
                    station, lead, string.Join(",", missingRequired));
                continue;
            }

            var dew = new double[N]; var rh = new double[N];
            var dewDep = new double[N]; var pres = new double[N];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                var t = pivot.Temp2m[ci];
                var d = pivot.Dew[ci];
                dew[i]    = d ?? double.NaN;
                rh[i]     = pivot.Rh[ci] ?? double.NaN;
                dewDep[i] = (t.HasValue && d.HasValue) ? t.Value - d.Value : double.NaN;
                pres[i]   = pivot.Pressure[ci] ?? double.NaN;
            }
            var runTime = valid.AddHours(-lead);
            var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain, runTime);
            // UA block lands in the rich half (richSpec.FeatureNames includes the
            // UA names when on, since UA precedes terrain in the rich-oro layout),
            // so ComposeRow fills it; terrain is appended after → [rich||UA||terrain].
            double[]? uaValues = richSpec.FeatureNames.Contains("t850_mean")
                ? PrecipFeatureBuilder.UpperAirValuesFor(
                    uaByLead.TryGetValue(lead, out var asof) ? asof : System.Array.Empty<(DateTime, double[])>(), valid)
                : null;
            var richRow = PrecipRichFeatureBuilder.ComposeRow(
                richSpec, valid, specPrecip, dew, rh, dewDep, pres,
                rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
                cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
                cloudHighMean: pivot.CloudHighMean,
                capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
                eaRainPrev24hMm: persist.Prev24hMm,
                eaRainPrev72hMm: persist.Prev72hMm,
                eaWetHoursLast24h: persist.WetHoursLast24h,
                eaDryHoursTrailing: persist.DryHoursTrailing,
                truthMmHour: 0.0,
                upperAir: uaValues);

            // Terrain block — needs NWP-mean wind/temp/dew/pressure at
            // this valid_time. The live-tree aux table was loaded once
            // above for the entire predict window; just index into it.
            if (!nwpMeanByValid.TryGetValue(valid, out var nwpMean))
            {
                _log.LogWarning("Station {Station} lead {Lead}h valid={Valid:yyyy-MM-dd HH:mm}Z: no NWP-mean aux row in live tree for terrain block; skipping.",
                    station, lead, valid);
                continue;
            }
            var terrain = PrecipRichOroFeatureBuilder.ComposeTerrainBlock(oro, nwpMean, stationIndex, includeStationId);

            // Concatenate rich + terrain into the final feature vector.
            var features = new float[spec.FeatureCount];
            Array.Copy(richRow.Features, features, richDim);
            for (int i = 0; i < PrecipRichOroFeatureBuilder.TerrainCountFor(includeStationId); i++)
                features[richDim + i] = terrain[i];
            var fullRow = new BinaryTrainingRow
            {
                ValidTimeUtc = valid,
                Features = features,
                Label = false,
                TruthMmHour = 0.0f,
            };

            // Primary member + any equal-weight policy blend members. 3o's
            // per-lead specs share model membership + layout (one pooled spec
            // per lead from the same builder), so the composed row is reused;
            // a layout mismatch drops the extra member instead of mis-composing.
            var pVals = new List<double>(modelLeads.Length)
            {
                PrecipOccurrenceTrainer.PredictVectorProbability(ml, ModelFor(modelLeads[0]), spec, new[] { fullRow })[0],
            };
            foreach (var extra in modelLeads.Skip(1))
            {
                if (specs.TryGetValue(extra, out var specX)
                    && specX.FeatureCount == spec.FeatureCount
                    && specX.Models.SequenceEqual(spec.Models, StringComparer.OrdinalIgnoreCase))
                    pVals.Add(PrecipOccurrenceTrainer.PredictVectorProbability(ml, ModelFor(extra), specX, new[] { fullRow })[0]);
                else
                    _log.LogWarning(
                        "Station {Station} lead {Lead}h: policy blend member m{Extra}h has an incompatible spec layout — using the remaining member(s) only.",
                        station, lead, extra);
            }
            var pWet = new[] { pVals.Average() };
            var climPWet = climatology.Predict(valid);

            var perModelPrecip = new double?[PrecipPredictionRow.PerModelFieldCount];
            var perModelRun = new DateTime?[PrecipPredictionRow.PerModelFieldCount];
            for (int i = 0; i < N; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                if (ci >= PrecipPredictionRow.PerModelFieldCount) continue;
                perModelPrecip[ci] = specPrecip[i];
                perModelRun[ci] = pivot.RunTime[ci];
            }

            var spreadStart = N;
            predictions.Add(new PrecipPredictionRow
            {
                LocationName = location.Name,
                TruthStation = outStation,
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
                PrecipMean = NanToNull(features[spreadStart + 0]),
                PrecipStd  = NanToNull(features[spreadStart + 1]),
                PrecipMax  = NanToNull(features[spreadStart + 2]),
                PrecipAgreementWet01 = NanToNull(features[spreadStart + 3]),
                FeatureVectorHash = FeatureHashing.HashFloats(features),
                // Per-lead conformal tags only apply to the plain bucket model
                // (same rationale as the 3c path).
                ConformalSetTag = modelLeads.Length == 1 && modelLeads[0] == lead
                    ? ModelArtifact.PredictConformalIfPresent(versionDir, lead, pWet[0])
                    : null,
                // 3o always carries the UA block; flag whether real UA values
                // were in hand for this valid-time (drives the site shading).
                UpperAirIncluded = richSpec.FeatureNames.Contains("t850_mean")
                    ? uaValues is not null && uaValues.Any(v => !double.IsNaN(v))
                    : (bool?)null,
            });
            _log.LogInformation("Station {Station} lead {Lead}h → P(wet) {P:0.000} (clim {Clim:0.000}, phase={Phase}, models=[{M}])",
                outStation, lead, pWet[0], climPWet, metadata.Phase, string.Join("+", modelLeads));
        }

        if (predictions.Count == 0)
        {
            _log.LogWarning("Station {Station}: no 3o predictions produced.", station);
            return false;
        }
        await WritePredictionsAsync(predictions, outStation, anchor, metadata.Version, ct);
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
    private static string? ResolveFriendlyStationName(string stationSlug, LocationConfig location)
    {
        foreach (var s in location.Rainfall.Stations)
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
        LocationConfig location,
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
        // Window starts a day BEFORE anchor's date so the forecast pivot
        // also covers recently-elapsed valid slots whose cycle has only
        // just been collected — see the leadTargets reach-back below.
        var windowStart = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc)
            .AddDays(-1);
        var windowEnd = anchor.AddHours(maxLead + 24);

        var pivotByLeadValid = QueryExactForecastRows(
            _cfg.Storage.ForecastsPath,
            location.Name,
            specs: specs,
            earliestValid: windowStart,
            latestValid: windowEnd,
            asOfRunTime: anchor,
            ct);

        foreach (var (lead, spec) in specs.OrderBy(kv => kv.Key))
        {
            // Target valid times span a 72h window: the day BEFORE
            // anchor.date + lead/24, through two days after. The −1-day
            // reach-back is load-bearing. Without it the 18:00Z slot of a
            // valid date is permanently orphaned: that slot's cycle (for
            // lead 24, the prior day's 18Z run) only publishes + gets
            // collected in the small hours of the NEXT day — after every
            // predict run still anchored on the prior day — and the runs
            // that finally hold the cycle had, pre-fix, moved their window
            // forward and stopped targeting that date. The +day-after
            // extension still covers a late-evening run's future lead-12
            // slots. Re-predicting an already-done exact-runtime slot is
            // idempotent, so the wider window is free; pivot-miss /
            // required-model checks drop slots that genuinely can't be made.
            var dayStart = new DateTime(anchor.Year, anchor.Month, anchor.Day, 0, 0, 0, DateTimeKind.Utc)
                .AddDays(lead / 24)
                .AddDays(-1);
            var leadTargets = Enumerable.Range(0, 72)
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
                // Upper-air block (3d-with-UA) — built from the pivot's per-model
                // pressure in the SAME model-major order the trainer used (see
                // PrecipExactFeatureBuilder.UpperAirValuesFromPerModel). Null when
                // the schema has no UA (legacy 3d bundles) → ComposeRow emits the
                // legacy-length vector. Missing pressure → NaN slots (graceful).
                double[]? uaValues = spec.FeatureNames.Contains("t850_mean")
                    ? PrecipExactFeatureBuilder.UpperAirValuesFromPerModel(spec, pivot.PerModelPressure)
                    : null;
                var row = PrecipExactFeatureBuilder.ComposeRow(
                    spec, valid, modelPrecip, truthMmHour: 0.0, ukvPrecip: ukvPrecip, upperAir: uaValues);

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
                    LocationName = location.Name,
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
        DateTime? UkvRunTime,
        // Per-model upper-air pressure values (10 cols in
        // PrecipExactFeatureBuilder.UaPressureColumnNames order), populated only
        // when a spec carries the UA block (3d-with-UA). Empty otherwise.
        Dictionary<string, double[]> PerModelPressure);

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

        // Upper-air: pull the 10 pressure cols per model when any spec carries
        // the UA block (3d-with-UA). Pressure is on the same exact rows, so it
        // rides through the existing pivot — no separate join. Cols land trailing
        // (ordinals 7..16) so the existing precip/ukv reader indices are untouched.
        var anyUa = specs.Values.Any(s => s.FeatureNames.Contains("t850_mean"));
        var uaCols = PrecipExactFeatureBuilder.UaPressureColumnNames;
        var uaLatestCols = anyUa ? ", " + string.Join(", ", uaCols) : "";
        var uaFinalCols = anyUa ? ", " + string.Join(", ", uaCols.Select(c => $"l.{c}")) : "";

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
    SELECT ValidTimeUtc, Model, LeadHours, RunTimeUtc, Precipitation{uaLatestCols},
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
       u.ukv_precip, u.ukv_run_time{uaFinalCols}
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
       CAST(NULL AS DOUBLE) AS ukv_precip, CAST(NULL AS TIMESTAMP) AS ukv_run_time{uaFinalCols}
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
                    UkvRunTime: ukvRunTime,
                    PerModelPressure: new Dictionary<string, double[]>(StringComparer.Ordinal));
                result[key] = existing;
            }
            existing.PerModelPrecip[model] = precip;
            if (precip.HasValue) existing.PerModelRunTime[model] = runTime;
            if (anyUa)
            {
                // Pressure cols are trailing (ordinals 7..7+uaCols.Count-1),
                // in PrecipExactFeatureBuilder.UaPressureColumnNames order.
                var pressure = new double[uaCols.Count];
                for (int j = 0; j < uaCols.Count; j++)
                    pressure[j] = NullableDouble(reader, 7 + j) ?? double.NaN;
                existing.PerModelPressure[model] = pressure;
            }
        }
        return result;
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);
}
