using System.Globalization;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Temp;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Exact12h;

namespace WeatherBlend.Commands;

/// <summary>
/// Cross-target dispatcher for the <c>train</c> CLI command. Validates the
/// shared arguments (target, leads, feature-set) and routes to the per-target
/// trainer:
/// <list type="bullet">
///   <item>temperature → in-class Phase 2b/2c/2d handlers below</item>
///   <item>precipitation → <see cref="PrecipTrainCommand"/></item>
///   <item>dry-window → <see cref="DryWindowTrainCommand"/></item>
///   <item>wind / humidity / shortwave-radiation / cloud-cover / wind-gust /
///         wind-speed-lgb → <see cref="ElementTrainCommand"/></item>
/// </list>
///
/// Renamed from <c>TempTrainCommand</c> on 2026-05-28 — the original name
/// reflected the temperature-only origin but the class has been the
/// cross-target dispatcher since the precipitation + dry-window + element
/// trainers were split out. The temperature-specific Phase 2b/2c/2d
/// handlers live in this file (RunPhase2bAsync / RunPhase2cAsync /
/// RunPhase2dAsync) and could be extracted into a TempTrainHandler in a
/// future cleanup, but the dispatcher role is the load-bearing public
/// surface.
///
/// Lead 120h applies to temperature + precipitation only — dry-window and
/// Element blenders stay capped at 72h pending a separate scoping decision.
/// </summary>
public sealed class TrainCommand : TrainCommandBase
{
    private readonly DryWindowTrainCommand _dryWindow;
    private readonly ElementTrainCommand _element;
    private readonly PrecipTrainCommand _precip;

    // Default leads for temperature + precipitation. Dry-window and Element
    // train commands keep their own narrower {24,48,72} arrays internally.
    // Sourced from Train.Common.Leads.Full so temp + precip + predict all share
    // a single definition. Dry-window + Element blenders use Leads.Short (their
    // own train commands set DefaultLeads = Leads.Short).
    private static readonly int[] DefaultLeads = Leads.Full;

    public TrainCommand(
        ILogger<TrainCommand> log,
        AppConfig cfg,
        DryWindowTrainCommand dryWindow,
        ElementTrainCommand element,
        PrecipTrainCommand precip)
        : base(log, cfg)
    {
        _dryWindow = dryWindow;
        _element = element;
        _precip = precip;
    }

    public Task<int> RunAsync(string target, string lead, string? station, string? window, string featureSet, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier: null, includeUkv: null, exactLeads: null, cycles: null, locationOverride: null, ct);

    public Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads: null, cycles: null, locationOverride: null, ct);

    public Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads, cycles: null, locationOverride: null, ct);

    public Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, int[]? cycles, CancellationToken ct)
        => RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads, cycles, locationOverride: null, ct);

    public async Task<int> RunAsync(
        string target, string lead, string? station, string? window, string featureSet,
        string? tier, bool? includeUkv, int[]? exactLeads, int[]? cycles, string? locationOverride, CancellationToken ct)
    {
        // tier + includeUkv + exactLeads are exact-runtime levers (Phase 2d /
        // 3d). Defaults (null, null, null) preserve historical behaviour: 2d
        // picks T2 + UKV + leads {12,24}; 3d picks P1 + UKV + leads {12,24}.
        // Bake-off variants pass non-default values to swap tiers (P1 vs P2),
        // toggle UKV, or extend leads to {48,72,96,120} for the long-range
        // 2b/3a comparison without code changes.
        var t = target.ToLowerInvariant();
        var elementTarget = ElementTargets.TryFromCli(t);
        // Non-element targets stay hard-coded; element targets are derived
        // from ElementTargets.All so adding a new element blender doesn't
        // require remembering to update this whitelist (which is exactly
        // how wind-gust silently failed under continue-on-error from
        // 2026-05-27 until this bug fix).
        var validTargets = new[] { "temperature", "precipitation", "dry-window" }
            .Concat(ElementTargets.All.Select(e => e.CliName))
            .ToArray();
        if (!validTargets.Contains(t))
        {
            _log.LogError("target must be one of [{Targets}] (got '{Target}')",
                string.Join(", ", validTargets), target);
            return 2;
        }

        var leads = ParseLeads(lead);
        if (leads is null)
        {
            _log.LogError("Invalid --lead value '{Lead}'. Expected 24, 48, 72, 96, 120, or all.", lead);
            return 2;
        }

        var fs = (featureSet ?? "lean").ToLowerInvariant();
        if (fs is not ("lean" or "rich" or "oro" or "oro-noid" or "exact" or "copula-mc" or "copula-mc-3c"))
        {
            _log.LogError("Invalid --feature-set value '{Fs}'. Expected lean | rich | oro | oro-noid | exact | copula-mc | copula-mc-3c " +
                "(oro = Phase 3o; oro-noid = Phase 3oni (3o minus station id); copula-mc = Phase 3p over 3o; copula-mc-3c = Phase 3q over 3c).", featureSet);
            return 2;
        }
        // "exact" = Phase 2d (temperature) or Phase 3d (precipitation) — same
        // feature-set tag, dispatched per target.
        if (fs == "exact" && t != "temperature" && t != "precipitation")
        {
            _log.LogError(
                "--feature-set {Fs} is only supported for targets temperature, precipitation.", fs);
            return 2;
        }
        // "oro" = Phase 3o, "oro-noid" = Phase 3oni (3o minus station id).
        // Both precipitation rich + terrain, 4-station Bonehill pool. Precip-only.
        if (fs is "oro" or "oro-noid" && t != "precipitation")
        {
            _log.LogError(
                "--feature-set {Fs} is only supported for target precipitation.", fs);
            return 2;
        }
        // "copula-mc" = Phase 3p (MC over 3o), "copula-mc-3c" = Phase 3q (MC
        // over 3c). Both dry-window-only.
        if (fs is "copula-mc" or "copula-mc-3c" && t != "dry-window")
        {
            _log.LogError(
                "--feature-set {Fs} is only supported for target dry-window.", fs);
            return 2;
        }
        if (elementTarget is not null && fs != "lean")
        {
            _log.LogError(
                "Element targets currently only support --feature-set lean (got '{Fs}'). " +
                "Rich variants are not defined for the per-variable blenders yet.", fs);
            return 2;
        }

        // Leads 96h + 120h are currently scoped to temperature + precipitation only.
        // Dry-window and Element targets stay capped at 72h until separately scoped.
        if (t == "dry-window" || elementTarget is not null)
        {
            foreach (var unsupported in new[] { 96, 120 })
            {
                if (leads.Contains(unsupported) && lead.Equals(unsupported.ToString(), StringComparison.OrdinalIgnoreCase))
                {
                    _log.LogError("--lead {N} is not supported for target '{T}' yet (temperature + precipitation only).", unsupported, t);
                    return 2;
                }
            }
            // "all" implicitly includes 96 + 120 — silently narrow rather than error.
            leads = leads.Where(l => l != 96 && l != 120).ToArray();
        }

        // Lead-0 "nowcast" model: trained on the hist_forecast archive (≈analysis
        // quality at lead 0; see NowcastSource) and appended to the SAME per-phase
        // bundle as the ≥24h leads, so the per-lead policy can select it at the
        // short bands. STANDARD (no flag, every location) for exactly the two
        // rich phases that carry it: temperature 2c (--feature-set rich) and
        // precipitation 3o (--feature-set oro) + 3c (--feature-set rich). 3c is
        // included so Membury/Sennen — whose precip champion falls back to 3c, not
        // 3o — also get a lead-0 candidate in their champion phase. Was an opt-in
        // --nowcast flag gated to Bonehill in CI; the per-site gate was the thing
        // that left the other sites without a nowcast. A location whose
        // hist_forecast surface archive isn't backfilled yet just trains too few
        // lead-0 rows and the phase trainer SKIPS that lead (see the IsNowcast
        // guard in RunPhase2cAsync / PrecipTrainCommand.RunPhase3cAsync /
        // RunPhase3oAsync), so this can never abort a retrain. APPENDED (not
        // prepended) so leads[0] stays a production lead — the RetrainGuard reads
        // its rows/label/feature basis from leads[0] and must not see the shorter
        // lead-0 window.
        var includeNowcast = (t == "temperature" && fs == "rich")
                          || (t == "precipitation" && (fs == "oro" || fs == "rich"));
        if (includeNowcast)
        {
            leads = leads.Where(l => l != Train.Common.NowcastSource.LeadHours)
                .Append(Train.Common.NowcastSource.LeadHours)
                .ToArray();
            _log.LogInformation("Nowcast lead {L}h appended (hist_forecast-sourced); leads now [{Leads}].",
                Train.Common.NowcastSource.LeadHours, string.Join(",", leads));
        }

        // Resolve --location into the active LocationConfig. Every target's
        // trainer reads from this — temperature 2b/2c/2d, precipitation
        // 3a/3c/3d/3o, dry-window, and the element blenders (Phase B,
        // commit 3). No trainer hardcodes _cfg.Location any more.
        Config.LocationConfig? location = _cfg.Location;
        if (!string.IsNullOrWhiteSpace(locationOverride))
        {
            location = _cfg.Locations.FirstOrDefault(l =>
                l.Name.Equals(locationOverride, StringComparison.OrdinalIgnoreCase));
            if (location is null)
            {
                _log.LogError("Location '{Name}' not found in config.yaml's `locations:` list. Available: [{All}]",
                    locationOverride,
                    string.Join(", ", _cfg.Locations.Select(l => l.Name)));
                return 2;
            }
        }

        return t switch
        {
            "temperature"   => fs switch
            {
                "rich"  => await RunPhase2cAsync(leads, location, ct),
                "exact" => await RunPhase2dAsync(tier, includeUkv, exactLeads, cycles, location, ct),
                _       => await RunPhase2bAsync(leads, location, ct),
            },
            "precipitation" => await _precip.RunAsync(
                                   leads, station, fs, tier, includeUkv,
                                   exactLeads, cycles, location, ct),
            // dry-window: lean / rich / default → Phase 3b (53 features);
            // copula-mc → 3p (MC over 3o), copula-mc-3c → 3q (MC over 3c).
            "dry-window"    => DryWindow3pPredictor.PhaseForFeatureSet(fs) is { } copulaPhase
                                   ? await _dryWindow.RunCopulaMcAsync(location, copulaPhase, ct)
                                   : await _dryWindow.RunAsync(
                                         station ?? "all", window ?? "all", leads,
                                         location, ct),
            // Per-variable element blenders: one dispatcher routes wind / humidity /
            // shortwave-radiation / cloud-cover to its dedicated IElementBlender.
            _ when elementTarget is not null
                   => await _element.RunAsync(elementTarget, leads, location, ct),
            _ => 2,
        };
    }

    // Scan-once temperature train cache: this location's forecasts + ERA5 into
    // one parquet each, so the per-lead BuildForLead calls are one-file reads
    // instead of re-globbing the whole forecast tree per lead (more so now that
    // --nowcast adds a lead-0 build and the hist_forecast backfill enlarged the
    // tree). All forecast sources kept; the builders filter internally.
    private (string FcPath, string EraPath) MaterializeTempTrainCache(string locationName, string tag)
    {
        var scratchRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "train_cache", tag);
        var fcPath = Path.Combine(scratchRoot, "fc"); Directory.CreateDirectory(Path.Combine(fcPath, "p"));
        var eraPath = Path.Combine(scratchRoot, "era"); Directory.CreateDirectory(Path.Combine(eraPath, "p"));
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        var fcSrc = Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=" + locationName, "**", "*.parquet"));
        var eraSrc = Esc(Path.Combine(_cfg.Storage.Era5Path, "location=" + locationName, "**", "*.parquet"));
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var c = conn.CreateCommand();
        c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcSrc}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(fcPath, "p", "fc.parquet"))}' (FORMAT PARQUET);";
        c.ExecuteNonQuery();
        c.CommandText = $"COPY (SELECT * FROM read_parquet('{eraSrc}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(eraPath, "p", "era.parquet"))}' (FORMAT PARQUET);";
        c.ExecuteNonQuery();
        _log.LogInformation("Scan-once temp train cache → {Scratch} (forecasts + ERA5, {Loc}).", scratchRoot, locationName);
        return (fcPath, eraPath);
    }

    private static int[]? ParseLeads(string lead) => lead.ToLowerInvariant() switch
    {
        "all" => DefaultLeads,
        "24"  => new[] { 24 },
        "48"  => new[] { 48 },
        "72"  => new[] { 72 },
        "96"  => new[] { 96 },
        "120" => new[] { 120 },
        _     => null,
    };

    private async Task<int> RunPhase2bAsync(int[] leads, Config.LocationConfig location, CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        // Station-keyed (location-keyed for temperature) layout post 2026-05-14:
        // bundles live under data/models/temperature/{location}/v{ts}/ to mirror
        // the per-station precipitation tree. The flat top-level layout was
        // single-location only; this prepares the manifest for a second location.
        var stationKey = location.Name;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "temperature", stationKey, now);
        var versionName = Path.GetFileName(versionDir);

        var hp = TempTrainer.Hyperparameters.Default();
        _log.LogInformation("Phase 2b — training per-lead blenders for leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // Buffer the first-lead train slice for training_summary.json
        // (Phase 1a of AUTO_RETRAIN_PLAN.md). Single-lead avoids mixing
        // lead-axis variability into the per-feature distribution bands
        // the retrain guard reads. Row counts ARE aggregated across leads.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        // Per-phase training-data cutoff (2026-05-26 — see PhaseRegistry).
        var minValidTime2b = PhaseRegistry.Default.AllPhases("temperature")
            .SingleOrDefault(p => p.Id == "2b")?.MinValidTime;
        if (minValidTime2b.HasValue)
            _log.LogInformation("Phase 2b training-data cutoff: ValidTimeUtc >= {Cutoff:yyyy-MM-dd} (from phases.yaml)", minValidTime2b.Value);

        var cache2b = MaterializeTempTrainCache(location.Name, $"2b_{location.Name}");

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = TempFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = TempFeatureBuilder.BuildForLead(
                cache2b.FcPath,
                cache2b.EraPath,
                location.Name,
                spec,
                minValidTime2b,
                ct);
            _log.LogInformation("Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);
            // Accumulate row counts across all leads (aggregate signal); buffer
            // the FIRST lead's train features for per-feature stats so the
            // distribution bands aren't blurred by per-lead variability.
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window — exclude from the RetrainGuard baseline so --nowcast can't
            // shift the rows/feature totals vs the production leads. (No-op for
            // non-nowcast leads, incl. exact-runtime's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            }

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            // Best-single: pick on val (no leakage), then re-score on TEST so the
            // Models page can compare blend-vs-best on the same chronological split.
            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = trained.FeatureImportance;

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
                BestSingle = best,
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPct = bestTestMae > 0 ? (blendStats.Mae - bestTestMae) / bestTestMae * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMae, bestValMae, deltaPct, testMonths);
        }

        // Shared artefacts — per-lead BlenderSpec is now the schema source of truth.
        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2b",
            LocationName = location.Name,
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Objective is L2 (squared error); MAE used only as early-stopping metric. Microsoft.ML.LightGbm 4.0 does not expose regression_l1.",
                "No monotone constraints on per-model temperature inputs. Microsoft.ML.LightGbm 4.0 does not expose monotone_constraints.",
                "Per-lead model artefacts named lead_{N}h.zip; vector-native LightGBM trainer (Features column = float[]).",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        // RetrainGuard (Phase 1b/1c of AUTO_RETRAIN_PLAN.md): build the
        // training_summary, compare against the previous 2b run's, and
        // abort BEFORE promotion if any tolerance band breached. Per-lead
        // bundles are already on disk by this point but the manifest
        // doesn't reference them until PromoteVersionAsChampion runs —
        // the orphan dir on guard-fail is invisible to predict + verify.
        var guardResult2b = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"temperature/{stationKey}", phase: "2b", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp0)
                ? sp0.FeatureNames.ToList() : Array.Empty<string>(),
            locationName: location.Name);
        if (!guardResult2b.Passed)
        {
            _log.LogError("Aborting Phase 2b retrain — sanity guard failed. Orphan dir {Dir} not promoted.", versionDir);
            return 4;
        }
        // Promote 2b: replaces any prior 2b entry in this station's Active and
        // sets Current. Any active 2c challenger survives untouched. Other
        // locations' Stations entries are left untouched.
        ModelArtifact.PromoteStationVersion(modelsRoot, "temperature", stationKey, versionName, newPhase: "2b");

        _log.LogInformation("Phase 2b artefacts → {Dir}", versionDir);
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    /// <summary>
    /// One-time escape hatch for a DELIBERATE, permanent training-window change
    /// that legitimately blows RetrainGuard's bands — e.g. the 2026-06-01 Phase
    /// 2c change making ukmo optional, which widened the rich window back to the
    /// lean 2024-02 floor. That shift trips TWO bands at once: the ±30% row band
    /// (~+33% rows) AND the per-feature NaN band (models whose Open-Meteo history
    /// doesn't span the added window go fully NaN over it — ukmo from 2024-09:
    /// 0→0.31; aifs from 2025-02: 0.43→0.64). So this relaxes the row band to
    /// <paramref name="WB_GUARD_ROWS_DELTA_OVERRIDE"/> AND effectively disables
    /// the NaN-fraction band (1.0 = any change passes). <see cref="GuardTolerances.FeaturesEffectiveDelta"/>
    /// stays 0 — a real column add/remove still aborts.
    ///
    /// Set <c>WB_GUARD_ROWS_DELTA_OVERRIDE=&lt;pct&gt;</c> (e.g. <c>0.8</c>) for the
    /// single retrain that establishes the new baseline. Once that run passes
    /// and writes its training_summary, the next cycle compares wide-vs-wide
    /// (rows AND NaN profile match) and the bands auto-return to default — so
    /// this is genuinely single-use and does NOT permanently weaken the guard.
    /// Returns null when unset/unparseable → <see cref="RetrainGuard.Defaults"/>.
    /// Applied only to the 2c call site; 2b/2d keep the default bands.
    /// </summary>
    private static GuardTolerances? DeliberateWindowChangeOverride()
    {
        var raw = Environment.GetEnvironmentVariable("WB_GUARD_ROWS_DELTA_OVERRIDE");
        if (string.IsNullOrWhiteSpace(raw)) return null;
        if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var pct) || pct <= 0)
            return null;
        return RetrainGuard.Defaults with { RowsDeltaPct = pct, NanPctAbsolute = 1.0 };
    }

    // ---- Phase 2c: rich-feature temperature blender (champion/challenger) ----------

    private async Task<int> RunPhase2cAsync(int[] leads, Config.LocationConfig location, CancellationToken ct)
    {
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var stationKey = location.Name;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "temperature", stationKey, now, suffix: "phase2c");
        var versionName = Path.GetFileName(versionDir);

        var hp = TempTrainer.Hyperparameters.Default();
        _log.LogInformation("Phase 2c — rich-feature blender (88 features), leads [{Leads}]",
            string.Join(",", leads));
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} esr={Esr} seed={Seed} (identical to Phase 2b — feature richness is the only variable)",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.EarlyStoppingRound, hp.Seed);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // Buffers for training_summary.json — same pattern as 2b. See
        // TrainingSummaryBuilder.BuildAndSave call after the loop.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        // Per-phase training-data cutoff (2026-05-26 — see PhaseRegistry).
        var minValidTime2c = PhaseRegistry.Default.AllPhases("temperature")
            .SingleOrDefault(p => p.Id == "2c")?.MinValidTime;
        if (minValidTime2c.HasValue)
            _log.LogInformation("Phase 2c training-data cutoff: ValidTimeUtc >= {Cutoff:yyyy-MM-dd} (from phases.yaml)", minValidTime2c.Value);

        var cache2c = MaterializeTempTrainCache(location.Name, $"2c_{location.Name}");

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = TempRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = TempRichFeatureBuilder.BuildForLead(
                cache2c.FcPath,
                cache2c.EraPath,
                location.Name,
                spec,
                minValidTime2c,
                ct);
            _log.LogInformation("Loaded {N} rich rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);

            if (rows.Count < 500)
            {
                // The lead-0 nowcast is appended automatically for 2c, but a
                // location whose hist_forecast surface archive isn't backfilled
                // yet has no lead-0 rows. Skip it (the bundle keeps its ≥24h
                // leads) rather than aborting the whole retrain — the policy fit
                // simply won't see a lead-0 candidate for that location.
                if (Train.Common.NowcastSource.IsNowcast(lead))
                {
                    _log.LogWarning("Lead {Lead}h (nowcast): only {N} rows — no hist_forecast surface "
                        + "data for this location yet; skipping the lead-0 model.", lead, rows.Count);
                    specsPerLead.Remove(lead);   // drop the orphan spec so the bundle has no model-less lead
                    continue;
                }
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train.", rows.Count, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), " +
                                "val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), " +
                                "test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window — exclude from the RetrainGuard baseline so --nowcast can't
            // shift the rows/feature totals vs the production leads. (No-op for
            // non-nowcast leads, incl. exact-runtime's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            }

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMaeRich = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = trained.FeatureImportance;

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
                BestSingle = best,
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMaeRich,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPctRich = bestTestMaeRich > 0 ? (blendStats.Mae - bestTestMaeRich) / bestTestMaeRich * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMaeRich, bestValMae, deltaPctRich, testMonths);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2c",
            LocationName = location.Name,
            DataSource = "previous_runs_api",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Objective is L2 (squared error); MAE used only as early-stopping metric. Microsoft.ML.LightGbm 4.0 does not expose regression_l1.",
                "No monotone constraints on per-model temperature inputs. Microsoft.ML.LightGbm 4.0 does not expose monotone_constraints.",
                "Persistence features (era5_temp_24h_ago, era5_temp_168h_ago) deliberately omitted: ERA5 has a ~5d release lag, so neither lookback exists at predict time for the 24/48/72h leads. Including them would teach the model to lean on data unavailable in production.",
                "Per-lead model artefacts named lead_{N}h.zip (ML.NET ITransformer pipeline). Same hyperparameters as Phase 2b — feature richness is the isolated variable.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        var guardOverride2c = DeliberateWindowChangeOverride();
        if (guardOverride2c is not null)
            _log.LogWarning(
                "Phase 2c: RetrainGuard bands relaxed (rows ±{Pct:P0}, per-feature NaN check off) via " +
                "WB_GUARD_ROWS_DELTA_OVERRIDE — single-use baseline reset for a deliberate training-window " +
                "change. Bands auto-return to default next cycle; feature-count check stays active.",
                guardOverride2c.RowsDeltaPct);
        var guardResult2c = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"temperature/{stationKey}", phase: "2c", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(leads[0], out var sp2c)
                ? sp2c.FeatureNames.ToList() : Array.Empty<string>(),
            tolerances: guardOverride2c,
            locationName: location.Name);
        if (!guardResult2c.Passed)
        {
            _log.LogError("Aborting Phase 2c retrain — sanity guard failed. Orphan dir {Dir} not promoted.", versionDir);
            return 4;
        }

        // Promote 2c as a challenger inside this station's entry: replaces any
        // prior 2c entry in Active (so re-training is idempotent) and leaves
        // Current = 2b champion. Predict + verify iterate both versions every
        // cycle.
        ModelArtifact.PromoteStationVersion(modelsRoot, "temperature", stationKey, versionName, newPhase: "2c");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "temperature", stationKey);

        _log.LogInformation("Phase 2c artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions now: [{Active}]", string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    // ---- Phase 2d: exact-runtime temperature blender (lead-12 champion) -------------

    /// <summary>
    /// 2d's tier (P1 in the precip-exact world / T2 in the temp-exact world):
    /// GFS + AIFS required, IFS oper + MO Global optional, UKV always
    /// optional. Single-lead training at <see cref="DefaultPhase2dLeads"/>;
    /// each lead gets its own per-lead artefact saved under
    /// <c>v{ts}_phase2d/lead_{N}h.zip</c>.
    /// </summary>
    private static readonly int[] DefaultPhase2dLeads = { 12, 24 };

    private async Task<int> RunPhase2dAsync(string? tierName, bool? includeUkvOpt, int[]? exactLeads, int[]? cycleHoursFilter, Config.LocationConfig location, CancellationToken ct)
    {
        var leads = exactLeads is { Length: > 0 } ? exactLeads : DefaultPhase2dLeads;
        var now = DateTime.UtcNow;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var stationKey = location.Name;
        var versionDir = ModelArtifact.BuildStationVersionDir(modelsRoot, "temperature", stationKey, now, suffix: "phase2d");
        var versionName = Path.GetFileName(versionDir);

        // Bake-off-tuned defaults: lr=0.05/leaves=31/min-leaf=50/feat-frac=1.0
        // is the no-UKV per-lead winner (also a reasonable middle-ground for
        // the UKV-included per-lead winners which differed at lead 12 vs 24).
        // Using one HP set across both leads keeps the artefact reproducible
        // from a single command — per-lead HP tuning can land later if the
        // delta justifies the complexity.
        var hp = TempTrainer.Hyperparameters.Default() with
        {
            LearningRate = 0.05,
            MinimumExampleCountPerLeaf = 50,
            FeatureFraction = 1.0,
        };

        var tier = Exact12hFeatureBuilder.AllTiers.First(t => t.Name == (tierName ?? "T2"));
        bool IncludeUkv = includeUkvOpt ?? true;

        // Cycle-filter logging — null/empty means "all cycles on disk",
        // i.e. production default. Comma-list logged when restricted so
        // bake-off runs leave a clear trail in the log of which cycles
        // contributed to the trained version.
        _log.LogInformation("Phase 2d — exact-runtime blender (tier {Tier}, UKV={Ukv}), leads [{Leads}], cycles [{Cycles}]",
            tier.Name, IncludeUkv, string.Join(",", leads),
            (cycleHoursFilter is { Length: > 0 }) ? string.Join(",", cycleHoursFilter) : "all");
        _log.LogInformation("Hyperparameters: iter={Iter} lr={Lr} leaves={Leaves} min-leaf={Ml} esr={Esr} seed={Seed} feat-frac={Ff}",
            hp.NumberOfIterations, hp.LearningRate, hp.NumberOfLeaves, hp.MinimumExampleCountPerLeaf,
            hp.EarlyStoppingRound, hp.Seed, hp.FeatureFraction);

        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        var importanceByLead = new Dictionary<int, IEnumerable<(string Name, double Gain)>>();
        var specsPerLead = new Dictionary<int, BlenderSpec>();
        // training_summary buffers (Phase 1a). 2d uses exact-runtime cycles
        // but the dataset shape + features are otherwise the same regression
        // as 2b/2c — same buffer-and-aggregate pattern.
        List<float[]>? firstLeadTrainFeatures = null;
        int totalTrainRows = 0, totalValRows = 0, totalTestRows = 0;

        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("--- Lead {Lead}h ---", lead);

            var spec = Exact12hFeatureBuilder.BuildSpec(tier, targetLead: lead, includeUkv: IncludeUkv);
            specsPerLead[lead] = spec;
            _log.LogInformation("Spec: {Spec}", spec);

            var rows = Exact12hFeatureBuilder.Build(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                location.Name,
                tier,
                spec,
                targetLead: lead,
                includeUkv: IncludeUkv,
                runCycleHoursFilter: cycleHoursFilter,
                ct: ct);
            _log.LogInformation("Loaded {N} rows spanning {S:yyyy-MM-dd} → {E:yyyy-MM-dd}",
                rows.Count,
                rows.Count == 0 ? DateTime.MinValue : rows[0].ValidTimeUtc,
                rows.Count == 0 ? DateTime.MinValue : rows[^1].ValidTimeUtc);
            if (rows.Count < 500)
            {
                _log.LogError("Only {N} rows for lead {Lead}h — too few to train (run s3-collect / backfill first).",
                    rows.Count, lead);
                return 3;
            }

            var ds = RegressionDataset.Split(rows);
            _log.LogInformation("Split → train {TN} ({T0:yyyy-MM-dd}..{T1:yyyy-MM-dd}), val {VN} ({V0:yyyy-MM-dd}..{V1:yyyy-MM-dd}), test {EN} ({E0:yyyy-MM-dd}..{E1:yyyy-MM-dd})",
                ds.Train.Count, ds.TrainStart, ds.TrainEnd,
                ds.Val.Count,   ds.ValStart,   ds.ValEnd,
                ds.Test.Count,  ds.TestStart,  ds.TestEnd);
            // Lead-0 nowcast rows train on a different (shorter, hist_forecast)
            // window — exclude from the RetrainGuard baseline so --nowcast can't
            // shift the rows/feature totals vs the production leads. (No-op for
            // non-nowcast leads, incl. exact-runtime's {12,24}.)
            if (!Train.Common.NowcastSource.IsNowcast(lead))
            {
                totalTrainRows += ds.Train.Count;
                totalValRows   += ds.Val.Count;
                totalTestRows  += ds.Test.Count;
                firstLeadTrainFeatures ??= ds.Train.Select(r => r.Features).ToList();
            }

            var trained = TempTrainer.TrainVector(ds.Train, ds.Val, spec, hp);

            var testActual = ds.Test.Select(x => (double)x.Label).ToArray();
            var testPred   = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, ds.Test);
            var blendStats = TempMetrics.Compute(testPred, testActual);

            var best = TempBaselines.BestSingle(spec, ds.Val);
            var bestValMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Val, best),
                ds.Val.Select(x => (double)x.Label).ToArray()).Mae;
            var bestTestMae = TempMetrics.Compute(
                TempBaselines.FromFeature(spec, ds.Test, best),
                testActual).Mae;

            ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
            importanceByLead[lead] = trained.FeatureImportance;

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
                BestSingle = best,
                BestSingleValMae  = bestValMae,
                BestSingleTestMae = bestTestMae,
                BlendTestMae  = blendStats.Mae,
                BlendTestRmse = blendStats.Rmse,
                BlendTestBias = blendStats.Bias,
            };

            var deltaPct = bestTestMae > 0 ? (blendStats.Mae - bestTestMae) / bestTestMae * 100 : double.NaN;
            _log.LogInformation(
                "Lead {Lead}h headline — blend MAE={Blend:0.000}°C, best[{Best}] test MAE={BestT:0.000}°C (val {BestV:0.000}°C), Δ {Delta:+0.0;-0.0;0.0}%, months={M}",
                lead, blendStats.Mae, best, bestTestMae, bestValMae, deltaPct, testMonths);
        }

        ModelArtifact.SaveBlenderSpecs(versionDir, specsPerLead);
        ModelArtifact.SavePerLeadFeatureImportance(versionDir, importanceByLead);

        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version = versionName,
            Target = "temperature",
            Phase = "2d",
            LocationName = location.Name,
            DataSource = "exact_runtime_s3_archive",
            TrainedAtUtc = now,
            Hyperparameters = BuildHpDict(hp),
            TestMae = perLead.ToDictionary(kv => $"lead_{kv.Key}h_blend", kv => kv.Value.BlendTestMae),
            PerLead = perLead,
            DeviationsFromBrief = new List<string>
            {
                "Trains on RunTimeSource='exact' rows from raw S3 archives (NOAA + ECMWF Open Data + MO AWS) — distinct from the 2b/2c offset_day path. Per-row provenance is rigorous (RunTimeUtc + ValidTimeUtc + LeadHours).",
                "Models = GFS + AIFS required, IFS oper + MO Global optional (tier T2). UKV always optional, sourced via per-V-hour conditional pull from 03Z + 15Z cycles to land at ~12h-ahead per ValidTime.",
                "Lead set restricted to {12, 24} — the leads with sufficient cycle coverage at the 4-cycle ValidTime grid {00,06,12,18}. 48/72/96/120 not trained; 2b/2c retain championship at those leads.",
                "Hyperparameters from the 2026-05-05 81-config grid search: lr=0.05, leaves=31, min-leaf=50, feat-frac=1.0 — the no-UKV per-lead winner, used as a pragmatic split between the per-lead UKV-included winners (lr=0.1/leaves=15 at L=12 vs lr=0.02/leaves=31 at L=24) which diverged.",
            },
        };
        ModelArtifact.SaveTrainingMetadata(versionDir, metadata);
        var firstLead2d = leads.Length > 0 ? leads[0] : 0;
        var guardResult2d = RetrainGuard.BuildCheckAndSave(_log,
            versionDir,
            composite: $"temperature/{stationKey}", phase: "2d", version: versionName,
            computedAtUtc: now,
            rowsTrain: totalTrainRows, rowsVal: totalValRows, rowsTest: totalTestRows,
            trainFeatures: firstLeadTrainFeatures,
            featureNames: specsPerLead.TryGetValue(firstLead2d, out var sp2d)
                ? sp2d.FeatureNames.ToList() : Array.Empty<string>(),
            locationName: location.Name);
        if (!guardResult2d.Passed)
        {
            _log.LogError("Aborting Phase 2d retrain — sanity guard failed. Orphan dir {Dir} not promoted (manifest unchanged).", versionDir);
            return 4;
        }

        // Promote 2d as a challenger. 2c is the champion since 2026-06-15 and its
        // per-lead LEAD_POLICY owns the <24hr (lead-12) bucket via the 0h nowcast.
        // 2d is no longer pinned as the lead-12 champion (the per-lead champion
        // override was removed 2026-06-15) — it stays an Active challenger, still
        // on the forecast/skill pages, just not the <24hr headline.
        // PromoteStationVersionAsChallenger is lead-set-aware (post-2026-05-08) so
        // a 2d retrain at e.g. {72,96,120} no longer clobbers a sibling 2d at
        // {12,24,48}; both stay Active when their lead-sets differ.
        ModelArtifact.PromoteStationVersion(modelsRoot, "temperature", stationKey, versionName, newPhase: "2d");
        var newActive = ModelArtifact.ResolveStationActive(modelsRoot, "temperature", stationKey);

        _log.LogInformation("Phase 2d artefacts → {Dir}", versionDir);
        _log.LogInformation("Active versions now: [{Active}]", string.Join(", ", newActive));
        _log.LogInformation("Summary — {Summary}",
            string.Join("; ", perLead.Select(kv =>
                $"lead {kv.Key}h: blend MAE {kv.Value.BlendTestMae:0.000}°C vs {kv.Value.BestSingle} val MAE {kv.Value.BestSingleValMae:0.000}°C")));

        await Task.CompletedTask;
        return 0;
    }

    private static Dictionary<string, object> BuildHpDict(TempTrainer.Hyperparameters hp) => new()
    {
        ["numberOfIterations"]           = hp.NumberOfIterations,
        ["learningRate"]                 = hp.LearningRate,
        ["numberOfLeaves"]               = hp.NumberOfLeaves,
        ["minimumExampleCountPerLeaf"]   = hp.MinimumExampleCountPerLeaf,
        ["l1Regularization"]             = hp.L1Regularization,
        ["l2Regularization"]             = hp.L2Regularization,
        ["earlyStoppingRound"]           = hp.EarlyStoppingRound,
        ["seed"]                         = hp.Seed,
        ["subsampleFraction"]            = hp.SubsampleFraction,
        ["subsampleFrequency"]           = hp.SubsampleFrequency,
        ["featureFraction"]              = hp.FeatureFraction,
        ["evaluationMetric"]             = "MeanAbsoluteError",
        ["objective"]                    = "regression (L2) — Microsoft.ML.LightGbm 4.0 does not expose regression_l1",
    };

}
