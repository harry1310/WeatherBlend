using System.Text.Json;
using Microsoft.ML;
using WeatherBlend.Models;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train;

/// <summary>
/// On-disk layout for a trained blender.
///
///   data/models/temperature/v{yyyy-MM-dd_HHmmss}/
///     lead_24h.zip / lead_48h.zip / lead_72h.zip   per-lead pipelines
///     feature_schema.json                          shared (13 features, same for all leads)
///     training_metadata.json                       holds PerLead map keyed by "24"/"48"/"72"
///     feature_importance.json                      PerLead map of [{name,gain}] arrays
///
/// MANIFEST.json lives at data/models/temperature/ and points Current → version dir.
/// Resolver returns the version dir; callers pick the per-lead file by name.
/// CI workflows rclone data/ to R2 after training runs.
/// </summary>
public static class ModelArtifact
{
    public const string ManifestFileName = "MANIFEST.json";
    public const string FeatureSchemaFileName = "feature_schema.json";
    public const string TrainingMetadataFileName = "training_metadata.json";
    public const string TrainingSummaryFileName = "training_summary.json";
    public const string FeatureImportanceFileName = "feature_importance.json";
    public const string ClimatologyFileName = "climatology.json";

    /// <summary>Per-lead artifact name, e.g. LeadModelFileName(24) → "lead_24h.zip".</summary>
    public static string LeadModelFileName(int leadHours) => $"lead_{leadHours}h.zip";

    /// <summary>
    /// MANIFEST.json for a target. Every target is Stations-keyed via
    /// <see cref="Stations"/>: precipitation and dry_window key by EA gauge /
    /// composite (one entry per truth source, e.g. <c>ea_bellever_dartmoor</c>),
    /// while temperature and the element_* single-model targets key by location
    /// (e.g. <c>bonehill_rocks</c>) with a single entry. There is no separate
    /// flat layout.
    ///
    /// There is no stored champion pointer. The headline version is derived:
    /// <see cref="ResolveStationChampionVersion"/> reads the phases.yaml
    /// champion phase and picks its newest Active version.
    /// </summary>
    public sealed class Manifest
    {
        public string Target { get; set; } = "";

        // Every target — temperature, precipitation, dry_window, and the
        // element_* single-model targets — is Stations-keyed. A target that
        // isn't gauge-partitioned (temperature, element) uses ONE entry keyed
        // by its location (e.g. "bonehill_rocks"); precipitation / dry_window
        // key by EA gauge / composite. There is no separate flat layout.
        public Dictionary<string, StationEntry> Stations { get; set; } = new();
    }

    public sealed class StationEntry
    {
        public List<string> Versions { get; set; } = new();

        /// <summary>
        /// Versions that should produce predictions/verify rows for this
        /// station. Used by the Phase 3c champion/challenger pattern: every
        /// shipping phase's latest version sits here so they all emit
        /// predictions every cycle. Which one is champion is decided by
        /// phases.yaml, not by this list.
        /// </summary>
        public List<string> Active { get; set; } = new();

        /// <summary>
        /// Per-lead champion override (Phase 3d, 2026-05-05). Maps lead-hours
        /// → version pinned as champion AT THAT LEAD ONLY for this station,
        /// so 3d (exact-runtime precip) can take over as champion at lead 12
        /// per station while the champion phase owns the other leads.
        /// </summary>
        public Dictionary<int, string> ChampionByLead { get; set; } = new();

        /// <summary>
        /// Configured location this station belongs to (e.g.
        /// <c>bonehill_rocks</c>, <c>membury_devon</c>). Populated by
        /// <see cref="PromoteStationVersion"/> from the
        /// freshly-trained bundle's <c>training_metadata.json</c>
        /// LocationName, and asserted never to change once set — a
        /// promote that would swap the location under an existing entry
        /// throws (Phase B multi-location safety, 2026-05-17). Empty on
        /// legacy manifests written before Phase B; the location
        /// backfill tool fills those in. Read via
        /// <see cref="ResolveStationLocation"/>.
        /// </summary>
        public string Location { get; set; } = "";
    }

    public sealed class FeatureSchema
    {
        /// <summary>
        /// Per-lead BlenderSpec (new canonical layout, post unify-model-membership refactor).
        /// Keys are stringified lead-hours ("24", "48", "72", "120").
        /// </summary>
        public Dictionary<string, LeadSchema> Leads { get; set; } = new();

        /// <summary>
        /// Legacy flat feature-name list. Populated by pre-refactor builders that
        /// haven't migrated to <see cref="SaveBlenderSpecs"/> yet. Removed after
        /// Phase 5 of the refactor lands.
        /// </summary>
        public List<string>? FeatureNames { get; set; }

        public string Dtype { get; set; } = "float32";
    }

    /// <summary>
    /// Per-lead schema entry persisted in feature_schema.json. Mirrors the runtime
    /// <see cref="BlenderSpec"/> with concrete <c>List&lt;string&gt;</c> collections so
    /// System.Text.Json round-trips cleanly.
    /// </summary>
    public sealed class LeadSchema
    {
        public string Target { get; set; } = "";
        public string FeatureSet { get; set; } = "";
        public int LeadHours { get; set; }
        public List<string> RequiredModels { get; set; } = new();
        public List<string> OptionalModels { get; set; } = new();
        public List<string> Models { get; set; } = new();
        public List<string> FeatureNames { get; set; } = new();

        /// <summary>One of <see cref="BlenderDataSource"/>'s constants.
        /// Added 2026-05-07 alongside the structured-spec migration —
        /// see BlenderSpec docstring. Empty on legacy schemas.</summary>
        public string DataSource { get; set; } = "";

        /// <summary>Tier label ("lean" / "rich" / "T2" / "P1" etc.) as
        /// the structured analogue of the trailing tier segment in
        /// <see cref="FeatureSet"/>. Empty on legacy schemas.</summary>
        public string Tier { get; set; } = "";

        /// <summary>UKV picker strategy when UKV is used. Persisted as a
        /// plain string ("Strict" / "Averaging" / "" for none) rather than
        /// the typed enum: <c>JsonStringEnumConverter</c> doesn't unwrap
        /// nullable enums, so a typed Nullable&lt;UkvPickStrategy&gt; here
        /// would round-trip the value but fail to deserialise it back on
        /// load (caught by BlenderSpecs_round_trip_through_disk... test
        /// 2026-05-07). The translation to the typed enum lives at the
        /// BlenderSpec boundary in SaveBlenderSpecs / LoadBlenderSpecs.</summary>
        public string UkvStrategy { get; set; } = "";
    }

    public sealed class TrainingMetadata
    {
        public string Version { get; set; } = "";
        public string Target { get; set; } = "";
        public string Phase { get; set; } = "";

        /// <summary>
        /// Configured location whose NWP fed this bundle's training data.
        /// Pinned at train time so predict can refuse to score the bundle
        /// against any other location's NWP (Phase A multi-location safety,
        /// 2026-05-12). Required at deserialise — every bundle on R2 was
        /// backfilled and every trainer writes it. A bundle without this
        /// field is a corrupt/incomplete write and should fail loudly at
        /// load time rather than slipping through to predict.
        /// </summary>
        [System.Text.Json.Serialization.JsonRequired]
        public string LocationName { get; set; } = "";

        /// <summary>Free-form tag — e.g. "previous_runs_api" so the report can explain data source at a glance.</summary>
        public string DataSource { get; set; } = "";

        public DateTime TrainedAtUtc { get; set; }

        public Dictionary<string, object> Hyperparameters { get; set; } = new();
        public Dictionary<string, double> TestMae { get; set; } = new();
        public List<string> DeviationsFromBrief { get; set; } = new();

        /// <summary>One entry per lead ("24","48","72").</summary>
        public Dictionary<string, PerLeadStats> PerLead { get; set; } = new();
    }

    public sealed class PerLeadStats
    {
        public int LeadHours { get; set; }
        public string DataRangeTrain { get; set; } = "";
        public string DataRangeVal { get; set; } = "";
        public string DataRangeTest { get; set; } = "";
        public int TrainRows { get; set; }
        public int ValRows { get; set; }
        public int TestRows { get; set; }
        public int TestCalendarMonths { get; set; }

        /// <summary>Best single model picked on val MAE for this lead.</summary>
        public string BestSingle { get; set; } = "";

        /// <summary>
        /// Best-single model val MAE. Nullable because Python-trained phases
        /// (4a per-cell BART, 5a INLA) don't compute a per-NWP best-single
        /// baseline — train_4a / extend_5a emit JSON `null` here. .NET
        /// trainers (2b/2c/2d/3a/3c/3d/3e/3b/3g/element) always write a real
        /// number. Pre-2026-05-12 this was non-nullable double; the
        /// deserialiser silently rejected 4a/5a's null in
        /// ModelMetadataRepository.GetPhaseByVersion (which reads the whole
        /// POCO just to grab the Phase string), so 4a's PhaseByVersion entry
        /// went missing and RenderPhase4aPanel filtered to 0 rows — chart
        /// went blank starting 2026-05-10 22:08 UTC when the per-cell 4a
        /// refactor minted a bundle with null at every lead. Models page
        /// already handles NaN/missing via the em-dash branch in
        /// SitePages.Models.cs:216-218 — no consumer change needed beyond
        /// unwrapping at the SitePages.PerLeadMetric construction site.
        /// </summary>
        public double? BestSingleValMae { get; set; }

        /// <summary>
        /// Same model identified by val MAE, scored on the test set. Lets the report
        /// answer "does the blend beat best-single on the SAME split?" without picking
        /// a different best per split (which would be cherry-picking). Nullable for
        /// the same reason as BestSingleValMae above (4a/5a don't emit it).
        /// </summary>
        public double? BestSingleTestMae { get; set; }

        /// <summary>Blend MAE on the held-out test set.</summary>
        public double BlendTestMae { get; set; }
        public double BlendTestRmse { get; set; }
        public double BlendTestBias { get; set; }

        /// <summary>
        /// Dry-window only (0.0 elsewhere or on legacy artefacts predating
        /// the field). Blend Brier on the held-out test set AFTER applying
        /// the per-lead isotonic calibrator fit on the validation set —
        /// always populated by the dry-window trainer regardless of whether
        /// the calibrator is actually shipped, so the metadata captures the
        /// raw-vs-calibrated experiment trail. Default 0.0 (not NaN) so
        /// the schema serialises cleanly through System.Text.Json without
        /// AllowNamedFloatingPointLiterals.
        /// </summary>
        public double CalibratedBlendTestMae { get; set; }
    }

    /// <summary>
    /// Station-scoped version dir: <c>{modelsRoot}/{target}/{station}/v{ts}/</c>.
    /// Every target lives under a station/location key — there is no flat
    /// <c>{modelsRoot}/{target}/v{ts}/</c> layout any more.
    /// </summary>
    public static string BuildStationVersionDir(string modelsRoot, string target, string station, DateTime nowUtc, string? suffix = null)
    {
        var ts = nowUtc.ToString("yyyy-MM-dd_HHmmss");
        var name = string.IsNullOrEmpty(suffix) ? $"v{ts}" : $"v{ts}_{suffix}";
        return Path.Combine(modelsRoot, target, station, name).Replace('\\', '/');
    }

    /// <summary>Save one pipeline for a specific lead bucket.</summary>
    public static void SaveLeadModel(
        MLContext ml,
        ITransformer model,
        DataViewSchema inputSchema,
        string versionDir,
        int leadHours)
    {
        Directory.CreateDirectory(versionDir);
        var path = Path.Combine(versionDir, LeadModelFileName(leadHours));
        ml.Model.Save(model, inputSchema, path);
    }

    /// <summary>Load one pipeline for a specific lead bucket.</summary>
    public static ITransformer LoadLeadModel(MLContext ml, string versionDir, int leadHours, out DataViewSchema schema)
    {
        var path = Path.Combine(versionDir, LeadModelFileName(leadHours));
        return ml.Model.Load(path, out schema);
    }

    /// <summary>Per-lead isotonic calibrator filename, e.g. <c>calibrator_24h.json</c>.</summary>
    public static string LeadCalibratorFileName(int leadHours) => $"calibrator_{leadHours}h.json";

    /// <summary>
    /// Save a fitted isotonic calibrator alongside the per-lead LightGBM zip.
    /// Optional — predict gracefully degrades to "raw probabilities, uncalibrated"
    /// when the file is absent (older artefacts, or models trained before the
    /// calibration layer landed).
    /// </summary>
    public static void SaveLeadCalibrator(WeatherBlend.Train.Common.IsotonicCalibrator calibrator,
                                          string versionDir, int leadHours)
    {
        Directory.CreateDirectory(versionDir);
        var path = Path.Combine(versionDir, LeadCalibratorFileName(leadHours));
        File.WriteAllText(path, calibrator.ToJson());
    }

    /// <summary>
    /// Load a per-lead isotonic calibrator if one was saved at training time.
    /// Returns <c>null</c> when absent — callers apply only when present so old
    /// artefacts predict identically to how they trained (just uncalibrated).
    /// </summary>
    public static WeatherBlend.Train.Common.IsotonicCalibrator? TryLoadLeadCalibrator(
        string versionDir, int leadHours)
    {
        var path = Path.Combine(versionDir, LeadCalibratorFileName(leadHours));
        if (!File.Exists(path)) return null;
        return WeatherBlend.Train.Common.IsotonicCalibrator.FromJson(File.ReadAllText(path));
    }

    /// <summary>Filename for the conformal calibrator at one lead. Distinct
    /// from the isotonic calibrator (which scales the probability) — conformal
    /// emits set-membership tags for the same probability without changing it.
    /// Both can coexist in one version dir.</summary>
    public static string LeadConformalCalibratorFileName(int leadHours)
        => $"conformal_calibrator_{leadHours}h.json";

    public static void SaveLeadConformalCalibrator(
        WeatherBlend.Train.Common.ConformalCalibrator cal, string versionDir, int leadHours)
    {
        Directory.CreateDirectory(versionDir);
        var path = Path.Combine(versionDir, LeadConformalCalibratorFileName(leadHours));
        File.WriteAllText(path, cal.ToJson());
    }

    public static WeatherBlend.Train.Common.ConformalCalibrator? TryLoadLeadConformalCalibrator(
        string versionDir, int leadHours)
    {
        var path = Path.Combine(versionDir, LeadConformalCalibratorFileName(leadHours));
        if (!File.Exists(path)) return null;
        return WeatherBlend.Train.Common.ConformalCalibrator.FromJson(File.ReadAllText(path));
    }

    /// <summary>
    /// Convenience used by every predict path that wants to attach a
    /// conformal-set tag to a row's probability: load the per-(versionDir,
    /// lead) calibrator if one was fitted (precip-conformal-fit /
    /// dry-window-conformal-fit) and return the SetTag enum's name as a
    /// string ("Dry" / "Wet" / "Ambiguous"); null when no calibrator is
    /// present so the parquet column degrades gracefully on legacy versions.
    /// </summary>
    public static string? PredictConformalIfPresent(string versionDir, int leadHours, double prob)
        => TryLoadLeadConformalCalibrator(versionDir, leadHours)?.Predict(prob).ToString();

    /// <summary>
    /// Legacy flat-list schema writer. Used by builders that haven't been migrated
    /// to <see cref="SaveBlenderSpecs"/> yet. Removed after Phase 5 of the
    /// unify-model-membership refactor.
    /// </summary>
    [Obsolete("Use SaveBlenderSpecs instead — per-lead schema captures model membership.")]
    public static void SaveFeatureSchema(string versionDir, IEnumerable<string> featureNames)
    {
        var obj = new FeatureSchema { FeatureNames = featureNames.ToList() };
        WriteJson(Path.Combine(versionDir, FeatureSchemaFileName), obj);
    }

    /// <summary>
    /// Persist per-lead BlenderSpecs to <c>feature_schema.json</c>. One file per
    /// version captures every lead in that artefact set so predict can look up
    /// the (models, feature ordering, requireAllModelsPresent) policy without
    /// consulting config.yaml — config might have changed between train and predict.
    /// </summary>
    public static void SaveBlenderSpecs(string versionDir, IReadOnlyDictionary<int, BlenderSpec> specsPerLead)
    {
        var schema = new FeatureSchema();
        foreach (var (lead, spec) in specsPerLead)
        {
            schema.Leads[lead.ToString(System.Globalization.CultureInfo.InvariantCulture)] = new LeadSchema
            {
                Target = spec.Target,
                FeatureSet = spec.FeatureSet,
                LeadHours = spec.LeadHours,
                RequiredModels = spec.RequiredModels.ToList(),
                OptionalModels = spec.OptionalModels.ToList(),
                Models = spec.Models.ToList(),
                FeatureNames = spec.FeatureNames.ToList(),
                DataSource = spec.DataSource,
                Tier = spec.Tier,
                // Persist enum as its string name ("Strict" / "Averaging");
                // empty string when null (no UKV). Plain string avoids the
                // nullable-enum deserialisation quirk in System.Text.Json.
                UkvStrategy = spec.UkvStrategy?.ToString() ?? "",
            };
        }
        WriteJson(Path.Combine(versionDir, FeatureSchemaFileName), schema);
    }

    /// <summary>
    /// Load every per-lead BlenderSpec from a version's <c>feature_schema.json</c>.
    /// Throws if the file is missing — predict cannot run without the schema
    /// (we'd have no idea which models the trained vector expects).
    /// </summary>
    public static IReadOnlyDictionary<int, BlenderSpec> LoadBlenderSpecs(string versionDir)
    {
        var path = Path.Combine(versionDir, FeatureSchemaFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Missing {FeatureSchemaFileName} in {versionDir} — artefact predates the per-lead schema layout. Retrain.");

        var schema = ReadJson<FeatureSchema>(path)
                     ?? throw new InvalidOperationException($"Could not parse {path}");

        var result = new Dictionary<int, BlenderSpec>();
        foreach (var (key, ls) in schema.Leads)
        {
            if (!int.TryParse(key, out var lead))
                throw new InvalidOperationException($"Non-integer lead key '{key}' in {path}");
            result[lead] = new BlenderSpec
            {
                Target = ls.Target,
                FeatureSet = ls.FeatureSet,
                LeadHours = ls.LeadHours,
                RequiredModels = ls.RequiredModels,
                OptionalModels = ls.OptionalModels,
                Models = ls.Models,
                FeatureNames = ls.FeatureNames,
                DataSource = ls.DataSource,
                Tier = ls.Tier,
                // Translate the saved string ("Strict" / "Averaging" / "")
                // back to the typed enum. Empty / unknown → null (no UKV).
                UkvStrategy = Enum.TryParse<WeatherBlend.Train.Exact12h.Exact12hFeatureBuilder.UkvPickStrategy>(
                    ls.UkvStrategy, ignoreCase: false, out var ukvParsed) ? ukvParsed : null,
            };
        }
        return result;
    }

    public static void SaveFeatureImportance(
        string versionDir,
        IEnumerable<(string Name, double Gain)> importance)
    {
        var dict = importance.Select(t => new Dictionary<string, object>
        {
            ["name"] = t.Name,
            ["gain"] = t.Gain,
        }).ToList();
        WriteJson(Path.Combine(versionDir, FeatureImportanceFileName), dict);
    }

    /// <summary>
    /// Persist one importance list per lead. File format is a dict keyed by lead
    /// string ("24","48","72") so one file covers the whole training run.
    /// </summary>
    public static void SavePerLeadFeatureImportance(
        string versionDir,
        IReadOnlyDictionary<int, IEnumerable<(string Name, double Gain)>> byLead)
    {
        var obj = byLead.ToDictionary(
            kv => kv.Key.ToString(),
            kv => kv.Value.Select(t => new Dictionary<string, object>
            {
                ["name"] = t.Name,
                ["gain"] = t.Gain,
            }).ToList());
        WriteJson(Path.Combine(versionDir, FeatureImportanceFileName), obj);
    }

    public static IReadOnlyDictionary<int, IReadOnlyList<(string Name, double Gain)>> LoadPerLeadFeatureImportance(string versionDir)
    {
        var path = Path.Combine(versionDir, FeatureImportanceFileName);
        if (!File.Exists(path))
            return new Dictionary<int, IReadOnlyList<(string, double)>>();

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var result = new Dictionary<int, IReadOnlyList<(string, double)>>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (!int.TryParse(prop.Name, out var lead)) continue;
            var list = new List<(string, double)>();
            foreach (var item in prop.Value.EnumerateArray())
            {
                var name = item.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
                var gain = item.TryGetProperty("gain", out var g) ? g.GetDouble() : 0.0;
                list.Add((name, gain));
            }
            result[lead] = list;
        }
        return result;
    }

    public static void SaveTrainingMetadata(string versionDir, TrainingMetadata metadata)
        => WriteJson(Path.Combine(versionDir, TrainingMetadataFileName), metadata);

    public static TrainingMetadata LoadTrainingMetadata(string versionDir)
        => ReadJson<TrainingMetadata>(Path.Combine(versionDir, TrainingMetadataFileName))
           ?? throw new InvalidOperationException($"Missing training metadata in {versionDir}");

    /// <summary>Persist a <see cref="WeatherBlend.Train.Common.TrainingSummary"/>
    /// alongside training_metadata.json. Called by the trainers right after
    /// SaveTrainingMetadata so the two files land atomically (best-effort —
    /// no transactional write across the two paths). Read by the upcoming
    /// retrain guard (Phase 1b of AUTO_RETRAIN_PLAN.md).</summary>
    public static void SaveTrainingSummary(string versionDir, WeatherBlend.Train.Common.TrainingSummary summary)
        => WriteJson(Path.Combine(versionDir, TrainingSummaryFileName), summary);

    /// <summary>Load a TrainingSummary if present. Returns null when the
    /// version dir has none (legacy artefacts written before Phase 1a, or
    /// trainers that haven't been wired yet). Caller should treat null as
    /// "first-ever summary; no prior baseline to compare against".</summary>
    public static WeatherBlend.Train.Common.TrainingSummary? TryLoadTrainingSummary(string versionDir)
    {
        var path = Path.Combine(versionDir, TrainingSummaryFileName);
        if (!File.Exists(path)) return null;
        return ReadJson<WeatherBlend.Train.Common.TrainingSummary>(path);
    }

    // ------------------------------------------------------------------
    // Champion resolution — derived, never stored
    //
    // The headline ("champion") version is computed from the canonical
    // phases.yaml registry (one champion phase per target) + the manifest's
    // Active set. There is no stored Current pointer: a mutable field any
    // trainer could clobber was exactly the bug this replaced — Phase 4b's
    // mint silently re-pointed the precipitation champion away from 3a.
    // ------------------------------------------------------------------

    /// <summary>
    /// The champion VERSION for a (target, station) — the newest Active
    /// entry whose trained phase equals the target's phases.yaml champion
    /// phase (<see cref="ActivePhasePolicy.ChampionPhase"/>). Replaces the
    /// retired per-station <c>Current</c> pointer. Empty when the station
    /// has no Active version of the champion phase.
    /// </summary>
    public static string ResolveStationChampionVersion(
        string modelsRoot, string target, string station)
        => ResolveStationPhaseVersion(
            modelsRoot, target, station, ActivePhasePolicy.ChampionPhase(target));

    /// <summary>
    /// The newest Active version of a specific <paramref name="phase"/> for
    /// a (target, station) — lexical max over Active entries whose
    /// <c>training_metadata</c> Phase matches. Empty when none match. Backs
    /// <see cref="ResolveStationChampionVersion"/>. (The dry-window MC source
    /// binding via <c>DryWindowMcSources</c> was retired 2026-05-25 in
    /// model-cleanup Phase 1; helper kept for analogous future bindings.)
    /// </summary>
    public static string ResolveStationPhaseVersion(
        string modelsRoot, string target, string station, string phase)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null || !manifest.Stations.TryGetValue(station, out var entry))
            return "";
        var stationDir = Path.Combine(modelsRoot, target, station);
        return entry.Active
            .Where(v => string.Equals(
                TryReadVersionPhase(Path.Combine(stationDir, v)), phase, StringComparison.Ordinal))
            .OrderByDescending(v => v, StringComparer.Ordinal)
            .FirstOrDefault() ?? "";
    }

    /// <summary>
    /// Set the per-lead champion override for one (station, lead). Pass an
    /// empty <paramref name="versionDirName"/> to clear the entry.
    /// Idempotent. Doesn't touch <see cref="StationEntry.Active"/>.
    /// </summary>
    public static void SetStationChampionForLead(
        string modelsRoot, string target, string station, int leadHours, string versionDirName)
    {
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Stations.TryGetValue(station, out var entry))
            {
                entry = new StationEntry();
                m.Stations[station] = entry;
            }
            if (string.IsNullOrWhiteSpace(versionDirName))
                entry.ChampionByLead.Remove(leadHours);
            else
                entry.ChampionByLead[leadHours] = versionDirName;
        });
    }

    /// <summary>
    /// Read-mutate-write the manifest under a cross-process lock. The lock
    /// serialises concurrent mutations so updates can't trample each other
    /// (chained <c>train --feature-set lean &amp;&amp; train --feature-set rich</c>
    /// invocations + parallel station trainers were the previous loss vector).
    /// The final write uses <c>File.Move(overwrite: true)</c>, which is a single
    /// atomic rename — readers (predict/verify/render-site) never see a missing
    /// or half-written manifest.
    /// </summary>
    private static void MutateManifest(string modelsRoot, string target, Action<Manifest> mutate)
    {
        var dir = Path.Combine(modelsRoot, target);
        Directory.CreateDirectory(dir);
        var manifestPath = Path.Combine(dir, ManifestFileName);
        var lockPath = manifestPath + ".lock";

        using (AcquireManifestLock(lockPath))
        {
            var manifest = File.Exists(manifestPath)
                ? ReadJson<Manifest>(manifestPath) ?? new Manifest()
                : new Manifest();

            manifest.Target = target;
            mutate(manifest);

            var tmp = manifestPath + ".tmp";
            WriteJson(tmp, manifest);
            AtomicReplace(tmp, manifestPath);
        }
    }

    // Lock-acquire backoff: polled retry up to ~5s. Manifest writes are tiny so
    // contention should resolve in a handful of milliseconds; the upper bound is
    // there only so a wedged lock-holder fails loudly instead of hanging forever.
    private const int LockAcquireMaxAttempts = 50;
    private const int LockAcquireBackoffMs = 100;

    /// <summary>
    /// Replace <paramref name="dst"/> with <paramref name="src"/> atomically.
    /// On Windows we prefer <see cref="File.Replace(string,string,string)"/>
    /// (ReplaceFile Win32 API) because it tolerates concurrent readers holding
    /// snapshot handles — <see cref="File.Move(string,string,bool)"/> with
    /// REPLACE_EXISTING is rejected when any handle is open on the destination,
    /// even readers opened with <c>FileShare.Delete</c>. On POSIX, both APIs
    /// reduce to the same <c>rename(2)</c>; we fall back to Move if Replace
    /// throws (e.g. cross-device or unsupported filesystem). Tiny per-call retry
    /// loop covers the rare narrow window where Windows hasn't released the
    /// previous reader handle yet.
    /// </summary>
    private static void AtomicReplace(string src, string dst)
    {
        if (!File.Exists(dst))
        {
            // Initial write — Replace requires an existing destination.
            File.Move(src, dst);
            return;
        }
        const int MaxAttempts = 20;
        for (int attempt = 0; attempt < MaxAttempts; attempt++)
        {
            try
            {
                File.Replace(src, dst, destinationBackupFileName: null);
                return;
            }
            catch (PlatformNotSupportedException)
            {
                // POSIX fallback — Move(overwrite) is atomic via rename(2).
                File.Move(src, dst, overwrite: true);
                return;
            }
            catch (IOException) when (attempt < MaxAttempts - 1)
            {
                // Transient Windows handle contention — back off briefly.
                Thread.Sleep(5);
            }
            catch (UnauthorizedAccessException) when (attempt < MaxAttempts - 1)
            {
                Thread.Sleep(5);
            }
        }
    }

    private static FileStream AcquireManifestLock(string lockPath)
    {
        for (int attempt = 0; attempt < LockAcquireMaxAttempts; attempt++)
        {
            try
            {
                // FileShare.None blocks any other process / thread that opens the
                // same lock file. DeleteOnClose tidies the sentinel after release
                // so the lock file doesn't accumulate in the model tree.
                return new FileStream(
                    lockPath,
                    FileMode.OpenOrCreate,
                    FileAccess.ReadWrite,
                    FileShare.None,
                    bufferSize: 1,
                    FileOptions.DeleteOnClose);
            }
            catch (IOException)
            {
                // Another process holds the lock — back off and retry.
                Thread.Sleep(LockAcquireBackoffMs);
            }
        }
        throw new IOException(
            $"Could not acquire manifest lock at '{lockPath}' after " +
            $"{LockAcquireMaxAttempts * LockAcquireBackoffMs}ms — " +
            "another writer is wedged or holding it for an unusually long time.");
    }

    /// <summary>
    /// Update the per-station entry in the target manifest. Atomic write via
    /// temp+move. Safe to call concurrently-ish (not truly thread-safe, but
    /// training runs are serialised).
    /// </summary>
    public static void UpdateStationManifest(string modelsRoot, string target, string station, string versionDirName)
    {
        var dir = Path.Combine(modelsRoot, target);
        Directory.CreateDirectory(dir);
        var manifestPath = Path.Combine(dir, ManifestFileName);

        var manifest = File.Exists(manifestPath)
            ? ReadJson<Manifest>(manifestPath) ?? new Manifest()
            : new Manifest();

        manifest.Target = target;
        if (!manifest.Stations.TryGetValue(station, out var entry))
        {
            entry = new StationEntry();
            manifest.Stations[station] = entry;
        }
        if (!entry.Versions.Contains(versionDirName))
            entry.Versions.Add(versionDirName);
        // Single-active semantics: a plain UpdateStationManifest resets Active
        // to [version]. For Phase 3c champion/challenger, call SetStationActive
        // afterwards with the full list.
        entry.Active = new List<string> { versionDirName };

        var tmp = manifestPath + ".tmp";
        WriteJson(tmp, manifest);
        if (File.Exists(manifestPath)) File.Delete(manifestPath);
        File.Move(tmp, manifestPath);
    }

    /// <summary>
    /// Append a station version to its history list without touching Active.
    /// </summary>
    public static void AppendStationVersion(string modelsRoot, string target, string station, string versionDirName)
    {
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Stations.TryGetValue(station, out var entry))
            {
                entry = new StationEntry();
                m.Stations[station] = entry;
            }
            if (!entry.Versions.Contains(versionDirName))
                entry.Versions.Add(versionDirName);
        });
    }

    /// <summary>
    /// Replace the Active list for a specific station. Predict + verify iterate this
    /// list when no specific version is requested. Caller is responsible for ensuring
    /// every listed version exists under <c>{modelsRoot}/{target}/{station}/</c>.
    /// </summary>
    public static void SetStationActive(string modelsRoot, string target, string station, IEnumerable<string> activeVersions)
    {
        var list = activeVersions.Distinct().ToList();
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Stations.TryGetValue(station, out var entry))
            {
                entry = new StationEntry();
                m.Stations[station] = entry;
            }
            entry.Active = list;
        });
    }

    // ------------------------------------------------------------------
    // Idempotent promote helper
    // ------------------------------------------------------------------
    //
    // The lower-level UpdateStationManifest / SetStationActive helpers
    // unconditionally REPLACE the Active list. That's wrong for any setup
    // running champion + challenger phases concurrently: retraining one phase
    // (say 2b) silently kicks the other (2c) out of Active, and predict +
    // verify stop scoring it on the next cycle. PromoteStationVersion below
    // is the right-shaped operation for "I just trained version V of phase
    // P" — it replaces any existing Active entry whose Phase matches P
    // (read from each entry's training_metadata.json) and adds V, leaving
    // every other phase untouched. Idempotent on re-run: training the same
    // phase twice converges to a one-entry-per-phase Active list rather
    // than accumulating dead versions or wiping siblings.
    //
    // There is no champion/challenger distinction at the manifest level any
    // more: promote just registers the version in Versions + Active (with
    // lead-set-aware replacement of the same phase). WHICH phase is champion
    // is decided solely by phases.yaml (ActivePhasePolicy.ChampionPhase); the
    // headline version is derived from that — see
    // ResolveStationChampionVersion. Pass the Phase string from the
    // training_metadata you just wrote — the helper doesn't infer it from the
    // version-name suffix because the suffix scheme isn't load-bearing.

    /// <summary>
    /// Register a freshly-trained version in a station entry of a target
    /// manifest. Adds to Versions and replaces any existing same-phase Active
    /// entry (lead-set-aware — see <see cref="ComposeActive"/>), leaving other
    /// phases untouched. Idempotent on re-run. Pins the station's location on
    /// first write and refuses a later location change (Phase B multi-location
    /// safety). Targets that aren't gauge-partitioned (temperature, element_*)
    /// pass their location as the station key.
    /// </summary>
    public static void PromoteStationVersion(
        string modelsRoot, string target, string station, string newVersionName, string newPhase)
    {
        var newLeadSet = TryReadVersionLeads(Path.Combine(modelsRoot, target, station, newVersionName));
        var newLocation = TryReadVersionLocation(Path.Combine(modelsRoot, target, station, newVersionName));
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Stations.TryGetValue(station, out var entry))
            {
                entry = new StationEntry();
                m.Stations[station] = entry;
            }
            // Pin the station's location from the freshly-trained bundle's
            // metadata. Once set it must never change: a promote that
            // would swap bonehill_rocks <-> membury_devon under an
            // existing entry is the catastrophic "wrong location's bundle
            // in this slot" scenario — throw rather than corrupt the
            // manifest. A bundle whose location can't be read (legacy /
            // unreadable metadata) leaves the entry's Location untouched.
            if (newLocation is not null)
            {
                if (!string.IsNullOrEmpty(entry.Location)
                    && !string.Equals(entry.Location, newLocation, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(
                        $"Refusing to promote version '{newVersionName}' into station " +
                        $"'{station}' of target '{target}': the bundle's location is " +
                        $"'{newLocation}' but the manifest entry is pinned to " +
                        $"'{entry.Location}'. A station never changes location — this is " +
                        "either the wrong bundle or a misrouted trainer.");
                entry.Location = newLocation;
            }
            if (!entry.Versions.Contains(newVersionName)) entry.Versions.Add(newVersionName);
            entry.Active = ComposeActive(entry.Active, newVersionName, newPhase, newLeadSet,
                phaseOfVersion: v => TryReadVersionPhase(Path.Combine(modelsRoot, target, station, v)),
                leadsOfVersion: v => TryReadVersionLeads(Path.Combine(modelsRoot, target, station, v)));
        });
    }

    /// <summary>
    /// Build the new Active list with lead-overlap-aware replacement.
    ///
    /// Replacement rule: drop an existing entry when its phase matches
    /// <paramref name="newPhase"/> AND its lead-set OVERLAPS with
    /// <paramref name="newLeadSet"/> on at least one lead. Two same-phase
    /// versions can only coexist when their leads are fully disjoint
    /// (e.g. 2d V_short {12,24,48} alongside 2d V_long {72,96,120} —
    /// both Active, neither emits at the other's leads, no duplicate
    /// predictions on the same (composite, lead, valid) tuple).
    ///
    /// Concretely:
    ///   * V_old {48,72} + V_new {12,24,48,72} → overlap on {48,72} → drop V_old.
    ///     V_new fully supersedes V_old's coverage.
    ///   * V_old {12,24,48} + V_new {72,96,120} → disjoint → both stay.
    ///   * V_old {12,24,48} + V_new {12,24,48} → full overlap → drop V_old
    ///     (re-train idempotency).
    ///   * V_old {12,24} + V_new {24} → overlap on {24} → drop V_old.
    ///     This is the operational footgun: a partial retrain that
    ///     overlaps an existing wider version supersedes it. Operator's
    ///     mental model is "training at lead L replaces any prior model
    ///     emitting at L"; so don't run partial-coverage retrains
    ///     unless that's what you want.
    ///
    /// Versions whose metadata is unreadable (missing dir, malformed JSON,
    /// missing schema) are PRESERVED — we don't silently retire something
    /// we can't introspect. Versions where lead-set is unknown but phase
    /// matches are also preserved.
    ///
    /// History:
    ///   * Pre-2026-05-08: dropped EVERY same-phase entry. A 2d retrain at
    ///     {72,96,120} silently removed an existing 2d at {12,24,48}.
    ///   * 2026-05-08 first attempt: lead-set EQUALITY check kept BOTH
    ///     versions even when one's coverage strictly contained the
    ///     other's, leading to duplicate predictions on shared leads.
    ///   * 2026-05-08 final: lead-set OVERLAP check (this rule). Disjoint
    ///     stays; overlap replaces.
    /// </summary>
    private static List<string> ComposeActive(
        IReadOnlyList<string> existing,
        string newVersionName,
        string newPhase,
        IReadOnlyList<int>? newLeadSet,
        Func<string, string?> phaseOfVersion,
        Func<string, IReadOnlyList<int>?> leadsOfVersion)
    {
        var preserved = new List<string>(existing.Count + 1);
        foreach (var v in existing)
        {
            if (string.Equals(v, newVersionName, StringComparison.Ordinal)) continue;  // dedupe later
            var phase = phaseOfVersion(v);
            if (phase is null || !string.Equals(phase, newPhase, StringComparison.Ordinal))
            {
                preserved.Add(v);
                continue;
            }
            // Same phase. Drop only when lead-sets OVERLAP. Disjoint
            // lead-sets coexist (both versions emit at non-conflicting
            // leads). Unknown lead-set on either side → preserve
            // conservatively.
            var existingLeadSet = leadsOfVersion(v);
            if (newLeadSet is null || existingLeadSet is null
                || !LeadSetsOverlap(existingLeadSet, newLeadSet))
            {
                preserved.Add(v);
                continue;
            }
            // Same phase + at least one shared lead → V_new supersedes
            // V_old at the shared leads (would emit duplicate predictions
            // otherwise). Drop V_old.
        }
        preserved.Add(newVersionName);
        return preserved;
    }

    private static bool LeadSetsOverlap(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        var aSet = new HashSet<int>(a);
        foreach (var l in b)
            if (aSet.Contains(l)) return true;
        return false;
    }

    /// <summary>
    /// Read the integer-keyed leads from a version's
    /// <c>feature_schema.json</c>. Used by <see cref="ComposeActive"/> to
    /// distinguish a same-phase re-train (same lead set → replace) from a
    /// same-phase complementary version (different lead set → coexist).
    /// Returns null on missing dir / file / parse error so callers can
    /// fall back to "preserve, don't risk silent removal".
    /// </summary>
    private static IReadOnlyList<int>? TryReadVersionLeads(string versionDir)
    {
        try
        {
            if (!Directory.Exists(versionDir)) return null;
            var path = Path.Combine(versionDir, FeatureSchemaFileName);
            if (!File.Exists(path)) return null;
            var schema = ReadJson<FeatureSchema>(path);
            if (schema is null) return null;
            var leads = new List<int>(schema.Leads.Count);
            foreach (var key in schema.Leads.Keys)
                if (int.TryParse(key, System.Globalization.NumberStyles.Integer,
                                 System.Globalization.CultureInfo.InvariantCulture, out var lead))
                    leads.Add(lead);
            return leads;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Read the <c>Phase</c> field from <c>training_metadata.json</c> at
    /// <paramref name="versionDir"/>. Returns null on missing dir, missing
    /// file, or any deserialise error — callers treat null as "unknown
    /// phase, don't touch this entry".
    /// </summary>
    private static string? TryReadVersionPhase(string versionDir)
    {
        try
        {
            if (!Directory.Exists(versionDir)) return null;
            var path = Path.Combine(versionDir, TrainingMetadataFileName);
            if (!File.Exists(path)) return null;
            var meta = LoadTrainingMetadata(versionDir);
            return string.IsNullOrEmpty(meta.Phase) ? null : meta.Phase;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Read the <c>LocationName</c> from <c>training_metadata.json</c> at
    /// <paramref name="versionDir"/>. Returns null on missing dir, missing
    /// file, or any deserialise error — callers treat null as "location
    /// unknown, leave the manifest entry's Location untouched". Phase B
    /// (2026-05-17): used by <see cref="PromoteStationVersion"/>
    /// to pin a station's location from the bundle it just trained.
    /// </summary>
    private static string? TryReadVersionLocation(string versionDir)
    {
        try
        {
            if (!Directory.Exists(versionDir)) return null;
            var path = Path.Combine(versionDir, TrainingMetadataFileName);
            if (!File.Exists(path)) return null;
            var meta = LoadTrainingMetadata(versionDir);
            return string.IsNullOrWhiteSpace(meta.LocationName) ? null : meta.LocationName;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Append a single version to the station's Active list (champion + challengers).
    /// Idempotent: a version that's already Active is left in place. Doesn't touch
    /// Current. Used by 3e to register itself as a challenger alongside the existing
    /// 3b champion without disturbing the latter.
    /// </summary>
    public static void AddStationActive(string modelsRoot, string target, string station, string versionDirName)
    {
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Stations.TryGetValue(station, out var entry))
            {
                entry = new StationEntry();
                m.Stations[station] = entry;
            }
            if (!entry.Active.Contains(versionDirName))
                entry.Active.Add(versionDirName);
        });
    }

    /// <summary>
    /// Versions that should produce predictions/verify rows for this station.
    /// Empty when the station isn't in the manifest or has no Active entries.
    /// </summary>
    public static IReadOnlyList<string> ResolveStationActive(string modelsRoot, string target, string station)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        if (!manifest.Stations.TryGetValue(station, out var entry)) return Array.Empty<string>();
        return entry.Active.Count > 0 ? entry.Active : Array.Empty<string>();
    }

    /// <summary>
    /// Location-checked variant of
    /// <see cref="ResolveStationActive(string,string,string)"/>. Resolves
    /// the station's Active versions but first asserts the manifest
    /// entry's <see cref="StationEntry.Location"/> matches
    /// <paramref name="location"/> — a predict / verify pass that names
    /// the location it expects gets a loud failure if the station has
    /// been re-homed under a different location (Phase B multi-location
    /// safety). An entry with an empty Location (legacy, pre-backfill) is
    /// permitted against any location since there's nothing to check.
    /// </summary>
    public static IReadOnlyList<string> ResolveStationActive(
        string modelsRoot, string target, string location, string station)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        if (!manifest.Stations.TryGetValue(station, out var entry)) return Array.Empty<string>();
        if (!string.IsNullOrEmpty(entry.Location)
            && !string.Equals(entry.Location, location, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                $"Station '{station}' in target '{target}' is pinned to location " +
                $"'{entry.Location}' but was resolved for location '{location}'. " +
                "Predict/verify must request the station's own location.");
        return entry.Active.Count > 0 ? entry.Active : Array.Empty<string>();
    }

    /// <summary>
    /// The location a station is pinned to, or empty string when the
    /// station isn't in the manifest / has no Location set (legacy
    /// pre-backfill entries). Phase B (2026-05-17); consumed by
    /// predict / verify / render fanout to route a station to its own
    /// location's NWP + truth.
    /// </summary>
    public static string ResolveStationLocation(string modelsRoot, string target, string station)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return "";
        return manifest.Stations.TryGetValue(station, out var entry) ? entry.Location : "";
    }

    /// <summary>
    /// Resolve a per-station version dir: <c>"current"</c> resolves to the
    /// station's champion version (<see cref="ResolveStationChampionVersion"/>
    /// — newest Active version of the phases.yaml champion phase); any other
    /// string is treated as an explicit <c>v…</c> folder name. Returned path
    /// always points inside the <c>{target}/{station}/</c> subtree.
    /// </summary>
    public static string ResolveStationVersionDir(string modelsRoot, string target, string station, string versionOrCurrent)
    {
        var stationDir = Path.Combine(modelsRoot, target, station);
        if (!string.Equals(versionOrCurrent, "current", StringComparison.OrdinalIgnoreCase))
            return Path.Combine(stationDir, versionOrCurrent);

        var champion = ResolveStationChampionVersion(modelsRoot, target, station);
        if (string.IsNullOrWhiteSpace(champion))
            throw new InvalidOperationException(
                $"No champion version for station '{station}' in target '{target}' — "
                + "no Active version of the champion phase.");
        return Path.Combine(stationDir, champion);
    }

    /// <summary>
    /// Stations currently ACTIVE in the manifest for this target — i.e. those
    /// with a non-empty <c>Active</c> list. Stations with an empty Active list
    /// (demoted-to-archive entries kept around so their historical parquets
    /// remain readable) are silently filtered out. Returns empty when the
    /// manifest is absent or flat-layout.
    ///
    /// Was previously "every key in manifest.Stations" — that broke
    /// PrecipPredictCommand on 2026-05-04 when a station-swap left a demoted
    /// entry behind: predict iterated all stations and then threw. Filter at
    /// the source so every caller (predict / verify / ablate) gets the
    /// demoted-archive semantics without re-encoding the rule.
    /// </summary>
    public static IReadOnlyList<string> ListStations(string modelsRoot, string target)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        return manifest.Stations
            .Where(kv => kv.Value.Active.Count > 0)
            .Select(kv => kv.Key)
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToArray();
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    private static void WriteJson<T>(string path, T value)
        => File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOpts));

    /// <summary>
    /// Read JSON with permissive file sharing so a concurrent writer can replace
    /// the file underneath us. Windows' <see cref="File.Replace(string,string,string)"/>
    /// has a microsecond window where the destination appears not-to-exist or
    /// throws sharing-violation to a reader that opens at the wrong instant —
    /// the swap itself is atomic at the filesystem level, but the path lookup
    /// flickers. We retry transient missing/sharing/not-found errors a handful
    /// of times so callers get a stable old-or-new snapshot. A path that's
    /// genuinely absent for the full retry window (no concurrent writer)
    /// returns <c>default</c> as before.
    /// </summary>
    private static T? ReadJson<T>(string path)
    {
        const int MaxAttempts = 50;
        for (int attempt = 0; attempt < MaxAttempts; attempt++)
        {
            try
            {
                using var fs = new FileStream(
                    path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete);
                return JsonSerializer.Deserialize<T>(fs);
            }
            catch (DirectoryNotFoundException)
            {
                // Containing dir never existed — nothing to retry.
                return default;
            }
            catch (FileNotFoundException) { /* retry below */ }
            catch (IOException)             { /* retry below */ }
            catch (UnauthorizedAccessException) { /* retry below */ }
            Thread.Sleep(2);
        }
        // After ~100ms of retries the file still isn't readable — treat as absent.
        return default;
    }
}
