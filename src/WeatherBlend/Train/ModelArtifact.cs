using System.Text.Json;
using Microsoft.ML;
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
    public const string FeatureImportanceFileName = "feature_importance.json";
    public const string ClimatologyFileName = "climatology.json";

    /// <summary>Per-lead artifact name, e.g. LeadModelFileName(24) → "lead_24h.zip".</summary>
    public static string LeadModelFileName(int leadHours) => $"lead_{leadHours}h.zip";

    /// <summary>
    /// MANIFEST.json for a target. Temperature uses the flat
    /// <see cref="Current"/>/<see cref="Versions"/> fields; precipitation uses
    /// <see cref="Stations"/> (one entry per truth source, e.g. <c>ea_bellever_dartmoor</c>)
    /// because each station is a distinct blender. Exactly one of the two sides
    /// is populated for a given target; the other stays at its default.
    /// </summary>
    public sealed class Manifest
    {
        public string Target { get; set; } = "";

        // Flat layout (temperature).
        public string Current { get; set; } = "";
        public List<string> Versions { get; set; } = new();

        /// <summary>
        /// Versions that predict + verify should treat as live. Empty means "fall back
        /// to [Current]" (back-compat for manifests written before Phase 2c). Used by
        /// the champion/challenger pattern: 2b stays as Current, 2c is added to Active
        /// alongside it so both produce predictions every cycle.
        /// </summary>
        public List<string> Active { get; set; } = new();

        // Per-station layout (precipitation).
        public Dictionary<string, StationEntry> Stations { get; set; } = new();
    }

    public sealed class StationEntry
    {
        public string Current { get; set; } = "";
        public List<string> Versions { get; set; } = new();

        /// <summary>
        /// Versions that should produce predictions/verify rows for this station. Empty
        /// means "fall back to [Current]" (back-compat). Used by Phase 3c champion/
        /// challenger: the 3a-lean stays as Current while 3c-rich is appended to Active
        /// so both versions emit predictions every cycle.
        /// </summary>
        public List<string> Active { get; set; } = new();
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
    }

    public sealed class TrainingMetadata
    {
        public string Version { get; set; } = "";
        public string Target { get; set; } = "";
        public string Phase { get; set; } = "";

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
        public double BestSingleValMae { get; set; }

        /// <summary>
        /// Same model identified by val MAE, scored on the test set. Lets the report
        /// answer "does the blend beat best-single on the SAME split?" without picking
        /// a different best per split (which would be cherry-picking). 0.0 on legacy
        /// metadata files predating the field.
        /// </summary>
        public double BestSingleTestMae { get; set; }

        /// <summary>Blend MAE on the held-out test set.</summary>
        public double BlendTestMae { get; set; }
        public double BlendTestRmse { get; set; }
        public double BlendTestBias { get; set; }
    }

    public static string BuildVersionDir(string modelsRoot, string target, DateTime nowUtc, string? suffix = null)
    {
        var ts = nowUtc.ToString("yyyy-MM-dd_HHmmss");
        var name = string.IsNullOrEmpty(suffix) ? $"v{ts}" : $"v{ts}_{suffix}";
        return Path.Combine(modelsRoot, target, name).Replace('\\', '/');
    }

    /// <summary>
    /// Station-scoped version dir: <c>{modelsRoot}/{target}/{station}/v{ts}/</c>.
    /// Used by precipitation so per-station blenders live in parallel trees rather
    /// than sharing a flat folder with station-suffixed version names.
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

    /// <summary>
    /// Write/update MANIFEST.json under data/models/{target}/. Atomic:
    /// serialize to temp file, then move over the existing manifest.
    /// Sets Current and resets Active = [Current] — single-active legacy semantics.
    /// Use <see cref="AppendVersion"/> + <see cref="SetActive"/> for champion/challenger
    /// flows where multiple versions should stay live concurrently.
    /// </summary>
    public static void UpdateManifest(string modelsRoot, string target, string versionDirName)
    {
        MutateManifest(modelsRoot, target, m =>
        {
            m.Current = versionDirName;
            if (!m.Versions.Contains(versionDirName))
                m.Versions.Add(versionDirName);
            m.Active = new List<string> { versionDirName };
        });
    }

    /// <summary>
    /// Append a version to the history list without touching Current or Active.
    /// Used by champion/challenger trainers that want to register a new artefact
    /// without making it the default pick.
    /// </summary>
    public static void AppendVersion(string modelsRoot, string target, string versionDirName)
    {
        MutateManifest(modelsRoot, target, m =>
        {
            if (!m.Versions.Contains(versionDirName))
                m.Versions.Add(versionDirName);
        });
    }

    /// <summary>
    /// Replace the Active list explicitly. Predict + verify iterate this list when
    /// no specific version is requested. Caller is responsible for ensuring every
    /// listed version exists under <c>{modelsRoot}/{target}/</c>.
    /// </summary>
    public static void SetActive(string modelsRoot, string target, IEnumerable<string> activeVersions)
    {
        var list = activeVersions.Distinct().ToList();
        MutateManifest(modelsRoot, target, m => m.Active = list);
    }

    /// <summary>
    /// Versions that should produce predictions/verify rows. Falls back to [Current]
    /// when Active is empty (legacy manifests). Returns empty if neither is populated.
    /// </summary>
    public static IReadOnlyList<string> ResolveActive(string modelsRoot, string target)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        if (manifest.Active.Count > 0) return manifest.Active;
        if (!string.IsNullOrWhiteSpace(manifest.Current)) return new[] { manifest.Current };
        return Array.Empty<string>();
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
            // File.Move(overwrite: true) is a single atomic rename on both Windows
            // (MoveFileEx with MOVEFILE_REPLACE_EXISTING) and POSIX (rename(2)).
            // Replaces the prior Delete-then-Move pattern that left a microsecond
            // window where the manifest didn't exist on disk.
            File.Move(tmp, manifestPath, overwrite: true);
        }
    }

    // Lock-acquire backoff: polled retry up to ~5s. Manifest writes are tiny so
    // contention should resolve in a handful of milliseconds; the upper bound is
    // there only so a wedged lock-holder fails loudly instead of hanging forever.
    private const int LockAcquireMaxAttempts = 50;
    private const int LockAcquireBackoffMs = 100;

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

    /// <summary>Resolve "current" / explicit version string → absolute directory path.</summary>
    public static string ResolveVersionDir(string modelsRoot, string target, string versionOrCurrent)
    {
        var dir = Path.Combine(modelsRoot, target);
        if (!string.Equals(versionOrCurrent, "current", StringComparison.OrdinalIgnoreCase))
            return Path.Combine(dir, versionOrCurrent);

        var manifest = ReadJson<Manifest>(Path.Combine(dir, ManifestFileName))
            ?? throw new InvalidOperationException($"No manifest at {dir} — train a model first.");
        if (string.IsNullOrWhiteSpace(manifest.Current))
            throw new InvalidOperationException("Manifest has no current pointer.");
        return Path.Combine(dir, manifest.Current);
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
        entry.Current = versionDirName;
        if (!entry.Versions.Contains(versionDirName))
            entry.Versions.Add(versionDirName);
        // Single-active legacy semantics (mirrors the flat-layout UpdateManifest): a
        // plain UpdateStationManifest resets Active to [Current]. For Phase 3c
        // champion/challenger, call SetStationActive afterwards with the full list.
        entry.Active = new List<string> { versionDirName };

        var tmp = manifestPath + ".tmp";
        WriteJson(tmp, manifest);
        if (File.Exists(manifestPath)) File.Delete(manifestPath);
        File.Move(tmp, manifestPath);
    }

    /// <summary>
    /// Append a station version to its history list without touching Current or Active.
    /// Mirrors <see cref="AppendVersion"/> for per-station manifests.
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

    /// <summary>
    /// Versions that should produce predictions/verify rows for this station. Falls
    /// back to [Current] when Active is empty (legacy per-station entries written
    /// before Phase 3c). Returns empty if neither is populated.
    /// </summary>
    public static IReadOnlyList<string> ResolveStationActive(string modelsRoot, string target, string station)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        if (!manifest.Stations.TryGetValue(station, out var entry)) return Array.Empty<string>();
        if (entry.Active.Count > 0) return entry.Active;
        if (!string.IsNullOrWhiteSpace(entry.Current)) return new[] { entry.Current };
        return Array.Empty<string>();
    }

    /// <summary>
    /// Resolve a per-station version dir: <c>"current"</c> reads the station's
    /// current pointer from MANIFEST.json; any other string is treated as an
    /// explicit <c>v…</c> folder name. Returned path always points inside the
    /// <c>{target}/{station}/</c> subtree.
    /// </summary>
    public static string ResolveStationVersionDir(string modelsRoot, string target, string station, string versionOrCurrent)
    {
        var stationDir = Path.Combine(modelsRoot, target, station);
        if (!string.Equals(versionOrCurrent, "current", StringComparison.OrdinalIgnoreCase))
            return Path.Combine(stationDir, versionOrCurrent);

        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName))
            ?? throw new InvalidOperationException($"No manifest for target '{target}' — train a model first.");
        if (!manifest.Stations.TryGetValue(station, out var entry) || string.IsNullOrWhiteSpace(entry.Current))
            throw new InvalidOperationException($"Manifest has no current pointer for station '{station}'.");
        return Path.Combine(stationDir, entry.Current);
    }

    /// <summary>Stations currently recorded in the manifest for this target. Empty when the manifest is absent or flat-layout.</summary>
    public static IReadOnlyList<string> ListStations(string modelsRoot, string target)
    {
        var manifest = ReadJson<Manifest>(Path.Combine(modelsRoot, target, ManifestFileName));
        if (manifest is null) return Array.Empty<string>();
        return manifest.Stations.Keys.OrderBy(s => s, StringComparer.Ordinal).ToArray();
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    private static void WriteJson<T>(string path, T value)
        => File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOpts));

    private static T? ReadJson<T>(string path)
        => File.Exists(path) ? JsonSerializer.Deserialize<T>(File.ReadAllText(path)) : default;
}
