using System.Text.Json;
using Microsoft.ML;

namespace WeatherBlend.Train;

/// <summary>
/// On-disk layout for a trained blender.
///
/// Phase 2a (one model, no lead bucketing):
///   data/models/temperature/v{yyyy-MM-dd_HHmmss}/
///     model.zip                       ITransformer pipeline (LightGBM embedded)
///     feature_schema.json
///     training_metadata.json
///     feature_importance.json
///
/// Phase 2b / "redo" (three models, one per lead):
///   data/models/temperature/v{yyyy-MM-dd_HHmmss}_phase2redo/
///     lead_24h.zip / lead_48h.zip / lead_72h.zip   per-lead pipelines
///     feature_schema.json                          shared (13 features, same for all leads)
///     training_metadata.json                       holds PerLead map keyed by "24"/"48"/"72"
///     feature_importance.json                      PerLead map of [{name,gain}] arrays
///
/// MANIFEST.json lives at data/models/temperature/ and points Current → version dir.
/// Resolver returns the version dir; callers pick the per-lead file by name.
/// CI workflows rclone data/ to R2 after training runs (phase 1 pattern).
/// </summary>
public static class ModelArtifact
{
    public const string ManifestFileName = "MANIFEST.json";
    public const string ModelFileName = "model.zip";                   // phase 2a
    public const string FeatureSchemaFileName = "feature_schema.json";
    public const string TrainingMetadataFileName = "training_metadata.json";
    public const string FeatureImportanceFileName = "feature_importance.json";

    /// <summary>Phase 2b per-lead artifact name, e.g. LeadModelFileName(24) → "lead_24h.zip".</summary>
    public static string LeadModelFileName(int leadHours) => $"lead_{leadHours}h.zip";

    public sealed class Manifest
    {
        public string Target { get; set; } = "";
        public string Current { get; set; } = "";
        public List<string> Versions { get; set; } = new();
    }

    public sealed class FeatureSchema
    {
        public List<string> FeatureNames { get; set; } = new();
        public string Dtype { get; set; } = "float32";
    }

    public sealed class TrainingMetadata
    {
        public string Version { get; set; } = "";
        public string Target { get; set; } = "";
        public string Phase { get; set; } = "";

        /// <summary>Free-form tag — e.g. "previous_runs_api" so the report can explain data source at a glance.</summary>
        public string DataSource { get; set; } = "";

        public DateTime TrainedAtUtc { get; set; }

        // Phase 2a (single-model) fields. Left populated for the 2a path; the 2b
        // path uses PerLead instead and leaves these empty.
        public string DataRangeTrain { get; set; } = "";
        public string DataRangeVal { get; set; } = "";
        public string DataRangeTest { get; set; } = "";
        public int TrainRows { get; set; }
        public int ValRows { get; set; }
        public int TestRows { get; set; }

        public Dictionary<string, object> Hyperparameters { get; set; } = new();
        public Dictionary<string, double> TestMae { get; set; } = new();
        public List<string> DeviationsFromBrief { get; set; } = new();

        /// <summary>Phase 2b: one entry per lead ("24","48","72"). Empty for phase 2a.</summary>
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

    public static void SaveModel(
        MLContext ml,
        ITransformer model,
        DataViewSchema inputSchema,
        string versionDir)
    {
        Directory.CreateDirectory(versionDir);
        var path = Path.Combine(versionDir, ModelFileName);
        ml.Model.Save(model, inputSchema, path);
    }

    public static ITransformer LoadModel(MLContext ml, string versionDir, out DataViewSchema schema)
    {
        var path = Path.Combine(versionDir, ModelFileName);
        return ml.Model.Load(path, out schema);
    }

    /// <summary>Phase 2b: save one pipeline for a specific lead bucket.</summary>
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

    /// <summary>Phase 2b: load one pipeline for a specific lead bucket.</summary>
    public static ITransformer LoadLeadModel(MLContext ml, string versionDir, int leadHours, out DataViewSchema schema)
    {
        var path = Path.Combine(versionDir, LeadModelFileName(leadHours));
        return ml.Model.Load(path, out schema);
    }

    public static void SaveFeatureSchema(string versionDir, IEnumerable<string> featureNames)
    {
        var obj = new FeatureSchema { FeatureNames = featureNames.ToList() };
        WriteJson(Path.Combine(versionDir, FeatureSchemaFileName), obj);
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

    public static IReadOnlyList<(string Name, double Gain)> LoadFeatureImportance(string versionDir)
    {
        var path = Path.Combine(versionDir, FeatureImportanceFileName);
        if (!File.Exists(path)) return Array.Empty<(string, double)>();
        var raw = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(File.ReadAllText(path))
                  ?? new List<Dictionary<string, JsonElement>>();
        return raw
            .Select(d => (
                Name: d.TryGetValue("name", out var n) ? n.GetString() ?? "" : "",
                Gain: d.TryGetValue("gain", out var g) ? g.GetDouble() : 0.0))
            .ToArray();
    }

    /// <summary>
    /// Phase 2b: persist one importance list per lead. File format is a dict
    /// keyed by lead string ("24","48","72") so one file covers the whole run.
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

        // File may be in phase-2a flat format OR phase-2b per-lead format. Sniff the first token.
        var text = File.ReadAllText(path);
        using var doc = JsonDocument.Parse(text);
        if (doc.RootElement.ValueKind == JsonValueKind.Array)
            return new Dictionary<int, IReadOnlyList<(string, double)>>();  // phase 2a file, not per-lead

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
    /// </summary>
    public static void UpdateManifest(string modelsRoot, string target, string versionDirName)
    {
        var dir = Path.Combine(modelsRoot, target);
        Directory.CreateDirectory(dir);
        var manifestPath = Path.Combine(dir, ManifestFileName);

        var manifest = File.Exists(manifestPath)
            ? ReadJson<Manifest>(manifestPath) ?? new Manifest()
            : new Manifest();

        manifest.Target = target;
        manifest.Current = versionDirName;
        if (!manifest.Versions.Contains(versionDirName))
            manifest.Versions.Add(versionDirName);

        var tmp = manifestPath + ".tmp";
        WriteJson(tmp, manifest);
        if (File.Exists(manifestPath)) File.Delete(manifestPath);
        File.Move(tmp, manifestPath);
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

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    private static void WriteJson<T>(string path, T value)
        => File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOpts));

    private static T? ReadJson<T>(string path)
        => File.Exists(path) ? JsonSerializer.Deserialize<T>(File.ReadAllText(path)) : default;
}
