using System.Text.Json;
using Microsoft.ML;

namespace WeatherBlend.Train;

/// <summary>
/// On-disk layout for a trained blender:
///   data/models/temperature/v{yyyy-MM-dd_HHmmss}/
///     model.zip                 Microsoft.ML ITransformer pipeline (LightGBM regressor embedded)
///     feature_schema.json       ordered feature names + dtypes (all float32)
///     training_metadata.json    training date, data range, hyperparameters, per-lead test MAE,
///                               deviations from the original brief (L2 objective, no monotone
///                               constraints — see 2a discussion)
///     MANIFEST.json             directory name + "current" pointer
///
/// CI workflows rclone data/ to R2 after training runs (phase 1 pattern).
/// </summary>
public static class ModelArtifact
{
    public const string ManifestFileName = "MANIFEST.json";
    public const string ModelFileName = "model.zip";
    public const string FeatureSchemaFileName = "feature_schema.json";
    public const string TrainingMetadataFileName = "training_metadata.json";
    public const string FeatureImportanceFileName = "feature_importance.json";

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
        public DateTime TrainedAtUtc { get; set; }
        public string DataRangeTrain { get; set; } = "";
        public string DataRangeVal { get; set; } = "";
        public string DataRangeTest { get; set; } = "";
        public int TrainRows { get; set; }
        public int ValRows { get; set; }
        public int TestRows { get; set; }
        public Dictionary<string, object> Hyperparameters { get; set; } = new();
        public Dictionary<string, double> TestMae { get; set; } = new();
        public List<string> DeviationsFromBrief { get; set; } = new();
    }

    public static string BuildVersionDir(string modelsRoot, string target, DateTime nowUtc)
    {
        var ts = nowUtc.ToString("yyyy-MM-dd_HHmmss");
        return Path.Combine(modelsRoot, target, $"v{ts}").Replace('\\', '/');
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
