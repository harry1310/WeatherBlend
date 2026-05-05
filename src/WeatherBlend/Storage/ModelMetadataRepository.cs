using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;

namespace WeatherBlend.Storage;

/// <summary>
/// Single source of truth for reads from <c>data/models/</c>: champion lookups
/// (<c>MANIFEST.json#Current</c>) and per-version <c>training_metadata.json</c>
/// loads. Lifts the four near-identical helpers that previously lived as
/// private methods on <c>RenderSiteCommand</c> + the verify commands:
///   <c>LoadCurrentTemperatureVersion</c>, <c>LoadCurrentPrecipByStation</c>,
///   <c>LoadPhaseByVersion</c>, <c>LoadMetadataForVersions</c>,
///   <c>LoadMetadataForKeys</c>.
///
/// Path layout it understands (mirrors <c>ModelArtifact</c>):
///   data/models/{target}/MANIFEST.json                   — flat-target champion + active list
///   data/models/{target}/{version}/training_metadata.json — temperature + element_*
///   data/models/{target}/{station}/{version}/...         — precipitation
///   data/models/{target}/{station}/window_{N}h/{version}/... — dry_window
///
/// Targets are passed as bare strings (<c>"temperature"</c>, <c>"precipitation"</c>,
/// <c>"dry_window"</c>, <c>"element_wind"</c>, etc.) — matches what
/// <see cref="ModelArtifact"/> expects, no enum needed.
///
/// Read-only — never writes. <c>data/models</c> is always present on a
/// rendered/verified workflow run because the predict step requires it; the
/// repo treats genuinely-missing directories as "no champion / no metadata"
/// with a logged warning, mirroring the pre-repo behaviour for the
/// fresh-checkout case.
/// </summary>
public sealed class ModelMetadataRepository
{
    private readonly ILogger<ModelMetadataRepository> _log;
    private readonly string _modelsRoot;

    public ModelMetadataRepository(ILogger<ModelMetadataRepository> log, AppConfig cfg)
    {
        _log = log;
        _modelsRoot = cfg.Storage.ModelsPath;
    }

    // ------------------------------------------------------------------
    // Champion (manifest) lookups
    // ------------------------------------------------------------------

    /// <summary>
    /// Read the raw <see cref="ModelArtifact.Manifest"/> for a target, or
    /// null when the file is missing/malformed. Use this when you need more
    /// than the champion pointer — the dry-window verify, for example,
    /// iterates every <c>Stations</c> entry. For the common case of "give
    /// me the champion", prefer <see cref="GetChampion"/> /
    /// <see cref="GetChampionsByStation"/>.
    /// </summary>
    public ModelArtifact.Manifest? TryGetManifest(string target) => TryReadManifest(target);

    /// <summary>
    /// Champion version for a flat-target manifest (<c>temperature</c>,
    /// <c>element_*</c>, etc.) — i.e. <c>MANIFEST.json#Current</c>. Returns
    /// empty string when the manifest is missing or has no champion; callers
    /// degrade to "no filter" rather than crashing.
    /// </summary>
    public string GetChampion(string target)
    {
        var manifest = TryReadManifest(target);
        return manifest?.Current ?? "";
    }

    /// <summary>
    /// Per-lead champion overrides for a flat-target manifest
    /// (<see cref="ModelArtifact.Manifest.ChampionByLead"/>). Maps each
    /// pinned lead to its champion-version. Empty dict on legacy manifests
    /// or when no per-lead pin is set; callers fall back to
    /// <see cref="GetChampion"/> for unpinned leads.
    /// </summary>
    public IReadOnlyDictionary<int, string> GetChampionByLead(string target)
    {
        var manifest = TryReadManifest(target);
        if (manifest?.ChampionByLead is null) return new Dictionary<int, string>();
        return manifest.ChampionByLead
            .Where(kv => !string.IsNullOrEmpty(kv.Value))
            .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    /// <summary>
    /// Per-station champions for the per-station manifest layout
    /// (<c>precipitation</c>, dry-window-style targets if they ever switch
    /// to a station axis). Returns an ordinal-keyed dict of
    /// (station-slug → champion-version), filtered to entries whose
    /// <c>Current</c> is non-empty. Empty dict on missing manifest.
    /// </summary>
    public IReadOnlyDictionary<string, string> GetChampionsByStation(string target)
    {
        var manifest = TryReadManifest(target);
        if (manifest?.Stations is null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        return manifest.Stations
            .Where(kv => !string.IsNullOrEmpty(kv.Value.Current))
            .ToDictionary(kv => kv.Key, kv => kv.Value.Current, StringComparer.Ordinal);
    }

    // ------------------------------------------------------------------
    // training_metadata.json loaders
    // ------------------------------------------------------------------

    /// <summary>
    /// Bulk-load training metadata for flat-target versions (temperature +
    /// element_*). Versions that don't have a metadata file on disk are
    /// silently skipped with a debug log — same behaviour as the old
    /// <c>LoadMetadataForVersions</c> helper.
    /// </summary>
    public IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata>
        GetTrainingMetadataForVersions(string target, IEnumerable<string> versions)
    {
        var result = new Dictionary<string, ModelArtifact.TrainingMetadata>(StringComparer.Ordinal);
        foreach (var v in versions.Distinct(StringComparer.Ordinal))
        {
            var dir = Path.Combine(_modelsRoot, target, v);
            var meta = TryLoadFromDir(dir, $"{target}/{v}");
            if (meta is not null) result[v] = meta;
        }
        return result;
    }

    /// <summary>
    /// Bulk-load training metadata for the per-station precipitation layout.
    /// Same skip-on-missing behaviour as the flat-target variant.
    /// </summary>
    public IReadOnlyDictionary<(string Station, string Version), ModelArtifact.TrainingMetadata>
        GetTrainingMetadataForPrecipitationKeys(IEnumerable<(string Station, string Version)> keys)
    {
        var result = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>();
        foreach (var (station, version) in keys.Distinct())
        {
            var dir = Path.Combine(_modelsRoot, "precipitation", station, version);
            var meta = TryLoadFromDir(dir, $"precipitation/{station}/{version}");
            if (meta is not null) result[(station, version)] = meta;
        }
        return result;
    }

    /// <summary>
    /// Targeted single-row metadata lookup for the dry-window per-(station,
    /// window) layout. Returns null on missing dir / missing file / read
    /// error (warning logged). Useful inside the dry-window verify's
    /// per-row loop where the bulk-load pattern doesn't fit (each row
    /// needs both the full metadata blob and a per-lead PerLead lookup).
    /// </summary>
    public ModelArtifact.TrainingMetadata? TryGetDryWindowTrainingMetadata(
        string station, int windowHours, string version)
    {
        var dir = Path.Combine(_modelsRoot, "dry_window", station, $"window_{windowHours}h", version);
        return TryLoadFromDir(dir, $"dry_window/{station}/{windowHours}h/{version}");
    }

    // ------------------------------------------------------------------
    // Phase-by-version map (the per-card / per-line grouping key)
    // ------------------------------------------------------------------

    /// <summary>
    /// Build a flat <c>{version → phase}</c> map covering every (version) /
    /// (station, version) / (station, window, version) tuple the caller has
    /// predictions for. Used by the renderer's per-phase grouping (rolling
    /// charts, verify-history match) so the 2026-05-02 retrain doesn't
    /// fragment lines onto two version stubs.
    ///
    /// Same version string can recur across stations or windows
    /// (<c>v..._phase3c</c> at all three precip stations) — phase is the same
    /// for all, so the second write wins identically and the resulting map is
    /// safe as a flat lookup.
    /// </summary>
    public IReadOnlyDictionary<string, string> GetPhaseByVersion(
        IEnumerable<string> temperatureVersions,
        IEnumerable<(string Station, string Version)> precipitationKeys,
        IEnumerable<(string Station, int WindowHours, string Version)> dryWindowKeys)
    {
        var phases = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var v in temperatureVersions.Distinct(StringComparer.Ordinal))
        {
            var meta = TryLoadFromDir(Path.Combine(_modelsRoot, "temperature", v), $"temperature/{v}");
            if (meta is not null && !string.IsNullOrWhiteSpace(meta.Phase))
                phases[v] = meta.Phase;
        }

        foreach (var (station, version) in precipitationKeys.Distinct())
        {
            var meta = TryLoadFromDir(
                Path.Combine(_modelsRoot, "precipitation", station, version),
                $"precipitation/{station}/{version}");
            if (meta is not null && !string.IsNullOrWhiteSpace(meta.Phase))
                phases[version] = meta.Phase;
        }

        foreach (var (station, window, version) in dryWindowKeys.Distinct())
        {
            var meta = TryLoadFromDir(
                Path.Combine(_modelsRoot, "dry_window", station, $"window_{window}h", version),
                $"dry_window/{station}/{window}h/{version}");
            if (meta is not null && !string.IsNullOrWhiteSpace(meta.Phase))
                phases[version] = meta.Phase;
        }

        return phases;
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// <summary>
    /// Read <c>data/models/{target}/MANIFEST.json</c> safely. Missing file →
    /// null + warning log; malformed file → null + warning log. Used by both
    /// champion accessors so they share one degradation path.
    /// </summary>
    private ModelArtifact.Manifest? TryReadManifest(string target)
    {
        var manifestPath = Path.Combine(_modelsRoot, target, ModelArtifact.ManifestFileName);
        if (!File.Exists(manifestPath)) return null;
        try
        {
            var json = File.ReadAllText(manifestPath);
            return System.Text.Json.JsonSerializer.Deserialize<ModelArtifact.Manifest>(json);
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to read {Target} manifest: {Msg}", target, ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Read <c>{dir}/training_metadata.json</c> safely. Returns null on
    /// missing dir, missing file, or any deserialise error — caller treats
    /// null as "no metadata available, skip this row". <paramref name="loggingKey"/>
    /// is the dotted path used in warning logs so a future grep can find the
    /// composite that broke.
    /// </summary>
    private ModelArtifact.TrainingMetadata? TryLoadFromDir(string dir, string loggingKey)
    {
        try
        {
            if (!Directory.Exists(dir)) return null;
            var path = Path.Combine(dir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(path)) return null;
            return ModelArtifact.LoadTrainingMetadata(dir);
        }
        catch (Exception ex)
        {
            _log.LogWarning("Failed to load training metadata for {Key}: {Msg}", loggingKey, ex.Message);
            return null;
        }
    }
}
