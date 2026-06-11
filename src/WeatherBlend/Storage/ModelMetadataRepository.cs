using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;

namespace WeatherBlend.Storage;

/// <summary>
/// Single source of truth for reads from <c>data/models/</c>: champion lookups
/// (derived from <c>MANIFEST.json</c> + phases.yaml) and per-version
/// <c>training_metadata.json</c> loads. Lifts the near-identical helpers that
/// previously lived as private methods on <c>RenderSiteCommand</c> + the
/// verify commands: <c>LoadCurrentTemperatureVersion</c>,
/// <c>LoadCurrentPrecipByStation</c>, <c>LoadPhaseByVersion</c>,
/// <c>LoadMetadataForVersions</c>, <c>LoadMetadataForKeys</c>.
///
/// Path layout it understands (mirrors <c>ModelArtifact</c>) — every target
/// manifest is station/location-keyed, there is no flat layout:
///   data/models/{target}/MANIFEST.json                       — per-station Active lists
///   data/models/{target}/{station}/{version}/...             — temperature, precipitation, element_*
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
    /// me the champion", prefer <see cref="GetChampionsByStation"/>.
    /// </summary>
    public ModelArtifact.Manifest? TryGetManifest(string target) => TryReadManifest(target);

    /// <summary>
    /// Per-(station, lead) champion overrides for a target manifest.
    /// Returns a (station-slug, lead-hours) → champion-version dict.
    /// Empty when no station has any per-lead pin set. Callers fall back to
    /// <see cref="GetChampionsByStation"/> for any (station, lead) absent
    /// from this dict.
    /// </summary>
    public IReadOnlyDictionary<(string Station, int LeadHours), string>
        GetChampionByStationLead(string target)
    {
        var result = new Dictionary<(string, int), string>();
        var manifest = TryReadManifest(target);
        if (manifest?.Stations is null) return result;
        foreach (var (station, entry) in manifest.Stations)
        {
            foreach (var (lead, version) in entry.ChampionByLead)
            {
                if (!string.IsNullOrEmpty(version))
                    result[(station, lead)] = version;
            }
        }
        return result;
    }

    /// <summary>
    /// Per-station champion versions for the per-station manifest layout
    /// (<c>precipitation</c>, <c>temperature</c>). Returns an ordinal-keyed
    /// dict of (station-slug → champion-version), where the champion is
    /// derived by walking the target's phases.yaml lineup champion-first and
    /// taking the newest Active version of the first phase the station has a
    /// bundle of (<see cref="ModelArtifact.ResolveStationChampionVersion"/>).
    /// For precipitation that lineup is a fallback chain (3o → 3c → 3a), so
    /// the per-station champion phase varies by location. Stations with no
    /// Active version of any lineup phase are omitted. Empty dict on a
    /// missing manifest.
    /// </summary>
    public IReadOnlyDictionary<string, string> GetChampionsByStation(string target)
    {
        var manifest = TryReadManifest(target);
        if (manifest?.Stations is null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var station in manifest.Stations.Keys)
        {
            var champion = ModelArtifact.ResolveStationChampionVersion(_modelsRoot, target, station);
            if (!string.IsNullOrEmpty(champion))
                result[station] = champion;
        }
        return result;
    }

    // ------------------------------------------------------------------
    // training_metadata.json loaders
    // ------------------------------------------------------------------

    /// <summary>
    /// Bulk-load training metadata for the element_* targets, whose bundles
    /// live under <c>data/models/{target}/{location}/{version}/</c> — the
    /// element manifests are location-keyed like every other target.
    /// Versions that don't have a metadata file on disk are silently skipped
    /// with a debug log.
    /// </summary>
    public IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata>
        GetTrainingMetadataForVersions(string target, string station, IEnumerable<string> versions)
    {
        var result = new Dictionary<string, ModelArtifact.TrainingMetadata>(StringComparer.Ordinal);
        foreach (var v in versions.Distinct(StringComparer.Ordinal))
        {
            var dir = Path.Combine(_modelsRoot, target, station, v);
            var meta = TryLoadFromDir(dir, $"{target}/{station}/{v}");
            if (meta is not null) result[v] = meta;
        }
        return result;
    }

    /// <summary>
    /// Bulk-load training metadata for the per-station temperature layout
    /// (post-2026-05-14). Walks every (station, version) combination in the
    /// supplied list; missing dirs / missing files / parse errors are skipped
    /// with a warning. Result keyed by version string only (station is
    /// recoverable from the underlying metadata.LocationName + the manifest
    /// shape, and the renderer wants flat lookups by version).
    /// </summary>
    public IReadOnlyDictionary<string, ModelArtifact.TrainingMetadata>
        GetTemperatureTrainingMetadataForVersions(IEnumerable<(string Station, string Version)> keys)
    {
        var result = new Dictionary<string, ModelArtifact.TrainingMetadata>(StringComparer.Ordinal);
        foreach (var (station, version) in keys.Distinct())
        {
            var dir = Path.Combine(_modelsRoot, "temperature", station, version);
            var meta = TryLoadFromDir(dir, $"temperature/{station}/{version}");
            if (meta is not null) result[version] = meta;
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
    ///
    /// <paramref name="elementKeys"/> (2026-06-10) covers the element
    /// blenders — (location, version) pairs from the element predictions
    /// trees (today only the wind page groups by phase, so the renderer
    /// passes its wind rows). Their bundles live under
    /// <c>data/models/{ElementTarget.ModelDirName}/{location}/{version}/</c>;
    /// the dir list is derived from <see cref="ElementTargets.All"/>, not a
    /// hardcoded string list (a hardcoded phase list in render code silently
    /// dropped a phase on 2026-05-05). Before this, the wind page sniffed
    /// version-string suffixes itself (the deleted <c>DeriveWindPhase</c>).
    /// </summary>
    public IReadOnlyDictionary<string, string> GetPhaseByVersion(
        IEnumerable<(string Station, string Version)> temperatureKeys,
        IEnumerable<(string Station, string Version)> precipitationKeys,
        IEnumerable<(string Station, int WindowHours, string Version)> dryWindowKeys,
        IEnumerable<(string Station, string Version)> elementKeys)
    {
        var phases = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var (station, version) in temperatureKeys.Distinct())
        {
            // Post-2026-05-14 layout: temperature/{station}/{version}/...
            var meta = TryLoadFromDir(
                Path.Combine(_modelsRoot, "temperature", station, version),
                $"temperature/{station}/{version}");
            if (meta is not null && !string.IsNullOrWhiteSpace(meta.Phase))
                phases[version] = meta.Phase;
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

        foreach (var (station, version) in elementKeys.Distinct())
        {
            if (phases.ContainsKey(version)) continue;

            // Composed phase: wind_blend is minted live at predict time
            // under the fixed v_wind_blend_live version — it has NO bundle
            // (phases.yaml `retrain: none`), so the metadata scan below can
            // never resolve it. Map it explicitly from the constants the
            // minter defines.
            if (version.Contains("_wind_blend", StringComparison.Ordinal))
            {
                phases[version] = Commands.WindBlendPredictCommand.PhaseTag;
                continue;
            }

            // Which element tree a version belongs to isn't part of the key,
            // so probe every element model dir — version dir names are
            // timestamped and unique, so at most one probe hits.
            var resolved = false;
            foreach (var target in Train.Element.ElementTargets.All)
            {
                var meta = TryLoadFromDir(
                    Path.Combine(_modelsRoot, target.ModelDirName, station, version),
                    $"{target.ModelDirName}/{station}/{version}");
                if (meta is null || string.IsNullOrWhiteSpace(meta.Phase)) continue;
                // Pre-rename bundles stamp "lean-{cli-name}" (e.g. lean-wind,
                // lean-cloud-cover) — the same lineage as today's PhaseTag,
                // renamed ~2026-05-24. Normalise so their still-in-window
                // predictions stay on the champion line instead of silently
                // dropping as an unknown phase (the version-string sniffing
                // this scan replaced mapped them to the champion implicitly).
                // Same legacy tag ModelArtifact.PromoteStationVersion evicts
                // from Active lists.
                phases[version] = string.Equals(meta.Phase, "lean-" + target.CliName, StringComparison.Ordinal)
                    ? target.PhaseTag
                    : meta.Phase;
                resolved = true;
                break;
            }
            if (resolved) continue;

            // Conservative version-string fallback for element versions whose
            // bundle metadata is not on disk at render time. That genuinely
            // happens: the Python-owned wind bundles (wind_speed_lgb minted by
            // WeatherProbabilistic, wind_mvn under data/models/wind_direction
            // which is not an ElementTarget) sync to R2 on a different cadence
            // from predictions, so a fresh predict cycle can land rows before
            // its bundle is pullable. Without this the wind page would
            // silently fold those rows into the wrong phase line. The suffix
            // convention is what every trainer stamps (VersionSuffix =
            // "_{phase}"); bare versions (no suffix) are left UNRESOLVED so a
            // non-wind element version can't get mislabelled — the wind page
            // defaults its own unresolved versions to the champion "wind"
            // phase (pre-VersionSuffix bundles), see SitePages.Wind.PhaseOf.
            if (version.Contains("_wind_speed_lgb", StringComparison.Ordinal))
                phases[version] = "wind_speed_lgb";
            else if (version.Contains("_wind_mvn", StringComparison.Ordinal))
                phases[version] = "wind_mvn";
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
