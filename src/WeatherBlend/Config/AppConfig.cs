namespace WeatherBlend.Config;

public sealed class AppConfig
{
    /// <summary>
    /// Configured locations the system can target. YAML key: <c>locations:</c>
    /// (a list). Backfill + Collect iterate all entries; train/predict/site
    /// commands currently operate on the FIRST (primary) location only via
    /// the back-compat <see cref="Location"/> accessor — adding a second
    /// location to the YAML doesn't break those single-location code paths,
    /// it just makes data available under the new <c>location=&lt;n&gt;</c>
    /// partitions so a future per-location predict/site can pick it up.
    ///
    /// Default: empty list. The YAML binder will populate when the file is
    /// loaded; tests / unit-tests of LocationConfig itself don't depend on
    /// this list being non-empty.
    /// </summary>
    public List<LocationConfig> Locations { get; set; } = new();

    /// <summary>
    /// Back-compat accessor for the historical single-location field. Returns
    /// the FIRST configured location (the primary). 29+ call sites across
    /// Commands/, Storage/, Train/ continue to use <c>_cfg.Location.X</c>;
    /// keeping this as a computed property over <see cref="Locations"/>
    /// keeps them all working unchanged after the 2026-05-11 multi-location
    /// shift. New code that needs ALL locations (Backfill, Collect) iterates
    /// <c>Locations</c> directly.
    /// </summary>
    public LocationConfig Location
        => Locations.Count > 0 ? Locations[0] : _emptyLocation;

    private static readonly LocationConfig _emptyLocation = new();

    public List<ModelConfig> Models { get; set; } = new();
    public VariablesConfig Variables { get; set; } = new();
    public int ForecastDays { get; set; } = 7;
    public StorageConfig Storage { get; set; } = new();
    public HttpConfig Http { get; set; } = new();
    public MetOfficeConfig MetOffice { get; set; } = new();
    public BlendersConfig Blenders { get; set; } = new();
    public DryWindowConfig DryWindow { get; set; } = new();
    public RockSurfaceConfig RockSurface { get; set; } = new();
}

/// <summary>
/// Per-target-family knobs for the dry-window blender:
///   * <see cref="AllowedWindow"/> — local-time hour range (e.g. 9–18) within
///     which dry-block searches happen for both label construction (truth)
///     and per-NWP feature aggregation. DST is handled at scan time via the
///     IANA tz id.
///   * <see cref="CalibrationStations"/> — opt-in PAV (isotonic) calibration
///     applied at predict time. Only stations whose raw blender is
///     materially mis-calibrated benefit; on well-calibrated stations PAV
///     overfits the small (~125-row) validation set and makes things worse
///     (the 2026-04-29 dry-daytime experiment confirmed this empirically).
///     Stations not listed here ship raw probabilities.
/// </summary>
public sealed class DryWindowConfig
{
    public DaytimeWindowConfig AllowedWindow { get; set; } = new();

    /// <summary>
    /// Station names (matched case-insensitively against
    /// <c>Location.Rainfall.Stations[].Name</c>) for which the trainer should
    /// save a PAV calibrator alongside the LightGBM artefact. Predict applies
    /// the calibrator iff the file is present on disk, so toggling a station
    /// off here just skips saving — no separate predict-side flag needed.
    /// </summary>
    public List<string> CalibrationStations { get; set; } = new();

    /// <summary>
    /// Resolve to a runtime <see cref="WeatherBlend.Train.DryWindow.DaytimeWindow"/>
    /// usable by the label builder and feature builder. Throws on a malformed
    /// tz id, which would otherwise surface as a confusing per-row exception.
    /// </summary>
    public WeatherBlend.Train.DryWindow.DaytimeWindow BuildDaytimeWindow()
        => new(AllowedWindow.StartLocalHour, AllowedWindow.EndLocalHour, AllowedWindow.Tz);

    /// <summary>True iff <paramref name="stationName"/> appears in <see cref="CalibrationStations"/>.</summary>
    public bool ShouldCalibrate(string stationName)
        => CalibrationStations.Any(s => string.Equals(s, stationName, StringComparison.OrdinalIgnoreCase));
}

public sealed class DaytimeWindowConfig
{
    public int StartLocalHour { get; set; } = 9;
    public int EndLocalHour { get; set; } = 18;
    public string Tz { get; set; } = "Europe/London";
}

public sealed class VariablesConfig
{
    /// <summary>Variables pulled per-model from Open-Meteo (Live + Previous Runs).</summary>
    public List<string> Forecast { get; set; } = new();

    /// <summary>Variables pulled from the ERA5 archive endpoint as training truth.</summary>
    public List<string> Era5 { get; set; } = new();

    /// <summary>Wave variables pulled per wave model from the Marine API
    /// (live + offset-day + hindcast). Sennen sea-state Phase 0, 2026-06-11.</summary>
    public List<string> Marine { get; set; } = new();

    /// <summary>Marine variables only <c>best_match</c> serves (secondary
    /// swell, sea_level_height_msl, sea_surface_temperature) — "undefined"
    /// on per-model requests, so they ride on the best_match pull only.</summary>
    public List<string> MarineSite { get; set; } = new();
}

public sealed class LocationConfig
{
    public string Name { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public double Latitude { get; set; }
    public double Longitude { get; set; }
    public double ElevationMeters { get; set; }
    public MetarConfig Metar { get; set; } = new();
    public RainfallConfig Rainfall { get; set; } = new();

    /// <summary>
    /// Optional marine (sea-state) collection block — Sennen sea-state
    /// Phase 0, 2026-06-11. Present only on sea-cliff locations; null means
    /// no marine collection (collect + backfill gate on it). Holds the
    /// PINNED marine grid cell — a different coordinate from the site
    /// itself, because the marine grid resolves coastal land points to its
    /// nearest sea cell and we want that choice explicit + stable in config
    /// rather than implicit in API snapping.
    /// </summary>
    public MarineConfig? Marine { get; set; }

    /// <summary>
    /// Per-location nav tabs surfaced on the site (Phase D, 2026-05-20). Each
    /// entry maps to one button in the location's site-nav and one set of
    /// per-target pages under <c>/{Name}/...</c>. Recognised values:
    /// <c>overview</c>, <c>temperature</c>, <c>rain</c>, <c>dry_window</c>,
    /// <c>skill</c>, <c>models</c>. Skill + Models pages auto-filter their
    /// internal target sub-navs to the intersection with this list (so a
    /// location with <c>skill</c> but no <c>dry_window</c> renders a skill
    /// page with no dry-window sub-tab). Empty list = render nothing for the
    /// location; useful only for a config that's mid-onboarding.
    /// </summary>
    public List<string> Tabs { get; set; } = new();

    /// <summary>
    /// Overview-page render knobs (Phase D, 2026-05-20). Today: just the
    /// visible-hour window for the home tile grid. Bonehill uses 4–22 to
    /// drop sleep hours; Membury uses 0–24 because the user cares about
    /// rainfall around the clock.
    /// </summary>
    public OverviewConfig Overview { get; set; } = new();

    /// <summary>
    /// Optional per-location rock-surface override block (Sennen sea-cliff
    /// extension, 2026-06-12 — docs/SENNEN_ROCK_TEMP_PLAN.md S2). Any subset
    /// of the global <c>rockSurface:</c> knobs, layered over the global block
    /// at predict time via <see cref="RockSurfaceConfig.ResolveFor"/>; also
    /// carries the per-location <c>enabled</c> gate. Null = inherit the global
    /// (Bonehill-calibrated) values wholesale.
    /// </summary>
    public RockSurfaceOverrideConfig? RockSurface { get; set; }

    /// <summary>
    /// Optional on-site rainfall calibration (Membury village, 2026-06-10).
    /// Weights map this location's EA gauge 3f forecasts onto a manual
    /// gauge AT the place of interest; the rain pages render a "village"
    /// composite section when present. Keys are rainfall station NAMES
    /// (matching <see cref="RainfallConfig.Stations"/>) — slugified at
    /// descriptor-build time the same way RainStationSlugs are.
    /// </summary>
    public VillageRainConfig? VillageRain { get; set; }

    /// <summary>True iff <paramref name="tab"/> appears in <see cref="Tabs"/> (case-insensitive).</summary>
    public bool HasTab(string tab) =>
        Tabs.Any(t => string.Equals(t, tab, StringComparison.OrdinalIgnoreCase));
}

/// <summary>
/// On-site rainfall calibration weights — see <see cref="LocationConfig.VillageRain"/>.
/// Provenance: MemburyDailyTruth/membury_true_rainfall_study.py (NNLS through
/// origin, daily 09:00Z rain-day vs the manual village gauge).
/// </summary>
public sealed class VillageRainConfig
{
    /// <summary>Human-facing section label, e.g. "Membury village".</summary>
    public string Label { get; set; } = "";

    /// <summary>Date the weights were last fitted (yyyy-MM-dd) — surfaced
    /// in the section's provenance line.</summary>
    public string FittedOn { get; set; } = "";

    /// <summary>Station NAME → weight. Applied to each gauge's 3f
    /// per-hour quantities and summed.</summary>
    public Dictionary<string, double> Weights { get; set; } = new();
}

public sealed class OverviewConfig
{
    /// <summary>Lowest UTC hour-of-day rendered on the Overview tile grid (inclusive). Default 0 (full day).</summary>
    public int FirstVisibleHourUtc { get; set; } = 0;

    /// <summary>Exclusive upper bound — <c>24</c> renders through hour 23. Default 24 (full day).</summary>
    public int LastVisibleHourUtcExclusive { get; set; } = 24;
}

public sealed class MetarConfig
{
    public string Primary { get; set; } = "";
    public string Fallback { get; set; } = "";
}

public sealed class RainfallConfig
{
    public List<RainfallStationConfig> Stations { get; set; } = new();
}

public sealed class RainfallStationConfig
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
}

public sealed class ModelConfig
{
    public string Id { get; set; } = "";
    public string DisplayName { get; set; } = "";
}

/// <summary>
/// Marine collection config for one location — see <see cref="LocationConfig.Marine"/>.
/// <see cref="Models"/> lists the per-model wave sources (MFWAM, ECMWF WAM, …);
/// the <c>best_match</c> pull that carries the site extras (tide, SST,
/// secondary swell) is implicit — every marine location gets it.
/// </summary>
public sealed class MarineConfig
{
    /// <summary>Pinned marine grid cell latitude (NOT the site latitude).</summary>
    public double Latitude { get; set; }

    /// <summary>Pinned marine grid cell longitude.</summary>
    public double Longitude { get; set; }

    /// <summary>
    /// Optional separate coordinate for the era5_ocean TRUTH fetch. ERA5's
    /// wave grid is 0.5° — far coarser than the forecast models — and at
    /// Sennen the cell nearest the forecast point sits SE of Land's End,
    /// SHELTERED from the W/NW Atlantic swell a west-facing cliff actually
    /// receives (validated vs the Sevenstones buoy 2026-06-11: −0.31 m bias
    /// in the dominant SW-W/W-NW bins, vs ~0.00 for the cell west of the
    /// peninsula). Null = fall back to Latitude/Longitude.
    /// </summary>
    public double? TruthLatitude { get; set; }
    public double? TruthLongitude { get; set; }

    /// <summary>Wave models to collect (Open-Meteo Marine API model ids).</summary>
    public List<ModelConfig> Models { get; set; } = new();

    /// <summary>Wave buoys providing verification truth (realtime via the
    /// open Cefas WaveNet API each collect cycle; QC'd archive via EMODnet
    /// ERDDAP through <c>backfill --source buoys</c>).</summary>
    public List<BuoyConfig> Buoys { get; set; } = new();

    /// <summary>Overview-tile sea-state badge thresholds. Null (no
    /// <c>seaStateBadge:</c> block in config.yaml) = the location renders
    /// no sea-state badge. See <see cref="SeaStateBadgeConfig"/>.</summary>
    public SeaStateBadgeConfig? SeaStateBadge { get; set; }
}

/// <summary>
/// Thresholds for the overview-tile sea-state badge (amber = one trigger
/// fired, red = two-or-more) — see <c>Site/Badges.cs</c> for the trigger
/// definitions. Config-driven, NOT code: these are per-cell physical
/// values that need recalibration against real sessions, never recompiles.
/// </summary>
public sealed class SeaStateBadgeConfig
{
    /// <summary>HIGH TIDE trigger: fires when the wave-predictions
    /// pass-through TideHeightMsl (m vs mean sea level) is ≥ this.</summary>
    public double TideHighMsl { get; set; }

    /// <summary>BIG WAVES trigger: fires when √Hs(m) × swell period(s)
    /// — a run-up scaling proxy — is ≥ this.</summary>
    public double RunUpProxy { get; set; }

    /// <summary>ONSHORE WIND trigger minimum speed, mph.</summary>
    public double WindMinMph { get; set; }

    /// <summary>Onshore sector start ("coming from" degrees). The sector
    /// runs clockwise from→to and may wrap through 360.</summary>
    public double WindSectorFromDeg { get; set; }

    /// <summary>Onshore sector end ("coming from" degrees).</summary>
    public double WindSectorToDeg { get; set; }
}

/// <summary>One wave buoy — see <see cref="MarineConfig.Buoys"/>.</summary>
public sealed class BuoyConfig
{
    /// <summary>Storage slug — the <c>source=</c> partition under
    /// data/truth/waves and the Source column value (e.g. "sevenstones_62107").</summary>
    public string Slug { get; set; } = "";

    public string DisplayName { get; set; } = "";

    /// <summary>Cefas WaveNet station id — numeric for third-party stations
    /// ("4" = Sevenstones), alphanumeric for Cefas-owned ("SWSCILLYWN").</summary>
    public string WavenetId { get; set; } = "";

    /// <summary>WaveNet source discriminator: "EXT" (Met Office / CCO
    /// stations mirrored into WaveNet) or "INT" (Cefas-owned).</summary>
    public string WavenetSource { get; set; } = "EXT";

    /// <summary>WMO/Copernicus platform code on EMODnet ERDDAP (e.g. "6200107").</summary>
    public string PlatformCode { get; set; } = "";
}

public sealed class StorageConfig
{
    public string ForecastsPath { get; set; } = "data/forecasts";
    public string ObservationsPath { get; set; } = "data/truth/metar";
    public string Era5Path { get; set; } = "data/truth/era5";
    public string RainfallPath { get; set; } = "data/truth/rainfall";
    public string PredictionsPath { get; set; } = "data/predictions";
    public string ReportsPath { get; set; } = "data/reports";
    public string MetOfficeObsPath { get; set; } = "data/truth/met_office_obs";
    public string MarinePath { get; set; } = "data/marine";
    public string WavesPath { get; set; } = "data/truth/waves";

    /// <summary>
    /// Root of the trained-model artefact tree
    /// (<c>{ModelsPath}/{target}/{station?}/{window?}/{version}/...</c>).
    /// Promoted to config 2026-05-02 — was hard-coded as
    /// <c>"data/models"</c> in 18+ call sites; now lives alongside every
    /// other tree path so a future retarget (e.g. to a SAN mount) is a
    /// one-line config edit. <see cref="ModelMetadataRepository"/> is the
    /// canonical reader; train and predict commands also write under it.
    /// </summary>
    public string ModelsPath { get; set; } = "data/models";
}

public sealed class MetOfficeConfig
{
    public bool Enabled { get; set; } = true;
    public string SpotModelTag { get; set; } = "met_office_spot";
    public string SpotKeyEnvVar { get; set; } = "MET_OFFICE_SPOT_API_KEY";
    // Relative to the process cwd (typically the repo root when `dotnet run` is
    // invoked there). Keys live one level above the repo, outside version control.
    // Env var wins over file (see MetOfficeSecrets.TryLoad), and CI uses env vars
    // exclusively — so this relative path only matters for local dev convenience.
    public string SpotKeyFile { get; set; } = "../MetOfficeSpotKey.txt";
    public string ObsKeyEnvVar { get; set; } = "MET_OFFICE_OBS_API_KEY";
    public string ObsKeyFile { get; set; } = "../MetOfficeObsKey.txt";

    /// <summary>
    /// Geohash of the Land Observations location nearest our site. The DataHub
    /// Land Observations API is geohash-addressed (not station-id) — call
    /// <c>/nearest?lat=...&amp;lon=...</c> once, pin the result here, and reuse.
    /// Left null until a bootstrap run fills it in; leaving it null simply skips
    /// obs collection without blocking the rest of the cycle.
    /// </summary>
    public string? ObsGeohash { get; set; }

    /// <summary>Human-readable area the geohash resolves to (e.g. "Devon"). Informational.</summary>
    public string? ObsArea { get; set; }

    /// <summary>
    /// Optional supplemental observation geohashes to collect alongside the
    /// primary <see cref="ObsGeohash"/> on every cycle. Used to capture
    /// physically-distinct station observations (e.g. Dunkeswell hilltop
    /// at <c>gcj9q6</c>) as additional truth signals for cross-validation
    /// and bake-offs against the primary site truth.
    ///
    /// Each entry is pulled with the same DataHub Land Observations API
    /// and stored under <c>{MetOfficeObsPath}/location={Label}/geohash={Geohash}/</c>
    /// so consumers can read by either geohash or label.
    ///
    /// Empty list (default) means no supplemental pulls — behaviour
    /// identical to the pre-2026-05-27 single-geohash flow.
    /// </summary>
    public List<SupplementalObsGeohash> SupplementalObsGeohashes { get; set; } = new();
}

/// <summary>
/// One supplemental observation source: a DataHub Land Observations
/// geohash to collect alongside the primary
/// <see cref="MetOfficeConfig.ObsGeohash"/>. The <see cref="Label"/> is
/// written into the <c>LocationName</c> column on stored rows so
/// downstream queries can distinguish supplemental sources from the
/// primary site without inspecting the geohash.
/// </summary>
public sealed class SupplementalObsGeohash
{
    /// <summary>DataHub geohash (e.g. "gcj9q6" for Dunkeswell).</summary>
    public string Geohash { get; set; } = "";

    /// <summary>Human-readable area returned by /nearest (e.g. "Devon"). Informational.</summary>
    public string Area { get; set; } = "";

    /// <summary>Friendly label for storage + logs (e.g. "dunkeswell_aerodrome").
    /// Becomes the <c>LocationName</c> column value on stored rows and the
    /// <c>location=</c> hive partition under <c>MetOfficeObsPath</c>.</summary>
    public string Label { get; set; } = "";
}

public sealed class HttpConfig
{
    public string UserAgent { get; set; } = "WeatherBlend-PoC/0.1";
    public int TimeoutSeconds { get; set; } = 60;
    public int RetryCount { get; set; } = 3;

    /// <summary>
    /// Delay between consecutive Open-Meteo Previous Runs backfill chunks, in seconds.
    /// That endpoint uses an hourly token bucket — a 30 min burst at ~5 calls/min
    /// tipped the limit on 2026-04-25 and locked the API for the rest of the hour.
    /// 15s keeps us at ~4 calls/min, slow enough to let the bucket refill while a
    /// long backfill is in flight. ERA5 has its own (much smaller) load profile and
    /// uses a hardcoded 2s delay; tune this knob up if 429s reappear on previous-runs.
    /// </summary>
    public int PreviousRunsBackfillDelaySeconds { get; set; } = 15;
}
