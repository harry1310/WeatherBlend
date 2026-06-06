using System.Globalization;
using System.Text;
using WeatherBlend.Models;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Site;

/// <summary>
/// Pure HTML renderers for the static site. No I/O, no DuckDB — takes already-loaded
/// model objects and returns strings. Callers (see <c>RenderSiteCommand</c>) own the
/// query-and-write orchestration.
///
/// Split across partial files by page: index, forecasts (per lead), dry-window,
/// skill (merged eyeball + rolling), models, about. This file holds the shared input
/// contract, page-chrome (<see cref="WrapPage"/>, <see cref="Stylesheet"/>) and small helpers.
/// </summary>
public static partial class SitePages
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;

    // Site lead horizons come from <see cref="WeatherBlend.Train.Common.Leads"/>:
    //   - Temp + precip pages (home cards, per-lead forecasts, rolling-MAE,
    //     Models): <c>Leads.Full</c> = {24, 48, 72, 96, 120}.
    //   - Dry-window page: <c>Leads.Short</c> = {24, 48, 72} — Phase 3b/3d
    //     blenders were never trained at 96/120h.

    // -------------------------------------------------------------------------
    // NWP display table — single source of truth for per-NWP labels + colours
    // and the small "what NWPs participate in this target" lists. Without it,
    // every chart that draws a per-NWP line had to re-spell GFS=#ef5350,
    // ECMWF=#42a5f5, ... and they drifted the moment a new model landed
    // (back when AIFS was #ec407a it took three edits per palette change,
    // four for JMA — see NwpPalette docstring for the current values).
    //
    // Per-NWP getter is row-type-specific (TempPredictionRow.TempGfs vs
    // PrecipForecastPoint.PrecipGfs), so each call site composes its own
    // <see cref="NwpDisplaySpec{TRow}"/> list from the shared label+colour
    // pair plus a row-shaped accessor lambda.
    // -------------------------------------------------------------------------

    /// <summary>One NWP's display metadata + the row accessor that pulls its
    /// value out of a target-specific row type (a <c>TempPredictionRow</c>'s
    /// <c>TempGfs</c>, a <c>PrecipForecastPoint</c>'s <c>PrecipGfs</c>, etc.).</summary>
    public sealed record NwpDisplaySpec<TRow>(string Label, string Color, Func<TRow, double?> Get);

    /// <summary>Per-NWP colour constants. Hue-separated so seven (or eight) NWP
    /// lines on a single chart stay legible. UKMO is indigo rather than purple
    /// to leave purple free for the brand-coloured Blend line. The five
    /// canonical models (GFS / ECMWF / ICON / MF / UKMO) keep Material 400-level
    /// colours; the three later additions are pushed to non-overlapping hues:
    /// GEM deep cyan (was teal — too close to ECMWF blue), AIFS pale pink
    /// (was hot pink — too close to GFS red), JMA brown (was amber — too close
    /// to MF orange). Updated 2026-05-04 after user-flagged clashes.</summary>
    internal static class NwpPalette
    {
        public const string Gfs   = "#ef5350";
        public const string Ecmwf = "#42a5f5";
        public const string Icon  = "#66bb6a";
        public const string Mf    = "#ffa726";
        public const string Ukmo  = "#5c6bc0";
        public const string Gem   = "#00838f";   // was #26a69a (teal-too-close-to-ECMWF) → cyan 800
        public const string Aifs  = "#f48fb1";   // was #ec407a (pink-too-close-to-GFS) → pink 200
        public const string Jma   = "#8d6e63";   // was #ffc107 (amber-too-close-to-MF) → brown 400

        /// <summary>Brand colour reserved for the blend's own line on per-NWP
        /// overlay charts. Not for any NWP.</summary>
        public const string Blend = "#7c4dff";

        /// <summary>Lighter shade of <see cref="Blend"/> for the challenger
        /// phase's line on champion+challenger overlay charts. Same hue family
        /// so the eye reads them as paired ("same prediction, two methods").
        /// Was a dashed-style variant of <see cref="Blend"/> until 2026-05-04;
        /// dashed lines were noisier than a saturation difference at the chart
        /// densities we render. Material deep-purple 200.</summary>
        public const string BlendChallenger = "#b39ddb";

        /// <summary>Vivid magenta for the exact-runtime challenger families
        /// (Phase 2d temperature, Phase 3d precipitation). Distinct hue from
        /// the deep-purple <see cref="Blend"/>/<see cref="BlendChallenger"/>
        /// pair because exact-runtime is a structurally different blender
        /// approach (per-cycle exact init time, not offset_day average) and
        /// shouldn't visually read as just "another challenger in the
        /// purple family". Material pink-700 — saturated enough to pop on
        /// both light and dark backgrounds, distinct from AIFS soft pink
        /// <see cref="Aifs"/> and GFS red <see cref="Gfs"/>.</summary>
        public const string BlendExactChallenger = "#c2185b";

        /// <summary>Met Office DataHub Spot product line colour on the
        /// skill charts. Was navy <c>#262261</c> originally — readable on
        /// paper but too close to the blend's purple family on dark-mode
        /// monitors, where the line merged into the +72h blend. Material
        /// green 800 <c>#2e7d32</c> is meaningfully separated from ICON's
        /// lighter <c>#66bb6a</c> green (rain-skill per-NWP panel), the
        /// red ERA5 + orange METAR truth lines, and every blend purple,
        /// so the eye reads MO Spot as a distinct comparison line on
        /// every chart it appears on.</summary>
        public const string MetOfficeSpot = "#2e7d32";
    }

    /// <summary>NWPs that feed the temperature blender — the seven that
    /// emit <c>Temperature2m</c> at hourly resolution. JMA is precip-only,
    /// not in this list.</summary>
    internal static IReadOnlyList<NwpDisplaySpec<Models.TempPredictionRow>> NwpsForTemperature() =>
        new NwpDisplaySpec<Models.TempPredictionRow>[]
        {
            new("GFS",   NwpPalette.Gfs,   p => p.TempGfs),
            new("ECMWF", NwpPalette.Ecmwf, p => p.TempEcmwf),
            new("ICON",  NwpPalette.Icon,  p => p.TempIcon),
            new("MF",    NwpPalette.Mf,    p => p.TempMf),
            new("UKMO",  NwpPalette.Ukmo,  p => p.TempUkmo),
            new("GEM",   NwpPalette.Gem,   p => p.TempGem),
            new("AIFS",  NwpPalette.Aifs,  p => p.TempAifs),
        };

    /// <summary>NWPs that feed the precipitation blender — the seven temp
    /// inputs plus JMA which contributes precip-only. Same colour scheme as
    /// <see cref="NwpsForTemperature"/> so a reader who learnt "ECMWF is blue"
    /// up there reads it the same way down here.</summary>
    internal static IReadOnlyList<NwpDisplaySpec<PrecipForecastPoint>> NwpsForPrecipitation() =>
        new NwpDisplaySpec<PrecipForecastPoint>[]
        {
            new("GFS",   NwpPalette.Gfs,   p => p.PrecipGfs),
            new("ECMWF", NwpPalette.Ecmwf, p => p.PrecipEcmwf),
            new("ICON",  NwpPalette.Icon,  p => p.PrecipIcon),
            new("MF",    NwpPalette.Mf,    p => p.PrecipMf),
            new("UKMO",  NwpPalette.Ukmo,  p => p.PrecipUkmo),
            new("GEM",   NwpPalette.Gem,   p => p.PrecipGem),
            new("AIFS",  NwpPalette.Aifs,  p => p.PrecipAifs),
            new("JMA",   NwpPalette.Jma,   p => p.PrecipJma),
        };

    /// <summary>
    /// Per-location descriptor surfaced on the site so the Rain forecast +
    /// Rain skill pages can show a Bonehill/Membury tab without leaking
    /// AppConfig into the rendering layer. Populated from
    /// <c>AppConfig.Locations</c> by <c>RenderSiteCommand</c>; consumed by
    /// the location sub-nav helpers + per-station filters in
    /// <c>SitePages.Forecasts</c> and <c>SitePages.Skill</c>.
    /// </summary>
    /// <param name="Name">Config slug (e.g. <c>bonehill_rocks</c>) — also the URL slug for non-primary location pages.</param>
    /// <param name="DisplayName">Human-facing label for the tab pill (e.g. <c>Bonehill Rocks, Dartmoor</c>).</param>
    /// <param name="RainStationSlugs">EA-prefixed station slugs from this location's rainfall config (e.g. <c>ea_bellever_dartmoor</c>).</param>
    /// <param name="IsPrimary">First location in config. Phase D made the URL scheme symmetric per-slug, so this is now only used by the location-switcher on the per-loc home (to pre-select an entry on the picker).</param>
    /// <param name="Tabs">Site-nav buttons from <see cref="LocationConfig.Tabs"/>. Order matches config.yaml; values are <c>overview</c>, <c>temperature</c>, <c>rain</c>, <c>dry_window</c>, <c>skill</c>, <c>models</c>.</param>
    /// <param name="OverviewFirstVisibleHourUtc">Per-loc Overview tile-grid lower bound (inclusive UTC hour). Bonehill 4, Membury 0.</param>
    /// <param name="OverviewLastVisibleHourUtcExclusive">Per-loc Overview tile-grid upper bound (exclusive UTC hour). Bonehill 22, Membury 24.</param>
    public sealed record LocationDescriptor(
        string Name,
        string DisplayName,
        IReadOnlyList<string> RainStationSlugs,
        bool IsPrimary,
        IReadOnlyList<string> Tabs,
        int OverviewFirstVisibleHourUtc,
        int OverviewLastVisibleHourUtcExclusive)
    {
        /// <summary>True iff <paramref name="tab"/> is in <see cref="Tabs"/> (case-insensitive).</summary>
        public bool HasTab(string tab) =>
            Tabs.Any(t => string.Equals(t, tab, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// First-pick location for legacy "single-location render" sites. Returns
    /// the <see cref="LocationDescriptor.IsPrimary"/> entry if one exists,
    /// otherwise the first entry, or <c>null</c> for an empty list. Phase D
    /// will drop this concept (every location is rendered the same way under
    /// its own URL prefix); until then, the 11 sites that need "the primary"
    /// route through here so the lookup is consistent.
    /// </summary>
    internal static LocationDescriptor? PrimaryLocation(IReadOnlyList<LocationDescriptor> locations)
        => locations.Count == 0
            ? null
            : locations.FirstOrDefault(l => l.IsPrimary) ?? locations[0];

    /// <summary>
    /// Resolve a named location for per-loc pages. <paramref name="locationName"/>
    /// null/empty falls through to <see cref="PrimaryLocation"/>; a non-empty
    /// name does a case-insensitive lookup and returns null if no match. Used
    /// by the Rain forecast + Rain skill pages where the location is the URL
    /// segment.
    /// </summary>
    internal static LocationDescriptor? ResolveActiveLocation(
        IReadOnlyList<LocationDescriptor> locations, string? locationName)
    {
        if (string.IsNullOrEmpty(locationName)) return PrimaryLocation(locations);
        return locations.FirstOrDefault(l =>
            l.Name.Equals(locationName, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Render input for one page.
    ///
    /// <para><b>Single-location invariant (2026-05-21):</b> every prediction /
    /// forecast / truth / NWP-overlay collection on this record is scoped to
    /// exactly one location — the one named by <see cref="ActiveLocationName"/>
    /// / <see cref="RenderingFor"/>. <c>RenderSiteCommand</c> does ALL the
    /// per-location filtering once, in its per-location loop, before
    /// constructing this object. Page builders therefore MUST NOT filter by
    /// location — they iterate their data directly. A page that forgets to
    /// filter can't leak another location's data because there is nothing
    /// else in the object to leak.</para>
    ///
    /// <para>The one deliberate exception is <see cref="VerifyHistory"/> — it
    /// carries every location's rows because the Models page matches them to
    /// per-station cards (<c>StationMatchesCard</c>), a finer granularity than
    /// location. See that field's remarks.</para>
    ///
    /// <para>The top-level pages (picker / specs / about) pass a degenerate
    /// instance with empty collections + <c>RenderingFor == null</c>.</para>
    /// </summary>
    public sealed record SiteInputs
    {
        /// <summary>Config slug (e.g. <c>bonehill_rocks</c>) of the location
        /// this SiteInputs is rendered for. Empty only for the top-level
        /// picker / specs / about pages. Every collection on this record is
        /// already scoped to this location (see the type-level invariant) —
        /// it's an identity tag, not a filter key.</summary>
        public string ActiveLocationName { get; init; } = "";

        /// <summary>
        /// Phase D (2026-05-20): the <see cref="LocationDescriptor"/> the
        /// current page is being rendered FOR, or <c>null</c> for top-level
        /// pages (picker / specs / about). Used by <c>WrapPage</c> to pick
        /// per-loc vs top-level chrome and by per-page nav builders to scope
        /// the active-tab highlight. When non-null, <see cref="AssetPrefix"/>
        /// is typically <c>"../"</c> (page lives one subdir deep).
        /// </summary>
        public LocationDescriptor? RenderingFor { get; init; } = null;

        /// <summary>
        /// Path prefix prepended to top-level asset hrefs (<c>styles.css</c>,
        /// <c>chart.js</c>, <c>index.html</c>, <c>specs.html</c>,
        /// <c>about.html</c>). Empty for top-level pages; <c>"../"</c> for
        /// per-loc pages under <c>/{slug}/</c>.
        /// </summary>
        public string AssetPrefix { get; init; } = "";

        public required string LocationDisplay { get; init; }
        public required double Latitude { get; init; }
        public required double Longitude { get; init; }
        public required double ElevationMeters { get; init; }

        /// <summary>
        /// Configured locations, primary first. Empty list = no Rain
        /// location sub-nav rendered (legacy single-location behaviour
        /// preserved for tests / older drives that don't populate it).
        /// </summary>
        public IReadOnlyList<LocationDescriptor> Locations { get; init; }
            = Array.Empty<LocationDescriptor>();

        /// <summary>ICAO of the primary METAR station (for the chart legend). Empty → no METAR.</summary>
        public required string MetarStation { get; init; }

        public required DateTime GeneratedAtUtc { get; init; }

        /// <summary>Earliest timestamp rendered in charts. Truth series are clamped to this.</summary>
        public required DateTime WindowStartUtc { get; init; }

        /// <summary>All prediction rows in the reporting window (typically last 30d).</summary>
        public required IReadOnlyList<TempPredictionRow> Predictions { get; init; }

        /// <summary>ERA5 truth by ValidTime for the same window.</summary>
        public required IReadOnlyDictionary<DateTime, double> TruthByTime { get; init; }

        /// <summary>METAR observations (time-sorted) from the primary station over the window.</summary>
        public required IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> MetarByTime { get; init; }

        /// <summary>
        /// Per-(version, lead) conformal threshold τ for the active dry-window
        /// blenders. Surfaced in the dry-window table's conformal cell so the
        /// reader can see the actual numbers (P + τ) behind a "confident wet
        /// day" / "confident dry day" tag rather than the tag alone. One value
        /// per (version, lead) — 3b's LightGBM probability is the only phase
        /// with a fitted conformal τ today; MC phases have no val-slice
        /// predictor that can replay deterministically. Empty when conformal
        /// artefacts haven't been fit yet.
        /// </summary>
        public IReadOnlyDictionary<(string Version, int LeadHours), double> DryWindowConformalTau { get; init; }
            = new Dictionary<(string, int), double>();

        /// <summary>
        /// Per-(station, version, lead) conformal threshold τ for the active
        /// precipitation blenders. Surfaced on the precip forecasts page so
        /// the conformal chip reads as "Confident wet · τ=70% · P=85%"
        /// rather than just the chip — a reader can then judge whether the
        /// model is confident-because-narrow-band (low τ) or
        /// confident-because-extreme-prob (high τ + extreme P). Keyed on
        /// station because precip artefact paths are per-station, so version
        /// strings aren't unique across stations. Empty when no calibrators
        /// are fitted yet.
        /// </summary>
        public IReadOnlyDictionary<(string Station, string Version, int LeadHours), double> PrecipConformalTau { get; init; }
            = new Dictionary<(string, string, int), double>();

        /// <summary>
        /// Same daytime window the dry-window labeller uses (default 09–18
        /// Europe/London). Surfaced here so the verify table's "Observed"
        /// column scans the same hours as the trainer's truth labels — a
        /// 24h scan would happily report "yes there was a 6h dry block"
        /// for a day that rained continuously through the daytime but was
        /// dry overnight (real bug we hit 2026-05-04). Defaults to the
        /// full UTC day for tests / standalone renders that don't supply it.
        /// </summary>
        public DaytimeWindow DryWindowDaytime { get; init; } = DryWindowLabelBuilder.FullUtcDay;

        /// <summary>
        /// Per-hour low-cloud / mist signal aggregated across NWP forecasts,
        /// keyed by valid time. Scoped to <see cref="ActiveLocationName"/> by
        /// RenderSiteCommand (the 2026-05-21 single-location-SiteInputs
        /// refactor — was a location-keyed outer dict). Two independent inner
        /// counts: how many of the visibility-publishing NWPs are forecasting
        /// sub-1km vis (mist or fog), and how many of all NWPs are forecasting
        /// saturated air (T-Td &lt; 1.5°C, an Espy proxy for cloud base below
        /// the 393m tor). Either threshold met → home tile renders a warning
        /// badge with hover-tooltip naming the firing signal(s).
        /// </summary>
        public IReadOnlyDictionary<DateTime, LowCloudSignal> LowCloudByValid { get; init; }
            = new Dictionary<DateTime, LowCloudSignal>();

        /// <summary>Met Office DataHub Land Observations temperature
        /// (time-sorted) for the configured geohash. Closer to Bonehill
        /// than EGTE METAR (~22 km vs ~30 km) with smaller elevation gap
        /// (~250 m vs ~360 m), so it doubles up on the temp skill chart
        /// as an independent cross-check. Empty list when the obs tree
        /// hasn't landed yet (capture started 2026-04-24). Defaults to
        /// empty so test fixtures and legacy callers don't have to set it.</summary>
        public IReadOnlyList<(DateTime ObservedTimeUtc, double Temperature2m)> MetOfficeObsByTime { get; init; }
            = Array.Empty<(DateTime, double)>();

        /// <summary>Rolling MAE per (version, lead) for the verify chart.</summary>
        public required IReadOnlyList<RollingMaePoint> RollingMae { get; init; }

        /// <summary>Per-(station, version, lead, day) rolling Brier for the
        /// precipitation blender. Empty when the predict tree hasn't aged
        /// into rainfall truth yet — the rain skill page silently shows an
        /// "MAE points pending" empty state in that case, mirroring temp.</summary>
        public IReadOnlyList<RollingBrierPoint> RollingBrier { get; init; }
            = Array.Empty<RollingBrierPoint>();

        /// <summary>Phase 3a P(wet) predictions across all stations / versions in the window.</summary>
        public required IReadOnlyList<PrecipForecastPoint> PrecipPredictions { get; init; }

        /// <summary>Phase 3b P(dry-window) predictions across all (station, window) blenders.</summary>
        public required IReadOnlyList<DryWindowForecastPoint> DryWindowPredictions { get; init; }

        /// <summary>
        /// Phase 3f rainfall_amount mixed-distribution predictions across all
        /// rainfall stations active for 3f. Each row carries the full mixed
        /// distribution (Pi + μ_log + σ_log + derived quantile fan +
        /// exceedance probabilities) for one (valid_time, lead). The
        /// rainfall_amount card on the rain tab consumes this directly. Empty
        /// list when 3f isn't trained for this location yet (Bonehill today)
        /// or when the predictions tree hasn't been synced from R2.
        /// </summary>
        public IReadOnlyList<RainfallAmountPredictionRow> RainfallAmountPredictions { get; init; }
            = Array.Empty<RainfallAmountPredictionRow>();

        /// <summary>
        /// training_metadata.Phase per ModelVersion that appears in Predictions. Used by
        /// the forecast-vs-truth page to bucket lines into "2b lean" vs "2c rich" groups.
        /// Missing entries (older versions, missing metadata) are treated as an empty phase
        /// by callers — those versions end up in a "other" group if present.
        /// </summary>
        public IReadOnlyDictionary<string, string> PhaseByVersion { get; init; }
            = new Dictionary<string, string>();

        /// <summary>
        /// Hourly observed rainfall (mm/h) per EA rainfall station slug. Aggregated from
        /// the 15-minute EA gauge parquet tree with the same 4-of-4 rule used by
        /// PrecipVerify — partial hours dropped so a half-observed hour can't flip wet↔dry.
        /// Key is the EA station slug (e.g. <c>ea_bellever_dartmoor</c>), matching the
        /// prediction <c>TruthStation</c> column.
        /// </summary>
        public IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> RainfallTruth { get; init; }
            = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>();

        /// <summary>
        /// Currently-active EA station slugs from config (rainfall.stations →
        /// ea-prefixed slugs). The renderer filters per-station tabs +
        /// pages to this set so a station demoted from the rotation
        /// doesn't render just because its historical parquets are still
        /// on disk. Empty
        /// set falls back to "render everything seen in the predictions"
        /// for backward-compat with older drives.
        /// </summary>
        public IReadOnlyCollection<string> ActiveStationSlugs { get; init; }
            = new HashSet<string>(StringComparer.Ordinal);

        /// <summary>
        /// Temperature champion version — the newest Active version of the
        /// phases.yaml champion phase (resolved, not a stored pointer). When
        /// set, the home page filters its forecast cards to this version so
        /// results are deterministic even while a challenger is also active.
        /// Empty string → no filter (fall back to "latest by PredictionMadeAt"
        /// across all versions).
        /// </summary>
        public string CurrentVersion { get; init; } = "";

        /// <summary>
        /// Per-lead champion override (Phase 2d, 2026-05-05). Maps lead-hours
        /// → version pinned as champion AT THAT LEAD ONLY. Lookups for any
        /// lead not present here fall back to <see cref="CurrentVersion"/>.
        /// Empty dict on legacy renders so behaviour is unchanged from before
        /// per-lead championship landed.
        /// </summary>
        public IReadOnlyDictionary<int, string> ChampionByLead { get; init; }
            = new Dictionary<int, string>();

        /// <summary>
        /// Per-station precipitation champion version, keyed by EA station
        /// slug (e.g. <c>ea_bellever_dartmoor</c>) — the newest Active version
        /// of the phases.yaml champion phase (3a), resolved per station. The
        /// home page pulls the P(wet) chip from this version only, so a
        /// challenger that happens to write the same (lead, valid_time) never
        /// leaks onto the headline card.
        /// </summary>
        public IReadOnlyDictionary<string, string> PrecipCurrentByStation { get; init; }
            = new Dictionary<string, string>();

        /// <summary>
        /// Per-(station, lead) precipitation champion override (Phase 3d,
        /// 2026-05-05). Maps (station-slug, lead-hours) → version pinned as
        /// champion AT THAT (STATION, LEAD) ONLY. Lookups for any (station,
        /// lead) not present here fall back to <see cref="PrecipCurrentByStation"/>.
        /// Mirrors <see cref="ChampionByLead"/> on the temperature side so 3d
        /// (exact-runtime precip) can take over as champion at lead 12 per
        /// station while 3a owns 24+. Empty dict when no per-lead pin is set.
        /// </summary>
        public IReadOnlyDictionary<(string Station, int LeadHours), string> PrecipChampionByStationLead { get; init; }
            = new Dictionary<(string, int), string>();

        /// <summary>
        /// Per-model held-out test scores (Blend vs ERA5 / EA rainfall) for every active
        /// blender, grouped by composite. One entry per (target, optional station, optional
        /// window) × version. Drives the Models page; empty when no training metadata is
        /// on disk (first-render, missing rclone sync, etc.).
        /// </summary>
        public IReadOnlyList<ModelSummary> ModelSummaries { get; init; }
            = Array.Empty<ModelSummary>();

        /// <summary>
        /// Auto-generated rows for the Models / Spec page. One row per
        /// (composite, version, lead) triple loaded from each active version's
        /// <c>feature_schema.json</c>. The renderer dedupes per
        /// (composite, phase, lead) keeping the freshest <see cref="FeatureSpecRow.TrainedAtUtc"/>
        /// so the table reads as "what's actually deployed at each lead today".
        /// Empty list when no schemas are on disk (first-render); page degrades
        /// to an empty-state message rather than crashing.
        /// </summary>
        public IReadOnlyList<FeatureSpecRow> FeatureSpecRows { get; init; }
            = Array.Empty<FeatureSpecRow>();

        /// <summary>
        /// Joined "feels-like" predictions per (lead, valid_time). Each row pairs the lean
        /// temperature blender output with the four element blenders (humidity / wind /
        /// shortwave radiation / cloud cover) at the same anchor and runs both Bröde 2012
        /// (UTCI) and Steadman 1994 (apparent temperature). Empty list when the feels-like
        /// prediction tree hasn't been synced yet — the home card falls back silently (no
        /// chip rendered).
        /// </summary>
        public IReadOnlyList<FeelsLikeForecastPoint> FeelsLikePredictions { get; init; }
            = Array.Empty<FeelsLikeForecastPoint>();

        /// <summary>Rock surface temperature + condensation hours (Phase P1).
        /// Empty when the rock_surface tree hasn't been synced — the overview
        /// chip + temp-tab chart fall back silently to nothing.</summary>
        public IReadOnlyList<RockSurfaceForecastPoint> RockSurfacePredictions { get; init; }
            = Array.Empty<RockSurfaceForecastPoint>();

        /// <summary>
        /// Wind-gust blender predictions per valid_time (m/s). Surfaced inline
        /// on the UTCI pop-out next to the 10 m wind row when gust is materially
        /// above wind. Empty when the wind-gust prediction tree hasn't been
        /// synced yet — the row falls back to wind-only rendering.
        /// </summary>
        public IReadOnlyDictionary<DateTime, double> WindGustByValidMs { get; init; }
            = new Dictionary<DateTime, double>();

        /// <summary>
        /// Per-(station, window, lead, target_date) start-hour curve points —
        /// one row per candidate start hour. Latest <c>PredictedAtUtc</c> per
        /// (Station, WindowHours, LeadHours, TargetDateUtc, StartHourUtc) wins
        /// at render time. Empty when the start-hour predict tree hasn't been
        /// synced yet — the dry-window page degrades silently to "no curve".
        /// </summary>
        public IReadOnlyList<StartHourForecastPoint> StartHourPredictions { get; init; }
            = Array.Empty<StartHourForecastPoint>();

        /// <summary>
        /// Met Office DataHub Spot forecasts for the configured location. Pre-
        /// filtered to "latest RunTime per ValidTime" by the renderer query so
        /// the chart code just plots them; future-target rows past the
        /// rendering window are also dropped on read.
        /// </summary>
        public IReadOnlyList<MetOfficeSpotForecastPoint> MetOfficeSpotForecasts { get; init; }
            = Array.Empty<MetOfficeSpotForecastPoint>();

        /// <summary>
        /// Per-NWP precipitation probability points (Open-Meteo's
        /// <c>precipitation_probability</c> field). Pre-deduped to freshest
        /// RunTime per (Model, ValidTime). Empty when the forecasts tree
        /// hasn't been synced yet — the rain-skill panel degrades silently
        /// to "no per-NWP data" rather than rendering empty.
        /// </summary>
        public IReadOnlyList<NwpPrecipProbForecastPoint> NwpPrecipProbabilities { get; init; }
            = Array.Empty<NwpPrecipProbForecastPoint>();

        /// <summary>Per-NWP precipitation rate (mm/h) at hourly granularity,
        /// pulled from the raw forecast tree (same path as NwpPrecipProbabilities).
        /// Used at the lead-12 page where the prediction-row-based NWP overlay
        /// is sparse (3d only emits at {0,6,12,18}Z); at lead 24+ the
        /// prediction parquet itself carries hourly per-NWP rate so this is
        /// only consulted on the 12h tab.</summary>
        public IReadOnlyList<NwpPrecipRateForecastPoint> NwpPrecipRates { get; init; }
            = Array.Empty<NwpPrecipRateForecastPoint>();

        /// <summary>Per-NWP 2m temperature at hourly granularity, pulled from
        /// the raw forecast tree. Same purpose as NwpPrecipRates: backfills
        /// the temperature lead-12 chart's per-NWP overlay where 2b/2c don't
        /// emit and 2d only emits at {0,6,12,18}Z.</summary>
        public IReadOnlyList<NwpTemperatureForecastPoint> NwpTemperatures { get; init; }
            = Array.Empty<NwpTemperatureForecastPoint>();

        /// <summary>Per-phase wind speed predictions for this location at every
        /// trained lead (24 / 48 / 72 h). One row per (Version, ValidTime); each
        /// row carries the blender output + per-NWP individual model speeds so
        /// the wind forecast page can render both the multi-phase top chart and
        /// the per-NWP overlay below it from the same source. Empty list when
        /// no <c>wind</c> element predictions have landed (fresh deploy).
        /// </summary>
        public IReadOnlyList<WindForecastPoint> WindPredictions { get; init; }
            = Array.Empty<WindForecastPoint>();

        /// <summary>Wind direction predictions (Phase 2 <c>wind_mvn</c>) for
        /// this location at every trained lead. Powers the per-lead wind
        /// page's speed CI ribbon (BlendSpeedCi95Lo/Hi) and the per-hour
        /// direction circles (BlendDirection + Ci95Lo/Hi). Empty until the
        /// WP predict-wind-direction workflow has pushed parquets to R2 +
        /// sync-render-inputs has pulled them locally.</summary>
        public IReadOnlyList<WindDirectionForecastPoint> WindDirectionPredictions { get; init; }
            = Array.Empty<WindDirectionForecastPoint>();

        /// <summary>
        /// Structured weekly verify history loaded from
        /// <c>data/reports/verify_*.json</c>. One entry per (target, asOf)
        /// run; the Models-page renderer filters per-card on (target,
        /// station, version, windowHours) to populate a "Verify history"
        /// table beneath each model card. Empty until the first weekly
        /// verify run lands JSON sidecars on R2.
        ///
        /// <para><b>The single-location-invariant exception:</b> this list
        /// carries every location's rows, not just <see cref="ActiveLocationName"/>'s.
        /// That's deliberate — the Models page matches rows to per-station
        /// cards via <c>StationMatchesCard</c> (a finer granularity than
        /// location: each location has multiple precip-gauge cards). Because
        /// the page's cards are themselves location-scoped (built from this
        /// SiteInputs' location-scoped <c>ModelSummaries</c>) and the match
        /// is exact station-key equality, a card never picks up another
        /// location's rows. Pre-filtering this list per-location would not
        /// remove the per-card matching logic, so it stays whole.</para>
        /// </summary>
        public IReadOnlyList<VerifyHistoryFile> VerifyHistory { get; init; }
            = Array.Empty<VerifyHistoryFile>();
    }

    /// <summary>
    /// Composite-neutral blender metric snapshot for the Models page. Pulls the subset
    /// of fields from <c>training_metadata.json</c> that actually appear in the site; the
    /// rest (hyperparameters, deviations-from-brief) are skipped to keep the Site layer
    /// from depending on <c>WeatherBlend.Train</c>.
    ///
    /// <c>Composite</c> names the (target, station?, window?) triple in a format the
    /// Models page can sort by: "temperature", "precipitation / bellever_dartmoor",
    /// "dry_window / bellever_dartmoor / 6h". Scoring metric per target: temperature is
    /// MAE (°C); precipitation is Brier score; dry-window is Brier score. The
    /// <see cref="MetricLabel"/> string identifies which.
    /// </summary>
    public sealed record ModelSummary(
        string Composite,
        string Version,
        string Phase,
        string DataSource,
        DateTime TrainedAtUtc,
        string MetricLabel,
        IReadOnlyDictionary<int, PerLeadMetric> PerLead);

    /// <summary>One (lead → metric) entry, kept neutral so the Site layer doesn't import Train types.</summary>
    public sealed record PerLeadMetric(
        int LeadHours,
        string BestSingle,
        double BestSingleValMae,
        double BestSingleTestMae,
        double BlendTestScore,
        double BlendTestRmse,
        double BlendTestBias,
        int TestRows,
        int TestCalendarMonths);

    /// <summary>
    /// One row in the auto-generated feature spec table on the Models / Spec
    /// page. Sourced from each Active version's <c>feature_schema.json</c> so
    /// the table reflects deployed truth — when the table and the hand-edited
    /// design doc disagree, the table is right by construction.
    ///
    /// <c>Composite</c> uses the same key the Models page uses
    /// (<c>"temperature"</c>, <c>"precipitation/{station}"</c>,
    /// <c>"dry_window/{station}/{N}h"</c>) so a future link from a card to
    /// "show me this composite's spec" is a string match. <c>FeatureSet</c>
    /// is the raw tag stored on disk (<c>"lean"</c>, <c>"exact-l24-T2"</c>,
    /// <c>"exact-l48-P1-ukv"</c>); the renderer derives data-source label
    /// + UKV mode from the tag prefix / suffix rather than re-storing them.
    /// </summary>
    public sealed record FeatureSpecRow(
        string Composite,
        string Phase,
        string Version,
        DateTime TrainedAtUtc,
        int LeadHours,
        string FeatureSet,
        IReadOnlyList<string> RequiredModels,
        IReadOnlyList<string> OptionalModels,
        IReadOnlyList<string> FeatureNames,
        // Structured-spec fields (added 2026-05-07). Sourced from the
        // BlenderSpec on disk (BlenderDataSource constants for DataSource;
        // tier name like "lean"/"T2" for Tier; "Strict"/"Averaging"/null
        // for UkvStrategy). Empty / null on legacy schemas — Phase 2 of
        // the structured-spec migration will switch the renderer to read
        // these directly with fallback to FeatureNames inference; for
        // Phase 1 they're plumbed through but not yet consumed.
        string DataSource = "",
        string Tier = "",
        string? UkvStrategy = null);

    /// <summary>One per (Phase, LeadHours, day-end) rolling-MAE point.
    /// Grouped on <c>Phase</c> rather than <c>ModelVersion</c> so a retrain
    /// — which mints a new version — doesn't fragment the chart into a
    /// short stub for the new version sitting next to the long history of
    /// the old one. When two versions of the same phase have predictions
    /// for the same valid hour, the freshest <c>PredictionMadeAtUtc</c>
    /// wins (same rule as everywhere else in the renderer).</summary>
    public sealed record RollingMaePoint(string Phase, int LeadHours, DateTime WindowEndUtc, double BlendMae, int N);

    /// <summary>Per-(Station, Phase, LeadHours, WindowEndUtc) rolling Brier
    /// score for the precip blender. <c>BlendBrier</c> is the mean of
    /// <c>(ProbWet − truthBinary)²</c> over the window's pairs, where
    /// truthBinary = 1 iff the EA gauge recorded ≥ 0.1 mm/h that hour. <c>N</c>
    /// is the pair count in the window — readers use it to gauge stability
    /// the same way they do on the temperature rolling MAE chart.
    /// Phase grouping mirrors <see cref="RollingMaePoint"/>.</summary>
    public sealed record RollingBrierPoint(string Station, string Phase, int LeadHours, DateTime WindowEndUtc, double BlendBrier, int N);

    public sealed record PrecipForecastPoint(
        string Station,
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double ProbWet,
        double ClimatologyPWet,
        double? PrecipGfs,
        double? PrecipEcmwf,
        double? PrecipIcon,
        double? PrecipMf,
        double? PrecipUkmo,
        double? PrecipGem,
        double? PrecipAifs,
        double? PrecipJma,
        // Per-NWP wet-vote fraction at training-time hourly resolution (0..1).
        // Confidence signal: when all NWPs agree, this saturates at 0 or 1;
        // when they're split, it sits near 0.5. Free byproduct of 3a's
        // training feature row, persisted on PrecipPredictionRow. Defaulted
        // to null so existing positional construction sites compile unchanged
        // — the renderer treats null as "legacy row, hide the chip".
        double? AgreementWet01 = null,
        // Conformal-prediction set tag from precip-conformal-fit
        // {"Dry", "Wet", "Ambiguous"}. Calibrated alternative to the
        // NWP-agreement chip — agreement is "do the NWPs agree?" (heuristic);
        // conformal is "does the model commit at the requested coverage
        // guarantee?" (distribution-free). Both shown side-by-side in the
        // hourly-detail table. Null on legacy rows pre-conformal-fit.
        string? ConformalSetTag = null,
        // Bayesian-uncertainty quantiles (added 2026-05-09). Persisted by
        // phases that compute a posterior per row — 4a (BART) writes them
        // for live predictions. Null on 3a/3c/3d/2x rows and on pre-2026-05-09
        // 4a rows. Render path uses Q05/Q95 to draw a dashed band around the
        // 4a champion's prediction line.
        double? ProbWetQ05 = null,
        double? ProbWetQ95 = null,
        double? Ci80Width  = null,
        double? Ci90Width  = null,
        // Location whose NWP fed the featureset (PrecipPredictCommand's
        // _activeLocation.Name). Carried through so multi-location renders
        // can drop rows where a station's predictions came from another
        // location's NWP — the right pairing is "row.LocationName matches
        // the station's home location" (see RowMatchesStationHomeLocation).
        // Defaults to "" so existing test fixtures + legacy positional
        // callers still build; production projection in RenderSiteCommand
        // assigns the real value via named arg.
        string LocationName = "");

    /// <summary>
    /// Per-valid-hour aggregate of NWP fog / low-cloud forecasts. Counts
    /// how many models passed each independent threshold:
    ///   <see cref="VisFiredCount"/> / <see cref="VisTotalCount"/> — sub-1km
    ///     surface visibility (mist / fog). Only the 6 NWPs that publish
    ///     visibility through Open-Meteo can fire this (UKMO + GFS + ICON +
    ///     two HARMONIE variants + MO Spot).
    ///   <see cref="CloudBaseFiredCount"/> / <see cref="CloudBaseTotalCount"/>
    ///     — temperature within 1.5°C of dew point (Espy proxy: cloud base
    ///     plausibly below 393m AMSL). All 11 NWPs publish T + Td so the
    ///     denominator is much larger.
    /// Render the badge when EITHER hits its threshold (3/6 vis or 6/11
    /// cloud-base) — they're complementary signals so a single trigger is
    /// enough.
    /// </summary>
    public sealed record LowCloudSignal(
        int VisFiredCount, int VisTotalCount,
        int CloudBaseFiredCount, int CloudBaseTotalCount);

    public sealed record FeelsLikeForecastPoint(
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double UtciC,
        string Band,
        double ApparentC,
        // Element-blender inputs that fed UTCI for this hour. Surfaced in the
        // home-page per-tile pop-out so the reader can see "why is UTCI this
        // value?" without leaving the home page. All defaulted so existing
        // positional construction sites compile unchanged.
        double? TemperatureC = null,
        double? RelativeHumidityPct = null,
        double? WindSpeed10mMs = null,
        double? ShortwaveDownWm2 = null,
        double? CloudCoverPct = null,
        /// <summary>Config slug of the location this UTCI row was sourced
        /// from (e.g. <c>bonehill_rocks</c>, <c>membury_devon</c>) so the
        /// per-loc home-card picks the right chip. Phase C commit 3a.
        /// Defaulted so any existing positional construction (mostly tests)
        /// keeps compiling.</summary>
        string LocationName = "");

    /// <summary>One rock surface temperature + condensation hour for the site
    /// (Phase P1). <see cref="GreasinessStatus"/> is dry / potentially_greasy /
    /// condensation. <see cref="LocationName"/> scopes the per-loc render.</summary>
    public sealed record RockSurfaceForecastPoint(
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double RockSurfaceTempC,
        double AirTempC,
        double DewPointC,
        double CondensationMarginC,
        string GreasinessStatus,
        string LocationName = "");

    public sealed record DryWindowForecastPoint(
        string Station,
        int WindowHours,
        string Version,
        DateTime PredictedAtUtc,
        DateTime TargetDateUtc,
        int LeadHours,
        double ProbHasDryWindow,
        double ClimatologyProbHasDryWindow,
        double? AgreementHasDryWindow,
        // MC aleatoric uncertainty (null on non-MC rows like 3b). Summary
        // of the per-MC-sample longest-dry-run distribution; narrow P10–P90
        // → headline P(any block) is robust across MC realisations under
        // the chosen copula; wide → fragile. Populated by 3p; defaulted to
        // null so existing positional construction sites compile unchanged.
        double? McMeanLongestDryRunHours = null,
        double? McP10LongestDryRunHours = null,
        double? McP50LongestDryRunHours = null,
        double? McP90LongestDryRunHours = null,
        // Optional epistemic envelope from a Bayesian CI join. No active
        // phase populates these today; left in place for future producers.
        double? EpistemicProbDryWindowMean = null,
        double? EpistemicProbDryWindowQ10  = null,
        double? EpistemicProbDryWindowQ90  = null,
        double? EpistemicSigmaUsed         = null,
        // Conformal-prediction set tag {"Dry", "Wet", "Ambiguous"} from
        // the per-(version, lead) ConformalCalibrator if one was fitted
        // (dry-window-conformal-fit). "Ambiguous" flags rows where the model
        // can't commit to a single class with the requested coverage
        // guarantee — a calibrated alternative to NWP-agreement confidence.
        string? ConformalSetTag = null);

    /// <summary>Per-NWP precipitation probability (Open-Meteo's
    /// <c>precipitation_probability</c> field, percent 0..100) for one
    /// model at one valid time. Caller has already deduped to the freshest
    /// RunTime per (Model, ValidTime). Only ~4 of our 8 NWPs publish this
    /// field (GFS / ECMWF / ICON / GEM); the rest emit nothing and are
    /// silently absent from the chart's legend.</summary>
    public sealed record NwpPrecipProbForecastPoint(
        string Model,
        DateTime ValidTimeUtc,
        double ProbabilityPercent,
        string LocationName);

    /// <summary>Per-NWP precipitation rate (mm/h) — Open-Meteo's
    /// <c>precipitation</c> field. Hourly granularity, deduped to the
    /// freshest RunTime per (Model, ValidTime). Used by the lead-12
    /// rain page's NWP rate chart where the prediction-row-based overlay
    /// is too sparse. <see cref="LocationName"/> carries the config slug
    /// of the location this forecast was sourced from so the renderer can
    /// route per active location tab (Bonehill vs Membury, etc.).</summary>
    public sealed record NwpPrecipRateForecastPoint(
        string Model,
        DateTime ValidTimeUtc,
        double PrecipMmPerHour,
        string LocationName);

    /// <summary>Per-NWP 2m temperature (°C). Same shape and purpose as
    /// NwpPrecipRateForecastPoint, used on the temp lead-12 page.</summary>
    public sealed record NwpTemperatureForecastPoint(
        string Model,
        DateTime ValidTimeUtc,
        double Temperature2m,
        string LocationName);

    /// <summary>One Met Office DataHub Spot forecast row at a single
    /// (RunTime, ValidTime). Fields nullable because the Spot product
    /// doesn't always emit every variable on every hour. Used as a
    /// comparison line on the temp + rain skill pages alongside the
    /// blend's prediction; never feeds the blender (it's a separate
    /// product to <c>ukmo_seamless</c>).</summary>
    public sealed record MetOfficeSpotForecastPoint(
        DateTime RunTimeUtc,
        DateTime ValidTimeUtc,
        double? Temperature2m,
        /// <summary>Precipitation probability in percent [0, 100], or null
        /// when the row didn't carry it. NB Met Office's spot PoP definition
        /// is "P(any measurable precip)", which is a slightly looser
        /// threshold than our blender's 0.1 mm/h training label — note that
        /// in any chart legend that overlays the two.</summary>
        double? PrecipitationProbabilityPercent,
        /// <summary>Config slug of the location this Spot row was sourced
        /// from (e.g. <c>bonehill_rocks</c>, <c>membury_devon</c>) so the
        /// skill-page renderer can pick the right rows per active tab.</summary>
        string LocationName,
        /// <summary>(ValidTime − RunTime) in hours. Used by the skill-page
        /// per-lead bucketing: chart's +Lh trace draws rows with
        /// LeadHours ∈ [L, L+24).</summary>
        int LeadHours);

    /// <summary>One row of the dry-window start-hour curve. Multiple rows
    /// share a (Station, WindowHours, LeadHours, TargetDateUtc) and differ
    /// only in <see cref="StartHourUtc"/> + the per-start probabilities.</summary>
    public sealed record StartHourForecastPoint(
        string Station,
        int WindowHours,
        string Version,
        DateTime PredictedAtUtc,
        DateTime TargetDateUtc,
        int LeadHours,
        int StartHourUtc,
        double RawProduct,
        double ConditionalProb,
        double CalibratedProb,
        double DailyProbAnyBlock,
        /// <summary>Config slug of the location this curve was sourced from
        /// (e.g. <c>bonehill_rocks</c>, <c>membury_devon</c>) so the per-loc
        /// dry-window page picks the right rows. Phase C commit 3a.
        /// Defaulted so any existing positional construction (mostly tests)
        /// keeps compiling.</summary>
        string LocationName = "");

    /// <summary>One row from a <c>wind</c> element prediction parquet
    /// (champion <c>wind</c> phase + challenger <c>wind_speed_lgb</c> /
    /// <c>wind_blend</c> phases all live under the same Element). Powers
    /// the per-lead wind forecast page's speed chart (multi-phase top line
    /// + per-NWP overlay).
    ///
    /// SpeedMs / per-NWP values keep the storage unit (m/s); the renderer
    /// converts to mph at draw time so the unit knob stays in one place.
    /// </summary>
    public sealed record WindForecastPoint(
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double SpeedMs,
        double? Gfs,
        double? Ecmwf,
        double? Icon,
        double? Mf,
        double? Ukmo,
        double? Gem,
        double? Aifs,
        string LocationName);

    /// <summary>One row from a wind_direction (<c>wind_mvn</c>) parquet.
    /// Powers the per-lead wind page's speed CI ribbon (CI lo/hi
    /// columns) and the per-hour direction circles (BlendDirection +
    /// dir CI lo/hi).
    /// </summary>
    public sealed record WindDirectionForecastPoint(
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double DirectionDeg,
        double DirectionCi95LoDeg,
        double DirectionCi95HiDeg,
        double SpeedMs,
        double SpeedCi95LoMs,
        double SpeedCi95HiMs,
        string LocationName,
        // CI80 fields added 2026-05-28. Nullable so older parquets that
        // only carry CI95 (anything written before this revision) still
        // deserialize; the renderer falls back to CI95 when these are
        // null. Same MC samples drive both bands — 10/90 percentiles for
        // CI80, 2.5/97.5 for CI95.
        double? DirectionCi80LoDeg = null,
        double? DirectionCi80HiDeg = null,
        double? SpeedCi80LoMs = null,
        double? SpeedCi80HiMs = null);

    public static string Stylesheet() => """
        :root { --brand: #7c4dff; --pwet: #0288d1; }
        body > main { padding-top: 1rem; padding-bottom: 3rem; }
        header > hgroup h1 a { color: inherit; text-decoration: none; }
        nav.site-nav { padding: 0.5rem 0 0.5rem; border-bottom: 1px solid var(--pico-muted-border-color); margin-bottom: 0.5rem; }
        /* Pico's default <section> has a top margin that combined with the
           site-nav bottom margin to leave a big empty band between the main
           and sub navs. Suppress it on the first section in main. */
        main > section:first-of-type { margin-top: 0; padding-top: 0; }
        nav.site-nav ul { display: flex; gap: 1rem; list-style: none; padding: 0; margin: 0; flex-wrap: wrap; }
        nav.site-nav a { text-decoration: none; }
        nav.site-nav a.active { font-weight: 600; color: var(--brand); }

        /* Phase D (2026-05-20) — cross-cutting top-nav (Locations / Specs / About) */
        nav.top-nav { padding: 0.25rem 0; margin-bottom: 0.25rem; font-size: 0.9em; }
        nav.top-nav ul { display: flex; gap: 0.75rem; list-style: none; padding: 0; margin: 0; }
        nav.top-nav a { text-decoration: none; color: var(--pico-muted-color); }
        nav.top-nav a:hover { color: var(--brand); }
        nav.top-nav a.active { font-weight: 600; color: var(--brand); }

        /* Phase D — location picker on the top-level home page (/index.html) */
        .loc-picker { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
        .loc-card { display: block; padding: 1.25rem; border-radius: 8px; background: var(--pico-card-background-color); border: 1px solid var(--pico-muted-border-color); text-decoration: none; color: inherit; transition: transform 0.1s; }
        .loc-card:hover { transform: translateY(-2px); border-color: var(--brand); }
        .loc-card h3 { margin: 0 0 0.5rem; color: var(--brand); }
        .loc-card .loc-tabs { margin: 0; font-size: 0.85em; color: var(--pico-muted-color); }

        /* auto-fill (not auto-fit) so empty tracks stay reserved at min
           width — a single tile renders at ~180px instead of expanding to
           fill the row, matching the size of the multi-tile case. Changed
           2026-05-07 after a low-tile-count day stretched one tile to
           full row width. */
        .forecast-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.75rem; }
        /* Per-lead wind page direction-circle row. One cell per UTC hour
           in [HomeFirstVisibleHourUtc, HomeLastVisibleHourUtcExclusive); each
           cell stacks hour label / SVG circle / speed text. Wraps on narrow
           viewports — minmax(64px, 1fr) keeps the cells readable. */
        .wind-dir-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(64px, 1fr)); gap: 0.4rem; margin: 0.5rem 0 1rem; }
        .wind-dir-cell { display: flex; flex-direction: column; align-items: center; }
        .wind-dir-hour { font-size: 0.75rem; color: var(--pico-muted-color); }
        .wind-dir-speed { font-size: 0.8rem; }
        .wind-dir-svg { display: block; }
        .wind-dir-empty { opacity: 0.6; }
        .wind-day-label { margin-top: 0.5rem; margin-bottom: 0.25rem; }
        /* Flex layout so the footer pins to the cell's lower edge regardless of
           whether the feels-like chip is present — cards with shorter content
           (e.g. 96/120h leads with no feels-like row) used to float the footer
           up, leaving made-time misaligned across the grid. */
        .forecast-card { padding: 0.75rem 0.85rem; display: flex; flex-direction: column; }
        .forecast-card h3 { margin: 0 0 0.1rem; font-size: 1rem; }
        .forecast-card header small { color: var(--pico-muted-color); font-size: 0.8rem; }
        /* Pico v2 wraps <article> > header in its own padded container; on
           a tight tile that opens an extra ~0.5rem gap below the time
           heading. Override to zero so the temperature sits snug under the
           time. Tightened 2026-05-07 per user feedback. */
        .forecast-card > header { margin-bottom: 0; padding-bottom: 0; }
        .forecast-card .temp { font-size: 2rem; font-weight: 700; color: var(--temp-color, var(--brand)); line-height: 1.1; margin: 0.1rem 0 0.25rem; }
        .forecast-card-empty .temp { color: var(--pico-muted-color); }
        .forecast-card .pwet { font-size: 0.9rem; color: var(--pwet); margin: 0 0 0.35rem; font-variant-numeric: tabular-nums; }
        .forecast-card .pwet strong { font-weight: 700; }
        .forecast-card .pwet .rain { margin-left: 0.25rem; }
        .forecast-card .feels { font-size: 0.85rem; color: var(--pico-muted-color); margin: 0 0 0.35rem; font-variant-numeric: tabular-nums; }
        .forecast-card .feels strong { font-weight: 700; }
        .forecast-card .feels small { margin-left: 0.35rem; }
        /* UTCI band on its own line under the UTCI value, italic + muted
           so it reads as a quiet caption rather than competing with the
           number. Margin pulls it tight under the value rather than the
           feels-row's default 0.35rem gap. */
        .forecast-card .feels .utci-band { margin-top: -0.1rem; font-style: italic; color: var(--pico-muted-color); }
        .forecast-card footer { margin-top: auto; padding-top: 0.4rem; border-top: 1px solid var(--pico-muted-border-color); }
        .forecast-card footer small { color: var(--pico-muted-color); font-size: 0.75rem; }
        .forecast-card h4 { margin: 0; font-size: 1rem; font-variant-numeric: tabular-nums; }
        /* Fixed header min-height pins the time row to a uniform height across
           the grid. (Badges no longer live in the header — they sit in the
           .tile-badges block below, whose own min-height is what now aligns the
           temperature across badged / badge-free tiles. See .tile-badges.) */
        .forecast-card header { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; min-height: 1.4rem; }
        /* Low-cloud / mist badge — slate pill in the tile header. Click to
           expand details (Pico <details>) listing which signal(s) fired and
           the per-NWP agreement count. Tap-friendly on mobile, no hover needed. */
        /* Low-cloud and UTCI pop-outs both float their open panel ON TOP of
           tile contents instead of expanding inline. Without this the
           panel inside the tile <header> shoved the temperature / feels-
           like / P(wet) rows down — tiles ended up different heights and
           the row alignment broke. position:relative on the <details>
           makes the inner panel's position:absolute anchor to it. The
           panel is invisible by default; details[open] flips it to
           visible. Click-to-toggle is the native <details> behaviour. */
        /* Badges live in a dedicated stacked block under the tile time (top of
           the card), one pill per line. Previously they were flex children of
           the <header> row, so the time + both pills laid out left-to-right and
           a card with BOTH a low-cloud and a rock badge overflowed its width
           onto the next tile. The column container + flex-start keeps each pill
           on its own line, sized to its content.
           ALWAYS rendered (even empty) with a fixed min-height sized to two
           stacked pills (low-cloud + rock) so the temperature aligns across
           every tile regardless of badge count — badge-free tiles show the
           reserved space as a small blank gap, by design (2026-06-06). Margins
           kept tight so the gap to the time above and temperature below is
           small (was 0.2/0.35 — read as too large). */
        .forecast-card .tile-badges { display: flex; flex-direction: column; align-items: flex-start; gap: 0.2rem; margin: 0.1rem 0 0.15rem; min-height: 2.25rem; }
        details.low-cloud-pop { display: block; position: relative; margin-top: 0; }
        details.low-cloud-pop > summary.low-cloud-badge { background: #455a64; color: #fff; font-size: 0.7rem; line-height: 1.15; padding: 0.1rem 0.4rem; border-radius: 999px; cursor: pointer; white-space: nowrap; list-style: none; display: inline-block; }
        /* Kill BOTH native marker and Pico's ::after chevron — Pico v2 adds
           the chevron via a background-image on summary::after, which the
           webkit rule alone doesn't reach. User wants these as plain pills. */
        details.low-cloud-pop > summary.low-cloud-badge::-webkit-details-marker { display: none; }
        details.low-cloud-pop > summary.low-cloud-badge::after { display: none !important; content: none !important; }
        details.low-cloud-pop > ul {
          position: absolute;
          top: calc(100% + 0.25rem);
          left: 0;
          z-index: 10;
          margin: 0;
          padding: 0.4rem 0.6rem 0.4rem 1.5rem;
          font-size: 0.78rem;
          line-height: 1.35;
          color: var(--pico-muted-color);
          background: var(--pico-card-background-color);
          border: 1px solid var(--pico-card-border-color);
          border-radius: 4px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
          width: max-content;
          max-width: 16rem;
        }
        details.low-cloud-pop > ul li { margin: 0; padding: 0.05rem 0; }

        /* Rock surface / condensation badge (Phase P1) — same pill + pop-out
           idiom as the low-cloud badge. Red = condensation (rock wet), amber =
           potentially greasy (rock near dew point). */
        details.rock-pop { display: block; position: relative; margin-top: 0; }
        details.rock-pop > summary.rock-badge { color: #fff; font-size: 0.7rem; line-height: 1.15; padding: 0.1rem 0.4rem; border-radius: 999px; cursor: pointer; white-space: nowrap; list-style: none; display: inline-block; }
        details.rock-pop > summary.rock-badge.rock-wet { background: #c62828; }
        /* Amber (not the old #ef6c00 deep-orange, which read as a second red).
           True amber needs dark text for legibility — override the white base. */
        details.rock-pop > summary.rock-badge.rock-greasy { background: #ffb300; color: #1f1300; }
        details.rock-pop > summary.rock-badge::-webkit-details-marker { display: none; }
        details.rock-pop > summary.rock-badge::after { display: none !important; content: none !important; }
        details.rock-pop > ul {
          position: absolute;
          top: calc(100% + 0.25rem);
          left: 0;
          z-index: 10;
          margin: 0;
          padding: 0.4rem 0.6rem 0.4rem 1.5rem;
          font-size: 0.78rem;
          line-height: 1.35;
          color: var(--pico-muted-color);
          background: var(--pico-card-background-color);
          border: 1px solid var(--pico-card-border-color);
          border-radius: 4px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
          width: max-content;
          max-width: 16rem;
        }
        details.rock-pop > ul li { margin: 0; padding: 0.05rem 0; }

        /* Day-grouped home layout — one block per UTC day with a summary line
           above the per-hour tile grid. Top + bottom margin so days read as
           visually separated blocks; the overall page still scrolls naturally. */
        .day-block { margin: 0 0 1.5rem; padding: 0; border: 0; }
        .day-block h3 { margin: 0.75rem 0 0.25rem; }
        .day-block .skill-line { margin: 0 0 0.5rem; }

        /* UTCI per-hour pop-out — Pico's <details> styled inline so the ⓘ
           summary sits next to the band tag instead of stacking on its own
           line. The panel itself opens below the tile contents (Pico handles
           the toggle) and shows a tight 2-column table of the element-blender
           values that fed UTCI. */
        details.utci-pop { display: inline; margin-left: 0.25rem; position: relative; }
        details.utci-pop summary { cursor: pointer; color: var(--pico-muted-color); font-size: 0.85rem; list-style: none; display: inline; }
        details.utci-pop summary::-webkit-details-marker { display: none; }
        /* Kill Pico v2's ::after chevron too — see low-cloud-pop. */
        details.utci-pop summary::after { display: none !important; content: none !important; }
        details.utci-pop[open] summary { color: var(--brand); }
        /* Float the table on top of subsequent tile content (P(wet) row +
           footer) for the same reason as low-cloud-pop's <ul> — keeps the
           tile height stable when expanded so the grid stays aligned. */
        .utci-pop-table {
          position: absolute;
          top: calc(100% + 0.25rem);
          left: 0;
          z-index: 10;
          margin: 0;
          font-size: 0.78rem;
          background: var(--pico-card-background-color);
          border: 1px solid var(--pico-card-border-color);
          border-radius: 4px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
          padding: 0.3rem 0.5rem;
          min-width: 14rem;
        }
        .utci-pop-table td { padding: 0.1rem 0.25rem; border: 0; }
        /* Keep "Wind 10 m | XX mph · gust YY mph" on a single line — the
           "· gust …" suffix was tipping that cell to two rows on tiles
           with a tight column width. nowrap on the value column lets the
           table widen as needed (min-width above absorbs the extra). */
        .utci-pop-table td.num { white-space: nowrap; }

        nav.lead-nav { margin: 0 0 1.25rem; padding: 0; }
        nav.lead-nav ul { display: flex; gap: 0.75rem; list-style: none; padding: 0; margin: 0; }
        nav.lead-nav a { text-decoration: none; padding: 0.25rem 0.75rem; border-radius: 4px; background: var(--pico-card-background-color); }
        nav.lead-nav a.active { background: var(--brand); color: white; font-weight: 600; }

        .skill-line { font-style: italic; color: var(--pico-muted-color); }

        /* Confidence chip used on hourly precip detail tables — reads as a
           glance-able classifier of "do all the NWPs agree on this hour?". */
        .conf { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
                font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
        .conf-high    { background: #e8f5e9; color: #2e7d32; }
        .conf-medium  { background: #fff8e1; color: #ef6c00; }
        .conf-low     { background: #ffebee; color: #c62828; }
        .conf-unknown { background: #f5f5f5; color: #757575; }
        details.hourly-detail { margin: -0.5rem 0 1.5rem; padding: 0; }
        details.hourly-detail summary { font-size: 0.85rem; color: var(--pico-muted-color);
                                        cursor: pointer; padding: 0.25rem 0; }
        details.hourly-detail figure { margin: 0.25rem 0 0.5rem; }
        details.hourly-detail table { margin: 0; }
        details.hourly-detail table th, details.hourly-detail table td { padding: 0.25rem 0.6rem; font-size: 0.85rem; }

        table td.num, table th.num { text-align: right; font-variant-numeric: tabular-nums; }
        table td.num.strong { font-weight: 700; color: var(--brand); }
        /* Drift colouring works on either a table cell or a bare span;
           the verify-history row stacks blend + best-NWP metrics in one
           cell with a span on each, and the prior td-only selector
           dropped the highlight from those spans. Generic class applies
           anywhere the marker class is used. */
        .delta-good { color: #2e7d32; }
        .delta-bad  { color: #c62828; }
        /* Δ vs previous train badge in blender-card header. Same colour
           semantics as the table-cell delta classes (green = improved,
           red = regressed) but rendered as a pill alongside the version
           tag rather than inside a table cell. */
        .delta-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.1rem 0.45rem; border-radius: 999px; margin-left: 0.4rem; vertical-align: middle; }
        .delta-badge.delta-good { background: rgba(46,125,50,0.12); color: #2e7d32; }
        .delta-badge.delta-bad  { background: rgba(198,40,40,0.12); color: #c62828; }
        table small { color: var(--pico-muted-color); }
        article.blender-card { padding: 1rem 1.25rem; margin: 0.75rem 0 1.25rem; }
        article.blender-card > header { margin-bottom: 0.5rem; }
        article.blender-card h4 { margin: 0; font-size: 1.05rem; }
        article.blender-card h4 code { font-size: 0.85em; }
        article.blender-card table { margin: 0; }
        article.blender-card table th, article.blender-card table td { padding: 0.35rem 0.6rem; }

        svg.chart { width: 100%; height: auto; max-width: 100%; background: var(--pico-card-background-color); border-radius: 4px; margin: 0.5rem 0 1.5rem; }
        .chart-cjs { position: relative; margin: 0.5rem 0 1.5rem; background: var(--pico-card-background-color); border-radius: 4px; padding: 0.5rem 0.5rem 0.25rem; }
        .chart-cjs canvas { width: 100% !important; height: 100% !important; }
        /* Stacked precip charts share an X axis: render flush with each other so
           the eye reads them as one panel split into prob/mm-h halves. */
        .chart-stack { margin: 0.5rem 0 1.5rem; }
        .chart-stack .chart-cjs { margin: 0; border-radius: 0; }
        .chart-stack .chart-cjs:first-child { border-top-left-radius: 4px; border-top-right-radius: 4px; padding-bottom: 0; }
        .chart-stack .chart-cjs:last-child { border-top: 1px solid var(--pico-muted-border-color); border-bottom-left-radius: 4px; border-bottom-right-radius: 4px; padding-top: 0; }
        .chart-title { font-size: 14px; font-weight: 600; fill: var(--pico-color); }
        .chart-grid { stroke: var(--pico-muted-border-color); stroke-width: 0.5; }
        .chart-frame { fill: none; stroke: var(--pico-muted-border-color); stroke-width: 1; }
        .chart-tick { font-size: 10px; fill: var(--pico-muted-color); font-family: ui-monospace, monospace; }
        .chart-tick-mark { stroke: var(--pico-muted-border-color); stroke-width: 1; }
        .chart-axis-label { font-size: 11px; fill: var(--pico-muted-color); }
        .chart-legend { font-size: 11px; fill: var(--pico-color); font-family: ui-monospace, monospace; }
        .chart-empty { font-size: 12px; fill: var(--pico-muted-color); font-style: italic; }

        footer.site-footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--pico-muted-border-color); color: var(--pico-muted-color); font-size: 0.875rem; }
        """;

    /// <summary>
    /// Chart.js bootstrapper: scans every <c>canvas[data-cjs]</c> on the page,
    /// builds a Chart.js v4 line chart from the embedded JSON config. Each config
    /// carries pre-baked tick-format hints (date kind, decimals, suffix, trim)
    /// so the JS layer doesn't need a date-fns/luxon adapter — we share a tiny
    /// inline UTC date formatter and a tiny number formatter, both keyed off
    /// metadata <see cref="LineChartRenderer.RenderChartJs"/> emits server-side.
    ///
    /// Only temperature charts go through here today (the user pivot from the
    /// hand-rolled SVG hover). Other pages still render via
    /// <see cref="LineChartRenderer.Render"/>; flip them over once the temp
    /// page is confirmed good.
    /// </summary>
    public static string ChartScript() => """
        (function () {
          const PAD = n => String(n).padStart(2, '0');
          function fmtDate(v, kind) {
            // 'hour' axis: v is a plain hour-of-day number (0-23), not a ms
            // timestamp — label it as a clock time.
            if (kind === 'hour') return PAD(Math.round(v)) + ':00Z';
            const d = new Date(v);
            const md = PAD(d.getUTCMonth() + 1) + '-' + PAD(d.getUTCDate());
            if (kind === 'datetime') return md + ' ' + PAD(d.getUTCHours()) + 'Z';
            return md;
          }
          function fmtNum(v, ycfg) {
            const dec = ycfg.dec | 0;
            let s = v.toFixed(dec);
            if (ycfg.trim && s.indexOf('.') >= 0) s = parseFloat(s).toString();
            return s + (ycfg.suffix || '');
          }
          // OADate (days since 1899-12-30) → Unix ms. Keeps server-side data tight.
          // Snap to the nearest minute: the OADate fraction can't represent a
          // whole hour exactly in a double, so (oa - 25569) * 86400000 lands a
          // few tenths of a ms BELOW the true instant — e.g. 10:00:00 comes out
          // as 09:59:59.9996, and getUTCHours() then floors it to 09, which is
          // why adjacent hourly tooltips read 09:00 → 09:00 → 11:00. Our data is
          // never finer than 15-min, so rounding to the minute is exact for real
          // values and removes the float drift.
          function oaToMs(oa) { return Math.round((oa - 25569) * 86400000 / 60000) * 60000; }

          function build(canvas) {
            if (!window.Chart) return;
            let cfg;
            try { cfg = JSON.parse(canvas.getAttribute('data-cjs')); }
            catch (e) { return; }
            if (!cfg || !cfg.datasets || !cfg.datasets.length) return;

            const ycfg = { dec: cfg.yDec, suffix: cfg.ySuffix, trim: cfg.yTrim };
            const xKind = cfg.xKind || 'date';
            // Date/datetime X arrives as an OADate and converts to Unix ms;
            // the 'hour' axis is already a plain number, so pass it through.
            const xVal = x => xKind === 'hour' ? x : oaToMs(x);

            const datasets = cfg.datasets.map(ds => ({
              label: ds.label,
              data: ds.points.map(p => ({ x: xVal(p[0]), y: p[1] })),
              borderColor: ds.color,
              // Ribbon fill (added 2026-05-27 for the rainfall_amount 3f
              // chart): when the server-side LineChartSpec.Ribbons names
              // this series as the low partner of a ribbon, `ds.fillTo`
              // is the dataset INDEX of its high partner and `ds.fillColor`
              // is the band's rgba colour. Chart.js draws a filled area
              // between the two datasets in `fillColor`. The series's
              // line stroke stays in `borderColor`. Other charts (no
              // ribbons) leave both undefined → backgroundColor stays
              // ds.color and fill stays false, matching pre-ribbon
              // behaviour exactly.
              backgroundColor: ds.fillColor || ds.color,
              fill: ds.fillTo != null ? ds.fillTo : false,
              borderWidth: 1.75,
              // Dashed series — used to overlay a challenger-phase line on top
              // of its champion in the same colour so the eye reads them as
              // paired ("same prediction, two methods"). [6, 4] is a moderate
              // dash that stays visually distinct from the solid champion at
              // chart resolutions we render (220-360px tall).
              borderDash: ds.dashed ? [6, 4] : [],
              // No markers on continuous series regardless of point count —
              // the previous "show markers if ≤30 points" branch was creating
              // visual asymmetry across stations (Bovey + rolling MAE / Brier
              // panels are sparser, landed in the marker branch; Bellever + denser
              // series didn't). Discrete-truth dots stay tiny so the 0/1 indicators
              // are still readable. Hover radius stays at 4 so the picker still
              // works on bare lines.
              pointRadius: ds.discrete ? 1.5 : 0,
              pointHoverRadius: 4,
              showLine: !ds.discrete,
              tension: 0,
              spanGaps: true,
            }));

            // Annotation plugin payload — wet bands and the "today" reference line.
            // Server emits cfg.annotations only when something's there, so most
            // charts pass an empty annotations object and the plugin no-ops.
            const annoCfg = {};
            if (cfg.annotations) {
              const ann = cfg.annotations;
              if (ann.bands && ann.bands.length) {
                ann.bands.forEach((b, i) => {
                  annoCfg['band' + i] = {
                    type: 'box',
                    xMin: xVal(b[0]),
                    xMax: xVal(b[1]),
                    yMin: -Infinity, yMax: Infinity,
                    backgroundColor: ann.bandColor || 'rgba(33,150,243,0.18)',
                    borderWidth: 0,
                    drawTime: 'beforeDatasetsDraw',
                  };
                });
              }
              if (ann.todayX != null) {
                annoCfg['today'] = {
                  type: 'line',
                  xMin: xVal(ann.todayX),
                  xMax: xVal(ann.todayX),
                  borderColor: 'rgba(102,102,102,0.7)',
                  borderWidth: 1.25,
                  borderDash: [4, 4],
                  // Label removed 2026-05-04 — the dashed line is self-explanatory
                  // ("everything past it is forecast-only" is repeated in the
                  // skill-line copy above each chart) and the floating tag
                  // sat on top of the freshest data points.
                };
              }
            }

            const chart = new Chart(canvas, {
              type: 'line',
              data: { datasets },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                normalized: true,
                interaction: { mode: 'nearest', axis: 'x', intersect: false },
                plugins: {
                  annotation: { annotations: annoCfg },
                  title: { display: !!cfg.title, text: cfg.title || '', font: { size: 14, weight: '600' }, padding: { top: 4, bottom: 8 } },
                  legend: { position: 'top', align: 'end', labels: { boxWidth: 12, boxHeight: 12, usePointStyle: false, font: { size: 11 } } },
                  // X-axis zoom + pan via chartjs-plugin-zoom. Wheel zoom
                  // and drag-pan; pinch zoom on touch. Y axis stays
                  // auto-scaled — zooming the time axis is what's useful
                  // here. Double-click resets to the chart's initial
                  // (xMin..xMax) window.
                  zoom: {
                    pan:  { enabled: true,  mode: 'x', modifierKey: null },
                    zoom: {
                      wheel: { enabled: true },
                      pinch: { enabled: true },
                      mode: 'x',
                    },
                    limits: cfg.xMin != null && cfg.xMax != null
                      ? { x: { min: xVal(cfg.xMin), max: xVal(cfg.xMax) } }
                      : undefined,
                  },
                  tooltip: {
                    backgroundColor: 'rgba(20,20,30,0.92)',
                    titleColor: '#fff', bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.15)', borderWidth: 1,
                    padding: 8, cornerRadius: 4,
                    titleFont: { family: 'ui-monospace, monospace', size: 11 },
                    bodyFont:  { family: 'ui-monospace, monospace', size: 11 },
                    callbacks: {
                      title: items => items.length ? fmtDate(items[0].parsed.x, xKind) : '',
                      label: item => item.dataset.label + ': ' + fmtNum(item.parsed.y, ycfg),
                    },
                  },
                },
                scales: {
                  x: {
                    type: 'linear',
                    // Server-supplied min/max pin the X axis so multiple charts
                    // in a section share the same time window. When unset, fall
                    // back to Chart.js's per-chart auto-scaling.
                    min: cfg.xMin != null ? xVal(cfg.xMin) : undefined,
                    max: cfg.xMax != null ? xVal(cfg.xMax) : undefined,
                    title: { display: !!cfg.xLabel, text: cfg.xLabel || '' },
                    ticks: {
                      autoSkip: true, maxTicksLimit: 8,
                      // Hour axis: force integer-spaced ticks so two adjacent
                      // ticks can't both round to the same "HH:00Z" label (the
                      // duplicate-label bug the SVG renderer had).
                      precision: xKind === 'hour' ? 0 : undefined,
                      font: { family: 'ui-monospace, monospace', size: 10 },
                      callback: v => fmtDate(v, xKind),
                    },
                  },
                  y: {
                    type: 'linear',
                    title: { display: !!cfg.yLabel, text: cfg.yLabel || '' },
                    ticks: {
                      font: { family: 'ui-monospace, monospace', size: 10 },
                      callback: v => fmtNum(v, ycfg),
                    },
                  },
                },
              },
            });

            // Double-click → reset zoom to the server-supplied (xMin, xMax)
            // window. Without this the only way back from a zoomed-in view
            // was page reload. Touch users get pinch-out which the plugin
            // already handles natively.
            canvas.addEventListener('dblclick', () => chart.resetZoom());
          }

          function init() {
            document.querySelectorAll('canvas[data-cjs]').forEach(build);
          }
          if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
          } else {
            init();
          }
        })();
        """;

    private static string WrapPage(SiteInputs input, string pageTitle, string pageId, string bodyHtml)
    {
        var prefix = input.AssetPrefix;
        var topNav = RenderTopLevelNav(input, pageId);
        var siteNav = input.RenderingFor is null
            ? ""  // top-level pages (picker, specs, about) carry only the cross-cutting nav
            : RenderLocationNav(input.RenderingFor, pageId);
        var subtitle = input.RenderingFor is null
            ? "Multi-model forecast blending"
            : $"Multi-model forecast blending for {Escape(input.RenderingFor.DisplayName)}";

        return $$"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>{{Escape(pageTitle)}} — WeatherBlend</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
              <link rel="stylesheet" href="{{prefix}}styles.css">
              <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
              <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
              <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
              <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js"></script>
              <script src="{{prefix}}chart.js" defer></script>
            </head>
            <body>
              <main class="container">
                <header>
                  <hgroup>
                    <h1><a href="{{prefix}}index.html">WeatherBlend</a></h1>
                    <p>{{subtitle}}</p>
                  </hgroup>
                  {{topNav}}
                  {{siteNav}}
                </header>

                {{bodyHtml}}

                <footer class="site-footer">
                  Rendered {{input.GeneratedAtUtc.ToString("yyyy-MM-dd HH:mm", Ci)}}Z ·
                  Training truth: ERA5 · Verification: ERA5 + METAR ·
                  <a href="{{prefix}}about.html">About this site</a>
                </footer>
              </main>
            </body>
            </html>
            """;
    }

    /// <summary>
    /// Cross-cutting nav: Locations picker · Specs · About. Renders on every
    /// page (top-level + per-loc); the active-tab highlight is on the link
    /// matching <paramref name="pageId"/> (one of "home", "specs", "about",
    /// or — for per-loc pages — any of the per-loc tab ids, in which case
    /// none of these links highlight).
    /// </summary>
    private static string RenderTopLevelNav(SiteInputs input, string pageId)
    {
        var prefix = input.AssetPrefix;
        return $$"""
              <nav class="top-nav">
                <ul>
                  <li><a href="{{prefix}}index.html"{{NavActive(pageId, "home")}}>Locations</a></li>
                  <li><a href="{{prefix}}specs.html"{{NavActive(pageId, "specs")}}>Specs</a></li>
                  <li><a href="{{prefix}}about.html"{{NavActive(pageId, "about")}}>About</a></li>
                </ul>
              </nav>
            """;
    }

    /// <summary>
    /// Per-location nav driven by <see cref="LocationDescriptor.Tabs"/>. Each
    /// recognised tab id maps to a single href + label. Unknown ids in the
    /// config are silently skipped (so adding a tab requires both a config
    /// edit + a code edit here, which is the right friction — new tabs
    /// always need new page builders).
    /// </summary>
    private static string RenderLocationNav(LocationDescriptor loc, string pageId)
    {
        // Tab id → (href relative to /{slug}/, label, pageId-for-active-highlight).
        // For the multi-target "skill" / "models" tabs, href points at the
        // temperature variant by default; locations without a temperature
        // tab fall back to the rain variant (handled below).
        var hasTemp = loc.HasTab("temperature");
        var skillHref = hasTemp ? "skill-temperature.html" : "skill-rainfall.html";
        var modelsHref = hasTemp ? "models-temp.html" : "models-rain.html";
        (string Href, string Label, string PageId)? Map(string tab) => tab.ToLowerInvariant() switch
        {
            "overview"     => ("index.html",                "Overview",     "overview"),
            "temperature"  => ("forecasts-temp-24h.html",   "Temperature",  "temperature"),
            "rain"         => ("forecasts-rain-24h.html",   "Rain",         "rain"),
            "wind"         => ("forecasts-wind-24h.html",   "Wind",         "wind"),
            "dry_window"   => ("forecasts-dry-window.html", "Dry window",   "dry-window"),
            "skill"        => (skillHref,                   "Skill",        "skill"),
            "models"       => (modelsHref,                  "Models",       "models"),
            _ => null,
        };

        var items = new StringBuilder();
        foreach (var tab in loc.Tabs)
        {
            var mapped = Map(tab);
            if (mapped is null) continue;
            var (href, label, mappedPageId) = mapped.Value;
            items.Append(Ci, $"""<li><a href="{href}"{NavActive(pageId, mappedPageId)}>{Escape(label)}</a></li>""");
        }

        // Location switcher — render only when more than one location is
        // configured. Sits below the tab nav so it doesn't dominate.
        var switcher = new StringBuilder();
        if (loc.Tabs.Count > 0)
        {
            // No-op when there's nothing to switch TO (single-location config).
        }

        return $"""
              <nav class="site-nav">
                <ul>{items}</ul>
              </nav>
            """;
    }

    /// <summary>
    /// Location switcher chip-row. Used by the top-level picker (<c>RenderHomePicker</c>)
    /// and rendered as a secondary nav on per-loc pages so a reader can
    /// jump from Bonehill to Membury without going via the home picker.
    /// Returns empty string when fewer than two locations exist.
    /// </summary>
    internal static string RenderLocationSwitcher(
        IReadOnlyList<LocationDescriptor> locations, string? activeLocName, string assetPrefix)
    {
        if (locations.Count < 2) return "";
        var items = new StringBuilder();
        foreach (var loc in locations)
        {
            var cls = string.Equals(loc.Name, activeLocName, StringComparison.OrdinalIgnoreCase)
                ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="{assetPrefix}{loc.Name}/index.html"{cls}>{Escape(loc.DisplayName)}</a></li>""");
        }
        return $"""<nav class="loc-switcher"><ul>{items}</ul></nav>""";
    }

    private static string NavActive(string current, string target)
        => current == target ? " class=\"active\"" : "";

    private static string FmtNullable(double? v, string fmt = "0.00")
        => v.HasValue ? v.Value.ToString(fmt, Ci) : "—";

    /// <summary>
    /// Map a station slug (e.g. <c>ea_bellever_dartmoor</c>) to a display name.
    /// Strips the data-source prefix and title-cases the rest.
    /// </summary>
    private static string PrettyStation(string slug)
    {
        if (string.IsNullOrWhiteSpace(slug)) return slug;
        var trimmed = slug.StartsWith("ea_", StringComparison.Ordinal) ? slug[3..] : slug;
        var parts = trimmed.Split('_', StringSplitOptions.RemoveEmptyEntries);
        return string.Join(' ', parts.Select(p => p.Length == 0 ? p : char.ToUpperInvariant(p[0]) + p[1..]));
    }

    private static string RenderEmptyChart(string title, string message) =>
        $"""<div class="chart-empty-box"><strong>{Escape(title)}</strong><p><em>{Escape(message)}</em></p></div>""";

    /// <summary>
    /// Short URL-safe slug for a station. Used to route <c>skill.html</c> /
    /// <c>skill-{slug}.html</c> and the dry-window equivalents. Returns the station
    /// id unchanged for anything we don't recognise — links would still work but
    /// look less tidy.
    /// </summary>
    internal static string StationSlug(string station) => station switch
    {
        "ea_bellever_dartmoor" => "bellever",
        "ea_dartmoor_nr_hexworthy" => "hexworthy",
        "ea_bovey_tracey" => "bovey",
        _ => station,
    };

    /// <summary>
    /// Render a per-station sub-nav for a section. The first station is the canonical
    /// one (its link is <c>{pageBase}.html</c>, matching the top-nav entry); the rest
    /// live at <c>{pageBase}-{slug}.html</c>. The current station is marked active.
    /// </summary>
    internal static string RenderStationSubNav(string pageBase, IReadOnlyList<string> stations, string currentStation)
    {
        if (stations.Count <= 1) return "";   // one station → sub-nav adds nothing

        var items = new StringBuilder();
        for (int i = 0; i < stations.Count; i++)
        {
            var s = stations[i];
            var href = i == 0 ? $"{pageBase}.html" : $"{pageBase}-{StationSlug(s)}.html";
            var cls = s == currentStation ? " class=\"active\"" : "";
            items.Append(Ci, $"""<li><a href="{href}"{cls}>{Escape(PrettyStation(s))}</a></li>""");
        }
        return $"""<nav class="lead-nav"><ul>{items}</ul></nav>""";
    }

    // RenderRainLocationSubNav removed in Phase D — location switching now
    // lives in the global chrome (top-nav "Locations" link + per-page
    // location-switcher), and Rain forecast / Skill pages live under their
    // location's URL prefix so the per-page pill row was redundant.

    /// <summary>
    /// Pretty-print a <see cref="WeatherBlend.Predict.Utci.UtciStress"/> enum name for
    /// the home-card chip. "ModerateHeat" → "Moderate heat"; "NoStress" → "No stress".
    /// </summary>
    internal static string PrettyUtciBand(string band)
    {
        if (string.IsNullOrEmpty(band)) return "";
        var sb = new StringBuilder(band.Length + 4);
        for (int i = 0; i < band.Length; i++)
        {
            var c = band[i];
            if (i == 0) sb.Append(char.ToUpper(c));
            else if (char.IsUpper(c)) { sb.Append(' '); sb.Append(char.ToLower(c)); }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Collapse hourly rainfall truth into a list of wet-run intervals
    /// (start, end), each in OADate. A wet hour is one with ≥ 0.1 mm rainfall —
    /// the same threshold the precip blender's training label uses. Consecutive
    /// wet hours merge into a single band so the skill page renders a continuous
    /// blue stripe rather than a row of abutting rectangles. End-of-run is the
    /// next hour's tick (start + N·1h), so a single wet hour at 14:00 covers
    /// the visual span 14:00..15:00.
    /// </summary>
    internal static List<(double XStart, double XEnd)> ComputeWetBands(
        IReadOnlyDictionary<DateTime, double> hourlyRainfallMm,
        DateTime windowStartUtc)
    {
        var bands = new List<(double, double)>();
        if (hourlyRainfallMm.Count == 0) return bands;
        var ordered = hourlyRainfallMm
            .Where(kv => kv.Key >= windowStartUtc)
            .OrderBy(kv => kv.Key)
            .ToList();
        DateTime? runStart = null;
        DateTime? runLast = null;
        foreach (var (hour, mm) in ordered)
        {
            if (mm >= 0.1)
            {
                if (runStart is null)
                {
                    runStart = hour;
                    runLast = hour;
                }
                else if (runLast is { } prev && hour == prev.AddHours(1))
                {
                    runLast = hour;
                }
                else
                {
                    // Gap — flush the previous run, start a new one.
                    bands.Add((runStart!.Value.ToOADate(), runLast!.Value.AddHours(1).ToOADate()));
                    runStart = hour;
                    runLast = hour;
                }
            }
            else if (runStart is not null)
            {
                bands.Add((runStart!.Value.ToOADate(), runLast!.Value.AddHours(1).ToOADate()));
                runStart = null;
                runLast = null;
            }
        }
        if (runStart is not null)
            bands.Add((runStart!.Value.ToOADate(), runLast!.Value.AddHours(1).ToOADate()));
        return bands;
    }

    private static string Escape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    /// <summary>
    /// CSS colour for a displayed temperature on a standard coolwarm
    /// (blue → white → red) ramp — the convention in scientific viz so a
    /// reader doesn't have to learn a custom palette. Anchors chosen for
    /// the Dartmoor climate: sub-zero is rare but plausible, 12°C is a
    /// typical-mild centre, and values above 25°C are heatwave territory.
    /// RGB-interpolated rather than HSL so the gradient walks the
    /// matplotlib coolwarm path directly without rotating through purple.
    /// </summary>
    internal static string TemperatureColor(double celsius)
    {
        // (t, r, g, b) anchors. The cold/white/hot triplet matches matplotlib
        // coolwarm (#3b4cc0, #f7f7f7, #b40426); the intermediate stops are
        // sampled along the same ramp so a 14°C cell reads as "lukewarm
        // pink", not "saturated halfway".
        (double t, int r, int g, int b)[] anchors =
        {
            (-5,  59,  76, 192),  // deep cobalt #3b4cc0
            ( 5, 146, 177, 244),  // light blue   #92b1f4
            (12, 247, 247, 247),  // near-white   #f7f7f7
            (18, 244, 154, 123),  // salmon       #f49a7b
            (25, 214,  82,  68),  // brick red    #d65244
            (32, 180,   4,  38),  // deep red     #b40426
        };
        if (double.IsNaN(celsius)) return "var(--pico-muted-color)";
        if (celsius <= anchors[0].t) return FormatRgb(anchors[0].r, anchors[0].g, anchors[0].b);
        if (celsius >= anchors[^1].t) return FormatRgb(anchors[^1].r, anchors[^1].g, anchors[^1].b);
        for (int i = 0; i < anchors.Length - 1; i++)
        {
            var a = anchors[i];
            var b = anchors[i + 1];
            if (celsius >= a.t && celsius <= b.t)
            {
                var k = (celsius - a.t) / (b.t - a.t);
                int r = (int)Math.Round(a.r + (b.r - a.r) * k);
                int g = (int)Math.Round(a.g + (b.g - a.g) * k);
                int bl = (int)Math.Round(a.b + (b.b - a.b) * k);
                return FormatRgb(r, g, bl);
            }
        }
        var last = anchors[^1];
        return FormatRgb(last.r, last.g, last.b);
    }

    private static string FormatRgb(int r, int g, int b)
        => $"rgb({r.ToString(Ci)} {g.ToString(Ci)} {b.ToString(Ci)})";

    /// <summary>
    /// CSS colour for a probability in <c>[0, 1]</c> on a "dry-window goodness"
    /// scale: 1.0 (dry block almost certain) is green, 0.0 (no dry block almost
    /// certain) is red, 0.5 sits at amber. Anchors are RGB-interpolated so the
    /// gradient walks smoothly across the table without jumping through brown.
    /// NaN renders as the muted text colour so missing cells don't shout.
    /// </summary>
    internal static string ProbabilityColor(double prob)
    {
        if (double.IsNaN(prob)) return "var(--pico-muted-color)";
        (double t, int r, int g, int b)[] anchors =
        {
            (0.00, 229,  57,  53),  // red    #e53935
            (0.50, 255, 167,  38),  // amber  #ffa726
            (1.00,  67, 160,  71),  // green  #43a047
        };
        if (prob <= anchors[0].t) return FormatRgb(anchors[0].r, anchors[0].g, anchors[0].b);
        if (prob >= anchors[^1].t) return FormatRgb(anchors[^1].r, anchors[^1].g, anchors[^1].b);
        for (int i = 0; i < anchors.Length - 1; i++)
        {
            var a = anchors[i];
            var b = anchors[i + 1];
            if (prob >= a.t && prob <= b.t)
            {
                var k = (prob - a.t) / (b.t - a.t);
                int r = (int)Math.Round(a.r + (b.r - a.r) * k);
                int g = (int)Math.Round(a.g + (b.g - a.g) * k);
                int bl = (int)Math.Round(a.b + (b.b - a.b) * k);
                return FormatRgb(r, g, bl);
            }
        }
        var last = anchors[^1];
        return FormatRgb(last.r, last.g, last.b);
    }

    /// <summary>
    /// CSS colour for a P(wet) value in <c>[0, 1]</c>. 
    /// Just the revers of ProbabilityColor for now
    /// </summary>
    internal static string PrecipProbColor(double prob)
    {
        if (double.IsNaN(prob)) return "var(--pico-muted-color)";
        (double t, int r, int g, int b)[] anchors =
        {
            (0.00,  67, 160,  71),  // green  #43a047
            (0.50, 255, 167,  38),  // amber  #ffa726
            (1.00, 229,  57,  53),  // red    #e53935
        };
        if (prob <= anchors[0].t) return FormatRgb(anchors[0].r, anchors[0].g, anchors[0].b);
        if (prob >= anchors[^1].t) return FormatRgb(anchors[^1].r, anchors[^1].g, anchors[^1].b);
        for (int i = 0; i < anchors.Length - 1; i++)
        {
            var a = anchors[i];
            var b = anchors[i + 1];
            if (prob >= a.t && prob <= b.t)
            {
                var k = (prob - a.t) / (b.t - a.t);
                int r = (int)Math.Round(a.r + (b.r - a.r) * k);
                int g = (int)Math.Round(a.g + (b.g - a.g) * k);
                int bl = (int)Math.Round(a.b + (b.b - a.b) * k);
                return FormatRgb(r, g, bl);
            }
        }
        var last = anchors[^1];
        return FormatRgb(last.r, last.g, last.b);
    }
}
