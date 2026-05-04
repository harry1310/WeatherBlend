using System.Globalization;
using System.Text;
using WeatherBlend.Models;

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

    public sealed record SiteInputs
    {
        public required string LocationDisplay { get; init; }
        public required double Latitude { get; init; }
        public required double Longitude { get; init; }
        public required double ElevationMeters { get; init; }

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
        /// (e.g. Princetown after the 2026-05-04 Bovey swap) doesn't render
        /// just because its historical parquets are still on disk. Empty
        /// set falls back to "render everything seen in the predictions"
        /// for backward-compat with older drives.
        /// </summary>
        public IReadOnlyCollection<string> ActiveStationSlugs { get; init; }
            = new HashSet<string>(StringComparer.Ordinal);

        /// <summary>
        /// Temperature <c>Manifest.Current</c> — the official champion. When set, the home
        /// page filters its forecast cards to this version so results are deterministic
        /// even while a challenger is also active. Empty string → no filter (fall back to
        /// "latest by PredictionMadeAt" across all versions).
        /// </summary>
        public string CurrentVersion { get; init; } = "";

        /// <summary>
        /// Per-station precipitation champion (<c>StationEntry.Current</c>) keyed by EA
        /// station slug (e.g. <c>ea_bellever_dartmoor</c>). The home page pulls the P(wet)
        /// chip from this version only, so a challenger that happens to write the same
        /// (lead, valid_time) never leaks onto the headline card.
        /// </summary>
        public IReadOnlyDictionary<string, string> PrecipCurrentByStation { get; init; }
            = new Dictionary<string, string>();

        /// <summary>
        /// Per-model held-out test scores (Blend vs ERA5 / EA rainfall) for every active
        /// blender, grouped by composite. One entry per (target, optional station, optional
        /// window) × version. Drives the Models page; empty when no training metadata is
        /// on disk (first-render, missing rclone sync, etc.).
        /// </summary>
        public IReadOnlyList<ModelSummary> ModelSummaries { get; init; }
            = Array.Empty<ModelSummary>();

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

        /// <summary>
        /// Structured weekly verify history loaded from
        /// <c>data/reports/verify_*.json</c>. One entry per (target, asOf)
        /// run; the Models-page renderer filters per-card on (target,
        /// station, version, windowHours) to populate a "Verify history"
        /// table beneath each model card. Empty until the first weekly
        /// verify run lands JSON sidecars on R2.
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
        string? ConformalSetTag = null);

    public sealed record FeelsLikeForecastPoint(
        string Version,
        DateTime PredictedAtUtc,
        DateTime ValidTimeUtc,
        int LeadHours,
        double UtciC,
        string Band,
        double ApparentC);

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
        // Phase 3g aleatoric uncertainty (null on 3b/3e/3c rows). Summary of
        // the per-MC-sample longest-dry-run distribution; narrow P10–P90 →
        // headline P(any block) is robust across MC realisations under
        // independence; wide → fragile. Defaulted to null so existing
        // positional construction sites compile unchanged.
        double? McMeanLongestDryRunHours = null,
        double? McP10LongestDryRunHours = null,
        double? McP50LongestDryRunHours = null,
        double? McP90LongestDryRunHours = null,
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
        double ProbabilityPercent);

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
        double? PrecipitationProbabilityPercent);

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
        double ConditionalProb,
        double CalibratedProb,
        double DailyProbAnyBlock);

    public static string Stylesheet() => """
        :root { --brand: #7c4dff; --pwet: #0288d1; }
        body > main { padding-top: 1rem; padding-bottom: 3rem; }
        nav.site-nav { padding: 0.5rem 0 1rem; border-bottom: 1px solid var(--pico-muted-border-color); margin-bottom: 1.5rem; }
        nav.site-nav ul { display: flex; gap: 1rem; list-style: none; padding: 0; margin: 0; }
        nav.site-nav a { text-decoration: none; }
        nav.site-nav a.active { font-weight: 600; color: var(--brand); }

        .forecast-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
        /* Flex layout so the footer pins to the cell's lower edge regardless of
           whether the feels-like chip is present — cards with shorter content
           (e.g. 96/120h leads with no feels-like row) used to float the footer
           up, leaving made-time misaligned across the grid. */
        .forecast-card { padding: 0.75rem 0.85rem; display: flex; flex-direction: column; }
        .forecast-card h3 { margin: 0 0 0.1rem; font-size: 1rem; }
        .forecast-card header small { color: var(--pico-muted-color); font-size: 0.8rem; }
        .forecast-card .temp { font-size: 2rem; font-weight: 700; color: var(--temp-color, var(--brand)); line-height: 1.1; margin: 0.35rem 0 0.25rem; }
        .forecast-card-empty .temp { color: var(--pico-muted-color); }
        .forecast-card .pwet { font-size: 0.9rem; color: var(--pwet); margin: 0 0 0.35rem; font-variant-numeric: tabular-nums; }
        .forecast-card .pwet strong { font-weight: 700; }
        .forecast-card .pwet .rain { margin-left: 0.25rem; }
        .forecast-card .feels { font-size: 0.85rem; color: var(--pico-muted-color); margin: 0 0 0.35rem; font-variant-numeric: tabular-nums; }
        .forecast-card .feels strong { font-weight: 700; }
        .forecast-card .feels small { margin-left: 0.35rem; }
        .forecast-card footer { margin-top: auto; padding-top: 0.4rem; border-top: 1px solid var(--pico-muted-border-color); }
        .forecast-card footer small { color: var(--pico-muted-color); font-size: 0.75rem; }

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
        table td.num.delta-good { color: #2e7d32; }
        table td.num.delta-bad  { color: #c62828; }
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
          function fmtDate(ms, kind) {
            const d = new Date(ms);
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
          function oaToMs(oa) { return (oa - 25569) * 86400000; }

          function build(canvas) {
            if (!window.Chart) return;
            let cfg;
            try { cfg = JSON.parse(canvas.getAttribute('data-cjs')); }
            catch (e) { return; }
            if (!cfg || !cfg.datasets || !cfg.datasets.length) return;

            const ycfg = { dec: cfg.yDec, suffix: cfg.ySuffix, trim: cfg.yTrim };
            const xKind = cfg.xKind || 'date';

            const datasets = cfg.datasets.map(ds => ({
              label: ds.label,
              data: ds.points.map(p => ({ x: oaToMs(p[0]), y: p[1] })),
              borderColor: ds.color,
              backgroundColor: ds.color,
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
                    xMin: oaToMs(b[0]),
                    xMax: oaToMs(b[1]),
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
                  xMin: oaToMs(ann.todayX),
                  xMax: oaToMs(ann.todayX),
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

            new Chart(canvas, {
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
                    min: cfg.xMin != null ? oaToMs(cfg.xMin) : undefined,
                    max: cfg.xMax != null ? oaToMs(cfg.xMax) : undefined,
                    title: { display: !!cfg.xLabel, text: cfg.xLabel || '' },
                    ticks: {
                      autoSkip: true, maxTicksLimit: 8,
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
        return $$"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>{{Escape(pageTitle)}} — WeatherBlend</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
              <link rel="stylesheet" href="styles.css">
              <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
              <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
              <script src="chart.js" defer></script>
            </head>
            <body>
              <main class="container">
                <header>
                  <hgroup>
                    <h1>WeatherBlend</h1>
                    <p>Multi-model forecast blending for {{Escape(input.LocationDisplay)}}</p>
                  </hgroup>
                  <nav class="site-nav">
                    <ul>
                      <li><a href="index.html"{{NavActive(pageId, "index")}}>Home</a></li>
                      <li><a href="forecasts-temp-24h.html"{{NavActive(pageId, "forecasts")}}>Forecasts</a></li>
                      <li><a href="skill-temperature.html"{{NavActive(pageId, "skill")}}>Skill</a></li>
                      <li><a href="models-temp.html"{{NavActive(pageId, "models")}}>Models</a></li>
                      <li><a href="about.html"{{NavActive(pageId, "about")}}>About</a></li>
                    </ul>
                  </nav>
                </header>

                {{bodyHtml}}

                <footer class="site-footer">
                  Rendered {{input.GeneratedAtUtc.ToString("yyyy-MM-dd HH:mm", Ci)}}Z ·
                  Training truth: ERA5 · Verification: ERA5 + METAR ·
                  <a href="about.html">About this site</a>
                </footer>
              </main>
            </body>
            </html>
            """;
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
        "ea_princetown" => "princetown",     // legacy — Princetown is no longer in active config (2026-05-04)
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
