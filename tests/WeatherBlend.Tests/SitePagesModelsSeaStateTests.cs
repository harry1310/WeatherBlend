using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Sea-state Models page tests (models-sea-state.html):
///   * the wave_height_lgb bundle card renders version / trained-at /
///     training rows + window / cal-slice MAE + band coverage / truth source
///     and the any-lead note
///   * the page degrades gracefully when the bundle (WaveModelCard) or the
///     verify sidecars are absent
///   * the buoy verify-history table renders MAE + coverage per lead bucket
///     from wave_height sidecars, scoped to this location + phase only
///   * the Models sub-nav gains the Sea state entry only for locations with
///     the sea_state tab
/// </summary>
public class SitePagesModelsSeaStateTests
{
    private static readonly DateTime GeneratedAt = new(2026, 6, 12, 12, 0, 0, DateTimeKind.Utc);

    private static SitePages.LocationDescriptor Sennen(IReadOnlyList<string>? tabs = null) =>
        new(
            Name: "sennen_cove",
            DisplayName: "Sennen, Cornwall",
            RainStationSlugs: new[] { "ea_trengwainton" },
            IsPrimary: false,
            Tabs: tabs ?? new[] { "overview", "temperature", "rain", "wind", "sea_state", "skill", "models" },
            OverviewFirstVisibleHourUtc: 6,
            OverviewLastVisibleHourUtcExclusive: 21);

    private static SitePages.WaveModelCard Card() => new(
        Version: "v2026-06-11_220452_wave_height_lgb",
        Phase: "wave_height_lgb",
        TrainedAtUtc: new DateTime(2026, 6, 11, 22, 4, 52, DateTimeKind.Utc),
        RowsTrain: 31046,
        RowsVal: 3881,
        RowsCal: 3881,
        WindowStart: "2022-01-01 00:00:00",
        WindowEnd: "2026-06-05 23:00:00",
        FeatureCount: 51,
        TruthSource: "era5_ocean",
        CalMae: 0.1067,
        CalBandCoverage: 0.90,
        NominalCoverage: 0.90,
        LeadMode: "any-lead-v1",
        CalNote: "Trained on lead-unlabelled hindcast; per-lead calibration planned once the offset_day archive matures (~2026-09).");

    private static VerifyHistoryFile History(
        string station = "sennen_cove",
        string phase = "wave_height_lgb",
        double mae = 0.31,
        double? coverage = 0.88,
        bool drift = false) =>
        new()
        {
            Target = "wave_height",
            AsOfUtc = new DateTime(2026, 6, 15, 9, 30, 0, DateTimeKind.Utc),
            WindowDays = 30,
            LatencyDays = 0,
            MetricLabel = "Hs MAE (m)",
            Rows = new List<VerifyHistoryRow>
            {
                new()
                {
                    Station = station,
                    ModelVersion = "v2026-06-11_220452_wave_height_lgb",
                    Phase = phase,
                    LeadHours = 24,
                    N = 53,
                    BlendMetric = mae,
                    DriftFlag = drift,
                    BandCoverage = coverage,
                },
                new()
                {
                    Station = station,
                    ModelVersion = "v2026-06-11_220452_wave_height_lgb",
                    Phase = phase,
                    LeadHours = 48,
                    N = 41,
                    BlendMetric = mae + 0.05,
                    DriftFlag = false,
                    BandCoverage = coverage,
                },
            },
        };

    private static SitePages.SiteInputs MakeInput(
        SitePages.LocationDescriptor loc,
        SitePages.WaveModelCard? card,
        IReadOnlyList<VerifyHistoryFile>? history = null) =>
        new()
        {
            ActiveLocationName = loc.Name,
            RenderingFor = loc,
            AssetPrefix = "../",
            LocationDisplay = loc.DisplayName,
            Latitude = 50.0786, Longitude = -5.7044, ElevationMeters = 35,
            MetarStation = "EGDR",
            GeneratedAtUtc = GeneratedAt,
            WindowStartUtc = GeneratedAt.AddDays(-30),
            Predictions = Array.Empty<TempPredictionRow>(),
            TruthByTime = new Dictionary<DateTime, double>(),
            MetarByTime = Array.Empty<(DateTime, double)>(),
            RollingMae = Array.Empty<SitePages.RollingMaePoint>(),
            PrecipPredictions = Array.Empty<SitePages.PrecipForecastPoint>(),
            DryWindowPredictions = Array.Empty<SitePages.DryWindowForecastPoint>(),
            Locations = new[] { loc },
            WaveModelCard = card,
            VerifyHistory = history ?? Array.Empty<VerifyHistoryFile>(),
        };

    [Fact]
    public void RenderModelsSeaState_renders_the_bundle_card()
    {
        var html = SitePages.RenderModelsSeaState(MakeInput(Sennen(), Card()));

        // Header: phase + version + trained date.
        html.Should().Contain("wave_height_lgb");
        html.Should().Contain("v2026-06-11_220452_wave_height_lgb");
        html.Should().Contain("2026-06-11");

        // Cal-slice metrics with the era5_ocean truth-source label.
        html.Should().Contain("0.107");          // cal MAE, 3 dp
        html.Should().Contain("0.900");          // cal band coverage (matched vs nominal 0.90)
        html.Should().Contain("era5_ocean");

        // Training rows + window + feature count.
        html.Should().Contain("31,046");
        html.Should().Contain("3,881");
        html.Should().Contain("2022-01-01");
        html.Should().Contain("2026-06-05");
        html.Should().Contain(">51<");

        // The any-lead-v1 note (bundle's own calibration note wins).
        html.Should().Contain("lead-unlabelled hindcast");
    }

    [Fact]
    public void RenderModelsSeaState_degrades_to_empty_state_without_a_bundle()
    {
        var html = SitePages.RenderModelsSeaState(MakeInput(Sennen(), card: null));

        html.Should().Contain("No wave_height bundle metadata on disk");
        // Sub-nav still renders so the reader can navigate away.
        html.Should().Contain("models-sea-state.html");
    }

    [Fact]
    public void RenderModelsSeaState_shows_no_runs_hint_before_first_verify()
    {
        var html = SitePages.RenderModelsSeaState(MakeInput(Sennen(), Card()));

        html.Should().Contain("no runs yet");
        html.Should().Contain("Sevenstones");
    }

    [Fact]
    public void RenderModelsSeaState_renders_verify_history_with_mae_and_coverage()
    {
        var html = SitePages.RenderModelsSeaState(
            MakeInput(Sennen(), Card(), new[] { History() }));

        html.Should().Contain("Buoy verify history");
        html.Should().Contain("0.310");          // ≤24h MAE
        html.Should().Contain("0.360");          // ≤48h MAE
        html.Should().Contain("cov 0.88");       // band coverage line
        html.Should().Contain("≤24h").And.Contain("≤48h");
        html.Should().NotContain("no runs yet");
        // The location-mismatch caveat ships with the table.
        html.Should().Contain("28 km");
    }

    [Fact]
    public void RenderModelsSeaState_ignores_other_locations_history()
    {
        var foreign = History(station: "some_other_cove", mae: 9.99);
        var html = SitePages.RenderModelsSeaState(
            MakeInput(Sennen(), Card(), new[] { foreign }));

        html.Should().NotContain("9.990");
        html.Should().Contain("no runs yet");
    }

    [Fact]
    public void RenderModelsSeaState_marks_drifted_cells()
    {
        var html = SitePages.RenderModelsSeaState(
            MakeInput(Sennen(), Card(), new[] { History(drift: true) }));

        html.Should().Contain("delta-bad");
    }

    [Fact]
    public void Models_sub_nav_gains_sea_state_entry_for_sennen_only()
    {
        // Sennen (has sea_state tab): the temperature models page links it.
        var sennenHtml = SitePages.RenderModels(MakeInput(Sennen(), Card()), "temperature");
        sennenHtml.Should().Contain("models-sea-state.html");

        // A Membury-shaped tab list: no Sea state entry anywhere.
        var membury = Sennen(tabs: new[] { "overview", "temperature", "rain", "skill", "models" });
        var memburyHtml = SitePages.RenderModels(MakeInput(membury, card: null), "temperature");
        memburyHtml.Should().NotContain("models-sea-state.html");
    }
}
