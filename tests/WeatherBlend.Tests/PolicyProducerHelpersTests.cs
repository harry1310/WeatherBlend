using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Config;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Fast coverage for the per-lead policy producers' helper methods in
/// <see cref="PrecipCrossLeadBakeoffCommand"/>. These DuckDB-reader helpers had
/// ZERO test coverage — the fit commands that use them aren't exercised by any
/// fast test or smoke — which let a regression slip through on 2026-06-15 (a
/// cleanup pass dropped the braces off the readers' <c>while (r.Read()) { … }</c>
/// loops; it built clean and the whole suite passed, because nothing ran them).
/// Each test below writes a tiny parquet fixture and asserts the reader returns
/// the right rows: a mangled read loop returns the wrong count (or throws), so
/// these fail loudly instead of silently shipping a crash to CI.
/// </summary>
public sealed class PolicyProducerHelpersTests : IDisposable
{
    private readonly string _root;

    public PolicyProducerHelpersTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-policy-helpers-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private async Task<string> SeedAsync<T>(string subdir, IReadOnlyList<T> rows)
    {
        var dir = Path.Combine(_root, subdir);
        Directory.CreateDirectory(dir);
        await ParquetSerializer.SerializeAsync(rows, Path.Combine(dir, "data.parquet"));
        return dir;
    }

    // ---- ResolveLocation (the --location override I/O the 3 producers share) ----

    [Fact]
    public void ResolveLocation_resolves_named_blank_and_unknown()
    {
        var cfg = new AppConfig
        {
            Locations =
            {
                new LocationConfig { Name = "bonehill_rocks" },
                new LocationConfig { Name = "membury_devon" },
            },
        };
        var cmd = new PrecipCrossLeadBakeoffCommand(
            NullLogger<PrecipCrossLeadBakeoffCommand>.Instance, cfg);

        cmd.ResolveLocation("membury_devon")!.Name.Should().Be("membury_devon");
        cmd.ResolveLocation(null)!.Name.Should().Be("bonehill_rocks");        // blank = primary
        cmd.ResolveLocation("  ")!.Name.Should().Be("bonehill_rocks");        // whitespace = primary
        cmd.ResolveLocation("MEMBURY_DEVON")!.Name.Should().Be("membury_devon"); // case-insensitive
        cmd.ResolveLocation("nope").Should().BeNull();                        // unknown = null (exit-2)
    }

    // ---- LoadEra5Temp ----

    private sealed record Era5Fix(string LocationName, DateTime ValidTimeUtc, double? Temperature2m);

    [Fact]
    public async Task LoadEra5Temp_returns_in_window_rows_for_the_location_ordered()
    {
        var b = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var path = await SeedAsync("era5", new[]
        {
            new Era5Fix("L", b.AddHours(3), 12.0),   // in window
            new Era5Fix("L", b.AddHours(1), 10.0),   // in window (earlier — must sort first)
            new Era5Fix("L", b.AddHours(2), 11.0),   // in window
            new Era5Fix("L", b,             9.0),    // == earliest → EXCLUDED (strictly >)
            new Era5Fix("L", b.AddHours(99), 99.0),  // after latest → excluded
            new Era5Fix("OTHER", b.AddHours(2), 50.0), // other location → excluded
            new Era5Fix("L", b.AddHours(2), null),   // null temp → excluded by IS NOT NULL (dup valid)
        });

        var rows = PrecipCrossLeadBakeoffCommand.LoadEra5Temp(
            path, "L", earliest: b, latest: b.AddHours(72), CancellationToken.None);

        // 3 in-window non-null rows for L, ordered by valid time. A mangled read
        // loop would return 1 row (or throw).
        rows.Select(r => r.TempC).Should().Equal(10.0, 11.0, 12.0);
        rows[0].Valid.Should().Be(b.AddHours(1));
    }

    // ---- LoadHourlyTruth ----

    private sealed record RainFix(string LocationName, string StationName, DateTime ObservedTimeUtc, double? Value15MinMm);

    [Fact]
    public async Task LoadHourlyTruth_sums_complete_hours_only()
    {
        var b = new DateTime(2026, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        RainFix R(int hour, int min, double mm) => new("L", "S", b.AddHours(hour).AddMinutes(min), mm);
        var path = await SeedAsync("rain", new[]
        {
            // hour 0: 4 complete 15-min readings → kept, summed (0.1+0.2+0.3+0.4 = 1.0)
            R(0, 0, 0.1), R(0, 15, 0.2), R(0, 30, 0.3), R(0, 45, 0.4),
            // hour 1: only 3 readings → dropped by HAVING COUNT(*) = 4
            R(1, 0, 5.0), R(1, 15, 5.0), R(1, 30, 5.0),
            // other station, complete hour → excluded by StationName filter
            new RainFix("L", "OTHER", b.AddMinutes(0), 9.0),
        });

        var rows = PrecipCrossLeadBakeoffCommand.LoadHourlyTruth(
            path, "L", "S", earliest: b, latest: b.AddHours(24), CancellationToken.None);

        rows.Should().HaveCount(1);
        rows[0].Valid.Should().Be(b);
        rows[0].Mm.Should().BeApproximately(1.0, 1e-9);
    }

    // ---- LoadHourlyTruthWeatherLink (WeatherLink cove gauge — e.g. Lands End for Sennen) ----

    private sealed record WlFix(string LocationName, DateTime ObservedTimeUtc, double? RainfallMm);

    [Fact]
    public async Task LoadHourlyTruthWeatherLink_reads_the_cove_gauge_the_EA_reader_cannot_serve()
    {
        var b = new DateTime(2026, 4, 1, 0, 0, 0, DateTimeKind.Utc);
        var wlPath = await SeedAsync("weatherlink", new[]
        {
            // Already-hourly RainfallMm — summed per hour, NO 4-of-4 completeness rule.
            new WlFix("lands_end", b,             0.5),
            new WlFix("lands_end", b.AddHours(1), 1.2),
            new WlFix("lands_end", b.AddHours(2), 0.0),
            new WlFix("gwennap_head", b, 9.0),   // other location → excluded by LocationName filter
        });
        // The EA tree carries the OTHER Sennen gauge (Trengwainton), not Lands End.
        var eaPath = await SeedAsync("rain", new[]
        {
            new RainFix("sennen_cove", "Trengwainton", b, 0.1),
            new RainFix("sennen_cove", "Trengwainton", b.AddMinutes(15), 0.1),
            new RainFix("sennen_cove", "Trengwainton", b.AddMinutes(30), 0.1),
            new RainFix("sennen_cove", "Trengwainton", b.AddMinutes(45), 0.1),
        });

        var wl = PrecipCrossLeadBakeoffCommand.LoadHourlyTruthWeatherLink(
            wlPath, "lands_end", earliest: b, latest: b.AddHours(24), CancellationToken.None);
        wl.Select(r => r.Mm).Should().Equal(0.5, 1.2, 0.0);

        // A WeatherLink station must flow through its OWN source: the EA reader can't serve
        // Lands End (it isn't in the EA tree). Routing a WeatherLink gauge through the EA
        // slug/truth is exactly what yielded 0 study rows and crashed the 2026-06-24
        // lead-policy fit — so this guards that regression.
        PrecipCrossLeadBakeoffCommand.LoadHourlyTruth(
            eaPath, "sennen_cove", "Lands End", earliest: b, latest: b.AddHours(24), CancellationToken.None)
            .Should().BeEmpty();
    }

    // ---- QueryLatestTempRows (the worst-broken helper: pivot + freshest-cycle dedup) ----

    private sealed record FcFix(
        string LocationName, string Model, DateTime RunTimeUtc, string? RunTimeSource, DateTime ValidTimeUtc,
        double? Temperature2m, double? DewPoint2m, double? RelativeHumidity2m, double? CloudCover,
        double? CloudCoverLow, double? CloudCoverMid, double? CloudCoverHigh,
        double? WindSpeed10m, double? WindDirection10m, double? WindGusts10m, double? SurfacePressure);

    [Fact]
    public async Task QueryLatestTempRows_picks_freshest_cycle_within_lead_bound()
    {
        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var model = canon[0];                       // a real canonical model id (passes the Model IN filter)
        var b = new DateTime(2026, 3, 1, 0, 0, 0, DateTimeKind.Utc);
        var valid = b.AddHours(48);
        const int lead = 24;                        // require RunTime <= valid − 24h = b+24h

        FcFix F(DateTime runTime, string? src, string mdl, double temp) =>
            new("L", mdl, runTime, src, valid, temp, 5, 80, 50, 10, 20, 30, 4, 180, 8, 1010);

        var path = await SeedAsync("fc", new[]
        {
            F(b.AddHours(0),  null, model, 10.0),         // valid cycle (older)
            F(b.AddHours(12), null, model, 20.0),         // valid cycle (FRESHEST within bound) → wins
            F(b.AddHours(30), null, model, 99.0),         // RunTime > valid−24h → excluded by lead bound
            F(b.AddHours(12), "offset_day", model, 88.0), // historical → excluded
            F(b.AddHours(12), null, "not_a_real_model", 77.0), // non-canonical → excluded by Model IN
        });

        var pivot = PrecipCrossLeadBakeoffCommand.QueryLatestTempRows(
            path, "L", earliestValid: b, latestValid: b.AddHours(72), asOfRunTime: b.AddHours(200),
            leadHoursLowerBound: lead, canonOrder: canon, ct: CancellationToken.None);

        pivot.Should().ContainKey(valid);
        var slot = canon.IndexOf(model);
        pivot[valid].Temp[slot].Should().Be(20.0,
            "the freshest cycle still satisfying RunTime <= valid−lead wins (ROW_NUMBER dedup), and the "
            + "too-fresh / offset_day / non-canonical rows are filtered out");
    }
}
