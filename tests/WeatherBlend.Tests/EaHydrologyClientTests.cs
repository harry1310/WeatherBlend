using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using Xunit;

namespace WeatherBlend.Tests;

public class EaHydrologyClientTests
{
    private const string Loc = "bonehill_rocks";
    private static readonly RainfallStationConfig Station = new()
    {
        Id = "723a8fc4-908b-4430-91c7-9990be86540a_363307",
        Name = "Bellever Dartmoor"
    };

    private static JsonElement Payload(string json) => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void Parse_maps_timestamp_value_and_quality_fields()
    {
        var json = """
        {
          "items": [
            {"dateTime":"2024-06-01T00:00:00","value":0.0,"quality":"Good","completeness":"Complete"},
            {"dateTime":"2024-06-01T00:15:00","value":0.2,"quality":"Good","completeness":"Complete"}
          ]
        }
        """;

        var rows = EaHydrologyClient.Parse(Payload(json), Loc, Station);

        rows.Should().HaveCount(2);
        rows[0].LocationName.Should().Be(Loc);
        rows[0].StationId.Should().Be(Station.Id);
        rows[0].StationName.Should().Be(Station.Name);
        rows[0].ObservedTimeUtc.Should().Be(new DateTime(2024, 6, 1, 0, 0, 0, DateTimeKind.Utc));
        rows[0].ObservedTimeUtc.Kind.Should().Be(DateTimeKind.Utc);
        rows[0].Value15MinMm.Should().BeApproximately(0.0, 1e-9);
        rows[0].Quality.Should().Be("Good");
        rows[0].Completeness.Should().Be("Complete");

        rows[1].ObservedTimeUtc.Should().Be(new DateTime(2024, 6, 1, 0, 15, 0, DateTimeKind.Utc));
        rows[1].Value15MinMm.Should().BeApproximately(0.2, 1e-9);
    }

    [Fact]
    public void Parse_preserves_suspect_and_unchecked_quality_flags_verbatim()
    {
        var json = """
        {
          "items": [
            {"dateTime":"2024-12-05T03:00:00","value":0.0,"quality":"Suspect","completeness":"Incomplete"},
            {"dateTime":"2024-12-05T03:15:00","value":0.4,"quality":"Unchecked","completeness":"Complete"},
            {"dateTime":"2024-12-05T03:30:00","value":1.1,"quality":"Estimated","completeness":"Complete"}
          ]
        }
        """;

        var rows = EaHydrologyClient.Parse(Payload(json), Loc, Station);

        rows.Select(r => r.Quality).Should().Equal("Suspect", "Unchecked", "Estimated");
        rows.Select(r => r.Completeness).Should().Equal("Incomplete", "Complete", "Complete");
    }

    [Fact]
    public void Parse_allows_missing_value_to_be_null()
    {
        // EA occasionally returns readings with no value (quality=Missing).
        // Value field may be null or absent entirely.
        var json = """
        {
          "items": [
            {"dateTime":"2024-06-01T10:00:00","value":null,"quality":"Missing","completeness":"Incomplete"},
            {"dateTime":"2024-06-01T10:15:00","quality":"Missing","completeness":"Incomplete"}
          ]
        }
        """;

        var rows = EaHydrologyClient.Parse(Payload(json), Loc, Station);

        rows.Should().HaveCount(2);
        rows[0].Value15MinMm.Should().BeNull();
        rows[0].Quality.Should().Be("Missing");
        rows[1].Value15MinMm.Should().BeNull();
        rows[1].Quality.Should().Be("Missing");
    }

    [Fact]
    public void Parse_returns_empty_when_items_array_missing_or_empty()
    {
        EaHydrologyClient.Parse(Payload("""{"meta":{}}"""), Loc, Station).Should().BeEmpty();
        EaHydrologyClient.Parse(Payload("""{"items":[]}"""), Loc, Station).Should().BeEmpty();
    }

    [Fact]
    public void Parse_interprets_unzoned_timestamps_as_utc()
    {
        // EA returns unzoned ISO strings ("2024-06-01T00:00:00"); API spec says UTC.
        // If we accidentally treated them as local time, this test would fail on
        // any non-UTC machine (including CI runners in other TZs).
        var json = """
        {
          "items": [
            {"dateTime":"2024-06-01T12:00:00","value":0.1,"quality":"Good","completeness":"Complete"}
          ]
        }
        """;

        var rows = EaHydrologyClient.Parse(Payload(json), Loc, Station);

        rows.Should().ContainSingle();
        rows[0].ObservedTimeUtc.Kind.Should().Be(DateTimeKind.Utc);
        rows[0].ObservedTimeUtc.Hour.Should().Be(12);
    }

    // A WeatherLink gauge (Lands End) has no EA measure id, so it used to build
    // ".../measures/-rainfall-t-900-mm-qualified/..." — a URL that can never resolve, yet
    // still burned three 60s attempt-timeouts before the resilience handler gave up. Three
    // minutes of dead wait per collect cycle, which is what tipped the 2026-08-12 02:45Z and
    // 08:45Z runs past their 30-minute job limit during that morning's EA outage. Callers
    // now filter via LocationConfig.EaRainfallStations; this guard is the backstop, and it
    // must throw BEFORE any HTTP call so a future miswire surfaces on the first run.
    [Theory]
    [InlineData(RainfallTruthSource.WeatherLink, "")]                  // Lands End as configured
    [InlineData(RainfallTruthSource.WeatherLink, "some-guid")]         // wrong source even with an id
    [InlineData(RainfallTruthSource.Ea, "")]                           // EA source but a blank id
    [InlineData(RainfallTruthSource.Ea, "   ")]                        // whitespace-only id
    public async Task FetchAsync_rejects_a_station_with_no_EA_measure_id(
        RainfallTruthSource source, string id)
    {
        var client = new EaHydrologyClient(new HttpClient(), NullLogger<EaHydrologyClient>.Instance);
        var station = new RainfallStationConfig { Id = id, Name = "Lands End", Source = source };

        var act = async () => await client.FetchAsync(
            "sennen_cove", station, new DateOnly(2026, 8, 12), new DateOnly(2026, 8, 12), CancellationToken.None);

        (await act.Should().ThrowAsync<ArgumentException>())
            .WithMessage("*no measure id*");
    }

    [Fact]
    public async Task FetchAsync_accepts_a_normal_EA_station_past_the_guard()
    {
        // Sanity check the guard isn't over-eager: a properly configured EA gauge must get
        // past it and fail (if at all) on the network, never on the argument check.
        var client = new EaHydrologyClient(new HttpClient { Timeout = TimeSpan.FromMilliseconds(1) },
                                           NullLogger<EaHydrologyClient>.Instance);

        var act = async () => await client.FetchAsync(
            Loc, Station, new DateOnly(2026, 8, 12), new DateOnly(2026, 8, 12), CancellationToken.None);

        await act.Should().NotThrowAsync<ArgumentException>();
    }
}
