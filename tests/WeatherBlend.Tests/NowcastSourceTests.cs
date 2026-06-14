using FluentAssertions;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// The lead-0 "nowcast" sourcing rules and — critically — the anti-leak
/// property: at lead 0 the rainfall-persistence anchor must sit strictly
/// before valid-time so the predicted hour's own gauge reading can't enter
/// the feature vector.
/// </summary>
public class NowcastSourceTests
{
    [Fact]
    public void IsNowcast_only_lead_zero()
    {
        NowcastSource.IsNowcast(0).Should().BeTrue();
        NowcastSource.IsNowcast(24).Should().BeFalse();
        NowcastSource.IsNowcast(120).Should().BeFalse();
    }

    [Theory]
    [InlineData(0, "hist_forecast")]
    [InlineData(24, "offset_day")]
    [InlineData(120, "offset_day")]
    public void SurfaceRunTimeSource_switches_to_hist_forecast_only_at_lead0(int lead, string expected)
        => NowcastSource.SurfaceRunTimeSource(lead).Should().Be(expected);

    [Theory]
    [InlineData(0, "hist_forecast")]
    [InlineData(24, "exact")]
    [InlineData(72, "exact")]
    public void UpperAirRunTimeSource_switches_to_hist_forecast_only_at_lead0(int lead, string expected)
        => NowcastSource.UpperAirRunTimeSource(lead).Should().Be(expected);

    [Fact]
    public void PersistenceAnchorHours_is_identity_for_long_leads_and_backshifts_lead0()
    {
        // ≥24h leads unchanged (max(lead, lag) == lead); lead 0 back-shifts to lag.
        NowcastSource.PersistenceAnchorHours(24).Should().Be(24);
        NowcastSource.PersistenceAnchorHours(120).Should().Be(120);
        NowcastSource.PersistenceAnchorHours(0).Should().Be(NowcastSource.MinPersistenceLagHours);
        NowcastSource.MinPersistenceLagHours.Should().BeGreaterThan(0,
            "a zero lag at lead 0 would fold the predicted hour's own rain into persistence");
    }

    [Fact]
    public void Lead0_persistence_anchor_excludes_the_predicted_hours_own_rain()
    {
        // The leak that the anchor exists to prevent: heavy rain AT valid (and
        // the couple of hours just before it, inside the lag window), zero
        // elsewhere. The persistence window is (runTime − N, runTime] with
        // runTime INCLUSIVE, so anchoring at valid would sum the 5mm into
        // Prev24hMm. Anchoring at valid − lag must exclude it entirely.
        var valid = new DateTime(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 96; h++) hourly[valid.AddHours(-h)] = 0.0;
        hourly[valid] = 5.0;
        hourly[valid.AddHours(-1)] = 5.0;
        hourly[valid.AddHours(-2)] = 5.0;

        var lag = NowcastSource.PersistenceAnchorHours(0);
        var anchored = PrecipRichFeatureBuilder.ComputePersistence(hourly, valid.AddHours(-lag));
        // Window (valid−lag−24, valid−lag] excludes valid, valid−1, valid−2.
        anchored.Prev24hMm.Should().Be(0.0, "the lag must exclude the predicted hour and its immediate neighbours");
        anchored.WetHoursLast24h.Should().Be(0);

        // Sanity: anchoring AT valid (the leak) WOULD include the 5mm.
        var leaked = PrecipRichFeatureBuilder.ComputePersistence(hourly, valid);
        leaked.Prev24hMm.Should().BeApproximately(15.0, 1e-9);
    }
}
