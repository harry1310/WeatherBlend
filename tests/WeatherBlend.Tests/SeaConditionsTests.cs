using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Sennen sea-state quality factor (2026-06-13) — tide / run-up / ONSHORE
/// SPRAY folded into one graded 0–1 conditions factor. No ground truth (a
/// heuristic, like the sea-state badge it shares thresholds with), so these
/// pin the qualitative behaviour Harry asked for: spray needs onshore wind +
/// strength + waves together, and the worst sub-factor names the verdict.
/// </summary>
public class SeaConditionsTests
{
    // Sennen draft thresholds (config.yaml): onshore sector 225–335°.
    private static readonly SeaStateBadgeSpec Spec =
        new(TideHighMsl: 1.71, RunUpProxy: 12.0, WindMinMph: 15.0,
            WindSectorFromDeg: 225, WindSectorToDeg: 335);

    [Fact]
    public void No_inputs_yields_no_factor()
        => SeaConditions.Evaluate(null, null, null, null, null, Spec).Should().BeNull();

    [Fact]
    public void Calm_sea_scores_near_perfect()
    {
        // Low tide, tiny long-period swell, light offshore wind.
        var f = SeaConditions.Evaluate(
            tideHeightMsl: 0.2, waveHeightM: 0.3, swellPeriodS: 6,
            windMph: 8, windDirDeg: 90 /* offshore (E) */, Spec)!.Value;
        f.Score.Should().BeGreaterThan(0.9);
        f.Detail.Should().Be("calm");
    }

    [Fact]
    public void Strong_onshore_wind_over_big_swell_drives_spray_and_names_it()
    {
        // 30 mph from the W (280°, in sector) over 2.5 m swell → heavy spray.
        var f = SeaConditions.Evaluate(
            tideHeightMsl: 0.5, waveHeightM: 2.5, swellPeriodS: 8,
            windMph: 30, windDirDeg: 280, Spec)!.Value;
        f.Score.Should().BeLessThan(0.3, "strong onshore wind + big waves = heavy spray");
        f.Detail.Should().Contain("spray");
    }

    [Fact]
    public void Offshore_wind_lofts_no_spray_even_when_strong()
    {
        // Same big waves + strong wind, but blowing OFFSHORE (from the E, 90°):
        // spray blows out to sea, not onto the rock. Run-up may still bite, but
        // the spray term itself must vanish.
        var onshore = SeaConditions.Evaluate(0.5, 2.0, 7, 30, 280, Spec)!.Value;
        var offshore = SeaConditions.Evaluate(0.5, 2.0, 7, 30, 90, Spec)!.Value;
        offshore.Score.Should().BeGreaterThan(onshore.Score);
        offshore.Detail.Should().NotContain("spray");
    }

    [Fact]
    public void Spray_needs_waves_not_just_wind()
    {
        // Gale onshore but a flat sea (0.2 m) → nothing to break into spray.
        var f = SeaConditions.Evaluate(0.4, 0.2, 5, 35, 280, Spec)!.Value;
        f.Detail.Should().NotContain("spray");
    }

    [Fact]
    public void High_tide_alone_penalises_and_names_tide()
    {
        var f = SeaConditions.Evaluate(
            tideHeightMsl: 2.4, waveHeightM: 0.3, swellPeriodS: 6,
            windMph: 6, windDirDeg: 90, Spec)!.Value;
        f.Score.Should().BeLessThan(0.5);
        f.Detail.Should().Contain("tide");
    }

    [Fact]
    public void Sea_factor_drags_the_overall_conditions_verdict()
    {
        // A cold, dry, light-wind day that would otherwise be fine, but heavy
        // onshore spray must pull the verdict down and surface as the limit.
        var midday = new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc);
        var sea = SeaConditions.Evaluate(0.5, 2.5, 9, 32, 280, Spec);
        var r = ClimbingConditions.Evaluate(
            midday, 50.08, -5.70, airTempC: 9, pWet: 0.05, windMph: 6, rock: null, sea: sea);
        r.Tier.Should().BeOneOf(ConditionsTier.Marginal, ConditionsTier.Poor);
        r.Reason.Should().Contain("sea");
        r.Factors.Select(f => f.Name).Should().Contain("Sea");
    }
}
