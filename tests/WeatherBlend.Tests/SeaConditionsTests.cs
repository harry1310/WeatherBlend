using FluentAssertions;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Sennen ONSHORE-SPRAY quality factor for the climbing index. After the
/// 2026-06-16 de-dup this factor carries ONLY spray (onshore wind × wave
/// energy) — tide + run-up moved wholly to the Alerts badge (SeaStateBadge).
/// No ground truth (a heuristic), so these pin the qualitative behaviour Harry
/// asked for: spray needs onshore wind + strength + waves together; offshore
/// or flat seas don't fire it.
/// </summary>
public class SeaConditionsTests
{
    // Sennen draft thresholds (config.yaml): onshore sector 225–335°.
    private static readonly SeaStateBadgeSpec Spec =
        new(TideHighMsl: 1.71, RunUpProxy: 12.0, WindMinMph: 15.0,
            WindSectorFromDeg: 225, WindSectorToDeg: 335);

    [Fact]
    public void Missing_spray_inputs_yield_no_factor()
    {
        SeaConditions.Evaluate(null, null, null, Spec).Should().BeNull();
        // Waves + wind but no direction → can't tell onshore, so omit.
        SeaConditions.Evaluate(2.0, 30, null, Spec).Should().BeNull();
    }

    [Fact]
    public void Offshore_or_calm_scores_near_perfect_and_is_named_spray()
    {
        // Small waves, light offshore (E, 90°) wind → no spray.
        var f = SeaConditions.Evaluate(waveHeightM: 0.3, windMph: 8, windDirDeg: 90, spec: Spec)!.Value;
        f.Name.Should().Be("Spray");
        f.Score.Should().BeGreaterThan(0.9);
        f.Detail.Should().Be("no onshore spray");
    }

    [Fact]
    public void Strong_onshore_wind_over_big_swell_drives_spray()
    {
        // 30 mph from the W (280°, in sector) over 2.5 m waves → heavy spray.
        var f = SeaConditions.Evaluate(waveHeightM: 2.5, windMph: 30, windDirDeg: 280, spec: Spec)!.Value;
        f.Score.Should().BeLessThan(0.3, "strong onshore wind + big waves = heavy spray");
        f.Detail.Should().Contain("spray");
    }

    [Fact]
    public void Offshore_wind_lofts_no_spray_even_when_strong()
    {
        var onshore  = SeaConditions.Evaluate(2.0, 30, 280, Spec)!.Value;  // from W = onshore
        var offshore = SeaConditions.Evaluate(2.0, 30, 90, Spec)!.Value;   // from E = offshore
        offshore.Score.Should().BeGreaterThan(onshore.Score);
        offshore.Detail.Should().Be("no onshore spray");
    }

    [Fact]
    public void Spray_needs_waves_not_just_wind()
    {
        // Gale onshore but a flat sea (0.2 m) → nothing to break into spray.
        var f = SeaConditions.Evaluate(0.2, 35, 280, Spec)!.Value;
        f.Detail.Should().Be("no onshore spray");
    }

    [Fact]
    public void Spray_factor_drags_the_overall_conditions_verdict()
    {
        // A cold, dry, light-wind day that would otherwise be fine, but heavy
        // onshore spray pulls the verdict down and surfaces as the limit.
        var midday = new DateTime(2026, 6, 21, 12, 0, 0, DateTimeKind.Utc);
        var sea = SeaConditions.Evaluate(2.5, 32, 280, Spec);
        var r = ClimbingConditions.Evaluate(
            midday, 50.08, -5.70, airTempC: 9, pWet: 0.05, windMph: 6, rock: null, sea: sea);
        r.Tier.Should().BeOneOf(ConditionsTier.Marginal, ConditionsTier.Poor);
        r.Reason.Should().Contain("spray");
        r.Factors.Select(f => f.Name).Should().Contain("Spray");
    }
}
