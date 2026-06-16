using FluentAssertions;
using WeatherBlend.Config;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Per-location rock-surface override merge (Sennen sea-cliff extension,
/// docs/SENNEN_ROCK_TEMP_PLAN.md S2): a location's rockSurface: block shadows
/// only the fields it sets; everything else inherits the global block, and a
/// null override leaves the global config untouched (Bonehill/Membury path).
/// </summary>
public class RockSurfaceConfigTests
{
    [Fact]
    public void NullOverride_ReturnsGlobalUnchanged()
    {
        var global = new RockSurfaceConfig { FSky = 0.9, MuScale = 0.35 };

        var resolved = global.ResolveFor(null);

        resolved.Should().BeSameAs(global);
        resolved.Enabled.Should().BeTrue();
    }

    [Fact]
    public void Override_ShadowsOnlySetFields()
    {
        var global = new RockSurfaceConfig(); // defaults = spike PARAMS
        var o = new RockSurfaceOverrideConfig { Enabled = false, FSky = 0.5 };

        var resolved = global.ResolveFor(o);

        resolved.Enabled.Should().BeFalse();
        resolved.FSky.Should().Be(0.5);

        // Everything unset inherits the global values.
        resolved.Albedo.Should().Be(global.Albedo);
        resolved.EpsRock.Should().Be(global.EpsRock);
        resolved.Lambda.Should().Be(global.Lambda);
        resolved.Rho.Should().Be(global.Rho);
        resolved.CpRock.Should().Be(global.CpRock);
        resolved.TauDaySeconds.Should().Be(global.TauDaySeconds);
        resolved.TauLongSeconds.Should().Be(global.TauLongSeconds);
        resolved.LwClearK.Should().Be(global.LwClearK);
        resolved.LwCloudK.Should().Be(global.LwCloudK);
        resolved.MuScale.Should().Be(global.MuScale);
        resolved.Substeps.Should().Be(global.Substeps);
        resolved.SpinupHours.Should().Be(global.SpinupHours);
        resolved.GreasyMarginC.Should().Be(global.GreasyMarginC);

        // Surface-water model: the global physics coefficients carry through.
        resolved.SurfaceWaterEnabled.Should().Be(global.SurfaceWaterEnabled);
        resolved.MaxSurfaceWaterMm.Should().Be(global.MaxSurfaceWaterMm);
        resolved.WetThresholdMm.Should().Be(global.WetThresholdMm);
        resolved.EvapVpdCoeff.Should().Be(global.EvapVpdCoeff);
        resolved.EvapWindBase.Should().Be(global.EvapWindBase);
        resolved.EvapWindSlope.Should().Be(global.EvapWindSlope);
        resolved.EvapRadCoeff.Should().Be(global.EvapRadCoeff);
        resolved.LatentHeatWm2PerMmHr.Should().Be(global.LatentHeatWm2PerMmHr);
    }

    [Fact]
    public void SurfaceWaterEnabled_Override_FlipsPerLocation_WithGlobalCoeffsIntact()
    {
        // The enable switch is per-location overridable; the physics knobs stay
        // global (so a config.yaml coefficient change still reaches an
        // override location). Global off, location on.
        var global = new RockSurfaceConfig { EvapVpdCoeff = 0.02, LatentHeatWm2PerMmHr = 700.0 };
        global.SurfaceWaterEnabled.Should().BeFalse("default is off until calibrated");

        var resolved = global.ResolveFor(new RockSurfaceOverrideConfig { SurfaceWaterEnabled = true });

        resolved.SurfaceWaterEnabled.Should().BeTrue();
        resolved.EvapVpdCoeff.Should().Be(0.02, "physics coefficients come from the global block");
        resolved.LatentHeatWm2PerMmHr.Should().Be(700.0);
        global.SurfaceWaterEnabled.Should().BeFalse("the override must not mutate the global block");
    }

    [Fact]
    public void Faces_Override_ReplacesWholesale_AndDefaultsToEmpty()
    {
        var global = new RockSurfaceConfig();
        global.Faces.Should().BeEmpty("no faces = whole-crag horizontal mode");

        var o = new RockSurfaceOverrideConfig
        {
            Faces = new List<RockSurfaceFaceConfig>
            {
                new() { Name = "west", AspectDeg = 270 },
                new() { Name = "south", AspectDeg = 180, SlopeDeg = 75 },
            },
        };

        var resolved = global.ResolveFor(o);

        resolved.Faces.Should().HaveCount(2);
        resolved.Faces[0].SlopeDeg.Should().Be(90.0, "vertical wall is the default steepness");
        resolved.Faces[1].SlopeDeg.Should().Be(75.0);
        global.Faces.Should().BeEmpty("the global block must stay face-free for Bonehill");
    }

    [Fact]
    public void Override_DoesNotMutateGlobal()
    {
        var global = new RockSurfaceConfig();
        var o = new RockSurfaceOverrideConfig { FSky = 0.5, GreasyMarginC = 1.5 };

        var resolved = global.ResolveFor(o);

        resolved.Should().NotBeSameAs(global);
        global.FSky.Should().Be(1.0);
        global.GreasyMarginC.Should().Be(3.0);
        resolved.GreasyMarginC.Should().Be(1.5);
    }
}
