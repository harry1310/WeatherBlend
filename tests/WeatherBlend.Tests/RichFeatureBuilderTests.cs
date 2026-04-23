using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class RichFeatureBuilderTests
{
    [Fact]
    public void FeatureNames_has_exactly_88_columns()
    {
        // 13 lean + 66 per-model secondaries (10 vars × 6 models, with wind_dir split to sin+cos = 60+12) + 9 aggregates.
        RichFeatureBuilder.FeatureNames.Should().HaveCount(88);
    }

    [Fact]
    public void FeatureNames_first_13_match_lean_layout_verbatim()
    {
        RichFeatureBuilder.FeatureNames.Take(13).Should()
            .Equal(FeatureBuilder.FeatureNames);
    }

    [Fact]
    public void FeatureNames_uses_sin_cos_columns_for_wind_direction()
    {
        RichFeatureBuilder.FeatureNames.Should().Contain("wind_dir_sin_gfs");
        RichFeatureBuilder.FeatureNames.Should().Contain("wind_dir_cos_gem");
        // Raw wind_dir per-model columns must NOT exist — the model sees sin/cos only.
        RichFeatureBuilder.FeatureNames.Should().NotContain("wind_dir_gfs");
    }

    [Fact]
    public void FeatureNames_ends_with_nine_cross_model_aggregates()
    {
        RichFeatureBuilder.FeatureNames.TakeLast(9).Should().Equal(
            "dew_mean", "dew_std",
            "rh_mean", "rh_std",
            "cloud_mean",
            "wind_speed_mean", "wind_speed_std",
            "pressure_mean", "pressure_std");
    }

    [Fact]
    public void ComposeRow_wind_dir_becomes_sin_cos_per_model()
    {
        // North = 0°; east = 90°; pick known angles for 6 models.
        var dirs = new[] { 0.0, 90.0, 180.0, 270.0, 45.0, 135.0 };
        var row = Compose(windDirs: dirs);

        row.WindDirSinGfs.Should().BeApproximately(0f, 1e-5f);   // sin(0) = 0
        row.WindDirCosGfs.Should().BeApproximately(1f, 1e-5f);   // cos(0) = 1

        row.WindDirSinEcmwf.Should().BeApproximately(1f, 1e-5f); // sin(90°)
        row.WindDirCosEcmwf.Should().BeApproximately(0f, 1e-5f); // cos(90°)

        row.WindDirSinIcon.Should().BeApproximately(0f,  1e-5f);  // sin(180°)
        row.WindDirCosIcon.Should().BeApproximately(-1f, 1e-5f);  // cos(180°)
    }

    [Fact]
    public void ComposeRow_aggregates_skip_NaN_values()
    {
        // 4 values present, 2 NaN → mean over the 4 present values.
        var dew = new[] { 10.0, 12.0, double.NaN, 14.0, double.NaN, 16.0 };
        var row = Compose(dewPoints: dew);

        row.DewMean.Should().BeApproximately(13f, 1e-4f);          // (10+12+14+16)/4
        // Population std over [10,12,14,16]: mean=13; var = ((-3)²+(-1)²+1²+3²)/4 = 5; std = √5 ≈ 2.2360
        row.DewStd.Should().BeApproximately(2.2360f, 1e-3f);
    }

    [Fact]
    public void ComposeRow_aggregate_is_NaN_when_every_model_missing_the_variable()
    {
        var allNaN = new[] { double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN };
        var row = Compose(dewPoints: allNaN);

        float.IsNaN(row.DewMean).Should().BeTrue();
        float.IsNaN(row.DewStd).Should().BeTrue();
    }

    [Fact]
    public void ComposeRow_preserves_per_model_NaN_on_secondary()
    {
        var rh = new[] { 80.0, double.NaN, 85.0, 75.0, double.NaN, 90.0 };
        var row = Compose(rhs: rh);

        float.IsNaN(row.RhEcmwf).Should().BeTrue();   // index 1
        float.IsNaN(row.RhUkmo).Should().BeTrue();    // index 4
        row.RhGfs.Should().BeApproximately(80f, 1e-4f);
        row.RhGem.Should().BeApproximately(90f, 1e-4f);
    }

    [Fact]
    public void ComposeRow_populates_label_and_lean_13_from_inputs()
    {
        var row = Compose();

        // Lean features must still be present, identical to FeatureBuilder output.
        row.TempGfs.Should().BeApproximately(10f, 1e-4f);
        row.TempMean.Should().BeApproximately(12.5f, 1e-4f);
        row.Era5Temp.Should().BeApproximately(15f, 1e-4f);
    }

    // Shared builder — each test overrides only what it needs.
    private static RichTrainingRow Compose(
        double[]? temps      = null,
        double[]? dewPoints  = null,
        double[]? rhs        = null,
        double[]? clouds     = null,
        double[]? cloudLows  = null,
        double[]? cloudMids  = null,
        double[]? cloudHighs = null,
        double[]? windSpeeds = null,
        double[]? windDirs   = null,
        double[]? windGusts  = null,
        double[]? pressures  = null,
        double    era5Temp   = 15.0)
        => RichFeatureBuilder.ComposeRow(
            validTimeUtc: new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            temps:        temps      ?? new[] { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 },
            dewPoints:    dewPoints  ?? new[] {  5.0,  6.0,  7.0,  8.0,  9.0, 10.0 },
            rhs:          rhs        ?? new[] { 80.0, 82.0, 84.0, 86.0, 88.0, 90.0 },
            clouds:       clouds     ?? new[] { 50.0, 60.0, 70.0, 40.0, 30.0, 20.0 },
            cloudLows:    cloudLows  ?? new[] { 10.0, 15.0, 20.0, 25.0, 30.0, 35.0 },
            cloudMids:    cloudMids  ?? new[] { 20.0, 25.0, 30.0, 15.0, 10.0,  5.0 },
            cloudHighs:   cloudHighs ?? new[] { 20.0, 20.0, 20.0,  0.0,  0.0,  0.0 },
            windSpeeds:   windSpeeds ?? new[] {  5.0,  6.0,  4.0,  7.0,  5.5,  6.5 },
            windDirsDeg:  windDirs   ?? new[] {180.0,190.0,200.0,210.0,220.0,230.0 },
            windGusts:    windGusts  ?? new[] { 10.0, 12.0,  8.0, 15.0, 11.0, 13.0 },
            pressures:    pressures  ?? new[] {1010.0,1012.0,1008.0,1014.0,1011.0,1009.0},
            era5Temp:     era5Temp);
}
