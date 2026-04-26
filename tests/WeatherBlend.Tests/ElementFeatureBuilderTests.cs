using FluentAssertions;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Common;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using WeatherBlend.Train.Element.Wind;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Per-element feature-builder smoke tests. Mirrors <see cref="FeatureBuilderTests"/>
/// for the temperature blender — verifies feature ordering, spread math, and cyclical
/// encodings on small fixtures so a regression in any of the four shows up at CI time.
/// </summary>
public class ElementFeatureBuilderTests
{
    private static readonly DateTime Noon =
        new(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc);

    // ---------- Wind ----------

    [Fact]
    public void Wind_PublicFeatureNames_count_22_in_expected_order()
    {
        WindFeatureBuilder.PublicFeatureNames.Should().HaveCount(22);
        WindFeatureBuilder.PublicFeatureNames.Should().StartWith(new[]
        {
            "wind_spd_gfs", "wind_spd_ecmwf", "wind_spd_icon", "wind_spd_ukmo", "wind_spd_gem",
        });
        WindFeatureBuilder.PublicFeatureNames.Should().EndWith(new[]
        {
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        });
    }

    [Fact]
    public void Wind_ComposeRow_computes_speed_spread_correctly()
    {
        var speeds     = new float[] { 4f, 5f, 6f, 7f, 8f }; // mean=6, range=4
        var directions = new float[] { 0f, 90f, 180f, 270f, 360f };
        var row = WindFeatureBuilder.ComposeRow(Noon, speeds, directions, era5WindSpeed: 5.5f);

        row.SpdMean.Should().BeApproximately(6f, 1e-4f);
        row.SpdRange.Should().BeApproximately(4f, 1e-4f);
        // Population std of {4,5,6,7,8}: sqrt(((-2)^2+(-1)^2+0+1+4)/5) = sqrt(10/5) = sqrt(2) ≈ 1.4142
        row.SpdStd.Should().BeApproximately(1.4142f, 1e-3f);
    }

    [Fact]
    public void Wind_direction_encoded_as_unit_vector_components()
    {
        // Direction 0° (north) → sin=0, cos=1; 90° (east) → sin=1, cos=0; etc.
        var speeds = new float[] { 5f, 5f, 5f, 5f, 5f };
        var dirs   = new float[] { 0f, 90f, 180f, 270f, 360f };
        var row = WindFeatureBuilder.ComposeRow(Noon, speeds, dirs, era5WindSpeed: 5f);

        row.DirSinGfs.Should().BeApproximately(0f,   1e-5f);
        row.DirCosGfs.Should().BeApproximately(1f,   1e-5f);
        row.DirSinEcmwf.Should().BeApproximately(1f, 1e-5f);
        row.DirCosEcmwf.Should().BeApproximately(0f, 1e-5f);
        row.DirSinIcon.Should().BeApproximately(0f,  1e-4f);  // sin(180°) ≈ 0
        row.DirCosIcon.Should().BeApproximately(-1f, 1e-5f);
        row.DirSinUkmo.Should().BeApproximately(-1f, 1e-5f);
        row.DirCosUkmo.Should().BeApproximately(0f,  1e-4f);  // cos(270°) ≈ 0
    }

    [Fact]
    public void Wind_ComposeRow_throws_on_wrong_array_length()
    {
        var bad = new float[] { 1f, 2f, 3f };  // Only 3 instead of 5.
        var goodDirs = new float[] { 0f, 0f, 0f, 0f, 0f };
        var act = () => WindFeatureBuilder.ComposeRow(Noon, bad, goodDirs, era5WindSpeed: 0f);
        act.Should().Throw<ArgumentException>();
    }

    // ---------- Humidity ----------

    [Fact]
    public void Humidity_PublicFeatureNames_count_19_in_expected_order()
    {
        HumidityFeatureBuilder.PublicFeatureNames.Should().HaveCount(19);
        HumidityFeatureBuilder.PublicFeatureNames.Should().StartWith(new[]
        {
            "rh_gfs", "rh_ecmwf", "rh_icon", "rh_mf", "rh_ukmo", "rh_gem",
            "dp_gfs", "dp_ecmwf", "dp_icon", "dp_mf", "dp_ukmo", "dp_gem",
        });
    }

    [Fact]
    public void Humidity_ComposeRow_spread_is_over_RH_only_not_dewpoint()
    {
        var rh = new float[] { 60f, 65f, 70f, 75f, 80f, 85f }; // mean=72.5
        var dp = new float[] { 5f, 6f, 7f, 8f, 9f, 10f };       // dewpoint plays no role in spread feature
        var row = HumidityFeatureBuilder.ComposeRow(Noon, rh, dp, era5Rh: 70f);

        row.RhMean.Should().BeApproximately(72.5f, 1e-4f);
        row.RhRange.Should().BeApproximately(25f, 1e-4f);
    }

    // ---------- Radiation ----------

    [Fact]
    public void Radiation_PublicFeatureNames_count_25_in_expected_order()
    {
        RadiationFeatureBuilder.PublicFeatureNames.Should().HaveCount(25);
        RadiationFeatureBuilder.PublicFeatureNames.Should().Contain(new[]
        {
            "sw_gfs", "direct_gfs", "diffuse_gfs",
            "sw_mean", "sw_std", "sw_range",
        });
    }

    [Fact]
    public void Radiation_ComposeRow_spread_skips_NaN_UKMO()
    {
        // UKMO null at long lead → its slot carries NaN. Spread should be over the 5 present values.
        var sw      = new float[] { 100f, 200f, 300f, 400f, float.NaN, 600f };
        var direct  = new float[] { 80f,  150f, 220f, 290f, float.NaN, 480f };
        var diffuse = new float[] { 20f,   50f,  80f, 110f, float.NaN, 120f };

        var row = RadiationFeatureBuilder.ComposeRow(Noon, sw, direct, diffuse, era5Sw: 250f);

        // Mean of {100,200,300,400,600} = 320
        row.SwMean.Should().BeApproximately(320f, 1e-4f);
        // Range = 600 − 100 = 500
        row.SwRange.Should().BeApproximately(500f, 1e-4f);
        // UKMO slot stays as NaN — LightGBM treats it as missing.
        float.IsNaN(row.SwUkmo).Should().BeTrue();
    }

    // ---------- Cloud cover ----------

    [Fact]
    public void Cloud_PublicFeatureNames_count_13_total_only()
    {
        CloudFeatureBuilder.PublicFeatureNames.Should().HaveCount(13);
        CloudFeatureBuilder.PublicFeatureNames.Should().NotContain(n =>
            n.Contains("low") || n.Contains("mid") || n.Contains("high"));
    }

    [Fact]
    public void Cloud_ComposeRow_spread_correct()
    {
        var cc = new float[] { 0f, 25f, 50f, 75f, 100f, 50f }; // mean=50, range=100
        var row = CloudFeatureBuilder.ComposeRow(Noon, cc, era5Cc: 60f);

        row.CcMean.Should().BeApproximately(50f, 1e-4f);
        row.CcRange.Should().BeApproximately(100f, 1e-4f);
    }

    // ---------- Common helpers ----------

    [Fact]
    public void InterModelSpread_skips_NaN_entries()
    {
        Span<float> values = stackalloc float[] { 1f, 2f, float.NaN, 4f, 5f };
        var spread = InterModelSpread.From(values);
        spread.Mean.Should().BeApproximately(3f, 1e-4f);   // (1+2+4+5)/4 = 3
        spread.Range.Should().BeApproximately(4f, 1e-4f);  // 5 − 1 = 4
    }

    [Fact]
    public void InterModelSpread_returns_NaN_when_all_inputs_NaN()
    {
        Span<float> values = stackalloc float[] { float.NaN, float.NaN, float.NaN };
        var spread = InterModelSpread.From(values);
        float.IsNaN(spread.Mean).Should().BeTrue();
        float.IsNaN(spread.Std).Should().BeTrue();
        float.IsNaN(spread.Range).Should().BeTrue();
    }

    [Theory]
    [InlineData(0,   0.0,   1.0)]
    [InlineData(6,   1.0,   0.0)]
    [InlineData(12,  0.0,  -1.0)]
    [InlineData(18, -1.0,   0.0)]
    public void CalendarFeatures_hour_encoding_matches_unit_circle(int hour, double expectedSin, double expectedCos)
    {
        var cal = CalendarFeatures.From(new DateTime(2025, 1, 1, hour, 0, 0, DateTimeKind.Utc));
        cal.HourSin.Should().BeApproximately((float)expectedSin, 1e-5f);
        cal.HourCos.Should().BeApproximately((float)expectedCos, 1e-5f);
    }

    // ---- Per-lead model membership ----
    //
    // MétéoFrance live forecasts cut off ~36h, so humidity & cloud train+predict use
    // 5 models at lead 48/72h and the full 6 at lead 24h. UKMO radiation drops out at
    // lead ≥48h, so radiation excludes UKMO at long lead but keeps it at 24h. These
    // tests pin the per-lead membership lists so a future change to ModelsForLead is
    // a deliberate edit rather than an accident.

    // UKMO is structurally ~26% null at every lead in Previous Runs (same valid-times
    // missing regardless of lead). Inconsistent presence creates training noise — empirical
    // test showed dropping UKMO improved every blender. So UKMO is excluded from training
    // for every element at every lead. The remaining per-lead variation is just MF: humidity
    // and cloud have MF at 24h (full coverage) but not at 48/72h (live API horizon).

    [Fact]
    public void Humidity_ModelsForLead_24h_is_5_models_no_UKMO()
    {
        HumidityFeatureBuilder.ModelsForLead(24).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless",
        });
    }

    [Theory]
    [InlineData(48)]
    [InlineData(72)]
    public void Humidity_ModelsForLead_long_lead_is_4_models_no_UKMO_no_MF(int lead)
    {
        HumidityFeatureBuilder.ModelsForLead(lead).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
        });
    }

    [Fact]
    public void Cloud_ModelsForLead_24h_is_5_models_no_UKMO()
    {
        CloudFeatureBuilder.ModelsForLead(24).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless",
        });
    }

    [Theory]
    [InlineData(48)]
    [InlineData(72)]
    public void Cloud_ModelsForLead_long_lead_is_4_models_no_UKMO_no_MF(int lead)
    {
        CloudFeatureBuilder.ModelsForLead(lead).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
        });
    }

    [Theory]
    [InlineData(24)]
    [InlineData(48)]
    [InlineData(72)]
    public void Radiation_ModelsForLead_is_5_models_no_UKMO_at_every_lead(int lead)
    {
        // Radiation can keep MF at every lead (MF is fully populated for radiation in
        // Previous Runs at all leads, and live MF radiation is also present where MF
        // covers — its 36h horizon doesn't typically truncate radiation predict).
        RadiationFeatureBuilder.ModelsForLead(lead).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless",
        });
    }

    [Theory]
    [InlineData(24)]
    [InlineData(48)]
    [InlineData(72)]
    public void Wind_ModelsForLead_is_4_models_no_UKMO_no_MF_at_every_lead(int lead)
    {
        // Wind already excluded MF (no MF wind data anywhere on Open-Meteo) and now also
        // excludes UKMO (~26% null pattern). 4-model-everywhere matches the cleanest
        // per-blender pattern.
        WindFeatureBuilder.ModelsForLead(lead).Should().BeEquivalentTo(new[]
        {
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
        });
    }
}
