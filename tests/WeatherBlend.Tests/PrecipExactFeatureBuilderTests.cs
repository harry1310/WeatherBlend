using FluentAssertions;
using WeatherBlend.Train.PrecipExact;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Tests for the precip exact-runtime feature builder. AIFS deliberately
/// excluded from canonical model order due to the units bug found
/// 2026-05-05 (parser fix lives in EcmwfClient; existing R2 data being
/// re-backfilled separately). When AIFS is reinstated, update these tests.
/// </summary>
public class PrecipExactFeatureBuilderTests
{
    [Fact]
    public void CanonicalModelOrder_includes_AIFS_after_units_fix()
    {
        PrecipExactFeatureBuilder.CanonicalModelOrder.Should().BeEquivalentTo(
            "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global");
    }

    [Fact]
    public void AllTiers_has_one_first_cut_tier()
    {
        PrecipExactFeatureBuilder.AllTiers.Should().HaveCount(1);
        var p1 = PrecipExactFeatureBuilder.AllTiers[0];
        p1.Name.Should().Be("P1");
        p1.Required.Should().BeEquivalentTo("gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper");
        p1.Optional.Should().BeEquivalentTo("met_office_global");
    }

    [Fact]
    public void BuildSpec_default_lead_12_columns_have_precip_prefix()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        spec.Target.Should().Be("precipitation");
        spec.LeadHours.Should().Be(12);
        spec.FeatureNames.Should().StartWith(new[] { "precip_gfs", "precip_ifs", "precip_aifs", "precip_moglobal" });
        // 4 per-model + 3 spread + 4 calendar = 11 features
        spec.FeatureNames.Should().HaveCount(11);
        spec.FeatureNames.Should().Contain("precip_mean");
    }

    [Fact]
    public void BuildSpec_lead_24_metadata()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0], targetLead: 24);
        spec.LeadHours.Should().Be(24);
        spec.FeatureSet.Should().Contain("l24");
    }

    [Fact]
    public void ShortName_round_trips_and_unknown_throws()
    {
        foreach (var m in PrecipExactFeatureBuilder.CanonicalModelOrder)
        {
            var sn = PrecipExactFeatureBuilder.ShortName(m);
            sn.Should().NotBeNullOrEmpty();
        }
        var ex = Record.Exception(() => PrecipExactFeatureBuilder.ShortName("not_a_real_model"));
        ex.Should().BeOfType<ArgumentException>();
    }

    [Fact]
    public void ComposeRow_features_in_declared_order()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 1.5, 2.0, 1.8, 0.8 },
            era5Precip: 1.7);
        row.Features.Should().HaveCount(spec.FeatureCount);
        // BinaryTrainingRow: Label is bool (= ERA5 precip ≥ 0.1 mm/h),
        // TruthMmHour carries the actual mm value as diagnostic.
        row.Label.Should().BeTrue(); // 1.7 mm ≥ 0.1 wet
        row.TruthMmHour.Should().Be(1.7f);
        row.Features[0].Should().Be(1.5f); // gfs
        row.Features[1].Should().Be(2.0f); // ifs
        row.Features[2].Should().Be(1.8f); // aifs
        row.Features[3].Should().Be(0.8f); // moglobal
    }

    [Fact]
    public void ComposeRow_label_false_when_below_wet_threshold()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 0.05, 0.0, 0.02, 0.0 },
            era5Precip: 0.05); // 0.05 < 0.1 = dry
        row.Label.Should().BeFalse();
        row.TruthMmHour.Should().Be(0.05f);
    }

    [Fact]
    public void ComposeRow_NaN_safe_when_optional_model_missing()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        // MO Global (optional) NaN — spread should be over GFS + IFS + AIFS only
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 1.5, 2.0, 1.0, double.NaN },
            era5Precip: 1.7);
        // Mean across {1.5, 2.0, 1.0} = 1.5
        var meanIdx = spec.FeatureNames.ToList().IndexOf("precip_mean");
        row.Features[meanIdx].Should().BeApproximately(1.5f, 1e-3f);
    }
}
