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
    public void AllTiers_has_p1_and_p2()
    {
        // P2 added 2026-05-07 as a no-IFS challenger to P1 — see
        // PrecipExactFeatureBuilder for the bake-off rationale. P1 stays as
        // index 0 to keep the existing default-tier behaviour stable for
        // every other test in this file.
        PrecipExactFeatureBuilder.AllTiers.Should().HaveCount(2);
        var p1 = PrecipExactFeatureBuilder.AllTiers[0];
        p1.Name.Should().Be("P1");
        p1.Required.Should().BeEquivalentTo("gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper");
        p1.Optional.Should().BeEquivalentTo("met_office_global");
        var p2 = PrecipExactFeatureBuilder.AllTiers[1];
        p2.Name.Should().Be("P2");
        p2.Required.Should().BeEquivalentTo("gfs_ncep", "ecmwf_aifs_oper");
        p2.Optional.Should().BeEquivalentTo("met_office_global");
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
            truthMmHour: 1.7);
        row.Features.Should().HaveCount(spec.FeatureCount);
        // BinaryTrainingRow: Label is bool (= truth precip ≥ 0.1 mm/h —
        // EA Hydrology gauge in production); TruthMmHour carries the actual
        // mm value as diagnostic.
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
            truthMmHour: 0.05); // 0.05 < 0.1 = dry
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
            truthMmHour: 1.7);
        // Mean across {1.5, 2.0, 1.0} = 1.5
        var meanIdx = spec.FeatureNames.ToList().IndexOf("precip_mean");
        row.Features[meanIdx].Should().BeApproximately(1.5f, 1e-3f);
    }

    // ----- UKV-included variant (2026-05-05) -------------------------------------

    [Fact]
    public void BuildSpec_with_UKV_appends_one_extra_feature_named_precip_ukv()
    {
        var noUkv = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        var withUkv = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0], includeUkv: true);

        withUkv.FeatureCount.Should().Be(noUkv.FeatureCount + 1);
        withUkv.FeatureNames.Should().Contain("precip_ukv");
        // precip_ukv slot sits immediately after the per-model precip block (4),
        // before the spread features — matches the temperature builder's ordering.
        withUkv.FeatureNames[noUkv.Models.Count].Should().Be("precip_ukv");
        // FeatureSet tag carries the -ukv suffix so persisted schemas are
        // self-describing about which variant produced them.
        withUkv.FeatureSet.Should().EndWith("-ukv");
    }

    [Fact]
    public void ComposeRow_with_UKV_places_ukv_value_after_per_model_block()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0], includeUkv: true);
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 1.5, 2.0, 1.8, 0.8 },
            truthMmHour: 1.7,
            ukvPrecip: 1.2);
        var ukvIdx = spec.FeatureNames.ToList().IndexOf("precip_ukv");
        row.Features[ukvIdx].Should().Be(1.2f);
    }

    [Fact]
    public void ComposeRow_with_UKV_NaN_when_unset_and_spread_unaffected()
    {
        // UKV is always-optional. When it's NaN, spread features (mean/std/range)
        // must still reflect ONLY the per-model precip block — UKV in the
        // spread would change semantics whenever someone toggles --include-ukv.
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0], includeUkv: true);
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 1.5, 2.0, 1.0, 1.5 },
            truthMmHour: 1.5,
            ukvPrecip: double.NaN);
        var ukvIdx = spec.FeatureNames.ToList().IndexOf("precip_ukv");
        float.IsNaN(row.Features[ukvIdx]).Should().BeTrue();
        // Mean across {1.5, 2.0, 1.0, 1.5} = 1.5; UKV must NOT participate
        var meanIdx = spec.FeatureNames.ToList().IndexOf("precip_mean");
        row.Features[meanIdx].Should().BeApproximately(1.5f, 1e-3f);
    }
}
