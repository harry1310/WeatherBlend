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
            "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global", "gefs_ncep_mean");
    }

    [Fact]
    public void AllTiers_has_p1_and_p2()
    {
        // P2 added 2026-05-07 as a no-IFS challenger to P1 — see
        // PrecipExactFeatureBuilder for the bake-off rationale. P1 stays as
        // index 0 to keep the existing default-tier behaviour stable for
        // every other test in this file. GEFS appended as Optional in both
        // tiers 2026-05-09 (mirrors temp 2d wiring). IFS moved from P1's
        // Required to Optional 2026-05-22 — P1 now requires only the
        // 4-cycle-publishing pair (GFS + AIFS), matching temp 2d's T2.
        PrecipExactFeatureBuilder.AllTiers.Should().HaveCount(2);
        var p1 = PrecipExactFeatureBuilder.AllTiers[0];
        p1.Name.Should().Be("P1");
        p1.Required.Should().BeEquivalentTo("gfs_ncep", "ecmwf_aifs_oper");
        p1.Optional.Should().BeEquivalentTo("ecmwf_ifs_oper", "met_office_global", "gefs_ncep_mean");
        var p2 = PrecipExactFeatureBuilder.AllTiers[1];
        p2.Name.Should().Be("P2");
        p2.Required.Should().BeEquivalentTo("gfs_ncep", "ecmwf_aifs_oper");
        p2.Optional.Should().BeEquivalentTo("met_office_global", "gefs_ncep_mean");
    }

    [Fact]
    public void BuildSpec_default_lead_12_columns_have_precip_prefix()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        spec.Target.Should().Be("precipitation");
        spec.LeadHours.Should().Be(12);
        spec.FeatureNames.Should().StartWith(new[] { "precip_gfs", "precip_ifs", "precip_aifs", "precip_moglobal", "precip_gefsmean" });
        // 5 per-model + 3 spread + 4 calendar = 12 features
        spec.FeatureNames.Should().HaveCount(12);
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
            perModelPrecip: new double[] { 1.5, 2.0, 1.8, 0.8, 1.6 },
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
        row.Features[4].Should().Be(1.6f); // gefsmean
    }

    [Fact]
    public void ComposeRow_label_false_when_below_wet_threshold()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 0.05, 0.0, 0.02, 0.0, 0.04 },
            truthMmHour: 0.05); // 0.05 < 0.1 = dry
        row.Label.Should().BeFalse();
        row.TruthMmHour.Should().Be(0.05f);
    }

    [Fact]
    public void ComposeRow_NaN_safe_when_optional_model_missing()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(PrecipExactFeatureBuilder.AllTiers[0]);
        // MO Global + GEFS (both optional) NaN — spread should be over GFS + IFS + AIFS only
        var row = PrecipExactFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelPrecip: new double[] { 1.5, 2.0, 1.0, double.NaN, double.NaN },
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
            perModelPrecip: new double[] { 1.5, 2.0, 1.8, 0.8, 1.6 },
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
            perModelPrecip: new double[] { 1.5, 2.0, 1.0, 1.5, 1.5 },
            truthMmHour: 1.5,
            ukvPrecip: double.NaN);
        var ukvIdx = spec.FeatureNames.ToList().IndexOf("precip_ukv");
        float.IsNaN(row.Features[ukvIdx]).Should().BeTrue();
        // Mean across {1.5, 2.0, 1.0, 1.5} = 1.5; UKV must NOT participate
        var meanIdx = spec.FeatureNames.ToList().IndexOf("precip_mean");
        row.Features[meanIdx].Should().BeApproximately(1.5f, 1e-3f);
    }

    // ----- Upper-air predict-time parity (2026-06-02) ----------------------------
    // UpperAirValuesFromPerModel must reproduce the trainer's model-major
    // (spec.Models × UaPressureCols) + t850_mean/rh850_mean order, so the live 3d
    // predict feeds the model its trained UA columns (column-parity bug = silent
    // prediction corruption).
    [Fact]
    public void UpperAirValuesFromPerModel_matches_training_model_major_order_with_means()
    {
        var spec = PrecipExactFeatureBuilder.BuildSpec(
            PrecipExactFeatureBuilder.AllTiers[0], targetLead: 24, includeUkv: false, withUpperAir: true);
        int mc = spec.Models.Count;
        // 10 pressure cols/model in UaPressureColumnNames order: t850 = idx 0,
        // rh850 = idx 9. Set t850=10+k, rh850=80+k per model; the rest 0.
        var perModel = new Dictionary<string, double[]>();
        for (int k = 0; k < mc; k++)
        {
            var vals = new double[10];
            vals[0] = 10 + k;   // t850
            vals[9] = 80 + k;   // rh850
            perModel[spec.Models[k]] = vals;
        }

        var ua = PrecipExactFeatureBuilder.UpperAirValuesFromPerModel(spec, perModel);
        ua.Length.Should().Be(10 * mc + 2);
        for (int k = 0; k < mc; k++)
        {
            ua[10 * k + 0].Should().Be(10 + k, "t850 is model-major col 0");
            ua[10 * k + 9].Should().Be(80 + k, "rh850 is model-major col 9");
        }
        ua[10 * mc].Should().BeApproximately(Enumerable.Range(0, mc).Average(k => 10.0 + k), 1e-9);     // t850_mean
        ua[10 * mc + 1].Should().BeApproximately(Enumerable.Range(0, mc).Average(k => 80.0 + k), 1e-9); // rh850_mean

        // Models absent from the dict → all-NaN slots (graceful; LightGBM-missing).
        var uaNaN = PrecipExactFeatureBuilder.UpperAirValuesFromPerModel(spec, new Dictionary<string, double[]>());
        uaNaN.Length.Should().Be(10 * mc + 2);
        uaNaN.Take(10 * mc).Should().OnlyContain(x => double.IsNaN(x));
        double.IsNaN(uaNaN[10 * mc]).Should().BeTrue();
    }
}
