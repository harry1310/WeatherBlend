using FluentAssertions;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Tests for <see cref="Exact12hFeatureBuilder"/> — the exact-runtime
/// (non-Open-Meteo) blender feature pipeline used by the 2d temperature
/// blender. Pure logic tests; SQL-dependent paths (Build) are covered
/// by integration / bake-off runs against real data.
/// </summary>
public class Exact12hFeatureBuilderTests
{
    private static readonly Exact12hFeatureBuilder.TierSpec TestTier = Exact12hFeatureBuilder.AllTiers[1]; // T2

    [Fact]
    public void AllTiers_has_three_tiers_with_distinct_names()
    {
        Exact12hFeatureBuilder.AllTiers.Should().HaveCount(3);
        Exact12hFeatureBuilder.AllTiers.Select(t => t.Name).Should().BeEquivalentTo("T1", "T2", "T3");
    }

    [Fact]
    public void AllTiers_T1_requires_all_5_models()
    {
        var t1 = Exact12hFeatureBuilder.AllTiers.Single(t => t.Name == "T1");
        t1.Required.Should().BeEquivalentTo(
            "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global", "gefs_ncep_mean");
        t1.Optional.Should().BeEmpty();
    }

    [Fact]
    public void AllTiers_T3_requires_only_GFS_and_starts_at_GFS_archive()
    {
        var t3 = Exact12hFeatureBuilder.AllTiers.Single(t => t.Name == "T3");
        t3.Required.Should().BeEquivalentTo("gfs_ncep");
        t3.Optional.Should().BeEquivalentTo(
            "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global", "gefs_ncep_mean");
        t3.StartDate.Should().Be(new DateOnly(2023, 1, 18));
    }

    [Fact]
    public void GetTier_unknown_throws()
    {
        var ex = Record.Exception(() => Exact12hFeatureBuilder.GetTier("T99"));
        ex.Should().BeOfType<ArgumentException>();
    }

    [Fact]
    public void BuildSpec_single_lead_default_columns()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier);
        spec.Target.Should().Be("temperature");
        spec.LeadHours.Should().Be(12);
        // T2 has all 5 models in its column vector (GFS+AIFS required, IFS+Global+GEFS optional)
        spec.Models.Should().BeEquivalentTo(
            "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global", "gefs_ncep_mean");
        // Single-lead column shape: temp_<model> (no _l suffix)
        spec.FeatureNames.Should().StartWith(new[] { "temp_gfs", "temp_ifs", "temp_aifs", "temp_moglobal", "temp_gefsmean" });
        // 5 per-model + 3 spread + 4 calendar = 12 features
        spec.FeatureNames.Should().HaveCount(12);
        spec.FeatureNames.Should().EndWith(new[] {
            "temp_mean", "temp_std", "temp_range",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"
        });
    }

    [Fact]
    public void BuildSpec_lead_24_metadata_reflects_target_lead()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier, targetLead: 24);
        spec.LeadHours.Should().Be(24);
        spec.FeatureSet.Should().Contain("l24");
    }

    [Fact]
    public void BuildSpec_multi_lead_columns_have_per_lead_suffixes()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier, targetLead: 12, inputLeads: new[] { 6, 12, 18 });
        // 5 models × 3 leads = 15 model columns + 7 stats = 22 features
        spec.FeatureNames.Should().HaveCount(22);
        spec.FeatureNames.Should().Contain("temp_gfs_l06");
        spec.FeatureNames.Should().Contain("temp_gfs_l12");
        spec.FeatureNames.Should().Contain("temp_gfs_l18");
        // Spread features stay singular (computed across canonical lead only)
        spec.FeatureNames.Should().Contain("temp_mean");
        spec.FeatureNames.Should().NotContain("temp_mean_l06");
    }

    [Fact]
    public void BuildSpec_multi_lead_must_contain_target_lead()
    {
        // Target 12 not in {6, 18} → builder must reject so canonical-lead
        // gating + spread features have a valid index to point at.
        var ex = Record.Exception(() =>
            Exact12hFeatureBuilder.BuildSpec(TestTier, targetLead: 12, inputLeads: new[] { 6, 18 }));
        ex.Should().BeOfType<ArgumentException>();
    }

    [Fact]
    public void BuildSpec_required_optional_overlap_throws()
    {
        var bad = new Exact12hFeatureBuilder.TierSpec(
            Name: "BAD",
            Required: new[] { "gfs_ncep" },
            Optional: new[] { "gfs_ncep" }, // overlap
            StartDate: new DateOnly(2024, 1, 1),
            Description: "test");
        var ex = Record.Exception(() => Exact12hFeatureBuilder.BuildSpec(bad));
        ex.Should().BeOfType<InvalidOperationException>();
    }

    [Fact]
    public void ComposeRow_produces_features_in_declared_order_with_correct_count()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier);
        // perModelLeadValues = same shape as Models (5 entries) for single-lead
        var values = new double[] { 9.5, 10.0, 9.8, 10.2, 9.6 };
        var row = Exact12hFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelLeadValues: values,
            canonicalPerModel: values, // single-lead → canonical == per-model
            windDirMeanDeg: 270.0,
            era5Temp: 10.1);

        row.Features.Should().HaveCount(spec.FeatureCount);
        row.Label.Should().Be(10.1f);
        // Per-model values land in slots 0..4 in canonical order
        row.Features[0].Should().Be(9.5f);   // gfs
        row.Features[1].Should().Be(10.0f);  // ifs
        row.Features[2].Should().Be(9.8f);   // aifs
        row.Features[3].Should().Be(10.2f);  // moglobal
        row.Features[4].Should().Be(9.6f);   // gefsmean
        // Mean lands at slot 5 (= 9.82)
        row.Features[5].Should().BeApproximately(9.82f, 1e-3f);
    }

    [Fact]
    public void ComposeRow_NaN_safe_spread_skips_missing_models()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier);
        // Three of five models NaN — spread is computed across present-only
        var values = new double[] { 9.5, double.NaN, 9.8, double.NaN, double.NaN };
        var row = Exact12hFeatureBuilder.ComposeRow(
            spec,
            new DateTime(2025, 6, 15, 12, 0, 0, DateTimeKind.Utc),
            perModelLeadValues: values,
            canonicalPerModel: values,
            windDirMeanDeg: double.NaN,
            era5Temp: 9.7);
        // Mean across {9.5, 9.8} = 9.65 (lands at slot 5 after 5 per-model slots)
        row.Features[5].Should().BeApproximately(9.65f, 1e-3f);
        // Range = 9.8 - 9.5 = 0.3 (slot 7 = mean+std+range follows the per-model block)
        row.Features[7].Should().BeApproximately(0.3f, 1e-3f);
    }

    [Fact]
    public void ComposeRow_calendar_features_match_validtime_hour_and_doy()
    {
        var spec = Exact12hFeatureBuilder.BuildSpec(TestTier);
        // Pick a valid time we can verify: noon UTC on the 60th day of year.
        var v = new DateTime(2025, 3, 1, 12, 0, 0, DateTimeKind.Utc); // March 1 = doy 60
        var row = Exact12hFeatureBuilder.ComposeRow(
            spec, v,
            perModelLeadValues: new double[] { 5, 5, 5, 5, 5 },
            canonicalPerModel:  new double[] { 5, 5, 5, 5, 5 },
            windDirMeanDeg: 0,
            era5Temp: 5);
        // hour_sin / hour_cos at noon (h=12, frac=0.5) → sin(π) ≈ 0, cos(π) = -1
        var names = spec.FeatureNames.ToList();
        var hourSinIdx = names.IndexOf("hour_sin");
        var hourCosIdx = names.IndexOf("hour_cos");
        row.Features[hourSinIdx].Should().BeApproximately(0.0f, 1e-5f);
        row.Features[hourCosIdx].Should().BeApproximately(-1.0f, 1e-5f);
    }

    [Fact]
    public void ShortName_round_trips_for_all_canonical_models()
    {
        foreach (var m in Exact12hFeatureBuilder.CanonicalModelOrder)
        {
            var sn = Exact12hFeatureBuilder.ShortName(m);
            sn.Should().NotBeNullOrEmpty();
            // Short name should appear in the lean spec's feature names
            var spec = Exact12hFeatureBuilder.BuildSpec(
                new Exact12hFeatureBuilder.TierSpec(
                    Name: "TMP",
                    Required: new[] { m },
                    Optional: Array.Empty<string>(),
                    StartDate: new DateOnly(2024, 1, 1),
                    Description: "test"));
            spec.FeatureNames[0].Should().Be($"temp_{sn}");
        }
    }

    [Fact]
    public void ShortName_unknown_model_throws()
    {
        var ex = Record.Exception(() => Exact12hFeatureBuilder.ShortName("not_a_real_model"));
        ex.Should().BeOfType<ArgumentException>();
    }

    // ---- UKV per-V-hour picks (target-lead-aware, strategy-aware) -------------

    [Fact]
    public void UkvPicksForLead_strict_lead_12_uses_direct_lead_12_per_V()
    {
        // Strict: each V-hour gets a single (cycle, lead 12) tuple from
        // UKV's {0, 6, 12, 18} cycle set. Used by temp 2d.
        var picks = Exact12hFeatureBuilder.UkvPicksForLead(12, Exact12hFeatureBuilder.UkvPickStrategy.Strict);
        picks.Should().HaveCount(4);
        picks.Select(p => p.VHour).Should().BeEquivalentTo(new[] { 0, 6, 12, 18 });
        picks.Should().AllSatisfy(p => p.LeadHours.Should().Be(12));
        picks.Should().AllSatisfy(p => p.RunHour.Should().BeOneOf(0, 6, 12, 18));
    }

    [Fact]
    public void UkvPicksForLead_strict_lead_24_uses_direct_lead_24_per_V()
    {
        var picks = Exact12hFeatureBuilder.UkvPicksForLead(24, Exact12hFeatureBuilder.UkvPickStrategy.Strict);
        picks.Should().HaveCount(4);
        picks.Should().AllSatisfy(p => p.LeadHours.Should().Be(24));
        picks.Should().AllSatisfy(p => p.RunHour.Should().BeOneOf(0, 6, 12, 18));
        picks.Should().AllSatisfy(p => p.DayOffset.Should().Be(1));
        // For lead-24 strict, runHour == VHour (24h-earlier same hour, prev day)
        picks.Should().AllSatisfy(p => p.RunHour.Should().Be(p.VHour));
    }

    [Fact]
    public void UkvPicksForLead_averaging_lead_12_averages_two_leads_per_V()
    {
        // Averaging: each V-hour gets a (cycle ∈ {3,15}, lead ∈ {9, 15})
        // tuple. Across V-hours the leads 9 and 15 alternate so the
        // average effective forecast horizon is 12h. Used by precip 3d.
        var picks = Exact12hFeatureBuilder.UkvPicksForLead(12, Exact12hFeatureBuilder.UkvPickStrategy.Averaging);
        picks.Should().HaveCount(4);
        picks.Average(p => p.LeadHours).Should().Be(12,
            "lead-12 averaging picks must average to 12h-ahead");
        picks.Should().AllSatisfy(p => p.RunHour.Should().BeOneOf(3, 15),
            "averaging picks come from UKV's {3, 15} cycle set only");
        picks.Should().AllSatisfy(p => p.LeadHours.Should().BeOneOf(9, 15));
    }

    [Fact]
    public void UkvPicksForLead_averaging_lead_24_averages_to_24h_ahead()
    {
        var picks = Exact12hFeatureBuilder.UkvPicksForLead(24, Exact12hFeatureBuilder.UkvPickStrategy.Averaging);
        picks.Should().HaveCount(4);
        picks.Average(p => p.LeadHours).Should().Be(24);
        picks.Should().AllSatisfy(p => p.RunHour.Should().BeOneOf(3, 15));
        picks.Should().AllSatisfy(p => p.LeadHours.Should().BeOneOf(21, 27));
        picks.Should().AllSatisfy(p => p.DayOffset.Should().Be(1));
    }

    [Fact]
    public void UkvPicksForLead_each_pick_lands_at_its_V_hour_under_both_strategies()
    {
        // Sanity: for each pick, (RunHour + LeadHours) mod 24 should equal VHour
        // — i.e. the (cycle, lead) tuple actually lands at the claimed ValidTime hour.
        foreach (var strategy in new[] { Exact12hFeatureBuilder.UkvPickStrategy.Strict, Exact12hFeatureBuilder.UkvPickStrategy.Averaging })
        {
            foreach (var lead in new[] { 12, 24 })
            {
                foreach (var p in Exact12hFeatureBuilder.UkvPicksForLead(lead, strategy))
                {
                    var landedHour = (p.RunHour + p.LeadHours) % 24;
                    landedHour.Should().Be(p.VHour,
                        $"strategy={strategy}, target={lead}, V={p.VHour}: runHour {p.RunHour} + lead {p.LeadHours} should land at V");
                }
            }
        }
    }

    [Fact]
    public void UkvPicksForLead_unsupported_target_throws_under_both_strategies()
    {
        // Picker tables now cover {12, 24, 48} for Strict and
        // {12, 24, 48, 72} for Averaging. Lead 96+ is unsupported by
        // either strategy (Strict caps at UKV's T+54h horizon; Averaging
        // would need a backfill of leads {93, 99} from cycles 3/15Z).
        foreach (var strategy in new[] { Exact12hFeatureBuilder.UkvPickStrategy.Strict, Exact12hFeatureBuilder.UkvPickStrategy.Averaging })
        {
            var ex = Record.Exception(() => Exact12hFeatureBuilder.UkvPicksForLead(96, strategy));
            ex.Should().BeOfType<ArgumentException>($"strategy={strategy} should reject unsupported lead");
        }
    }

    [Fact]
    public void UkvPerVOrClause_strict_emits_lead_12_or_24_only()
    {
        var lead12 = Exact12hFeatureBuilder.UkvPerVOrClause(12, Exact12hFeatureBuilder.UkvPickStrategy.Strict);
        lead12.Should().Contain("LeadHours = 12");
        lead12.Should().NotContain("LeadHours = 9");
        lead12.Should().NotContain("LeadHours = 15");
        lead12.Should().NotContain("LeadHours = 24");

        var lead24 = Exact12hFeatureBuilder.UkvPerVOrClause(24, Exact12hFeatureBuilder.UkvPickStrategy.Strict);
        lead24.Should().Contain("LeadHours = 24");
        lead24.Should().NotContain("LeadHours = 12");
        lead24.Should().NotContain("LeadHours = 21");
        lead24.Should().NotContain("LeadHours = 27");
    }

    [Fact]
    public void UkvPerVOrClause_averaging_emits_brackets_around_target()
    {
        var lead12 = Exact12hFeatureBuilder.UkvPerVOrClause(12, Exact12hFeatureBuilder.UkvPickStrategy.Averaging);
        lead12.Should().Contain("LeadHours = 9");
        lead12.Should().Contain("LeadHours = 15");
        lead12.Should().NotContain("LeadHours = 12");
        lead12.Should().NotContain("LeadHours = 21");

        var lead24 = Exact12hFeatureBuilder.UkvPerVOrClause(24, Exact12hFeatureBuilder.UkvPickStrategy.Averaging);
        lead24.Should().Contain("LeadHours = 21");
        lead24.Should().Contain("LeadHours = 27");
        lead24.Should().NotContain("LeadHours = 24");
        lead24.Should().NotContain("LeadHours = 9");
    }
}
