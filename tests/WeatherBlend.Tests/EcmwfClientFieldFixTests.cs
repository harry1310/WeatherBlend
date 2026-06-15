using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Unit coverage for the ECMWF collector's accumulated-field deaccumulation and
/// cloud-cover normalisation (2026-06-15). The raw bug: <c>ssrd</c> is published
/// accumulated-since-forecast-start, so dividing the cumulative by 3600 grew the
/// stored shortwave ~linearly with lead (≈7900 W/m² at +24h); and AIFS publishes
/// <c>tcc</c> already as a percentage, so the shared ×100 VarMap over-scaled it
/// to &gt;100% (≈9274 for 92.74%). Both fields are unused by any blender, but the
/// data was visibly wrong.
/// </summary>
public sealed class EcmwfClientFieldFixTests
{
    [Fact]
    public void DeaccumulateMeanFlux_first_lead_returns_zero_to_lead_mean()
    {
        // Stored cum value at lead 12 = mean_{0..12} × 12 = 35.25 × 12 ≈ 423.
        EcmwfClient.DeaccumulateMeanFlux(423.0, cumPrevLead: null, leadHours: 12, prevLeadHours: 0)
            .Should().BeApproximately(35.25, 1e-6);
    }

    [Fact]
    public void DeaccumulateMeanFlux_window_returns_per_window_mean()
    {
        // cum 423 at lead 12, cum 7341 at lead 24 → 12–24h window mean
        // = (7341 − 423) / (24 − 12) = 576.5 W/m² (a sane cloudy-midday value,
        // not the raw 7341 the cumulative-/3600 bug produced).
        EcmwfClient.DeaccumulateMeanFlux(7341.0, cumPrevLead: 423.0, leadHours: 24, prevLeadHours: 12)
            .Should().BeApproximately(576.5, 1e-6);
    }

    [Fact]
    public void DeaccumulateMeanFlux_clamps_negative_and_handles_nulls_and_zero_window()
    {
        EcmwfClient.DeaccumulateMeanFlux(null, 100.0, 24, 12).Should().BeNull();
        // A cumulative that went backwards (numerical) → clamp ≥ 0.
        EcmwfClient.DeaccumulateMeanFlux(100.0, 200.0, 24, 12).Should().Be(0.0);
        // Non-positive window (can't deaccumulate) → value unchanged, clamped.
        EcmwfClient.DeaccumulateMeanFlux(500.0, 100.0, 12, 12).Should().Be(500.0);
    }

    [Fact]
    public void DeaccumulateSum_returns_per_window_total_not_a_rate()
    {
        // tp cumulative: 1.80mm at lead 12, 1.82mm at lead 24 → the 12–24h window
        // got just 0.02mm (NOT the cumulative 1.82, and NOT divided by hours).
        EcmwfClient.DeaccumulateSum(1.82, cumPrevLead: 1.80, leadHours: 24, prevLeadHours: 12)
            .Should().BeApproximately(0.02, 1e-9);
        // First lead → the 0→lead total.
        EcmwfClient.DeaccumulateSum(1.80, cumPrevLead: null, leadHours: 12, prevLeadHours: 0)
            .Should().BeApproximately(1.80, 1e-9);
        // Null / backwards / zero-window guards.
        EcmwfClient.DeaccumulateSum(null, 1.0, 24, 12).Should().BeNull();
        EcmwfClient.DeaccumulateSum(1.0, 2.0, 24, 12).Should().Be(0.0);
        EcmwfClient.DeaccumulateSum(5.0, 1.0, 12, 12).Should().Be(5.0);
    }

    [Fact]
    public void NormalizeCloudPct_undoes_the_aifs_double_scale()
    {
        // AIFS: stored 9274.61 (= 92.7461% × 100) → 92.7461%.
        EcmwfClient.NormalizeCloudPct(9274.61, "ecmwf_aifs_oper").Should().BeApproximately(92.7461, 1e-4);
        // AIFS genuine overcast 100% (raw 100 × 100 = 10000) → clamps to 100.
        EcmwfClient.NormalizeCloudPct(10000.0, "ecmwf_aifs_oper").Should().Be(100.0);
    }

    [Fact]
    public void NormalizeCloudPct_leaves_ifs_and_clamps()
    {
        // IFS tcc is already correctly ×100 to a percentage — leave it.
        EcmwfClient.NormalizeCloudPct(34.375, "ecmwf_ifs_oper").Should().Be(34.375);
        EcmwfClient.NormalizeCloudPct(100.0, "ecmwf_ifs_oper").Should().Be(100.0);
        // Defensive clamp + null passthrough.
        EcmwfClient.NormalizeCloudPct(150.0, "ecmwf_ifs_oper").Should().Be(100.0);
        EcmwfClient.NormalizeCloudPct(-5.0, "ecmwf_ifs_oper").Should().Be(0.0);
        EcmwfClient.NormalizeCloudPct(null, "ecmwf_aifs_oper").Should().BeNull();
    }
}
