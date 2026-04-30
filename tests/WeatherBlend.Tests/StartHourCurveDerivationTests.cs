using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Predict.StartHour;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin <see cref="StartHourCurveDerivation.Derive"/>. The math is small but
/// load-bearing — the home + dry-window pages will read the curve directly,
/// and the verify command will score π_s × DailyProbAnyBlock against truth
/// using the same shape these tests exercise.
/// </summary>
public class StartHourCurveDerivationTests
{
    private const string Loc = "bonehill_rocks";
    private const string Station = "ea_bellever_dartmoor";
    private const string V = "v1";
    private const string PrecipV = "v3a-champion";
    private const string DryV = "v3b-champion";
    private static readonly DateTime Made = new(2026, 4, 30, 10, 35, 0, DateTimeKind.Utc);
    private static readonly DateTime Target = new(2026, 5, 1, 0, 0, 0, DateTimeKind.Utc);

    private static IReadOnlyDictionary<int, double> Hours(params (int Hour, double Q)[] items)
        => items.ToDictionary(x => x.Hour, x => x.Q);

    [Fact]
    public void Derive_uniform_dry_forecast_yields_uniform_curve()
    {
        // q = 0 across the whole daytime span → every p_s = 1, π_s = 1/N for
        // every candidate start. Calibrated curve is just dailyP / N (which is
        // what an honest "I have no idea where the block is" answer looks like).
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
                     (13, 0), (14, 0), (15, 0), (16, 0));
        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, windowHours: 6, V, Made, leadHours: 24, Target,
            daytimeStartUtc: 8, daytimeEndUtc: 17, q,
            dailyProbAnyBlock: 0.95, PrecipV, DryV);

        rows.Should().HaveCount(4);
        rows.Select(r => r.StartHourUtc).Should().Equal(8, 9, 10, 11);
        rows.Select(r => r.ConditionalProb).Should().AllBeEquivalentTo(0.25);
        rows.Select(r => r.CalibratedProb).Should().AllBeEquivalentTo(0.25 * 0.95);
        rows.Select(r => r.RawProduct).Should().AllBeEquivalentTo(1.0);
    }

    [Fact]
    public void Derive_concentrates_curve_away_from_wet_hours()
    {
        // Wet at 13:00 (q=0.71), forecast otherwise dry. A 6-hour block can
        // only avoid hour 13 by starting at 14 (block 14..19) — but 19 is past
        // the 17:00 window end, so the effective best is "start as early as
        // possible" (block doesn't include hour 13 if start ≤ 7, but the
        // earliest start is 8). Actually with 6h window and hours 8..16 inside
        // 13:00-wet, every candidate window includes hour 13 (s=8 covers 8..13,
        // s=11 covers 11..16). So the curve favours starts whose 6-window
        // contains hour 13 ONCE and other dry hours — every start does. With
        // hour 13 the only wet hour, p_s is the same for every s. The wet
        // hour appears in s=8..s=11's windows uniformly. Let's verify the
        // raw products are equal and conditional uniform under that pattern.
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
                     (13, 0.71), (14, 0), (15, 0), (16, 0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, windowHours: 6, V, Made, leadHours: 24, Target,
            8, 17, q, 0.5, PrecipV, DryV);

        rows.Should().HaveCount(4);
        // Hour 13 is in EVERY 6-hour window starting at 8..11 (s=8 covers
        // 8..13, s=11 covers 11..16). Every p_s = (1-0.71) = 0.29, conditional
        // uniform 0.25. Approximate match because (1-0.71) materialises as
        // 0.29000…04 in IEEE 754.
        rows.Select(r => r.RawProduct).Should().AllSatisfy(p =>
            p.Should().BeApproximately(0.29, 1e-9));
        rows.Select(r => r.ConditionalProb).Should().AllBeEquivalentTo(0.25);
    }

    [Fact]
    public void Derive_favours_starts_that_skip_a_wet_hour_that_only_some_windows_avoid()
    {
        // Smaller window (N=4) and a wet hour at 16: only start hours 8..12
        // produce 4-hour windows fully inside [8..17). Of those:
        //   s=8 → 8..11 (no wet hour)        → p = 1.0
        //   s=9 → 9..12                       → p = 1.0
        //   s=10 → 10..13                     → p = 1.0
        //   s=11 → 11..14                     → p = 1.0
        //   s=12 → 12..15                     → p = 1.0
        // Hour 16 (q=1.0) only appears in windows starting at 13..16 — which
        // for daytime end 17 are out of range (we stop at s=12 since s+N=16<=17).
        // So no window contains the wet hour; uniform.
        // Let's instead put the wet hour at hour 12 with q=1.0 — that hour is
        // in windows starting 9, 10, 11, 12 but NOT s=8 (8..11 ends before 12).
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0), (12, 1.0),
                     (13, 0), (14, 0), (15, 0), (16, 0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, windowHours: 4, V, Made, leadHours: 24, Target,
            8, 17, q, 0.6, PrecipV, DryV);

        // 6 starts: 8..13.
        rows.Should().HaveCount(6);
        rows.Single(r => r.StartHourUtc == 8).RawProduct.Should().Be(1.0);
        // s=9..12 all hit hour 12 → p=0.
        foreach (var s in new[] { 9, 10, 11, 12 })
            rows.Single(r => r.StartHourUtc == s).RawProduct.Should().Be(0.0);
        // s=13 → 13..16, no wet hour → p=1.0.
        rows.Single(r => r.StartHourUtc == 13).RawProduct.Should().Be(1.0);

        // Conditional probability is split between the two non-zero starts
        // (s=8 and s=13) at 50/50.
        rows.Single(r => r.StartHourUtc == 8).ConditionalProb.Should().BeApproximately(0.5, 1e-9);
        rows.Single(r => r.StartHourUtc == 13).ConditionalProb.Should().BeApproximately(0.5, 1e-9);
        rows.Where(r => r.RawProduct == 0).Select(r => r.ConditionalProb).Should().AllBeEquivalentTo(0.0);

        // Calibrated curve sums to dailyProbAnyBlock (= 0.6).
        rows.Sum(r => r.CalibratedProb).Should().BeApproximately(0.6, 1e-9);
    }

    [Fact]
    public void Derive_returns_empty_list_when_any_daytime_hour_is_missing()
    {
        // Hour 12 missing → entire composite dropped for the day. Better to
        // produce no curve than to silently impute zero or some "neutral"
        // value, which would bias the shape.
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0),
                     /* 12 missing */ (13, 0), (14, 0), (15, 0), (16, 0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, 6, V, Made, 24, Target, 8, 17, q, 0.5, PrecipV, DryV);

        rows.Should().BeEmpty();
    }

    [Fact]
    public void Derive_returns_empty_list_when_window_exceeds_daytime_span()
    {
        // 9-hour daytime span, 10-hour window → no candidate start has room.
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
                     (13, 0), (14, 0), (15, 0), (16, 0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, windowHours: 10, V, Made, 24, Target,
            8, 17, q, 0.5, PrecipV, DryV);

        rows.Should().BeEmpty();
    }

    [Fact]
    public void Derive_falls_back_to_uniform_when_every_window_has_a_certain_wet_hour()
    {
        // Every daytime hour wet with q=1.0 — the "rain forecast all day"
        // case. p_s = 0 for every start; Σ = 0 → uniform fallback so
        // downstream rendering still has a shape to draw. Calibrated curve
        // then = uniform × dailyProbAnyBlock and still sums to dailyP.
        var q = Hours((8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0), (12, 1.0),
                     (13, 1.0), (14, 1.0), (15, 1.0), (16, 1.0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, windowHours: 4, V, Made, 24, Target,
            8, 17, q, 0.6, PrecipV, DryV);

        rows.Should().HaveCount(6);
        rows.Select(r => r.RawProduct).Should().AllBeEquivalentTo(0.0);
        rows.Select(r => r.ConditionalProb).Should().AllSatisfy(p =>
            p.Should().BeApproximately(1.0 / 6, 1e-12));
        rows.Sum(r => r.CalibratedProb).Should().BeApproximately(0.6, 1e-9);
    }

    [Fact]
    public void Derive_clamps_q_outside_zero_one_to_keep_products_valid()
    {
        // q values from upstream blenders shouldn't be < 0 or > 1, but a numeric
        // glitch (sigmoid overflow, junk row) shouldn't crash the derivation.
        // Clamp keeps p_s in [0, 1].
        var q = Hours((8, -0.1), (9, 0.0), (10, 0.0), (11, 0.0), (12, 0.0),
                     (13, 1.5), (14, 0.0), (15, 0.0), (16, 0.0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, 6, V, Made, 24, Target, 8, 17, q, 0.5, PrecipV, DryV);

        // Hour 13 clamped to 1.0 → every window through it has p=0; uniform fallback.
        rows.Should().HaveCount(4);
        rows.Select(r => r.RawProduct).Should().AllBeEquivalentTo(0.0);
        rows.Select(r => r.ConditionalProb).Should().AllBeEquivalentTo(0.25);
    }

    [Fact]
    public void Derive_propagates_provenance_and_anchor_metadata_unchanged()
    {
        // Every row should carry the input version strings + the per-row
        // identity unchanged. The curve consumer reads these to attribute
        // skill back to specific champions.
        var q = Hours((8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
                     (13, 0), (14, 0), (15, 0), (16, 0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, 6, V, Made, 24, Target, 8, 17, q, 0.95, PrecipV, DryV);

        rows.Should().AllSatisfy(r =>
        {
            r.LocationName.Should().Be(Loc);
            r.TruthStation.Should().Be(Station);
            r.WindowHours.Should().Be(6);
            r.ModelVersion.Should().Be(V);
            r.LeadHours.Should().Be(24);
            r.PredictionMadeAtUtc.Should().Be(Made);
            r.TargetDateUtc.Should().Be(Target);
            r.PrecipVersion.Should().Be(PrecipV);
            r.DryWindowVersion.Should().Be(DryV);
            r.DailyProbAnyBlock.Should().Be(0.95);
        });
    }

    [Fact]
    public void Derive_calibrated_sums_to_dailyP_when_curve_is_well_defined()
    {
        // Well-defined = at least one start has p > 0. Σ π = 1 by construction,
        // so Σ (π × dailyP) = dailyP. This is the property that makes the curve
        // "calibrated to 3b" — readers who add the bars get the daily marginal
        // back, modulo float rounding.
        var q = Hours((8, 0.1), (9, 0.05), (10, 0.0), (11, 0.0), (12, 0.2),
                     (13, 0.0), (14, 0.05), (15, 0.0), (16, 0.0));

        var rows = StartHourCurveDerivation.Derive(
            Loc, Station, 6, V, Made, 24, Target, 8, 17, q, 0.42, PrecipV, DryV);

        rows.Sum(r => r.CalibratedProb).Should().BeApproximately(0.42, 1e-9);
    }
}
