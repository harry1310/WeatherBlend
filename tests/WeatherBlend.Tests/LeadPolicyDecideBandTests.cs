using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Fast coverage for <see cref="PrecipCrossLeadBakeoffCommand.DecideBand"/> — the
/// shared per-band model chooser for the temperature (2c) and precip (3c/3o) lead
/// policies. The headline test is the 2026-06-15 regression: an incumbent band
/// whose model had fallen outside the ±24h candidate window was looked up with a
/// throwing <c>candidates.First(...)</c>, which killed the whole precip fit before
/// any policy was written (the precip and temp copies had drifted — temp used the
/// safe FirstOrDefault, precip did not). These run the chooser directly with tiny
/// score maps; no fixtures, no I/O.
/// </summary>
public sealed class LeadPolicyDecideBandTests
{
    private static (string, string, List<int>) Single(int lead) => ($"s{lead}", "single", new List<int> { lead });
    private static (string, string, List<int>) Blend(int lo, int hi) => ($"b{lo}x{hi}", "blend", new List<int> { lo, hi });

    private static PrecipLeadPolicy.BandEntry Inc(string kind, params int[] leads) =>
        new() { Kind = kind, Leads = leads.ToList() };

    private static readonly PrecipLeadPolicy.ThresholdsBlock Thr =
        new() { MarginPct = 0.75, HysteresisPct = 0.5 };

    private static PrecipCrossLeadBakeoffCommand.BandPick Decide(
        IReadOnlyList<(string, string, List<int>)> cands,
        IReadOnlyDictionary<string, double> sel, IReadOnlyDictionary<string, double> sco,
        int baseLead, double baseSco, PrecipLeadPolicy.BandEntry? inc)
        => PrecipCrossLeadBakeoffCommand.DecideBand(
            cands, s => sel.GetValueOrDefault(s, double.NaN), s => sco.GetValueOrDefault(s, double.NaN),
            baseLead, baseSco, inc, Thr);

    [Fact]
    public void OutOfWindow_incumbent_does_not_throw_and_keeps_select_pick()
    {
        // The crash: band 84-90h has in-window candidates {s72, s96}, but the
        // incumbent (from the old global file) names s48 — outside ±24h, so not a
        // candidate. Must NOT throw; the incumbent can't be kept, so we fall back
        // to the SELECT winner (which here is the baseline m72 → no deviation).
        var cands = new List<(string, string, List<int>)> { Single(72), Single(96) };
        var sel = new Dictionary<string, double> { ["s72"] = 0.10, ["s96"] = 0.12 };
        var sco = new Dictionary<string, double> { ["s72"] = 0.10, ["s96"] = 0.12, ["s48"] = 0.09 };
        var inc = Inc("single", 48);   // out of window — not in cands

        var act = () => Decide(cands, sel, sco, baseLead: 72, baseSco: 0.10, inc);
        act.Should().NotThrow("an out-of-window incumbent must fall back, not crash the fit");

        var pick = act();
        pick.Series.Should().Be("s72");
        pick.Passes.Should().BeFalse("the SELECT winner is the baseline bucket model — no deviation");
    }

    [Fact]
    public void Clear_blend_winner_beats_baseline_and_deviates()
    {
        var cands = new List<(string, string, List<int>)> { Single(24), Single(48), Blend(24, 48) };
        var sel = new Dictionary<string, double> { ["s24"] = 0.10, ["s48"] = 0.11, ["b24x48"] = 0.07 };
        var sco = new Dictionary<string, double> { ["s24"] = 0.10, ["s48"] = 0.11, ["b24x48"] = 0.085 };

        var pick = Decide(cands, sel, sco, baseLead: 24, baseSco: 0.10, inc: null);

        pick.Series.Should().Be("b24x48");
        pick.Kind.Should().Be("blend");
        pick.Passes.Should().BeTrue();
        pick.DeltaPct.Should().BeApproximately(15.0, 1e-9);   // (0.10-0.085)/0.10
    }

    [Fact]
    public void Baseline_is_the_select_winner_means_no_deviation()
    {
        var cands = new List<(string, string, List<int>)> { Single(24), Single(48) };
        var sel = new Dictionary<string, double> { ["s24"] = 0.08, ["s48"] = 0.09 };
        var sco = new Dictionary<string, double> { ["s24"] = 0.10, ["s48"] = 0.11 };

        var pick = Decide(cands, sel, sco, baseLead: 24, baseSco: 0.10, inc: null);

        pick.Series.Should().Be("s24");
        pick.Passes.Should().BeFalse("the baseline bucket model itself won — never recorded as a deviation");
    }

    [Fact]
    public void Margin_gate_blocks_a_sub_threshold_win()
    {
        // s48 wins SELECT and edges baseline on SCORE, but by < MarginPct (0.75%).
        var cands = new List<(string, string, List<int>)> { Single(24), Single(48) };
        var sel = new Dictionary<string, double> { ["s24"] = 0.11, ["s48"] = 0.07 };
        var sco = new Dictionary<string, double> { ["s24"] = 0.10, ["s48"] = 0.0995 };   // only 0.5% better

        var pick = Decide(cands, sel, sco, baseLead: 24, baseSco: 0.10, inc: null);

        pick.Series.Should().Be("s48");
        pick.Passes.Should().BeFalse("0.5% < 0.75% margin → stay on the baseline");
    }

    [Fact]
    public void In_window_incumbent_is_kept_when_challenger_does_not_clear_hysteresis()
    {
        // SELECT prefers s24, but the live incumbent is s48 and s24 doesn't beat it
        // on SCORE by HysteresisPct (0.5%) — so s48 stays, and it clears the
        // reduced (margin − hysteresis) re-qualify bar vs the baseline.
        var cands = new List<(string, string, List<int>)> { Single(24), Single(48) };
        var sel = new Dictionary<string, double> { ["s24"] = 0.07, ["s48"] = 0.08 };
        var sco = new Dictionary<string, double> { ["s24"] = 0.0905, ["s48"] = 0.0900 };
        var inc = Inc("single", 48);   // in window at this band

        var pick = Decide(cands, sel, sco, baseLead: 24, baseSco: 0.0950, inc);

        pick.Series.Should().Be("s48", "challenger didn't clear hysteresis — incumbent holds");
        pick.Passes.Should().BeTrue();
    }

    [Fact]
    public void In_window_incumbent_is_dropped_when_challenger_clearly_wins()
    {
        // Same setup but the incumbent s48 is now clearly worse on SCORE than the
        // SELECT winner s48... use a non-baseline challenger s72 that beats it.
        var cands = new List<(string, string, List<int>)> { Single(48), Single(72) };
        var sel = new Dictionary<string, double> { ["s48"] = 0.09, ["s72"] = 0.07 };
        var sco = new Dictionary<string, double> { ["s48"] = 0.0980, ["s72"] = 0.0900 };
        var inc = Inc("single", 48);

        var pick = Decide(cands, sel, sco, baseLead: 48, baseSco: 0.1000, inc);

        pick.Series.Should().Be("s72", "challenger clears hysteresis vs the incumbent");
        pick.Passes.Should().BeTrue();
    }
}
