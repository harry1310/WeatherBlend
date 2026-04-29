using FluentAssertions;
using WeatherBlend.Commands;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pin small pure helpers that live inside Command classes — string parsers,
/// slug matchers, mean-of-slots aggregators. These bits are leaf logic that's
/// easy to break in passing (renaming a manifest key format, adding a model
/// to the canonical order, etc.) and easy to test without parquet fixtures.
///
/// Heavier integration coverage (RunAsync end-to-end with parquet fixtures)
/// lives in <see cref="RenderSiteCommandTests"/>.
/// </summary>
public class CommandHelperTests
{
    // -------- DryWindowPredictCommand.ParseCompositeKey --------
    //
    // Parses manifest keys like "ea_bellever_dartmoor/window_3h" into a
    // (slug, hours) tuple. The format has been stable for months but it's
    // the join key between the manifest, the predict pipeline and the on-
    // disk artefact tree — a typo here means a whole composite gets
    // silently skipped.

    [Theory]
    [InlineData("ea_bellever_dartmoor/window_3h",     "ea_bellever_dartmoor", 3)]
    [InlineData("ea_bellever_dartmoor/window_4h",     "ea_bellever_dartmoor", 4)]
    [InlineData("ea_bellever_dartmoor/window_6h",     "ea_bellever_dartmoor", 6)]
    [InlineData("ea_princetown/window_12h",           "ea_princetown",        12)]
    [InlineData("ea_dartmoor_nr_hexworthy/window_6h", "ea_dartmoor_nr_hexworthy", 6)]
    public void ParseCompositeKey_extracts_slug_and_window_hours(string key, string expectedSlug, int expectedHours)
    {
        var parsed = DryWindowPredictCommand.ParseCompositeKey(key);
        parsed.Should().NotBeNull();
        parsed!.Value.StationSlug.Should().Be(expectedSlug);
        parsed.Value.WindowHours.Should().Be(expectedHours);
    }

    [Theory]
    [InlineData("")]                                   // empty
    [InlineData("ea_bellever_dartmoor")]                // no /window_Xh suffix
    [InlineData("ea_bellever/window_3")]                // missing 'h'
    [InlineData("ea_bellever/window_h")]                // missing digits
    [InlineData("/window_3h")]                          // empty slug
    [InlineData("ea_bellever_dartmoor/window_3h/extra")] // trailing junk
    public void ParseCompositeKey_returns_null_for_malformed_keys(string key)
    {
        DryWindowPredictCommand.ParseCompositeKey(key).Should().BeNull();
    }

    // -------- DryWindowPredictCommand.SlugMatches --------
    //
    // Match a CLI argument against a stored station slug. Caller passes in
    // either the bare slug ("ea_bellever_dartmoor"), the human form
    // ("Bellever Dartmoor"), or the prefix-stripped slug ("bellever_dartmoor").
    // Mismatch here means a `--truth-station bellever_dartmoor` invocation
    // silently runs zero composites.

    [Theory]
    [InlineData("ea_bellever_dartmoor", "ea_bellever_dartmoor")]    // exact slug
    [InlineData("ea_bellever_dartmoor", "bellever_dartmoor")]       // prefix-stripped
    [InlineData("ea_bellever_dartmoor", "Bellever Dartmoor")]       // human form
    [InlineData("ea_bellever_dartmoor", "bellever dartmoor")]       // case-insensitive
    [InlineData("ea_princetown",         "Princetown")]
    [InlineData("ea_princetown",         "princetown")]
    public void SlugMatches_accepts_slug_variants_and_human_form(string slug, string arg)
    {
        DryWindowPredictCommand.SlugMatches(slug, arg).Should().BeTrue();
    }

    [Theory]
    [InlineData("ea_bellever_dartmoor", "princetown")]              // wrong station
    [InlineData("ea_bellever_dartmoor", "")]                        // empty
    [InlineData("ea_bellever_dartmoor", "bellever")]                // partial → derived "bellever" ≠ "ea_bellever_dartmoor"
    public void SlugMatches_rejects_unrelated_arguments(string slug, string arg)
    {
        DryWindowPredictCommand.SlugMatches(slug, arg).Should().BeFalse();
    }

    // -------- PrecipPredictCommand.MeanOfSlots --------
    //
    // Averages a per-NWP slot array, ignoring nulls. Used to compute the
    // "ensemble mean" feature the rich blender consumes; bug here moves the
    // ensemble bias on every prediction row.

    [Fact]
    public void MeanOfSlots_returns_NaN_when_every_slot_is_null()
    {
        var slots = new double?[] { null, null, null };
        PrecipPredictCommand.MeanOfSlots(slots).Should().Be(double.NaN);
    }

    [Fact]
    public void MeanOfSlots_averages_only_populated_slots()
    {
        // 3 of 8 slots populated — nulls must not pull the mean toward zero.
        var slots = new double?[] { 1.0, null, 3.0, null, null, 5.0, null, null };
        PrecipPredictCommand.MeanOfSlots(slots).Should().BeApproximately(3.0, 1e-9);
    }

    [Fact]
    public void MeanOfSlots_preserves_zero_values_in_the_average()
    {
        // 0.0 is a valid populated slot (e.g. NWP reports 0 mm) — it's the
        // null check that should gate inclusion, not value-truthiness.
        var slots = new double?[] { 0.0, 0.0, 6.0 };
        PrecipPredictCommand.MeanOfSlots(slots).Should().BeApproximately(2.0, 1e-9);
    }

    // -------- PrecipPredictCommand.MeanOfDepressions --------
    //
    // Mean of (Ta − Td) across paired NWP slots. Drop slots where either
    // temp or dewpoint is null — partial pairs would skew the depression.

    [Fact]
    public void MeanOfDepressions_drops_pairs_with_either_side_null()
    {
        // Slot 0: both populated (10−5=5). Slot 1: temp only (skip).
        // Slot 2: dew only (skip). Slot 3: both populated (20−12=8).
        // Mean of {5, 8} = 6.5.
        var temps = new double?[] { 10.0, 15.0,  null, 20.0 };
        var dews  = new double?[] {  5.0, null, 10.0, 12.0 };

        PrecipPredictCommand.MeanOfDepressions(temps, dews).Should().BeApproximately(6.5, 1e-9);
    }

    [Fact]
    public void MeanOfDepressions_returns_NaN_when_no_complete_pair_exists()
    {
        var temps = new double?[] { 10.0, null };
        var dews  = new double?[] { null, 5.0 };

        PrecipPredictCommand.MeanOfDepressions(temps, dews).Should().Be(double.NaN);
    }
}
