using System.Globalization;
using System.Text.Json;
using FluentAssertions;
using WeatherBlend.Train.Common;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Regression tests for the HpInt / HpString extensions on
/// training_metadata.Hyperparameters dictionaries.
///
/// What happened on 2026-05-26: PrecipConformalFitCommand's
/// post-train auto-fit hook crashed with
///
///   System.InvalidCastException: Unable to cast object of type
///   'System.Text.Json.JsonElement' to type 'System.IConvertible'.
///      at System.Convert.ToInt32(Object value, IFormatProvider provider)
///
/// Root cause: `training_metadata.json` round-trips through
/// System.Text.Json so every value in the Hyperparameters dictionary
/// comes back as a <c>JsonElement</c>. <c>Convert.ToInt32(jsonElement,
/// ...)</c> throws because JsonElement doesn't implement IConvertible.
/// The extensions <see cref="HyperparameterExtensions.HpInt"/> /
/// <see cref="HyperparameterExtensions.HpString"/> wrap the JsonElement
/// case explicitly — this test fixes that contract so a future caller
/// that reaches for <c>Convert.ToInt32</c> instead has a clear regression
/// signal pointing at the right helper.
/// </summary>
public class HyperparameterExtensionsTests
{
    [Fact]
    public void HpInt_unwraps_JsonElement_after_System_Text_Json_round_trip()
    {
        // Mimic the exact path that broke conformal fit: build a dict
        // in-memory, serialise to JSON, deserialise back to
        // Dictionary<string, object> — which is what
        // ModelArtifact.LoadTrainingMetadata produces.
        var original = new Dictionary<string, object> { ["phase3o_station_index"] = 3 };
        var json = JsonSerializer.Serialize(original);
        var roundtripped = JsonSerializer.Deserialize<Dictionary<string, object>>(json);

        roundtripped.Should().NotBeNull();
        roundtripped!.HpInt("phase3o_station_index").Should().Be(3,
            "HpInt must unwrap JsonElement-typed values so callers don't crash on " +
            "metadata read from disk (2026-05-26 retrain failure mode).");

        // Direct Convert.ToInt32 on the roundtripped value is the
        // EXACT pre-fix line that crashed — pinning it as a sanity
        // check so a future regression that reaches for Convert
        // doesn't sneak past CI.
        var roundtrippedValue = roundtripped["phase3o_station_index"];
        roundtrippedValue.Should().BeOfType<JsonElement>(
            "roundtrip MUST land as JsonElement — if this changes, the JSON " +
            "serialiser default has shifted and the HpInt unwrap might need an update.");
        Assert.Throws<InvalidCastException>(() =>
            Convert.ToInt32(roundtrippedValue, CultureInfo.InvariantCulture));
    }

    [Fact]
    public void HpInt_handles_raw_int_without_JSON_serialisation()
    {
        // Fresh in-memory metadata (e.g. mid-train, BEFORE the dict is
        // persisted) has raw int values, NOT JsonElement. HpInt must
        // handle both shapes — same code path serving both train-time
        // writers and predict-time readers.
        var dict = new Dictionary<string, object> { ["mc_samples"] = 20_000 };
        dict.HpInt("mc_samples").Should().Be(20_000);
    }

    [Fact]
    public void HpInt_returns_null_for_missing_or_null_or_wrong_type()
    {
        var dict = new Dictionary<string, object>
        {
            ["str"] = "not-a-number",
            ["null_value"] = null!,
        };
        dict.HpInt("missing").Should().BeNull();
        dict.HpInt("null_value").Should().BeNull();
        dict.HpInt("str").Should().BeNull();
        (((IReadOnlyDictionary<string, object>?)null)).HpInt("any").Should().BeNull();
    }

    [Fact]
    public void HpString_unwraps_JsonElement_after_round_trip()
    {
        var original = new Dictionary<string, object> { ["precip_3a_version"] = "v2026-05-25_141845" };
        var json = JsonSerializer.Serialize(original);
        var roundtripped = JsonSerializer.Deserialize<Dictionary<string, object>>(json);

        roundtripped!.HpString("precip_3a_version").Should().Be("v2026-05-25_141845");
    }
}
