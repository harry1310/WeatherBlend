using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

public class DryWindowPredictCommandTests
{
    private static DryWindowTrainingRow MakeRow(float offset = 0f) => new()
    {
        TargetDateUtc = new DateTime(2026, 4, 23, 0, 0, 0, DateTimeKind.Utc),
        WindowHours = 3,
        PrecipSumGfs   = 4.5f + offset, PrecipSumEcmwf = 3.2f, PrecipSumIcon = 5.1f,
        PrecipSumMf    = 4.8f, PrecipSumUkmo  = 4.0f, PrecipSumGem   = 3.7f,
        PrecipMaxHourGfs   = 0.9f, PrecipMaxHourEcmwf = 0.7f, PrecipMaxHourIcon = 1.1f,
        PrecipMaxHourMf    = 0.85f, PrecipMaxHourUkmo  = 0.8f, PrecipMaxHourGem  = 0.6f,
        WetHourCountGfs   = 12f, WetHourCountEcmwf = 10f, WetHourCountIcon = 14f,
        WetHourCountMf    = 11f, WetHourCountUkmo  = 13f, WetHourCountGem  = 9f,
        LongestDryRunGfs   = 4f, LongestDryRunEcmwf = 5f, LongestDryRunIcon = 3f,
        LongestDryRunMf    = 4f, LongestDryRunUkmo  = 4f, LongestDryRunGem  = 6f,
        HasDryWindowGfs   = 1f, HasDryWindowEcmwf = 1f, HasDryWindowIcon = 0f,
        HasDryWindowMf    = 1f, HasDryWindowUkmo  = 1f, HasDryWindowGem  = 1f,
        ProbMaxGfs   = 75f, ProbMaxEcmwf = 60f, ProbMaxIcon = 80f,
        ProbMaxMf    = 70f, ProbMaxUkmo  = 65f, ProbMaxGem  = 55f,
        PrecipSumMean = 4.22f, PrecipSumStd = 0.65f, PrecipSumMax = 5.1f,
        AgreementHasDryWindow = 0.83f, LongestDryRunMean = 4.33f, WetHourCountMean = 11.5f,
        RhMean = 88f, RhMin = 70f, DewDepressionMax = 4.5f,
        CloudLowMean = 70f, CloudMidMean = 55f, CloudHighMean = 40f,
        CapeMax = 0f, WindMean = 18f, WindMax = 30f,
        DoySin = 0.5f, DoyCos = 0.5f,
        HasDryWindow = false,
        PrecipMmDay = 0f,
    };

    [Fact]
    public void HashFeatures_is_deterministic_for_identical_rows()
    {
        var a = MakeRow();
        var b = MakeRow();

        DryWindowPredictCommand.HashFeatures(a).Should().Be(DryWindowPredictCommand.HashFeatures(b));
    }

    [Fact]
    public void HashFeatures_changes_when_any_feature_changes()
    {
        var a = MakeRow();
        var b = MakeRow(offset: 0.0001f);

        DryWindowPredictCommand.HashFeatures(a).Should().NotBe(DryWindowPredictCommand.HashFeatures(b));
    }

    [Fact]
    public void HashFeatures_output_is_64_char_lowercase_hex_sha256()
    {
        var h = DryWindowPredictCommand.HashFeatures(MakeRow());

        h.Should().HaveLength(64);
        h.Should().MatchRegex("^[0-9a-f]{64}$");
    }

    [Fact]
    public void HashFeatures_ignores_label_and_target_date()
    {
        var a = MakeRow();
        var b = MakeRow();
        b.HasDryWindow = true;
        b.PrecipMmDay = 99f;
        b.TargetDateUtc = new DateTime(2030, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        // The 53-float feature vector only includes inference-time inputs;
        // the label and the raw target date must not leak into the hash.
        DryWindowPredictCommand.HashFeatures(a).Should().Be(DryWindowPredictCommand.HashFeatures(b));
    }
}
