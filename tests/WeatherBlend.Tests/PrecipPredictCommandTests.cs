using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipPredictCommandTests
{
    private static PrecipTrainingRow MakeRow(float offset = 0f) => new()
    {
        ValidTimeUtc = new DateTime(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc),
        PrecipGfs = 0.2f + offset, PrecipEcmwf = 0.1f, PrecipIcon = 0f,
        PrecipMf = 0.3f, PrecipUkmo = 0.15f, PrecipGem = 0.05f,
        ProbGfs = 60f, ProbEcmwf = 50f, ProbIcon = 40f,
        ProbMf = 70f, ProbUkmo = 55f, ProbGem = 45f,
        PrecipMean = 0.13f, PrecipStd = 0.1f, PrecipMax = 0.3f, PrecipAgreementWet01 = 0.67f,
        RhMean = 88f, DewDepressionMean = 1.5f,
        CloudLowMean = 80f, CloudMidMean = 60f, CloudHighMean = 40f,
        CapeMean = 0f, WindSpeedMean = 15f,
        HourSin = 0f, HourCos = 1f, DoySin = 0.5f, DoyCos = 0.5f,
        WetBinary = false,
        PrecipMmHour = 0f,
    };

    [Fact]
    public void HashFeatures_is_deterministic_for_identical_rows()
    {
        var a = MakeRow();
        var b = MakeRow();

        PrecipPredictCommand.HashFeatures(a).Should().Be(PrecipPredictCommand.HashFeatures(b));
    }

    [Fact]
    public void HashFeatures_changes_when_any_feature_changes()
    {
        var a = MakeRow();
        var b = MakeRow(offset: 0.0001f);

        PrecipPredictCommand.HashFeatures(a).Should().NotBe(PrecipPredictCommand.HashFeatures(b));
    }

    [Fact]
    public void HashFeatures_output_is_64_char_lowercase_hex_sha256()
    {
        var h = PrecipPredictCommand.HashFeatures(MakeRow());

        h.Should().HaveLength(64);
        h.Should().MatchRegex("^[0-9a-f]{64}$");
    }

    [Fact]
    public void HashFeatures_ignores_label_and_valid_time()
    {
        var a = MakeRow();
        var b = MakeRow();
        b.WetBinary = true;
        b.PrecipMmHour = 99f;
        b.ValidTimeUtc = new DateTime(2030, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        // The 27-float feature vector includes only inputs used at inference time;
        // the truth label and raw timestamp must not leak into the hash.
        PrecipPredictCommand.HashFeatures(a).Should().Be(PrecipPredictCommand.HashFeatures(b));
    }
}
