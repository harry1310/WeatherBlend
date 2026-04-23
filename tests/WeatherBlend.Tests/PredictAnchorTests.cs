using FluentAssertions;
using WeatherBlend.Predict;
using Xunit;

namespace WeatherBlend.Tests;

public class PredictAnchorTests
{
    [Fact]
    public void Compute_live_truncates_utcNow_to_top_of_hour()
    {
        var now = new DateTime(2026, 4, 23, 14, 37, 42, 913, DateTimeKind.Utc);

        var anchor = PredictAnchor.Compute(now, forDate: null);

        anchor.Should().Be(new DateTime(2026, 4, 23, 14, 0, 0, DateTimeKind.Utc));
        anchor.Kind.Should().Be(DateTimeKind.Utc);
    }

    [Fact]
    public void Compute_live_at_exact_hour_is_noop_truncation()
    {
        var now = new DateTime(2026, 4, 23, 9, 0, 0, DateTimeKind.Utc);

        PredictAnchor.Compute(now, forDate: null).Should().Be(now);
    }

    [Fact]
    public void Compute_backfill_uses_0800_utc_regardless_of_now()
    {
        var now = new DateTime(2026, 4, 23, 14, 37, 42, DateTimeKind.Utc);
        var forDate = new DateOnly(2026, 1, 15);

        var anchor = PredictAnchor.Compute(now, forDate);

        anchor.Should().Be(new DateTime(2026, 1, 15, 8, 0, 0, DateTimeKind.Utc));
        anchor.Kind.Should().Be(DateTimeKind.Utc);
    }

    [Fact]
    public void BackfillAnchor_is_0800_utc()
    {
        PredictAnchor.BackfillAnchor.Should().Be(new TimeSpan(8, 0, 0));
    }
}
