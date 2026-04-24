using FluentAssertions;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class PrecipRichFeatureBuilderTests
{
    private static readonly DateTime Valid = new(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc);

    [Fact]
    public void OccurrenceFeatureNames_is_55_and_starts_with_the_lean_27()
    {
        PrecipRichFeatureBuilder.OccurrenceFeatureNames.Count.Should().Be(55);
        PrecipRichFeatureBuilder.OccurrenceFeatureNames
            .Take(PrecipFeatureBuilder.OccurrenceFeatureNames.Count)
            .Should().Equal(PrecipFeatureBuilder.OccurrenceFeatureNames);
        PrecipRichFeatureBuilder.OccurrenceFeatureNames.Should().EndWith(new[]
        {
            "ea_rain_prev_24h_mm", "ea_rain_prev_72h_mm",
            "ea_wet_hours_last_24h", "ea_dry_hours_trailing",
        });
    }

    [Fact]
    public void ComposeRow_delegates_lean_features_to_PrecipFeatureBuilder()
    {
        // Build a lean row and a rich row from the same inputs; every lean field
        // on the rich row should match the lean row exactly.
        var precip = new[] { 0.5, 1.0, 0.0, 0.2, double.NaN, 0.8 };
        var probs  = new[] { 0.6, 0.8, 0.1, 0.3, double.NaN, 0.7 };
        var dew    = new[] { 8.0, 7.5, 8.2, 7.8, 8.1, 7.9 };
        var rh     = new[] { 82.0, 85.0, 80.0, 83.0, 81.0, 84.0 };
        var dewDep = new[] { 1.5, 1.2, 1.8, 1.4, 1.6, 1.3 };
        var pres   = new[] { 1012.0, 1012.5, 1011.8, 1012.2, 1012.1, 1012.3 };

        var lean = PrecipFeatureBuilder.ComposeRow(
            Valid, precip, probs,
            rhMean: 82.5, dewDepressionMean: 1.5,
            cloudLowMean: 60, cloudMidMean: 40, cloudHighMean: 20,
            capeMean: 100, windSpeedMean: 5,
            truthMmHour: 0.3);

        var rich = PrecipRichFeatureBuilder.ComposeRow(
            Valid, precip, probs, dew, rh, dewDep, pres,
            rhMean: 82.5, dewDepressionMean: 1.5,
            cloudLowMean: 60, cloudMidMean: 40, cloudHighMean: 20,
            capeMean: 100, windSpeedMean: 5,
            eaRainPrev24hMm: 2.4, eaRainPrev72hMm: 8.1,
            eaWetHoursLast24h: 3, eaDryHoursTrailing: 6,
            truthMmHour: 0.3);

        rich.PrecipMean.Should().Be(lean.PrecipMean);
        rich.PrecipStd.Should().Be(lean.PrecipStd);
        rich.PrecipMax.Should().Be(lean.PrecipMax);
        rich.PrecipAgreementWet01.Should().Be(lean.PrecipAgreementWet01);
        rich.WetBinary.Should().Be(lean.WetBinary);
        rich.HourSin.Should().Be(lean.HourSin);
        rich.DoyCos.Should().Be(lean.DoyCos);
    }

    [Fact]
    public void ComposeRow_assigns_rich_per_model_columns_in_declared_order()
    {
        var dew    = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var rh     = new[] { 70.0, 71.0, 72.0, 73.0, 74.0, 75.0 };
        var dewDep = new[] { 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
        var pres   = new[] { 1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0 };

        var row = PrecipRichFeatureBuilder.ComposeRow(
            Valid,
            perModelPrecip: new[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 },
            perModelProb:   new[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 },
            dew, rh, dewDep, pres,
            rhMean: 72.5, dewDepressionMean: 0.75,
            cloudLowMean: 0, cloudMidMean: 0, cloudHighMean: 0,
            capeMean: 0, windSpeedMean: 0,
            eaRainPrev24hMm: 0, eaRainPrev72hMm: 0,
            eaWetHoursLast24h: 0, eaDryHoursTrailing: 0,
            truthMmHour: 0);

        // Ordering is gfs, ecmwf, icon, mf, ukmo, gem — load-bearing.
        row.DewGfs.Should().Be(1f); row.DewGem.Should().Be(6f);
        row.RhEcmwf.Should().Be(71f); row.RhUkmo.Should().Be(74f);
        row.DewDepressionIcon.Should().Be(0.7f);
        row.PressureMf.Should().Be(1003f);
    }

    [Fact]
    public void ComposeRow_preserves_NaN_in_rich_columns()
    {
        var dew = new[] { double.NaN, 7.5, 8.2, 7.8, 8.1, 7.9 };

        var row = PrecipRichFeatureBuilder.ComposeRow(
            Valid,
            perModelPrecip: new[] { 0.5, 1.0, 0.0, 0.2, double.NaN, 0.8 },
            perModelProb:   new[] { 0.6, 0.8, 0.1, 0.3, double.NaN, 0.7 },
            dew,
            perModelRh:             new double[6],
            perModelDewDepression:  new double[6],
            perModelPressure:       new double[6],
            rhMean: 0, dewDepressionMean: 0,
            cloudLowMean: 0, cloudMidMean: 0, cloudHighMean: 0,
            capeMean: 0, windSpeedMean: 0,
            eaRainPrev24hMm: 0, eaRainPrev72hMm: 0,
            eaWetHoursLast24h: 0, eaDryHoursTrailing: 0,
            truthMmHour: 0);

        float.IsNaN(row.DewGfs).Should().BeTrue();
        row.DewEcmwf.Should().Be(7.5f);
    }

    [Fact]
    public void ComputePersistence_sums_full_24h_and_72h_windows()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 96; h++) hourly[runTime.AddHours(-h)] = 0.1;  // 0.1 mm/h everywhere

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.Prev24hMm.Should().BeApproximately(0.1 * 24, 1e-9);   // 24 hours inclusive of runTime
        p.Prev72hMm.Should().BeApproximately(0.1 * 72, 1e-9);
    }

    [Fact]
    public void ComputePersistence_counts_wet_hours_strictly_at_or_above_threshold()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        // 0.1 mm hits the wet threshold; 0.09 does not.
        for (int h = 0; h < 24; h++)
            hourly[runTime.AddHours(-h)] = h < 5 ? 0.1 : 0.09;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.WetHoursLast24h.Should().Be(5);
    }

    [Fact]
    public void ComputePersistence_trailing_dry_walks_back_through_dry_hours_only()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        // Last 4h dry, then a wet hour, then more dry. Trailing run = 4.
        for (int h = 0; h < 20; h++)
        {
            double mm = h switch
            {
                < 4 => 0.05,     // dry
                4 => 1.0,         // wet — stops the run
                _ => 0.0
            };
            hourly[runTime.AddHours(-h)] = mm;
        }

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(4);
    }

    [Fact]
    public void ComputePersistence_returns_NaN_for_24h_sum_when_coverage_incomplete()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        for (int h = 0; h < 72; h++) hourly[runTime.AddHours(-h)] = 0.0;
        hourly.Remove(runTime.AddHours(-10)); // drop one hour inside the 24h window

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        double.IsNaN(p.Prev24hMm).Should().BeTrue();
        double.IsNaN(p.Prev72hMm).Should().BeTrue(); // 24h gap also breaks 72h coverage
        double.IsNaN(p.WetHoursLast24h).Should().BeTrue();
    }

    [Fact]
    public void ComputePersistence_trailing_dry_stops_at_missing_reading()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double>();
        // 3 dry hours, then a gap. We can't claim "dry since 4h ago" without the reading.
        for (int h = 0; h < 3; h++) hourly[runTime.AddHours(-h)] = 0.0;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(3);
    }

    [Fact]
    public void ComputePersistence_trailing_dry_zero_when_runtime_hour_is_wet()
    {
        var runTime = new DateTime(2026, 4, 23, 6, 0, 0, DateTimeKind.Utc);
        var hourly = new Dictionary<DateTime, double> { [runTime] = 2.0 };
        for (int h = 1; h < 20; h++) hourly[runTime.AddHours(-h)] = 0.0;

        var p = PrecipRichFeatureBuilder.ComputePersistence(hourly, runTime);

        p.DryHoursTrailing.Should().Be(0);
    }

    [Fact]
    public void LoadHourlyRain_throws_diagnosable_error_when_rainfall_tree_is_empty()
    {
        // Regression: predict.yml used to skip 'data/truth/rainfall' in its R2 pull,
        // so Phase 3c predict would crash deep inside DuckDB with a raw
        // "No files found" IOException that gave CI operators no clue what tree to
        // populate. LoadHourlyRain now translates that into a targeted
        // InvalidOperationException naming the path and the action to take.
        using var tmp = new TempDirectory();
        var missingPath = Path.Combine(tmp.Path, "does_not_exist");

        var act = () => PrecipRichFeatureBuilder.LoadHourlyRain(
            rainfallPath: missingPath,
            locationName: "Bonehill Rocks, Dartmoor",
            stationName: "Bellever, Dartmoor",
            ct: CancellationToken.None);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Rainfall truth tree is empty*")
            .WithMessage("*does_not_exist*")   // path fragment we injected, slash-agnostic
            .WithMessage("*data/truth/rainfall*");
    }

    private sealed class TempDirectory : IDisposable
    {
        public string Path { get; }
        public TempDirectory()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "wb_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { /* best-effort */ }
        }
    }
}
