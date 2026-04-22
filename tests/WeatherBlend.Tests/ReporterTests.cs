using FluentAssertions;
using WeatherBlend.Evaluate;
using WeatherBlend.Train;
using Xunit;

namespace WeatherBlend.Tests;

public class ReporterTests
{
    // Deterministic seed so every helper produces the same rows each test run.
    private const int Seed = 1234;

    [Fact]
    public void BuildPhase2bMarkdown_is_byte_identical_for_identical_inputs()
    {
        var a = Reporter.BuildPhase2bMarkdown(BuildInput(leads: new[] { 24, 48, 72 }));
        var b = Reporter.BuildPhase2bMarkdown(BuildInput(leads: new[] { 24, 48, 72 }));

        a.Should().Be(b, "two runs with equivalent inputs must produce the same markdown");
    }

    [Fact]
    public void BuildPhase2bMarkdown_iterates_leads_in_input_order_not_sorted()
    {
        // Input order [48, 24, 72] — each lead heading should appear in that order.
        var report = Reporter.BuildPhase2bMarkdown(BuildInput(leads: new[] { 48, 24, 72 }));

        var i48 = report.IndexOf("## Lead 48h", StringComparison.Ordinal);
        var i24 = report.IndexOf("## Lead 24h", StringComparison.Ordinal);
        var i72 = report.IndexOf("## Lead 72h", StringComparison.Ordinal);

        i48.Should().BeGreaterThan(-1);
        i24.Should().BeGreaterThan(i48, "input order is 48 before 24");
        i72.Should().BeGreaterThan(i24, "input order is 24 before 72");
    }

    [Fact]
    public void BuildPhase2bMarkdown_renders_stratified_month_keys_in_ascending_order()
    {
        // TestRows span months 11, 3, 7 — month table should render 3, 7, 11.
        var report = Reporter.BuildPhase2bMarkdown(BuildInput(
            leads: new[] { 24 },
            testMonths: new[] { 11, 3, 7 }));

        // Find the "by month" table and extract the leading month column of each data row.
        var monthSection = ExtractSection(report, "### Stratified MAE — by month");
        var monthOrder = DataRowFirstCell(monthSection).ToList();

        monthOrder.Should().Equal("3", "7", "11");
    }

    [Fact]
    public void BuildPhase2bMarkdown_renders_feature_importance_in_input_order()
    {
        var importance = new[]
        {
            ("temp_ecmwf", 0.42),
            ("temp_gfs",   0.31),
            ("hour_sin",   0.04),
        };

        var report = Reporter.BuildPhase2bMarkdown(BuildInput(
            leads: new[] { 24 },
            importance: importance));

        var section = ExtractSection(report, "### Feature importance (gain)");
        var names = DataRowFirstCell(section).Select(s => s.Trim('`')).ToList();

        names.Should().Equal("temp_ecmwf", "temp_gfs", "hour_sin");
    }

    [Fact]
    public void BuildPhase2bMarkdown_hour_bin_stratification_uses_four_sorted_bins()
    {
        // 8 rows distributed across all four 6-hour bins (00-05, 06-11, 12-17, 18-23).
        var hours = new[] { 3, 4, 7, 9, 14, 16, 20, 22 };
        var report = Reporter.BuildPhase2bMarkdown(BuildInput(
            leads: new[] { 24 },
            testHours: hours));

        var section = ExtractSection(report, "### Stratified MAE — by hour-of-day bin");
        var bins = DataRowFirstCell(section).ToList();

        bins.Should().Equal("0", "1", "2", "3");
    }

    // ------------------------------------------------------------------------
    // Helpers: build a minimal-but-valid Phase2bReportInput.
    // ------------------------------------------------------------------------

    private static Reporter.Phase2bReportInput BuildInput(
        int[] leads,
        int[]? testMonths = null,
        int[]? testHours = null,
        (string Name, double Gain)[]? importance = null)
    {
        var bundles = leads.Select(l => BuildBundle(l, testMonths, testHours, importance)).ToList();
        return new Reporter.Phase2bReportInput
        {
            GeneratedAtUtc = new DateTime(2026, 4, 22, 10, 0, 0, DateTimeKind.Utc),
            ModelVersion = "v-test",
            Metadata = new ModelArtifact.TrainingMetadata
            {
                Version = "v-test",
                Target = "temperature",
                Phase = "2b",
                DataSource = "previous_runs_api",
                TrainedAtUtc = new DateTime(2026, 4, 21, 20, 0, 0, DateTimeKind.Utc),
            },
            Leads = bundles,
        };
    }

    private static Reporter.LeadBundle BuildBundle(
        int leadHours,
        int[]? testMonths,
        int[]? testHours,
        (string Name, double Gain)[]? importance)
    {
        // Deterministic fake rows. If months/hours given, use them verbatim;
        // otherwise spread across a calendar year at noon so all strat buckets
        // are non-empty.
        var rng = new Random(Seed + leadHours);
        List<TrainingRow> testRows;
        if (testMonths is not null)
        {
            testRows = testMonths
                .Select((m, i) => FakeRow(new DateTime(2026, m, 15, (testHours?[i] ?? 12), 0, 0, DateTimeKind.Utc), rng))
                .ToList();
        }
        else if (testHours is not null)
        {
            testRows = testHours
                .Select(h => FakeRow(new DateTime(2026, 6, 15, h, 0, 0, DateTimeKind.Utc), rng))
                .ToList();
        }
        else
        {
            testRows = Enumerable.Range(0, 24)
                .Select(i => FakeRow(new DateTime(2026, 1 + (i % 12), 1 + (i % 28), (i * 3) % 24, 0, 0, DateTimeKind.Utc), rng))
                .ToList();
        }

        var actual = testRows.Select(r => (double)r.Era5Temp).ToArray();
        var blend  = actual.Select(v => v + 0.10).ToArray();  // fixed +0.1°C bias
        var gfs    = actual.Select(v => v + 0.30).ToArray();
        var ecmwf  = actual.Select(v => v + 0.20).ToArray();
        var mean   = actual.Select(v => v + 0.25).ToArray();

        var baselines = new List<Reporter.ModelPrediction>
        {
            new("temp_gfs",       gfs),
            new("temp_ecmwf",     ecmwf),
            new("temp_icon",      gfs),
            new("temp_mf",        gfs),
            new("temp_ukmo",      gfs),
            new("temp_gem",       gfs),
            new("mean_of_models", mean),
            new($"persistence_-{leadHours}h", actual),       // perfect persistence = 0 MAE
            new("climatology(month,hour)",    mean),
        };

        return new Reporter.LeadBundle
        {
            LeadHours = leadHours,
            Stats = new ModelArtifact.PerLeadStats
            {
                LeadHours = leadHours,
                DataRangeTrain = "2024-06-01 → 2025-10-15",
                DataRangeVal   = "2025-10-15 → 2025-12-31",
                DataRangeTest  = "2026-01-01 → 2026-03-31",
                TrainRows = 6000, ValRows = 1000, TestRows = testRows.Count,
                TestCalendarMonths = 3,
                BestSingle = "temp_ecmwf",
                BestSingleValMae = 1.60,
                BlendTestMae = 1.23, BlendTestRmse = 1.70, BlendTestBias = -0.05,
            },
            TestRows = testRows,
            BlendTest = new Reporter.ModelPrediction("blend", blend),
            BaselinesTest = baselines,
            BestSingleName = "temp_ecmwf",
            FeatureImportance = importance ?? new[]
            {
                ("temp_ecmwf", 0.42),
                ("temp_gfs",   0.31),
            },
            PersistenceDropped = 0,
        };
    }

    private static TrainingRow FakeRow(DateTime validUtc, Random rng)
    {
        var era5 = 10f + (float)(rng.NextDouble() * 10.0);
        return new TrainingRow
        {
            ValidTimeUtc = validUtc,
            TempGfs = era5, TempEcmwf = era5, TempIcon = era5, TempMf = era5, TempUkmo = era5, TempGem = era5,
            TempMean = era5, TempStd = 0.1f, TempRange = 0.3f,
            HourSin = 0, HourCos = 1, DoySin = 0, DoyCos = 1,
            WindDirMean = 90f,   // E quadrant → non-"?" wind-quadrant key
            Era5Temp = era5,
        };
    }

    /// <summary>Grab everything from a heading line up to the next blank heading-prefix line.</summary>
    private static string ExtractSection(string report, string heading)
    {
        var idx = report.IndexOf(heading, StringComparison.Ordinal);
        idx.Should().BeGreaterThan(-1, $"report should contain heading '{heading}'");
        var rest = report.Substring(idx);
        var nextHeading = rest.IndexOf("\n## ", StringComparison.Ordinal);
        var nextSub     = rest.IndexOf("\n### ", 1, StringComparison.Ordinal);
        var end = new[] { nextHeading, nextSub }.Where(i => i > 0).DefaultIfEmpty(rest.Length).Min();
        return rest.Substring(0, end);
    }

    /// <summary>From a markdown table section, yield the first cell of each data row (skipping header + separator).</summary>
    private static IEnumerable<string> DataRowFirstCell(string section)
    {
        var lines = section.Split('\n');
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("|")) continue;
            if (trimmed.StartsWith("|---")) continue;
            // Skip header rows — they contain alpha like "MAE" or "Feature". We
            // detect data rows as "first cell parses as int/string literal" by
            // filtering out the first | row containing multiple header words.
            var cells = trimmed.Trim('|').Split('|');
            if (cells.Length < 2) continue;
            var first = cells[0].Trim();
            // Header rows have "MAE" / "Feature" / "Month" etc. as later cells;
            // data rows have numeric bin keys or backtick-wrapped names in cell 0.
            if (first == "Month" || first == "Hour bin" || first == "Quadrant"
                || first == "Spread Q" || first == "Feature" || first == "Predictor")
                continue;
            yield return first;
        }
    }
}
