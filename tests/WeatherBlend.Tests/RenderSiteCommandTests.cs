using DuckDB.NET.Data;
using FluentAssertions;
using Parquet.Serialization;
using WeatherBlend.Commands;
using WeatherBlend.Site;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Integration tests pinning the original <c>ApparentTemperatureC</c> regression
/// against the schema-probe path now exposed by
/// <see cref="WeatherBlend.Storage.ParquetReader.HasColumn"/>. When no parquet
/// in the tree carried the column, <c>union_by_name=true</c> excluded it from
/// the unified schema and DuckDB binder-errored on any SELECT referencing it.
/// Probing first prevents the crash; these tests pin the probe's behaviour
/// across the variants the renderer cares about.
/// </summary>
public class RenderSiteCommandTests : IDisposable
{
    private readonly string _root;

    public RenderSiteCommandTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "wb-rendersite-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    // ---------- ParquetSchemaHasColumn ----------

    [Fact]
    public void ParquetSchemaHasColumn_returns_false_when_glob_matches_no_files()
    {
        // Empty directory → read_parquet throws "No files found" — the helper
        // must catch it and return false rather than propagate the exception.
        // Renderer relies on this to render the no-data fallback chip silently
        // when the prediction tree hasn't been synced yet.
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeFalse();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_returns_false_when_no_file_carries_the_column()
    {
        // Old-schema parquets only — none carry ApparentTemperatureC. This is the
        // exact state that caused the binder error: union_by_name unifies columns
        // present in at least one file, and "zero files" means the column is
        // absent from the unified schema and SELECTing it explodes.
        await WriteParquetAsync("data1.parquet", new[]
        {
            new SimpleRow { Name = "a", UtciC = 1.0 },
            new SimpleRow { Name = "b", UtciC = 2.0 },
        });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeFalse();
        // Sanity: a column that DOES exist still resolves true on the same tree.
        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "UtciC")
            .Should().BeTrue();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_returns_true_when_at_least_one_file_carries_it()
    {
        // Mixed schema: one old file (no ApparentTemperatureC), one new file
        // (carrying it). The post-rename steady state where the writer has
        // emitted some new-schema rows but old rows survive on R2. The probe
        // must say "yes, the column is in the unified schema" so the SELECT
        // can reference it (NULL for old rows, populated for new).
        await WriteParquetAsync("old.parquet", new[]
        {
            new SimpleRow { Name = "old", UtciC = 1.0 },
        });
        await WriteParquetAsync("new.parquet", new[]
        {
            new ExtendedRow { Name = "new", UtciC = 2.0, ApparentTemperatureC = 5.5 },
        });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "ApparentTemperatureC")
            .Should().BeTrue();
    }

    [Fact]
    public async Task ParquetSchemaHasColumn_handles_quoted_column_names_safely()
    {
        // Column name containing a quote would otherwise inject into the SQL
        // and either crash or match nothing. The helper escapes the literal.
        await WriteParquetAsync("d.parquet", new[] { new SimpleRow { Name = "x", UtciC = 1.0 } });
        var glob = Path.Combine(_root, "**", "*.parquet").Replace('\\', '/');
        using var conn = OpenConn();

        // No injection — the predicate just doesn't match anything.
        WeatherBlend.Storage.ParquetReader.HasColumn(conn, glob, "Bobby'); DROP TABLE--")
            .Should().BeFalse();
    }

    // ---------- ComputeRollingMae ----------

    private static WeatherBlend.Models.TempPredictionRow Pred(
        string version, int leadHours, DateTime validTime, double blendTemperature)
        => new()
        {
            LocationName = "bonehill_rocks",
            ModelVersion = version,
            PredictionMadeAtUtc = validTime.AddHours(-leadHours),
            ValidTimeUtc = validTime,
            LeadHours = leadHours,
            BlendTemperature = blendTemperature,
            FeatureVectorHash = "",
        };

    [Fact]
    public void ComputeRollingMae_returns_empty_when_no_pairs()
    {
        // No predictions at all → empty. Same with predictions whose valid
        // times are missing from the truth dict — pairing drops them.
        var truth = new Dictionary<DateTime, double>();
        RenderSiteCommand.ComputeRollingMae(
                Array.Empty<WeatherBlend.Models.TempPredictionRow>(),
                truth, windowDays: 14)
            .Should().BeEmpty();

        var pred = new[] { Pred("v1", 24,
            new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc), 10.0) };
        RenderSiteCommand.ComputeRollingMae(pred, truth, 14).Should().BeEmpty();
    }

    [Fact]
    public void ComputeRollingMae_emits_partial_window_points_when_data_shorter_than_window()
    {
        // Bug fix 2026-04-30: with 3 days of paired data and a 14-day rolling
        // window, the loop used to start at minDate + windowDays - 1 (= 14
        // days into the future) and never enter the body, leaving the chart
        // empty even though pairs existed. Now we emit a point per paired
        // day from minDate, with a partial window where needed.
        var t1 = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var t2 = new DateTime(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc);
        var t3 = new DateTime(2026, 4, 24, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Pred("v1", 24, t1, 10.0),
            Pred("v1", 24, t2, 11.0),
            Pred("v1", 24, t3, 12.0),
        };
        var truth = new Dictionary<DateTime, double>
        {
            [t1] = 9.0, [t2] = 10.0, [t3] = 13.0,
        };

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, windowDays: 14);

        // One point per paired day. Cumulative MAE because the window covers
        // all of the paired data behind each day.
        points.Should().HaveCount(3);
        points.Select(p => p.WindowEndUtc.Date).Should().Equal(t1.Date, t2.Date, t3.Date);
        points.Select(p => p.N).Should().Equal(1, 2, 3);
        points[0].BlendMae.Should().BeApproximately(1.0, 1e-9);   // |10-9|
        points[1].BlendMae.Should().BeApproximately(1.0, 1e-9);   // (1 + 1) / 2
        points[2].BlendMae.Should().BeApproximately(1.0, 1e-9);   // (1 + 1 + 1) / 3
    }

    [Fact]
    public void ComputeRollingMae_window_slides_so_old_pairs_drop_off_after_windowDays()
    {
        // Five paired days, 3-day rolling window: the oldest pair leaves the
        // window once we get to day 4, so day-4's MAE is computed over days
        // 2..4 (not days 1..4).
        var t1 = new DateTime(2026, 4, 1, 12, 0, 0, DateTimeKind.Utc);
        var preds = Enumerable.Range(0, 5)
            .Select(i => Pred("v1", 24, t1.AddDays(i), 10.0 + i))
            .ToArray();
        var truth = preds.ToDictionary(p => p.ValidTimeUtc, p => p.BlendTemperature - 1.0);

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, windowDays: 3);

        // 5 points, one per day. N caps at 3 from day 3 onwards.
        points.Should().HaveCount(5);
        points.Select(p => p.N).Should().Equal(1, 2, 3, 3, 3);
        // Each pair has |pred - truth| = 1, so MAE = 1 across every window.
        points.Select(p => p.BlendMae).Should().AllBeEquivalentTo(1.0);
    }

    // ---------- ComputeRollingBrier ----------

    private static SitePages.PrecipForecastPoint Precip(
        string station, string version, int leadHours, DateTime validTime, double probWet)
        => new(
            Station: station,
            Version: version,
            PredictedAtUtc: validTime.AddHours(-leadHours),
            ValidTimeUtc: validTime,
            LeadHours: leadHours,
            ProbWet: probWet,
            ClimatologyPWet: 0.5,
            PrecipGfs: null, PrecipEcmwf: null, PrecipIcon: null,
            PrecipMf: null, PrecipUkmo: null, PrecipGem: null,
            PrecipAifs: null, PrecipJma: null);

    private static IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>>
        TruthFor(string station, IReadOnlyDictionary<DateTime, double> truthByTime)
        => new Dictionary<string, IReadOnlyDictionary<DateTime, double>> { [station] = truthByTime };

    [Fact]
    public void ComputeRollingBrier_returns_empty_when_no_pairs()
    {
        // No predictions OR no station-truth-dict overlap → empty list. Same
        // failure modes as ComputeRollingMae's "no pairs" early-return path.
        RenderSiteCommand.ComputeRollingBrier(
                Array.Empty<SitePages.PrecipForecastPoint>(),
                new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(),
                windowDays: 30)
            .Should().BeEmpty();

        var preds = new[] { Precip("ea_bellever_dartmoor", "v3a", 24,
            new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc), 0.5) };
        RenderSiteCommand.ComputeRollingBrier(preds,
                new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(), 30)
            .Should().BeEmpty();
    }

    [Fact]
    public void ComputeRollingBrier_thresholds_truth_at_0_1mm_and_computes_brier()
    {
        // Two predictions: P(wet)=0.8 with truth 0.5mm (wet, binary 1) →
        // squared error = 0.04. P(wet)=0.2 with truth 0.05mm (dry, binary 0)
        // → squared error = 0.04. Both windows include both, so daily Brier
        // = 0.04 once the second pair is in the window. The 0.1mm threshold
        // is the blender's training label boundary; pinning it here so a
        // future code-side threshold change immediately surfaces.
        var t1 = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var t2 = new DateTime(2026, 4, 23, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Precip("ea_bellever_dartmoor", "v3a", 24, t1, 0.8),
            Precip("ea_bellever_dartmoor", "v3a", 24, t2, 0.2),
        };
        var truth = TruthFor("ea_bellever_dartmoor",
            new Dictionary<DateTime, double> { [t1] = 0.5, [t2] = 0.05 });

        var points = RenderSiteCommand.ComputeRollingBrier(preds, truth, windowDays: 30);

        points.Should().HaveCount(2);
        points[0].BlendBrier.Should().BeApproximately(0.04, 1e-9);
        points[1].BlendBrier.Should().BeApproximately(0.04, 1e-9);
        points.Select(p => p.N).Should().Equal(1, 2);
        points.Select(p => p.Station).Should().AllBeEquivalentTo("ea_bellever_dartmoor");
    }

    [Fact]
    public void ComputeRollingBrier_window_slides_and_old_pairs_drop_off()
    {
        // 5 daily pairs, 3-day rolling window. Each pair has P(wet) = 0.5
        // against a dry hour (truth 0.0mm < 0.1mm threshold → binary 0).
        // Squared error per pair = (0.5 − 0)² = 0.25. Window cap of 3
        // means N goes 1, 2, 3, 3, 3 and Brier stays at 0.25 throughout
        // (mean of identical 0.25 values).
        var t1 = new DateTime(2026, 4, 1, 12, 0, 0, DateTimeKind.Utc);
        var preds = Enumerable.Range(0, 5)
            .Select(i => Precip("ea_bellever_dartmoor", "v3a", 24, t1.AddDays(i), 0.5))
            .ToArray();
        var truth = TruthFor("ea_bellever_dartmoor",
            preds.ToDictionary(p => p.ValidTimeUtc, _ => 0.0));

        var points = RenderSiteCommand.ComputeRollingBrier(preds, truth, windowDays: 3);

        points.Should().HaveCount(5);
        points.Select(p => p.N).Should().Equal(1, 2, 3, 3, 3);
        points.Select(p => p.BlendBrier).Should().AllSatisfy(b =>
            b.Should().BeApproximately(0.25, 1e-9));
    }

    [Fact]
    public void ComputeRollingBrier_separates_per_station_per_version_per_lead()
    {
        // Two stations × two versions × two leads → eight series, no
        // cross-talk. Mirrors the ComputeRollingMae per-(version, lead)
        // separation but with a station axis on top.
        var t = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Precip("ea_bellever_dartmoor", "v3a", 24, t, 0.6),
            Precip("ea_bellever_dartmoor", "v3a", 48, t, 0.55),
            Precip("ea_bellever_dartmoor", "v3c", 24, t, 0.65),
            Precip("ea_bellever_dartmoor", "v3c", 48, t, 0.50),
            Precip("ea_princetown",        "v3a", 24, t, 0.40),
            Precip("ea_princetown",        "v3a", 48, t, 0.45),
            Precip("ea_princetown",        "v3c", 24, t, 0.42),
            Precip("ea_princetown",        "v3c", 48, t, 0.48),
        };
        var truth = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>
        {
            ["ea_bellever_dartmoor"] = new Dictionary<DateTime, double> { [t] = 0.5 },
            ["ea_princetown"]        = new Dictionary<DateTime, double> { [t] = 0.0 },
        };

        var points = RenderSiteCommand.ComputeRollingBrier(preds, truth, windowDays: 30);

        points.Should().HaveCount(8);
        points.Select(p => (p.Station, p.ModelVersion, p.LeadHours)).Should().BeEquivalentTo(
            new (string, string, int)[]
            {
                ("ea_bellever_dartmoor", "v3a", 24),
                ("ea_bellever_dartmoor", "v3a", 48),
                ("ea_bellever_dartmoor", "v3c", 24),
                ("ea_bellever_dartmoor", "v3c", 48),
                ("ea_princetown",        "v3a", 24),
                ("ea_princetown",        "v3a", 48),
                ("ea_princetown",        "v3c", 24),
                ("ea_princetown",        "v3c", 48),
            });
    }

    [Fact]
    public void ComputeRollingMae_separates_per_version_and_per_lead()
    {
        // Two versions × two leads → four series of points, no cross-talk.
        var t = new DateTime(2026, 4, 22, 12, 0, 0, DateTimeKind.Utc);
        var preds = new[]
        {
            Pred("v1", 24, t, 10.0),
            Pred("v1", 48, t, 11.0),
            Pred("v2", 24, t, 12.0),
            Pred("v2", 48, t, 13.0),
        };
        var truth = new Dictionary<DateTime, double> { [t] = 10.0 };

        var points = RenderSiteCommand.ComputeRollingMae(preds, truth, 14);
        points.Should().HaveCount(4);
        points.Select(p => (p.ModelVersion, p.LeadHours)).Should().BeEquivalentTo(
            new (string, int)[] { ("v1", 24), ("v1", 48), ("v2", 24), ("v2", 48) });
    }

    // ---------- helpers ----------

    private static DuckDBConnection OpenConn()
    {
        var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        return conn;
    }

    private async Task WriteParquetAsync<T>(string filename, IEnumerable<T> rows) where T : class
    {
        var path = Path.Combine(_root, filename);
        await ParquetSerializer.SerializeAsync(rows.ToList(), path);
    }

    // Tiny POCOs purely for fixture-writing — keep them flat so the parquet
    // serializer's reflection-based mapping doesn't drag in unrelated types.
    public sealed class SimpleRow
    {
        public string Name { get; set; } = "";
        public double UtciC { get; set; }
    }

    public sealed class ExtendedRow
    {
        public string Name { get; set; } = "";
        public double UtciC { get; set; }
        public double ApparentTemperatureC { get; set; }
    }
}
