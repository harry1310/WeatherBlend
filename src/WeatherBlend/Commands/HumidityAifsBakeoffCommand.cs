using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Humidity;

namespace WeatherBlend.Commands;

/// <summary>
/// EXPERIMENT (2026-06-10): why does the humidity blend lose to raw AIFS at
/// every lead — and what fixes it? The live bundle's own test slice shows
/// AIFS beating the blend at 24/48/72h, and AIFS is NaN in ~63% of the
/// bundle's training rows (AIFS data starts 2025-02-17; training reaches
/// back to 2024-02), so the trees learn on a mostly-AIFS-missing
/// distribution while predict always has AIFS — the same train/predict
/// mismatch the wind-speed permissive-build bake-off proved harmful.
///
/// Four arms per lead, all scored on the SAME held-out OOS rows
/// (ValidTime ≥ --test-start, default 2026-03-01, ERA5 truth):
///   control     — current 17-feature spec, full training window (≈ live).
///   aifs-dense  — same spec, training floored at 2025-02-17 (AIFS present
///                 in every training row; ~half the rows).
///   rich        — +6 cross-variable NWP-ensemble means (temp, dew-point
///                 depression, cloud low/mid/high, wind) on the full window —
///                 the same trick that won the radiation bake-off (2026-06-03).
///   dense+rich  — both.
/// Yardstick: raw AIFS RH MAE on the same test rows. The blend should beat
/// it; today's live bundle doesn't.
///
/// Deliberately NOT tested: making AIFS a required model — required inputs
/// turn a feed outage into a zeroed blender (the 2026-06-01/05 GEM
/// incidents); the dense-window floor achieves the same training-density
/// goal as a train-time-only constraint.
/// </summary>
public sealed class HumidityAifsBakeoffCommand
{
    private readonly ILogger<HumidityAifsBakeoffCommand> _log;
    private readonly AppConfig _cfg;

    public HumidityAifsBakeoffCommand(ILogger<HumidityAifsBakeoffCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    private static readonly string[] RichAuxNames =
        { "temp_mean", "dewdep_mean", "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean", "wind_mean" };

    private sealed record Row(DateTime Valid, double[] Rh, double[] Dp, double[] Aux, double Truth);

    private sealed record ArmScore(string Arm, int NTrain, double Mae);

    public async Task<int> RunAsync(string? testStartStr, CancellationToken ct)
    {
        await Task.Yield();
        var loc = _cfg.Location.Name;
        var testStart = DateOnly.TryParse(testStartStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 1, 0, 0, 0, DateTimeKind.Utc);
        var aifsDenseFrom = new DateTime(2025, 2, 17, 0, 0, 0, DateTimeKind.Utc);
        int[] leads = { 24, 48, 72 };
        var hp = TempTrainer.Hyperparameters.Default();

        _log.LogInformation(
            "Humidity/AIFS bake-off — loc={Loc}, common OOS test ≥ {T:yyyy-MM-dd}, dense floor {F:yyyy-MM-dd}",
            loc, testStart, aifsDenseFrom);

        Console.WriteLine();
        Console.WriteLine($"=== humidity vs AIFS bake-off (ERA5 truth, common OOS test ≥ {testStart:yyyy-MM-dd}, {loc}) ===");
        Console.WriteLine("  RH MAE (%), lower = better. 'raw AIFS' is the same rows' ecmwf_aifs025_single RH vs truth.");
        Console.WriteLine($"  {"lead",5} {"N_test",7} {"raw AIFS",9} | {"arm",11} {"N_train",8} {"MAE",8} {"Δ vs AIFS",10}");

        var any = false;
        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            var spec = HumidityFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var rows = BuildRows(spec, loc, ct);
            var aifsIdx = -1;
            for (int i = 0; i < spec.Models.Count; i++)
                if (spec.Models[i] == "ecmwf_aifs025_single") { aifsIdx = i; break; }
            if (aifsIdx < 0) { _log.LogError("AIFS not in humidity spec at lead {L}h — abort.", lead); return 2; }

            var test = rows.Where(r => r.Valid >= testStart).ToList();
            var trainFull = rows.Where(r => r.Valid < testStart).ToList();
            var trainDense = trainFull.Where(r => r.Valid >= aifsDenseFrom).ToList();
            if (test.Count < 200 || trainDense.Count < 1000)
            {
                _log.LogWarning("lead {L}h: too few rows (test={T} dense={D}) — skipping.", lead, test.Count, trainDense.Count);
                continue;
            }

            // Yardstick: raw AIFS on the test rows.
            var aifsErrs = test.Where(r => !double.IsNaN(r.Rh[aifsIdx]))
                               .Select(r => Math.Abs(r.Rh[aifsIdx] - r.Truth)).ToList();
            var aifsMae = aifsErrs.Average();

            var richSpec = WithRichAux(spec);
            var arms = new (string Name, List<Row> Train, BlenderSpec Spec, bool Rich)[]
            {
                ("control",    trainFull,  spec,     false),
                ("aifs-dense", trainDense, spec,     false),
                ("rich",       trainFull,  richSpec, true),
                ("dense+rich", trainDense, richSpec, true),
            };

            var first = true;
            foreach (var arm in arms)
            {
                var trainRows = arm.Train.Select(r => Compose(arm.Spec, r, arm.Rich)).ToList();
                var testRows = test.Select(r => Compose(arm.Spec, r, arm.Rich)).ToList();
                var mae = TrainAndScore(trainRows, testRows, arm.Spec, hp);
                var dPct = (mae - aifsMae) / aifsMae * 100.0;
                Console.WriteLine(
                    $"  {(first ? lead + "h" : ""),5} {(first ? test.Count.ToString() : ""),7} {(first ? aifsMae.ToString("F4") : ""),9} | " +
                    $"{arm.Name,11} {arm.Train.Count,8} {mae,8:F4} {dPct,8:+0.0;-0.0;0.0}%");
                first = false;
                any = true;
            }
        }
        Console.WriteLine();
        Console.WriteLine("  Δ vs AIFS: negative = the blend arm beats raw AIFS on the held-out window.");
        return any ? 0 : 3;
    }

    /// <summary>One full-span pull per lead: per-model RH + dew point pivots
    /// (the production layout) PLUS NWP-ensemble aux means for the rich arms,
    /// joined to ERA5 RH truth. Same requiredNotNull gate as the production
    /// builder so every arm trains/scores on production-eligible rows.</summary>
    private List<Row> BuildRows(BlenderSpec spec, string locationName, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var fcGlob = SqlGlob.Escape(Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet"));
        var eraGlob = SqlGlob.Escape(Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet"));
        var modelIn = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        int n = spec.Models.Count;
        string Short(string m) => TempFeatureBuilder.ShortName(m);

        var pivotRh = string.Join(",\n        ", spec.Models.Select(m =>
            $"MAX(CASE WHEN Model = '{m}' THEN RelativeHumidity2m END) AS rh_{Short(m)}"));
        var pivotDp = string.Join(",\n        ", spec.Models.Select(m =>
            $"MAX(CASE WHEN Model = '{m}' THEN DewPoint2m END) AS dp_{Short(m)}"));
        var selRh = string.Join(", ", spec.Models.Select(m => $"p.rh_{Short(m)}"));
        var selDp = string.Join(", ", spec.Models.Select(m => $"p.dp_{Short(m)}"));
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m =>
                $"p.rh_{Short(m)} IS NOT NULL AND p.dp_{Short(m)} IS NOT NULL"))
            : "TRUE";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RelativeHumidity2m, DewPoint2m, Temperature2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh, WindSpeed10m,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND RelativeHumidity2m IS NOT NULL
      AND DewPoint2m IS NOT NULL
      AND Model IN {modelIn}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotRh},
        {pivotDp},
        AVG(Temperature2m)                  AS temp_mean,
        AVG(Temperature2m - DewPoint2m)     AS dewdep_mean,
        AVG(CloudCoverLow)                  AS cloud_low_mean,
        AVG(CloudCoverMid)                  AS cloud_mid_mean,
        AVG(CloudCoverHigh)                 AS cloud_high_mean,
        AVG(WindSpeed10m)                   AS wind_mean
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, RelativeHumidity2m AS truth
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}' AND RelativeHumidity2m IS NOT NULL
)
SELECT p.ValidTimeUtc, {selRh}, {selDp},
       p.temp_mean, p.dewdep_mean, p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean, p.wind_mean,
       e.truth
FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
WHERE ({requiredNotNull})
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();
        var rows = new List<Row>();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var rh = new double[n];
            var dp = new double[n];
            var aux = new double[RichAuxNames.Length];
            for (int i = 0; i < n; i++) rh[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            for (int i = 0; i < n; i++) dp[i] = r.IsDBNull(1 + n + i) ? double.NaN : r.GetDouble(1 + n + i);
            for (int i = 0; i < aux.Length; i++)
                aux[i] = r.IsDBNull(1 + 2 * n + i) ? double.NaN : r.GetDouble(1 + 2 * n + i);
            rows.Add(new Row(r.GetDateTime(0), rh, dp, aux, r.GetDouble(1 + 2 * n + aux.Length)));
        }
        _log.LogInformation("lead {L}h: {N} full-span rows ({S:yyyy-MM-dd}..{E:yyyy-MM-dd})",
            spec.LeadHours, rows.Count,
            rows.Count > 0 ? rows[0].Valid : default, rows.Count > 0 ? rows[^1].Valid : default);
        return rows;
    }

    private static BlenderSpec WithRichAux(BlenderSpec s) => new()
    {
        Target = s.Target,
        FeatureSet = "rich-aux",
        LeadHours = s.LeadHours,
        RequiredModels = s.RequiredModels,
        OptionalModels = s.OptionalModels,
        Models = s.Models,
        FeatureNames = s.FeatureNames.Concat(RichAuxNames).ToList(),
        DataSource = s.DataSource,
        Tier = "rich-aux",
        UkvStrategy = s.UkvStrategy,
    };

    private static RegressionTrainingRow Compose(BlenderSpec spec, Row row, bool rich)
    {
        var baseRow = HumidityFeatureBuilder.ComposeRow(
            // ComposeRow validates against ITS spec's count — always pass the
            // production-shaped 17-feature layout, then append aux for rich.
            spec.FeatureNames.Count == row.Rh.Length * 2 + 7
                ? spec
                : StripAux(spec),
            row.Valid, row.Rh, row.Dp, row.Truth);
        if (!rich) return baseRow;
        var features = new float[spec.FeatureCount];
        Array.Copy(baseRow.Features, features, baseRow.Features.Length);
        for (int i = 0; i < RichAuxNames.Length; i++)
            features[baseRow.Features.Length + i] = (float)row.Aux[i];
        return new RegressionTrainingRow { ValidTimeUtc = row.Valid, Features = features, Label = baseRow.Label };
    }

    private static BlenderSpec StripAux(BlenderSpec s) => new()
    {
        Target = s.Target,
        FeatureSet = HumidityFeatureBuilder.SpecFeatureSet,
        LeadHours = s.LeadHours,
        RequiredModels = s.RequiredModels,
        OptionalModels = s.OptionalModels,
        Models = s.Models,
        FeatureNames = s.FeatureNames.Take(s.FeatureNames.Count - RichAuxNames.Length).ToList(),
        DataSource = s.DataSource,
        Tier = HumidityFeatureBuilder.SpecFeatureSet,
        UkvStrategy = s.UkvStrategy,
    };

    private static double TrainAndScore(
        List<RegressionTrainingRow> pool, List<RegressionTrainingRow> test,
        BlenderSpec spec, TempTrainer.Hyperparameters hp)
    {
        int valStart = (int)(pool.Count * 0.85);
        var train = pool.Take(valStart).ToList();
        var val = pool.Skip(valStart).ToList();
        var trained = TempTrainer.TrainVector(train, val, spec, hp);
        var pred = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, test);
        double sum = 0; int nn = 0;
        for (int i = 0; i < test.Count; i++)
        {
            if (double.IsNaN(pred[i]) || float.IsNaN(test[i].Label)) continue;
            sum += Math.Abs(pred[i] - test[i].Label); nn++;
        }
        return nn == 0 ? double.NaN : sum / nn;
    }
}
