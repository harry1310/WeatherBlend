using System.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Element;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Commands;

/// <summary>
/// Rolling verification for the per-variable element blenders. One command serves
/// every element via target dispatch — the only per-element bit is which ERA5 column
/// is read as truth. Output: a markdown report per element under
/// <c>data/reports/verify_element_{element}_{yyyy-MM-dd}.md</c>.
///
/// METAR secondary metric is intentionally not included here. The brief calls for
/// it but each element needs bespoke handling (RH derivation for humidity, no METAR
/// signal for radiation, parser extension for cloud cover) — wiring it in now would
/// expand scope substantially. The reporter footnotes the deferral.
/// </summary>
public sealed class ElementVerifyCommand
{
    private readonly ILogger<ElementVerifyCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;

    public ElementVerifyCommand(ILogger<ElementVerifyCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
    }

    public async Task<int> RunAsync(
        ElementTarget target,
        DateOnly? asOf,
        int windowDays,
        int era5LatencyDays,
        double driftThreshold,
        CancellationToken ct)
    {
        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-era5LatencyDays);

        _log.LogInformation(
            "Verify {Target}: as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×",
            target.CliName, asOfUtc, windowStart, windowEnd, era5LatencyDays, driftThreshold);

        var predictions = QueryPredictions(target, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);

        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        var truthCol = TruthColumnFor(target);
        // Phase C commit 3: locationName is now required on GetEra5Hourly.
        // 3a passes primary explicitly here; 3b will loop _cfg.Locations and
        // emit per-loc verify rows. See [[project_phase_c_status]].
        var truth = _truth.GetEra5Hourly(_cfg.Location.Name, windowStart, windowEnd, ct, truthCol);
        _log.LogInformation("Loaded {N} ERA5 truth points (column={Col}).", truth.Count, truthCol);

        var metadata = _metadata.GetTrainingMetadataForVersions(
            target.ModelDirName, _cfg.Location.Name,
            predictions.Select(p => p.ModelVersion));
        _log.LogInformation("Loaded metadata for {N} versions.", metadata.Count);

        var rows = ElementVerifier.Compute(new ElementVerifier.Inputs
        {
            Predictions = predictions,
            TruthByTime = truth,
            MetadataByVersion = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            Era5LatencyDays = era5LatencyDays,
            DriftThreshold = driftThreshold,
            DriftThresholdSlopePer24h = TempVerifyCommand.DriftThresholdSlopeDefault,
            MinDriftN = TempVerifyCommand.MinDriftNDefault,
        });

        var md = ElementVerifyReporter.BuildMarkdown(
            target, asOfUtc, windowDays, era5LatencyDays, driftThreshold, rows, metadata);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_element_{target.ModelDirName}_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        // Structured sidecar for the Models-page "Verify history" table.
        // Per-element targets share the schema with temp (single-station,
        // MAE-style metric) plus an "element_<name>" target identifier so
        // the four element families surface as separate Models-page rows.
        var history = new WeatherBlend.Models.VerifyHistoryFile
        {
            Target = $"element_{target.ModelDirName}",
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = era5LatencyDays,
            MetricLabel = $"MAE ({target.Units})",
            Rows = rows.Select(r => new WeatherBlend.Models.VerifyHistoryRow
            {
                Station = null,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue(r.ModelVersion, out var meta) ? meta.Phase : null,
                LeadHours = r.LeadHours,
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendMae,
                ClimMetric = null,
                MeanOfModelsMetric = r.MeanMae,
                BestSingleName = r.BestSingleName,
                BestSingleMetric = r.BestSingleMae,
                ReferenceTrainingMetric = r.ReferenceTestMae,
                DriftFlag = r.DriftFlag,
            }).ToList(),
        };
        await WeatherBlend.Evaluate.VerifyHistoryWriter.WriteAsync(_cfg.Storage.ReportsPath, history, ct);

        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Version} lead {Lead}h — n={N}, blend MAE {Mae:0.000} {U}, bias {Bias:+0.000;-0.000;0.000}{Drift}",
                r.ModelVersion, r.LeadHours, r.N, r.BlendMae, target.Units, r.BlendBias,
                r.DriftFlag ? "  [DRIFT]" : "");
        }

        return rows.Any(r => r.DriftFlag) ? 4 : 0;
    }

    /// <summary>ERA5 column name (matches <see cref="Era5Row"/> property names) for each element target.</summary>
    private static string TruthColumnFor(ElementTarget target) => target.CliName switch
    {
        "wind"                => "WindSpeed10m",
        "humidity"            => "RelativeHumidity2m",
        "shortwave-radiation" => "ShortwaveRadiation",
        "cloud-cover"         => "CloudCover",
        _ => throw new ArgumentException($"No ERA5 truth column mapped for element '{target.CliName}'."),
    };

    private IReadOnlyList<ElementPredictionRow> QueryPredictions(
        ElementTarget target, DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, target.ModelDirName, "**", "*.parquet"));

        var sql = $@"
SELECT LocationName, Element, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendValue,
       ModelGfs, ModelEcmwf, ModelIcon, ModelMf, ModelUkmo, ModelGem, ModelAifs,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem, RunTimeAifs,
       Mean, Std, Range,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, r => new ElementPredictionRow
        {
            LocationName        = r.GetString(0),
            Element             = r.GetString(1),
            ModelVersion        = r.GetString(2),
            PredictionMadeAtUtc = r.GetDateTime(3),
            ValidTimeUtc        = r.GetDateTime(4),
            LeadHours           = r.GetInt32(5),
            BlendValue          = r.GetDouble(6),
            ModelGfs   = NullableDouble(r,  7),
            ModelEcmwf = NullableDouble(r,  8),
            ModelIcon  = NullableDouble(r,  9),
            ModelMf    = NullableDouble(r, 10),
            ModelUkmo  = NullableDouble(r, 11),
            ModelGem   = NullableDouble(r, 12),
            ModelAifs  = NullableDouble(r, 13),
            RunTimeGfs   = NullableDate(r, 14),
            RunTimeEcmwf = NullableDate(r, 15),
            RunTimeIcon  = NullableDate(r, 16),
            RunTimeMf    = NullableDate(r, 17),
            RunTimeUkmo  = NullableDate(r, 18),
            RunTimeGem   = NullableDate(r, 19),
            RunTimeAifs  = NullableDate(r, 20),
            Mean  = NullableDouble(r, 21),
            Std   = NullableDouble(r, 22),
            Range = NullableDouble(r, 23),
            FeatureVectorHash = r.IsDBNull(24) ? "" : r.GetString(24),
        }, _log, $"Predictions tree empty for {target.CliName} — nothing to verify.", ct);
    }

    private static double? NullableDouble(IDataReader r, int ord) => r.IsDBNull(ord) ? null : r.GetDouble(ord);
    private static DateTime? NullableDate(IDataReader r, int ord) => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
