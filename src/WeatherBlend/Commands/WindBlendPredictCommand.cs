using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 3 Slice C of WIND_BLENDER_PLAN — per-cycle synthesis of the
/// final wind speed from two sibling phases:
///   * <c>wind_speed_lgb</c> (LightGBM, Dunkeswell-truth, ERA5-free) —
///     produced by <see cref="Train.Element.Wind.WindSpeedLgbPredictPipeline"/>,
///     lands at <c>predictions/wind/model_version=v..._wind_speed_lgb/...</c>.
///   * <c>wind_mvn</c> (PyTorch bivariate-normal MLP) — produced by
///     WeatherProbabilistic's <c>predict_wind_mvn.py</c>, lands at
///     <c>predictions/wind_direction/{location}/model_version=v..._wind_mvn/...</c>.
///     The python side stores the MVN's speed magnitude in
///     <see cref="WindDirectionPredictionRow.BlendSpeedMagnitude"/>.
///
/// Inner-joins both on (ValidTime, Lead) and applies the sigmoid blend
/// the production-scope bake-off picked 2026-05-27:
/// <code>
///   w_mvn = 1 / (1 + exp((lgb_speed - center) / scale))
///   final = w_mvn · mvn_speed + (1 - w_mvn) · lgb_speed
/// </code>
/// with center=2.50, scale=3.00 (the plan defaults — calibration.json
/// from the wind_mvn bundle CAN override per-lead, deferred to a follow-up
/// when the bake-off shows lead-specific tuning helps).
///
/// Output schema: <see cref="ElementPredictionRow"/> with Element="wind"
/// and BlendValue = blended speed. Lands at
/// <c>predictions/wind/model_version=v_wind_blend_live/date={anchor}/predictions.parquet</c>.
/// The fixed <c>v_wind_blend_live</c> version is intentional for v1:
/// avoids requiring a wind_blend MANIFEST entry (no mint command yet),
/// so the predict path can ship before the bundle/render plumbing.
/// Bundle + MANIFEST integration is a follow-up slice.
///
/// Soft-skip exit codes (treated as non-fatal by predict-tail):
///   2 — no locations configured
///   3 — at least one component (lgb or mvn) missing for every location
/// </summary>
public sealed class WindBlendPredictCommand
{
    private readonly ILogger<WindBlendPredictCommand> _log;
    private readonly AppConfig _cfg;

    /// <summary>Fixed version tag for the blended output during v1.
    /// Stable across cycles so the daily date-partitioned parquets stack
    /// under one model_version dir rather than scattering.</summary>
    public const string VersionTag = "v_wind_blend_live";

    /// <summary>Sigmoid composition center (m/s) — lgb speed where the
    /// MVN weight = 0.5. Plan default from 2026-05-27 bake-off.</summary>
    public const double DefaultBlendCenter = 2.50;
    /// <summary>Sigmoid composition scale (m/s) — width of the transition
    /// band around <see cref="DefaultBlendCenter"/>.</summary>
    public const double DefaultBlendScale = 3.00;

    public WindBlendPredictCommand(ILogger<WindBlendPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
    {
        if (_cfg.Locations.Count == 0)
        {
            _log.LogError("No locations configured — cannot synthesise wind_blend.");
            return 2;
        }

        var predictionMadeAt = DateTime.UtcNow;
        var anchor = forDate ?? DateOnly.FromDateTime(predictionMadeAt);
        var dateStr = anchor.ToString("yyyy-MM-dd");

        _log.LogInformation(
            "wind_blend synthesis — anchor={Anchor:yyyy-MM-dd}, synthesis_time={Synth:yyyy-MM-dd HH:mm}Z",
            anchor, predictionMadeAt);

        int nOk = 0, nSkip = 0;
        foreach (var loc in _cfg.Locations)
        {
            ct.ThrowIfCancellationRequested();
            var (ok, note) = await SynthesiseLocationAsync(loc, anchor, dateStr, predictionMadeAt, ct);
            _log.LogInformation("{Tag} {Loc}: {Note}", ok ? "OK  " : "SKIP", loc.Name, note);
            if (ok) nOk++; else nSkip++;
        }
        _log.LogInformation("Done. Wrote: {OK}, skipped: {Skip}.", nOk, nSkip);
        return nOk > 0 ? 0 : 3;
    }

    private async Task<(bool ok, string note)> SynthesiseLocationAsync(
        LocationConfig location, DateOnly anchor, string dateStr,
        DateTime predictionMadeAt, CancellationToken ct)
    {
        var predRoot = _cfg.Storage.PredictionsPath;

        // wind_speed_lgb lives in WB's predictions/wind tree (no location
        // subdir — model_version disambiguates phases under the same
        // physical target). Pick the newest *_wind_speed_lgb parquet for
        // this anchor date.
        var lgbGlob = Path.Combine(predRoot, "wind",
            "model_version=*_wind_speed_lgb", $"date={dateStr}", "predictions.parquet");
        var lgbRows = ReadLgbRows(lgbGlob, location.Name, ct);
        if (lgbRows.Count == 0)
            return (false, $"no wind_speed_lgb predictions at {lgbGlob}");

        // wind_mvn lives in WP's predictions/wind_direction/{loc}/ tree —
        // WP uses a per-location subdir + suffixed model_version. Glob
        // across every *_wind_mvn version dir under this location for
        // this anchor date; DuckDB ROW_NUMBER picks the freshest cycle.
        var mvnGlob = Path.Combine(predRoot, "wind_direction", location.Name,
            "model_version=*_wind_mvn", $"date={dateStr}", "predictions.parquet");
        var mvnByKey = ReadMvnRows(mvnGlob, ct);
        if (mvnByKey.Count == 0)
            return (false, $"no wind_mvn predictions at {mvnGlob}");

        // Inner join on (ValidTime, Lead). Drive from lgb — it's the
        // canonical "what physical cells are we synthesising for". Cells
        // without an mvn match get dropped (mvn predicts at fixed
        // {24,48,72} leads, same as lgb under production scope, so
        // overlap should be 1.0 in practice).
        var blended = new List<ElementPredictionRow>(lgbRows.Count);
        int matched = 0;
        foreach (var l in lgbRows)
        {
            if (!mvnByKey.TryGetValue((l.ValidTimeUtc, l.LeadHours), out var m))
                continue;
            matched++;
            var lgbSpd = l.BlendValue;
            var mvnSpd = m.SpeedMs;
            var wMvn = 1.0 / (1.0 + Math.Exp((lgbSpd - DefaultBlendCenter) / DefaultBlendScale));
            var blendSpd = wMvn * mvnSpd + (1.0 - wMvn) * lgbSpd;

            blended.Add(new ElementPredictionRow
            {
                LocationName = location.Name,
                Element = "wind",
                ModelVersion = VersionTag,
                PredictionMadeAtUtc = predictionMadeAt,
                ValidTimeUtc = l.ValidTimeUtc,
                LeadHours = l.LeadHours,
                BlendValue = blendSpd,
                // Per-NWP slots: copy from the lgb row so downstream verify
                // can still see the underlying per-NWP wind contributions.
                // wind_mvn's per-NWP run-times are also present but the
                // lgb side carries the same NWPs (production scope is
                // shared) — single source is fine.
                ModelGfs = l.ModelGfs, ModelEcmwf = l.ModelEcmwf, ModelIcon = l.ModelIcon,
                ModelMf  = l.ModelMf,  ModelUkmo  = l.ModelUkmo,  ModelGem  = l.ModelGem,
                ModelAifs = l.ModelAifs,
                RunTimeGfs = l.RunTimeGfs, RunTimeEcmwf = l.RunTimeEcmwf,
                RunTimeIcon = l.RunTimeIcon, RunTimeMf = l.RunTimeMf,
                RunTimeUkmo = l.RunTimeUkmo, RunTimeGem = l.RunTimeGem,
                RunTimeAifs = l.RunTimeAifs,
                Mean = l.Mean, Std = l.Std, Range = l.Range,
                FeatureVectorHash = $"wind_blend:{l.FeatureVectorHash}",
            });
        }
        if (blended.Count == 0)
            return (false,
                $"no (ValidTime, Lead) overlap between {lgbRows.Count} lgb rows and {mvnByKey.Count} mvn rows");

        var outPath = Path.Combine(predRoot, "wind",
            $"model_version={VersionTag}", $"date={dateStr}", "predictions.parquet");
        var total = await PredictionParquetWriter.WriteAsync(
            outPath, blended,
            dedupKey:  r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc),
            freshness: r => r.PredictionMadeAtUtc,
            orderBy:   rows => rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours),
            ct);
        return (true,
            $"wrote {blended.Count} blended rows (matched {matched}/{lgbRows.Count} lgb cells; file now holds {total})");
    }

    /// <summary>Read lgb predictions for a single location at this date.
    /// ROW_NUMBER over (V, L) picks the freshest cycle if multiple
    /// PredictionMadeAtUtc cycles wrote to the same date partition.</summary>
    private IReadOnlyList<ElementPredictionRow> ReadLgbRows(
        string lgbGlob, string locationName, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(lgbGlob);
        var loc = locationName.Replace("'", "''");
        var sql = $@"
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', union_by_name = true)
    WHERE LocationName = '{loc}'
)
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendValue,
       ModelGfs, ModelEcmwf, ModelIcon, ModelMf, ModelUkmo, ModelGem, ModelAifs,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem, RunTimeAifs,
       Mean, Std, Range, FeatureVectorHash
FROM ranked WHERE rn = 1
ORDER BY ValidTimeUtc, LeadHours";
        return ParquetReader.Query(sql, r => new ElementPredictionRow
        {
            LocationName = r.GetString(0),
            Element = "wind",  // sentinel — overwritten by caller
            ModelVersion = r.GetString(1),
            PredictionMadeAtUtc = r.GetDateTime(2),
            ValidTimeUtc = r.GetDateTime(3),
            LeadHours = r.GetInt32(4),
            BlendValue = r.GetDouble(5),
            ModelGfs   = r.IsDBNull(6)  ? null : r.GetDouble(6),
            ModelEcmwf = r.IsDBNull(7)  ? null : r.GetDouble(7),
            ModelIcon  = r.IsDBNull(8)  ? null : r.GetDouble(8),
            ModelMf    = r.IsDBNull(9)  ? null : r.GetDouble(9),
            ModelUkmo  = r.IsDBNull(10) ? null : r.GetDouble(10),
            ModelGem   = r.IsDBNull(11) ? null : r.GetDouble(11),
            ModelAifs  = r.IsDBNull(12) ? null : r.GetDouble(12),
            RunTimeGfs   = r.IsDBNull(13) ? null : r.GetDateTime(13),
            RunTimeEcmwf = r.IsDBNull(14) ? null : r.GetDateTime(14),
            RunTimeIcon  = r.IsDBNull(15) ? null : r.GetDateTime(15),
            RunTimeMf    = r.IsDBNull(16) ? null : r.GetDateTime(16),
            RunTimeUkmo  = r.IsDBNull(17) ? null : r.GetDateTime(17),
            RunTimeGem   = r.IsDBNull(18) ? null : r.GetDateTime(18),
            RunTimeAifs  = r.IsDBNull(19) ? null : r.GetDateTime(19),
            Mean  = r.IsDBNull(20) ? null : r.GetDouble(20),
            Std   = r.IsDBNull(21) ? null : r.GetDouble(21),
            Range = r.IsDBNull(22) ? null : r.GetDouble(22),
            FeatureVectorHash = r.IsDBNull(23) ? "" : r.GetString(23),
        }, _log, $"wind_speed_lgb tree empty at {lgbGlob}", ct);
    }

    /// <summary>Per-cell wind_mvn read — we only need (V, L, speed_ms)
    /// from BlendSpeedMagnitude. ROW_NUMBER picks freshest cycle if
    /// multiple mvn dirs partition this anchor.</summary>
    private Dictionary<(DateTime ValidTime, int Lead), (double SpeedMs, DateTime PredictionMadeAt)>
        ReadMvnRows(string mvnGlob, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(mvnGlob);
        var sql = $@"
WITH ranked AS (
    SELECT ValidTimeUtc, LeadHours, BlendSpeedMagnitude, PredictionMadeAtUtc,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', union_by_name = true)
)
SELECT ValidTimeUtc, LeadHours, BlendSpeedMagnitude, PredictionMadeAtUtc
FROM ranked WHERE rn = 1";
        var rows = ParquetReader.Query(sql, r => (
            ValidTime: r.GetDateTime(0),
            Lead: r.GetInt32(1),
            SpeedMs: r.GetDouble(2),
            PredictionMadeAt: r.GetDateTime(3)
        ), _log, $"wind_mvn tree empty at {mvnGlob}", ct);
        var dict = new Dictionary<(DateTime, int), (double, DateTime)>(rows.Count);
        foreach (var row in rows)
            dict[(row.ValidTime, row.Lead)] = (row.SpeedMs, row.PredictionMadeAt);
        return dict;
    }
}
