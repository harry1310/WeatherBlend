using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Per-cycle synthesis of the final wind speed as the EQUAL-WEIGHT MEAN of two
/// blenders that predict 10 m speed against different truth:
///   * champion <c>wind</c> (LightGBM, ERA5 WindSpeed10m truth) — the element
///     blender, lands at <c>predictions/wind/model_version={champion}/...</c>.
///   * <c>wind_speed_lgb</c> (LightGBM, Dunkeswell SYNOP truth, ERA5-free) —
///     produced by <see cref="Train.Element.Wind.WindSpeedLgbPredictPipeline"/>,
///     lands at <c>predictions/wind/model_version=v..._wind_speed_lgb/...</c>.
///
/// Inner-joins both on (ValidTime, Lead): <c>final = 0.5·champion + 0.5·lgb</c>.
/// The cross-truth bake-off (2026-06-09) showed the 50/50 mean beats either
/// single on REAL (Dunkeswell) wind, and a val-fitted weight overfit and lost to
/// 50/50 at every lead — so the weight is FIXED, not learned. This replaced the
/// 2026-05-27 sigmoid(lgb, wind_mvn-speed) composition: wind_mvn's speed/CI broke
/// in the 06-07 retrain (val NLL doubled), so MVN is now direction-only and its
/// speed is no longer a blend input.
///
/// Output schema: <see cref="ElementPredictionRow"/> with Element="wind"
/// and BlendValue = blended speed. Lands at
/// <c>predictions/wind/model_version=v_wind_blend_live/date={anchor}/predictions.parquet</c>.
/// The fixed <c>v_wind_blend_live</c> version avoids requiring a wind_blend
/// MANIFEST entry (no mint command); wind_blend stays a challenger, not champion.
///
/// Soft-skip exit codes (treated as non-fatal by predict-tail):
///   2 — no locations configured
///   3 — at least one member (champion wind or lgb) missing for every location
/// </summary>
public sealed class WindBlendPredictCommand
{
    private readonly ILogger<WindBlendPredictCommand> _log;
    private readonly AppConfig _cfg;

    /// <summary>Fixed version tag for the blended output during v1.
    /// Stable across cycles so the daily date-partitioned parquets stack
    /// under one model_version dir rather than scattering.</summary>
    public const string VersionTag = "v_wind_blend_live";

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

        // Champion `wind` (ERA5-truth element blender) — the other blend member.
        // Resolve its version from the wind manifest (champion phase = "wind", so
        // this is NOT the *_wind_speed_lgb sibling nor the v_wind_blend_live output)
        // and read its predictions from the same predictions/wind/ tree.
        var championVer = ModelArtifact.ResolveStationChampionVersion(
            _cfg.Storage.ModelsPath, "wind", location.Name);
        if (string.IsNullOrEmpty(championVer))
            return (false, "no champion wind version in manifest");
        var champGlob = Path.Combine(predRoot, "wind",
            $"model_version={championVer}", $"date={dateStr}", "predictions.parquet");
        var champByKey = ReadChampionWindRows(champGlob, location.Name, ct);
        if (champByKey.Count == 0)
            return (false, $"no champion wind ({championVer}) predictions at {champGlob}");

        // Inner join on (ValidTime, Lead). Drive from lgb — it's the
        // canonical "what physical cells are we synthesising for". Cells
        // without an mvn match get dropped (mvn predicts at fixed
        // {24,48,72} leads, same as lgb under production scope, so
        // overlap should be 1.0 in practice).
        var blended = new List<ElementPredictionRow>(lgbRows.Count);
        int matched = 0;
        foreach (var l in lgbRows)
        {
            if (!champByKey.TryGetValue((l.ValidTimeUtc, l.LeadHours), out var champSpd))
                continue;
            matched++;
            var lgbSpd = l.BlendValue;
            // Equal-weight mean of the champion (ERA5-truth) and lgb (Dunkeswell-
            // truth) speeds. The cross-truth bake-off (2026-06-09) showed the 50/50
            // mean beats either single on REAL wind, and a val-fitted weight overfit
            // and lost to 50/50 at every lead — so the weight is fixed, not learned.
            var blendSpd = 0.5 * (champSpd + lgbSpd);

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
                $"no (ValidTime, Lead) overlap between {lgbRows.Count} lgb rows and {champByKey.Count} champion wind rows");

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

    /// <summary>Per-cell champion `wind` (ERA5-truth) speed read — we only need
    /// (V, L, speed) from BlendValue. ROW_NUMBER picks the freshest cycle if
    /// multiple cycles wrote to this anchor date.</summary>
    private Dictionary<(DateTime ValidTime, int Lead), double>
        ReadChampionWindRows(string champGlob, string locationName, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(champGlob);
        var loc = locationName.Replace("'", "''");
        var sql = $@"
WITH ranked AS (
    SELECT ValidTimeUtc, LeadHours, BlendValue, PredictionMadeAtUtc,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours
                              ORDER BY PredictionMadeAtUtc DESC) AS rn
    FROM read_parquet('{glob}', union_by_name = true)
    WHERE LocationName = '{loc}'
)
SELECT ValidTimeUtc, LeadHours, BlendValue
FROM ranked WHERE rn = 1";
        var rows = ParquetReader.Query(sql, r => (
            ValidTime: r.GetDateTime(0),
            Lead: r.GetInt32(1),
            SpeedMs: r.GetDouble(2)
        ), _log, $"champion wind tree empty at {champGlob}", ct);
        var dict = new Dictionary<(DateTime, int), double>(rows.Count);
        foreach (var row in rows)
            dict[(row.ValidTime, row.Lead)] = row.SpeedMs;
        return dict;
    }
}
