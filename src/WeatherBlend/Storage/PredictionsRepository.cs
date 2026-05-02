using System.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Models;

namespace WeatherBlend.Storage;

/// <summary>
/// Single source of truth for reads from the three blender prediction trees:
///
///   data/predictions/temperature/{model_version}/{date}/{run}.parquet     → <see cref="TempPredictionRow"/>
///   data/predictions/precipitation/{station}/{model_version}/...          → <see cref="PrecipPredictionRow"/>
///   data/predictions/dry_window/{station}/window_{N}h/{model_version}/... → <see cref="DryWindowPredictionRow"/>
///
/// Each tree had its full SELECT projection + 20-line row mapper duplicated
/// across <c>RenderSiteCommand</c> and the matching verify command — verify
/// adds an optional per-station / per-(station, window) filter, render
/// scans the whole subtree. The repo collapses both shapes into one method
/// per tree by accepting an optional filter list.
///
/// Returns the canonical domain row types from <c>WeatherBlend.Models</c>;
/// the renderer projects them to its lighter <c>SitePages.*</c> records
/// in-memory at the call site (cheap, keeps the storage layer free of UI
/// type knowledge).
///
/// Element predictions and start-hour curves stay outside the repo for now —
/// element has a per-target glob the repo can't easily encode without
/// gaining a target enum, and start-hour has bespoke projections per caller.
/// Lift them in a follow-up if/when they grow a third caller.
/// </summary>
public sealed class PredictionsRepository
{
    private readonly ILogger<PredictionsRepository> _log;
    private readonly AppConfig _cfg;

    public PredictionsRepository(ILogger<PredictionsRepository> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    // -----------------------------------------------------------------
    // Temperature
    // -----------------------------------------------------------------

    /// <summary>
    /// Every temperature prediction for the configured location whose
    /// <c>ValidTimeUtc</c> falls in [<paramref name="start"/>, <paramref name="end"/>].
    /// Drops rows with null <c>BlendTemperature</c> — they're meaningless and
    /// the renderer / verify both want to ignore them. Order is
    /// <c>(ModelVersion, LeadHours, ValidTimeUtc)</c>; callers re-sort if
    /// they need a different order.
    /// </summary>
    public IReadOnlyList<TempPredictionRow> GetTemperaturePredictions(
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Scope to the temperature subtree only — sibling subtrees (precipitation,
        // dry_window) have different schemas and would surface as nulls or wrong
        // columns under union_by_name.
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "temperature", "**", "*.parquet"));

        // hive_partitioning=false — the `model_version=` hive key collides with the
        // in-file ModelVersion column under case-insensitive resolution. Same rule
        // applies across the codebase (see CLAUDE.md gotcha). In-file column wins.
        var sql = $@"
SELECT LocationName, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       BlendTemperature,
       TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem, TempAifs,
       RunTimeGfs, RunTimeEcmwf, RunTimeIcon, RunTimeMf, RunTimeUkmo, RunTimeGem, RunTimeAifs,
       TempMean, TempStd, TempRange,
       FeatureVectorHash
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND BlendTemperature IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, MapTemperatureRow,
            _log, "Predictions tree empty — temperature reads will return no rows.", ct);
    }

    private static TempPredictionRow MapTemperatureRow(IDataReader r) => new()
    {
        LocationName        = r.GetString(0),
        ModelVersion        = r.GetString(1),
        PredictionMadeAtUtc = r.GetDateTime(2),
        ValidTimeUtc        = r.GetDateTime(3),
        LeadHours           = r.GetInt32(4),
        BlendTemperature    = r.GetDouble(5),
        TempGfs   = NullableDouble(r,  6),
        TempEcmwf = NullableDouble(r,  7),
        TempIcon  = NullableDouble(r,  8),
        TempMf    = NullableDouble(r,  9),
        TempUkmo  = NullableDouble(r, 10),
        TempGem   = NullableDouble(r, 11),
        TempAifs  = NullableDouble(r, 12),
        RunTimeGfs   = NullableDate(r, 13),
        RunTimeEcmwf = NullableDate(r, 14),
        RunTimeIcon  = NullableDate(r, 15),
        RunTimeMf    = NullableDate(r, 16),
        RunTimeUkmo  = NullableDate(r, 17),
        RunTimeGem   = NullableDate(r, 18),
        RunTimeAifs  = NullableDate(r, 19),
        TempMean  = NullableDouble(r, 20),
        TempStd   = NullableDouble(r, 21),
        TempRange = NullableDouble(r, 22),
        FeatureVectorHash = r.IsDBNull(23) ? "" : r.GetString(23),
    };

    // -----------------------------------------------------------------
    // Precipitation
    // -----------------------------------------------------------------

    /// <summary>
    /// Precipitation predictions for the configured location, optionally
    /// scoped to a list of <paramref name="stations"/>. Pass <c>null</c> or
    /// an empty list to scan the whole precipitation subtree (renderer
    /// pattern); pass an explicit list to limit to those station partitions
    /// (verify pattern — cheaper scan, less cross-station noise).
    /// </summary>
    public IReadOnlyList<PrecipPredictionRow> GetPrecipitationPredictions(
        IReadOnlyList<string>? stations,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Use a list-form read_parquet([...]) when stations are scoped so DuckDB
        // can prune partitions at planning time; fall back to the wildcard glob
        // otherwise. Both forms surface "No files found" the same way, so
        // ParquetReader.Query degrades identically on a missing tree.
        var fromClause = (stations is null || stations.Count == 0)
            ? $"read_parquet('{ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "precipitation", "**", "*.parquet"))}', hive_partitioning = false, union_by_name = true)"
            : "read_parquet([" + string.Join(", ", stations.Select(s =>
                "'" + ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "precipitation", s, "**", "*.parquet")) + "'"))
                + "], hive_partitioning = false, union_by_name = true)";

        var sql = $@"
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem, PrecipAifs, PrecipJma,
       PrecipAgreementWet01,
       FeatureVectorHash
FROM {fromClause}
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, MapPrecipRow,
            _log, "Precipitation predictions tree empty — precip reads will return no rows.", ct);
    }

    private static PrecipPredictionRow MapPrecipRow(IDataReader r) => new()
    {
        LocationName        = r.GetString(0),
        TruthStation        = r.GetString(1),
        ModelVersion        = r.GetString(2),
        PredictionMadeAtUtc = r.GetDateTime(3),
        ValidTimeUtc        = r.GetDateTime(4),
        LeadHours           = r.GetInt32(5),
        ProbWet             = r.GetDouble(6),
        ClimatologyPWet     = r.GetDouble(7),
        PrecipGfs   = NullableDouble(r,  8),
        PrecipEcmwf = NullableDouble(r,  9),
        PrecipIcon  = NullableDouble(r, 10),
        PrecipMf    = NullableDouble(r, 11),
        PrecipUkmo  = NullableDouble(r, 12),
        PrecipGem   = NullableDouble(r, 13),
        PrecipAifs  = NullableDouble(r, 14),
        PrecipJma   = NullableDouble(r, 15),
        PrecipAgreementWet01 = NullableDouble(r, 16),
        FeatureVectorHash    = r.IsDBNull(17) ? "" : r.GetString(17),
    };

    // -----------------------------------------------------------------
    // Dry window
    // -----------------------------------------------------------------

    /// <summary>
    /// Dry-window predictions for the configured location, anchored on
    /// <c>TargetDateUtc</c> (UTC midnight of the labelled day) — pass the
    /// <em>target-date</em> bounds, not valid-time. Optionally scoped to a
    /// list of (station, window-hours) cells; pass <c>null</c> / empty for
    /// the wildcard form the renderer uses.
    /// </summary>
    public IReadOnlyList<DryWindowPredictionRow> GetDryWindowPredictions(
        IReadOnlyList<(string Station, int WindowHours)>? cells,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var fromClause = (cells is null || cells.Count == 0)
            ? $"read_parquet('{ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "dry_window", "**", "*.parquet"))}', hive_partitioning = false, union_by_name = true)"
            : "read_parquet([" + string.Join(", ", cells.Select(c =>
                "'" + ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "dry_window", c.Station, $"window_{c.WindowHours}h", "**", "*.parquet")) + "'"))
                + "], hive_partitioning = false, union_by_name = true)";

        var sql = $@"
SELECT LocationName, TruthStation, WindowHours, ModelVersion,
       PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       ProbHasDryWindow, ClimatologyProbHasDryWindow,
       AgreementHasDryWindow, PrecipSumMean, LongestDryRunMean, WetHourCountMean,
       HasDryWindowGfs, HasDryWindowEcmwf, HasDryWindowIcon,
       HasDryWindowMf,  HasDryWindowUkmo,  HasDryWindowGem,  HasDryWindowAifs,  HasDryWindowJma,
       PrecipSumGfs, PrecipSumEcmwf, PrecipSumIcon, PrecipSumMf, PrecipSumUkmo, PrecipSumGem,
       PrecipSumAifs, PrecipSumJma,
       FeatureVectorHash
FROM {fromClause}
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND TargetDateUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND TargetDateUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, WindowHours, ModelVersion, LeadHours, TargetDateUtc";

        return ParquetReader.Query(sql, MapDryWindowRow,
            _log, "Dry-window predictions tree empty — dry-window reads will return no rows.", ct);
    }

    private static DryWindowPredictionRow MapDryWindowRow(IDataReader r) => new()
    {
        LocationName        = r.GetString(0),
        TruthStation        = r.GetString(1),
        WindowHours         = r.GetInt32(2),
        ModelVersion        = r.GetString(3),
        PredictionMadeAtUtc = r.GetDateTime(4),
        TargetDateUtc       = r.GetDateTime(5),
        LeadHours           = r.GetInt32(6),
        ProbHasDryWindow            = r.GetDouble(7),
        ClimatologyProbHasDryWindow = r.GetDouble(8),
        AgreementHasDryWindow       = NullableDouble(r,  9),
        PrecipSumMean       = NullableDouble(r, 10),
        LongestDryRunMean   = NullableDouble(r, 11),
        WetHourCountMean    = NullableDouble(r, 12),
        HasDryWindowGfs     = NullableDouble(r, 13),
        HasDryWindowEcmwf   = NullableDouble(r, 14),
        HasDryWindowIcon    = NullableDouble(r, 15),
        HasDryWindowMf      = NullableDouble(r, 16),
        HasDryWindowUkmo    = NullableDouble(r, 17),
        HasDryWindowGem     = NullableDouble(r, 18),
        HasDryWindowAifs    = NullableDouble(r, 19),
        HasDryWindowJma     = NullableDouble(r, 20),
        PrecipSumGfs        = NullableDouble(r, 21),
        PrecipSumEcmwf      = NullableDouble(r, 22),
        PrecipSumIcon       = NullableDouble(r, 23),
        PrecipSumMf         = NullableDouble(r, 24),
        PrecipSumUkmo       = NullableDouble(r, 25),
        PrecipSumGem        = NullableDouble(r, 26),
        PrecipSumAifs       = NullableDouble(r, 27),
        PrecipSumJma        = NullableDouble(r, 28),
        FeatureVectorHash   = r.IsDBNull(29) ? "" : r.GetString(29),
    };

    // -----------------------------------------------------------------
    // Shared
    // -----------------------------------------------------------------

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
