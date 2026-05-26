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
    /// Temperature predictions whose <c>ValidTimeUtc</c> falls in
    /// [<paramref name="start"/>, <paramref name="end"/>], optionally scoped to a
    /// list of <paramref name="locations"/>. Pass <c>null</c> or an empty list to
    /// scan every location partition under the temperature subtree (multi-loc
    /// renderer pattern); pass an explicit list to limit to those locations
    /// (verify pattern — cheaper scan, no cross-location noise).
    /// Drops rows with null <c>BlendTemperature</c> — they're meaningless and
    /// the renderer / verify both want to ignore them. Order is
    /// <c>(ModelVersion, LeadHours, ValidTimeUtc)</c>; callers re-sort if
    /// they need a different order.
    /// </summary>
    public IReadOnlyList<TempPredictionRow> GetTemperaturePredictions(
        IReadOnlyList<string>? locations,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Scope to the temperature subtree only — sibling subtrees (precipitation,
        // dry_window) have different schemas and would surface as nulls or wrong
        // columns under union_by_name.
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "temperature", "**", "*.parquet"));

        var locationClause = locations is { Count: > 0 }
            ? "AND LocationName IN (" + string.Join(", ",
                  locations.Select(l => $"'{l.Replace("'", "''")}'")) + ")"
            : ""; // null/empty = no location filter, scan everything

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
WHERE BlendTemperature IS NOT NULL
  {locationClause}
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

        // Predictions are tagged with the LocationName whose NWP fed the
        // featureset (see PrecipPredictCommand: `_activeLocation.Name`).
        // Multi-location PoC (2026-05-11) means the tree carries rows for
        // every configured location; build the IN list from `_cfg.Locations`
        // so adding a third location doesn't need a code change here.
        // Per-station home-location filtering happens downstream in
        // SitePages.Forecasts so a Membury station never picks up
        // Bonehill-NWP rows of the same model version (or vice versa).
        var locationInList = string.Join(", ",
            _cfg.Locations.Select(l => "'" + l.Name.Replace("'", "''") + "'"));
        if (string.IsNullOrEmpty(locationInList))
        {
            // Defensive: empty config → match nothing rather than throw, so
            // tests / smoke runs without a Locations block degrade to "no rows".
            locationInList = "''";
        }

        // ConformalSetTag persisted only since precip-conformal-fit (2026-05-03).
        // ProbWetStd / ProbWetQ05 / ProbWetQ95 / Ci80Width / Ci90Width persisted
        // since the Bayesian-uncertainty alignment 2026-05-09 (4a).
        // union_by_name in the read clause silently fills NULL for older parquets
        // missing any of these columns — handled by NullableDouble in MapPrecipRow.
        var sql = $@"
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem, PrecipAifs, PrecipJma,
       PrecipAgreementWet01,
       FeatureVectorHash,
       ConformalSetTag,
       ProbWetStd, ProbWetQ05, ProbWetQ95, Ci80Width, Ci90Width
FROM {fromClause}
WHERE LocationName IN ({locationInList})
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
  -- Defensive: predictors occasionally write null-band rows (cycle×lead
  -- combos where Open-Meteo features were missing). MapPrecipRow's
  -- GetDouble(6) assumes non-null and would throw — filter at SQL.
  AND ProbWet IS NOT NULL
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
        // ClimatologyPWet null-safe so older parquets without the column
        // pass through with NaN; verify still computes Brier from ProbWet
        // vs truth.
        ClimatologyPWet     = r.IsDBNull(7) ? double.NaN : r.GetDouble(7),
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
        ConformalSetTag      = r.IsDBNull(18) ? null : r.GetString(18),
        // Bayesian-uncertainty columns (4a). NullableDouble returns null
        // for older parquets (pre-2026-05-09 4a, all 3a/3c/3d).
        ProbWetStd  = NullableDouble(r, 19),
        ProbWetQ05  = NullableDouble(r, 20),
        ProbWetQ95  = NullableDouble(r, 21),
        Ci80Width   = NullableDouble(r, 22),
        Ci90Width   = NullableDouble(r, 23),
    };

    // -----------------------------------------------------------------
    // Rainfall amount (Phase 3f distributional forecast)
    // -----------------------------------------------------------------

    /// <summary>
    /// Phase 3f rainfall_amount predictions whose <c>ValidTimeUtc</c> falls in
    /// [<paramref name="start"/>, <paramref name="end"/>], optionally scoped to a
    /// list of <paramref name="stations"/>. Pass <c>null</c> or an empty list to
    /// scan the whole rainfall_amount subtree; pass an explicit list to limit to
    /// those station partitions (verify pattern).
    ///
    /// Schema mirrors <see cref="RainfallAmountPredictionRow"/>. The Python
    /// writer (predict_3f.py) emits these column names verbatim — any
    /// rename here MUST be made in lockstep with the Python side or the
    /// DuckDB read trips on a missing column.
    /// </summary>
    public IReadOnlyList<RainfallAmountPredictionRow> GetRainfallAmountPredictions(
        IReadOnlyList<string>? stations,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var fromClause = (stations is null || stations.Count == 0)
            ? $"read_parquet('{ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "rainfall_amount", "**", "*.parquet"))}', hive_partitioning = false, union_by_name = true)"
            : "read_parquet([" + string.Join(", ", stations.Select(s =>
                "'" + ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "rainfall_amount", s, "**", "*.parquet")) + "'"))
                + "], hive_partitioning = false, union_by_name = true)";

        var locationInList = string.Join(", ",
            _cfg.Locations.Select(l => "'" + l.Name.Replace("'", "''") + "'"));
        if (string.IsNullOrEmpty(locationInList)) locationInList = "''";

        var sql = $@"
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       Pi, MuLog, SigmaLog,
       MeanMmPerHr, MedianMmPerHr,
       P2_5MmPerHr, P10MmPerHr, P50MmPerHr, P90MmPerHr, P97_5MmPerHr,
       PExceed0_1, PExceed1, PExceed5, PExceed10,
       Precip3aVersion
FROM {fromClause}
WHERE LocationName IN ({locationInList})
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
  -- Defensive: predict_3f silently skips cells where the (μ, σ) NGBoost
  -- inference failed; those rows shouldn't reach the parquet, but
  -- filter on Pi anyway so a half-written cycle can't crash MapRainfallAmountRow.
  AND Pi IS NOT NULL
ORDER BY TruthStation, ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(sql, MapRainfallAmountRow,
            _log, "Rainfall amount predictions tree empty — 3f reads will return no rows.", ct);
    }

    private static RainfallAmountPredictionRow MapRainfallAmountRow(IDataReader r) => new()
    {
        LocationName        = r.GetString(0),
        TruthStation        = r.GetString(1),
        ModelVersion        = r.GetString(2),
        PredictionMadeAtUtc = r.GetDateTime(3),
        ValidTimeUtc        = r.GetDateTime(4),
        LeadHours           = r.GetInt32(5),
        Pi                  = r.GetDouble(6),
        MuLog               = r.GetDouble(7),
        SigmaLog            = r.GetDouble(8),
        MeanMmPerHr         = r.GetDouble(9),
        MedianMmPerHr       = r.GetDouble(10),
        P2_5MmPerHr         = r.GetDouble(11),
        P10MmPerHr          = r.GetDouble(12),
        P50MmPerHr          = r.GetDouble(13),
        P90MmPerHr          = r.GetDouble(14),
        P97_5MmPerHr        = r.GetDouble(15),
        PExceed0_1          = r.GetDouble(16),
        PExceed1            = r.GetDouble(17),
        PExceed5            = r.GetDouble(18),
        PExceed10           = r.GetDouble(19),
        Precip3aVersion     = r.IsDBNull(20) ? "" : r.GetString(20),
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

        // Multi-location filter (mirrors the precip query above).
        var locationInListDw = string.Join(", ",
            _cfg.Locations.Select(l => "'" + l.Name.Replace("'", "''") + "'"));
        if (string.IsNullOrEmpty(locationInListDw)) locationInListDw = "''";

        // McMean/P10/P50/P90LongestDryRunHours: persisted only by Phase 3g
        // since 2026-05-03. Older parquets lack the columns; union_by_name
        // already on the read silently fills them as NULL for those rows.
        var sql = $@"
SELECT LocationName, TruthStation, WindowHours, ModelVersion,
       PredictionMadeAtUtc, TargetDateUtc, LeadHours,
       ProbHasDryWindow, ClimatologyProbHasDryWindow,
       AgreementHasDryWindow, PrecipSumMean, LongestDryRunMean, WetHourCountMean,
       HasDryWindowGfs, HasDryWindowEcmwf, HasDryWindowIcon,
       HasDryWindowMf,  HasDryWindowUkmo,  HasDryWindowGem,  HasDryWindowAifs,  HasDryWindowJma,
       PrecipSumGfs, PrecipSumEcmwf, PrecipSumIcon, PrecipSumMf, PrecipSumUkmo, PrecipSumGem,
       PrecipSumAifs, PrecipSumJma,
       FeatureVectorHash,
       McMeanLongestDryRunHours, McP10LongestDryRunHours,
       McP50LongestDryRunHours,  McP90LongestDryRunHours,
       EpistemicProbDryWindowMean, EpistemicProbDryWindowQ10,
       EpistemicProbDryWindowQ90,  EpistemicSigmaUsed,
       ConformalSetTag
FROM {fromClause}
WHERE LocationName IN ({locationInListDw})
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
        McMeanLongestDryRunHours = NullableDouble(r, 30),
        McP10LongestDryRunHours  = NullableDouble(r, 31),
        McP50LongestDryRunHours  = NullableDouble(r, 32),
        McP90LongestDryRunHours  = NullableDouble(r, 33),
        EpistemicProbDryWindowMean = NullableDouble(r, 34),
        EpistemicProbDryWindowQ10  = NullableDouble(r, 35),
        EpistemicProbDryWindowQ90  = NullableDouble(r, 36),
        EpistemicSigmaUsed         = NullableDouble(r, 37),
        ConformalSetTag          = r.IsDBNull(38) ? null : r.GetString(38),
    };

    // -----------------------------------------------------------------
    // Shared
    // -----------------------------------------------------------------

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static DateTime? NullableDate(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDateTime(ord);
}
