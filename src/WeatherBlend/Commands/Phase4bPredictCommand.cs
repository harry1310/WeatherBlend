using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 4b per-cycle synthesis — the 2-way arithmetic mean of phase
/// 4a (BART) and phase 3e (MLP) P(wet) for each (station, valid_time,
/// lead) where both components have a prediction. Identified as the
/// best production stack in the 2026-05-12 LightGBM-meta bake-off
/// (mean Brier 0.0830, wins 14/15 station,lead cells).
///
/// Runs as an inline step inside predict-and-render after the .NET
/// predict-all step writes 3e and after the R2 sync has pulled the
/// most recent 4a parquets. Reads both parquets per station,
/// inner-joins on (valid_time, lead) taking the freshest
/// PredictionMadeAtUtc per phase, averages ProbWet, copies the rest
/// of each row from 4a so downstream render sees the standard
/// PrecipPredictionRow shape.
///
/// Output: <c>data/predictions/precipitation/{station}/
/// model_version={4b_version}/date={anchor}/predictions.parquet</c>.
/// 4b's bundle version comes from MANIFEST.Active (one *_phase4b
/// entry per station, refreshed by Phase4bMintCommand post-retrain).
///
/// Soft-skip exit codes (treated as non-fatal by predict-and-render):
///   2 — no stations configured
///   3 — at least one component missing for every station
/// </summary>
public sealed class Phase4bPredictCommand
{
    private readonly ILogger<Phase4bPredictCommand> _log;
    private readonly AppConfig _cfg;

    public Phase4bPredictCommand(ILogger<Phase4bPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
    {
        var stations = ResolveStations();
        if (stations.Count == 0)
        {
            _log.LogError("No precipitation stations configured — cannot synthesise 4b.");
            return 2;
        }

        var anchor = forDate ?? DateOnly.FromDateTime(DateTime.UtcNow);
        var synthesisTime = DateTime.UtcNow;

        _log.LogInformation(
            "Phase 4b synthesis — anchor={Anchor:yyyy-MM-dd}, synthesis_time={Synth:yyyy-MM-dd HH:mm}Z, stations=[{Stations}]",
            anchor, synthesisTime, string.Join(", ", stations));

        int nOk = 0, nSkip = 0;
        foreach (var station in stations)
        {
            ct.ThrowIfCancellationRequested();
            var (ok, note) = await SynthesiseStationAsync(station, anchor, synthesisTime, ct);
            _log.LogInformation("{Tag} {Station}: {Note}", ok ? "OK  " : "SKIP", station, note);
            if (ok) nOk++; else nSkip++;
        }
        _log.LogInformation("Done. Wrote: {OK}, skipped: {Skip}.", nOk, nSkip);
        return nOk > 0 ? 0 : 3;
    }

    /// <summary>EA station slugs for every rainfall station across every
    /// configured location. Bonehill-only today; Membury 4b waits for
    /// Phase B (Membury 3e + 4a not yet trained).</summary>
    private IReadOnlyList<string> ResolveStations()
    {
        var slugs = new List<string>();
        foreach (var loc in _cfg.Locations)
        {
            foreach (var s in loc.Rainfall.Stations)
                slugs.Add(StationSlug.WithEaPrefix(s.Name));
        }
        return slugs;
    }

    private async Task<(bool ok, string note)> SynthesiseStationAsync(
        string station, DateOnly anchor, DateTime synthesisTime, CancellationToken ct)
    {
        var modelsRoot = _cfg.Storage.ModelsPath;
        var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
        var version4b = active.FirstOrDefault(v => v.EndsWith("_phase4b", StringComparison.Ordinal));
        if (version4b is null)
            return (false, "MANIFEST.Active has no *_phase4b entry (run mint-4b first)");

        var predRoot = _cfg.Storage.PredictionsPath;
        var stationDir = Path.Combine(predRoot, "precipitation", station);
        var parquet4a = FindLatestPredictionParquet(stationDir, "_phase4a", anchor);
        var parquet3e = FindLatestPredictionParquet(stationDir, "_phase3e", anchor);
        if (parquet4a is null) return (false, $"no 4a predictions parquet for date={anchor:yyyy-MM-dd}");
        if (parquet3e is null) return (false, $"no 3e predictions parquet for date={anchor:yyyy-MM-dd}");

        // Read both with DuckDB rather than ParquetSerializer<PrecipPredictionRow> —
        // predict_4a.py writes from Python and only carries the bare minimum
        // columns (ProbWet + quantile bands + metadata, no ClimatologyPWet or
        // per-NWP precip). 3e is .NET-written with the full PrecipPredictionRow
        // schema, so we carry every covariate column from 3e and just plug in
        // 4a's ProbWet for the mean.
        var (rows3e, rows4aProbByKey) = await ReadBothPhasesAsync(parquet4a, parquet3e, ct);
        if (rows3e.Count == 0) return (false, $"3e parquet at {parquet3e} read as empty");
        if (rows4aProbByKey.Count == 0) return (false, $"4a parquet at {parquet4a} read as empty");

        // Inner-join on (valid_time, lead). For each match: ProbWet =
        // mean(p_3e, p_4a), every other covariate column carried from
        // 3e (per-NWP precip, climatology, agreement chip data —
        // downstream render reads these for the chart's NWP overlay
        // and chip annotations). Quantile / CI columns nulled —
        // they're 4a's BART posterior outputs, no defined meaning
        // for an unfit arithmetic mean.
        // PrecipPredictionRow is a sealed class (not a record), so `with`
        // doesn't work — construct each output row explicitly. Verbose
        // but the property list is stable.
        var joined = new List<PrecipPredictionRow>(rows3e.Count);
        foreach (var r in rows3e)
        {
            if (!rows4aProbByKey.TryGetValue((r.ValidTimeUtc, r.LeadHours), out var p4a))
                continue;
            joined.Add(new PrecipPredictionRow
            {
                LocationName         = r.LocationName,
                TruthStation         = r.TruthStation,
                ModelVersion         = version4b,
                PredictionMadeAtUtc  = synthesisTime,
                ValidTimeUtc         = r.ValidTimeUtc,
                LeadHours            = r.LeadHours,
                ProbWet              = (r.ProbWet + p4a) / 2.0,
                ClimatologyPWet      = r.ClimatologyPWet,
                PrecipGfs            = r.PrecipGfs,
                PrecipEcmwf          = r.PrecipEcmwf,
                PrecipIcon           = r.PrecipIcon,
                PrecipMf             = r.PrecipMf,
                PrecipUkmo           = r.PrecipUkmo,
                PrecipGem            = r.PrecipGem,
                PrecipAifs           = r.PrecipAifs,
                PrecipJma            = r.PrecipJma,
                PrecipMean           = r.PrecipMean,
                PrecipStd            = r.PrecipStd,
                PrecipMax            = r.PrecipMax,
                PrecipAgreementWet01 = r.PrecipAgreementWet01,
                FeatureVectorHash    = r.FeatureVectorHash,
                // Quantile + CI columns are 4a's BART posterior outputs —
                // no defined meaning for an unfit arithmetic mean, so
                // null them. Exact-runtime cols + per-NWP run timestamps
                // left default (not relevant for 4b consumers).
                ConformalSetTag      = null,
                ProbWetStd           = null,
                ProbWetQ05           = null,
                ProbWetQ95           = null,
                Ci80Width            = null,
                Ci90Width            = null,
            });
        }
        if (joined.Count == 0)
            return (false, "inner-join of latest 4a + 3e produced 0 rows (no shared valid_time + lead — different anchor times?)");

        // Same merge-on-write as PrecipPredictCommand.MergeRows.
        var dateStr = anchor.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(predRoot, "precipitation", station,
            $"model_version={version4b}", $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");
        List<PrecipPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<PrecipPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<PrecipPredictionRow>();
        var merged = existing.Concat(joined)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.LeadHours, r.ValidTimeUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.LeadHours)
            .ToList();
        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        return (true, $"wrote {joined.Count} new rows (file now holds {merged.Count}); model_version={version4b}");
    }

    /// <summary>
    /// Read both parquets via DuckDB, inner-joining on (ValidTime,
    /// Lead). Returns:
    ///   - rows3e: full PrecipPredictionRow rows from 3e (which carries
    ///     the rich .NET-side schema — ClimatologyPWet, per-NWP precip,
    ///     agreement chip, FeatureVectorHash, etc).
    ///   - rows4aProbByKey: dict of (ValidTime, Lead) -> ProbWet from
    ///     4a (predict_4a.py's parquet only carries the bare minimum
    ///     columns, hence we pull just P(wet) and ignore the rest).
    ///
    /// Both are deduped by latest PredictionMadeAtUtc per cell so a
    /// multi-cycle parquet collapses to one row per (ValidTime, Lead).
    /// Caller carries 3e's rich columns into the synthesised 4b rows;
    /// only ProbWet differs.
    /// </summary>
    private static async Task<(List<PrecipPredictionRow> rows3e,
                               Dictionary<(DateTime, int), double> rows4aProbByKey)>
        ReadBothPhasesAsync(string parquet4a, string parquet3e, CancellationToken ct)
    {
        await Task.Yield();   // keep async signature for symmetry; DuckDB is sync
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var p4aGlob = parquet4a.Replace("\\", "/");
        var p3eGlob = parquet3e.Replace("\\", "/");

        // 3e: full PrecipPredictionRow schema (.NET-written from
        // PrecipPredictCommand). ROW_NUMBER picks the freshest PMT per
        // (ValidTime, Lead) so a multi-cycle file collapses to one row.
        var rows3e = new List<PrecipPredictionRow>();
        var sql3e = $@"
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours ORDER BY PredictionMadeAtUtc DESC) AS rn
  FROM read_parquet('{p3eGlob}', union_by_name = true)
)
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem, PrecipAifs, PrecipJma,
       PrecipMean, PrecipStd, PrecipMax, PrecipAgreementWet01,
       FeatureVectorHash
FROM ranked WHERE rn = 1 ORDER BY ValidTimeUtc, LeadHours";
        using (var cmd3e = conn.CreateCommand())
        {
            cmd3e.CommandText = sql3e;
            using var r = cmd3e.ExecuteReader();
            while (r.Read())
            {
                rows3e.Add(new PrecipPredictionRow
                {
                    LocationName        = r.GetString(0),
                    TruthStation        = r.GetString(1),
                    ModelVersion        = r.GetString(2),
                    PredictionMadeAtUtc = r.GetDateTime(3),
                    ValidTimeUtc        = r.GetDateTime(4),
                    LeadHours           = (int)r.GetInt64(5),
                    ProbWet             = r.GetDouble(6),
                    ClimatologyPWet     = r.IsDBNull(7)  ? 0.0  : r.GetDouble(7),
                    PrecipGfs           = r.IsDBNull(8)  ? null : (double?)r.GetDouble(8),
                    PrecipEcmwf         = r.IsDBNull(9)  ? null : (double?)r.GetDouble(9),
                    PrecipIcon          = r.IsDBNull(10) ? null : (double?)r.GetDouble(10),
                    PrecipMf            = r.IsDBNull(11) ? null : (double?)r.GetDouble(11),
                    PrecipUkmo          = r.IsDBNull(12) ? null : (double?)r.GetDouble(12),
                    PrecipGem           = r.IsDBNull(13) ? null : (double?)r.GetDouble(13),
                    PrecipAifs          = r.IsDBNull(14) ? null : (double?)r.GetDouble(14),
                    PrecipJma           = r.IsDBNull(15) ? null : (double?)r.GetDouble(15),
                    PrecipMean          = r.IsDBNull(16) ? null : (double?)r.GetDouble(16),
                    PrecipStd           = r.IsDBNull(17) ? null : (double?)r.GetDouble(17),
                    PrecipMax           = r.IsDBNull(18) ? null : (double?)r.GetDouble(18),
                    PrecipAgreementWet01 = r.IsDBNull(19) ? null : (double?)r.GetDouble(19),
                    FeatureVectorHash   = r.IsDBNull(20) ? "" : r.GetString(20),
                });
            }
        }

        // 4a: just the key + ProbWet — predict_4a.py's parquet only
        // writes the bare minimum (ValidTimeUtc/LeadHours/ProbWet +
        // quantile bands + metadata). The rich covariate columns come
        // from 3e above.
        var rows4a = new Dictionary<(DateTime, int), double>();
        var sql4a = $@"
WITH ranked AS (
  SELECT ValidTimeUtc, LeadHours, ProbWet, PredictionMadeAtUtc,
         ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours ORDER BY PredictionMadeAtUtc DESC) AS rn
  FROM read_parquet('{p4aGlob}', union_by_name = true)
)
SELECT ValidTimeUtc, LeadHours, ProbWet
FROM ranked WHERE rn = 1";
        using (var cmd4a = conn.CreateCommand())
        {
            cmd4a.CommandText = sql4a;
            using var r = cmd4a.ExecuteReader();
            while (r.Read())
            {
                var vt = r.GetDateTime(0);
                var lead = (int)r.GetInt64(1);
                var p = r.GetDouble(2);
                rows4a[(vt, lead)] = p;
            }
        }
        return (rows3e, rows4a);
    }

    private static string? FindLatestPredictionParquet(string stationDir, string phaseSuffix, DateOnly anchor)
    {
        if (!Directory.Exists(stationDir)) return null;
        var versionDirs = Directory.GetDirectories(stationDir)
            .Where(d => Path.GetFileName(d).StartsWith("model_version=", StringComparison.Ordinal)
                     && Path.GetFileName(d).EndsWith(phaseSuffix, StringComparison.Ordinal))
            .OrderByDescending(d => Path.GetFileName(d), StringComparer.Ordinal)
            .ToList();
        foreach (var vdir in versionDirs)
        {
            var dateDir = Path.Combine(vdir, $"date={anchor:yyyy-MM-dd}");
            if (!Directory.Exists(dateDir)) continue;
            var parquet = Path.Combine(dateDir, "predictions.parquet");
            if (File.Exists(parquet)) return parquet;
        }
        return null;
    }
}
