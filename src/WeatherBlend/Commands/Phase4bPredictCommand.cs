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
        // 4a: drive only from THIS cycle's parquet — today's 4a is the
        // canonical source of "what physical times are we synthesising
        // 4b for right now". Older 4a parquets would just produce
        // duplicate (V, L) keys (their forward grids overlap with
        // today's), the dedup would keep today's anyway, and we'd
        // emit zero extra 4b rows for the work.
        var parquet4a = FindLatestPredictionParquet(stationDir, "_phase4a", anchor);
        if (parquet4a is null) return (false, $"no 4a predictions parquet for date={anchor:yyyy-MM-dd}");

        // 3e: scan EVERY anchor's parquet for the station. Reason:
        // 3e's per-cycle parquet at lead L covers exactly one calendar
        // day (anchor + L/24 days, 24 hourly slots), while 4a's
        // per-cycle parquet at lead L spans the full forecast horizon.
        // So today's 4a row at (V=2026-05-13 06:00, L=24) needs to be
        // paired with the 3e cycle whose anchor was 2026-05-12 (lead-24
        // bucket = the day starting 2026-05-13) — yesterday's parquet,
        // not today's. We glob across all date= partitions and let the
        // DuckDB window pick the freshest PredictionMadeAtUtc per
        // (V, L). If multiple parquets agree on a cell, freshest wins.
        var parquet3eGlob = Path.Combine(stationDir, "model_version=*_phase3e", "date=*", "predictions.parquet");
        var anyTouched3e = Directory.EnumerateDirectories(stationDir, "model_version=*_phase3e")
            .Any(d => Directory.EnumerateFiles(d, "predictions.parquet", SearchOption.AllDirectories).Any());
        if (!anyTouched3e) return (false, "no 3e predictions parquets on disk for this station");

        // Read both with DuckDB rather than ParquetSerializer<PrecipPredictionRow> —
        // predict_4a.py writes from Python and only carries the bare minimum
        // columns (ProbWet + quantile bands + metadata, no ClimatologyPWet or
        // per-NWP precip). 3e is .NET-written with the full PrecipPredictionRow
        // schema, so we carry every covariate column from 3e and just plug in
        // 4a's ProbWet for the mean.
        var (rows4a, rows3eByKey) = ReadBothPhases(parquet4a, parquet3eGlob, ct);
        if (rows4a.Count == 0) return (false, $"4a parquet at {parquet4a} read as empty");
        if (rows3eByKey.Count == 0) return (false, "3e parquet glob produced 0 rows across all anchors");

        // Drive from 4a: each row in today's 4a becomes a candidate 4b
        // synthesis cell. Look up the freshest 3e row at the same
        // (ValidTime, Lead) — possibly from an older anchor — and emit
        // the mean if both sides have a value. Covariates ride along
        // from the matched 3e row (slightly stale vs today's NWP cycle
        // when the match is from an earlier anchor, but render only
        // consumes ClimatologyPWet + PrecipAgreementWet01 and both are
        // robust to one-cycle staleness).
        // PrecipPredictionRow is a sealed class (not a record), so `with`
        // doesn't work — construct each output row explicitly. Verbose
        // but the property list is stable.
        var joined = new List<PrecipPredictionRow>(rows4a.Count);
        int matchedCount = 0;
        foreach (var (key, p4a) in rows4a)
        {
            if (!rows3eByKey.TryGetValue(key, out var r3e))
                continue;
            matchedCount++;
            joined.Add(new PrecipPredictionRow
            {
                LocationName         = r3e.LocationName,
                TruthStation         = r3e.TruthStation,
                ModelVersion         = version4b,
                PredictionMadeAtUtc  = synthesisTime,
                ValidTimeUtc         = key.ValidTime,
                LeadHours            = key.Lead,
                ProbWet              = (r3e.ProbWet + p4a) / 2.0,
                ClimatologyPWet      = r3e.ClimatologyPWet,
                PrecipGfs            = r3e.PrecipGfs,
                PrecipEcmwf          = r3e.PrecipEcmwf,
                PrecipIcon           = r3e.PrecipIcon,
                PrecipMf             = r3e.PrecipMf,
                PrecipUkmo           = r3e.PrecipUkmo,
                PrecipGem            = r3e.PrecipGem,
                PrecipAifs           = r3e.PrecipAifs,
                PrecipJma            = r3e.PrecipJma,
                PrecipMean           = r3e.PrecipMean,
                PrecipStd            = r3e.PrecipStd,
                PrecipMax            = r3e.PrecipMax,
                PrecipAgreementWet01 = r3e.PrecipAgreementWet01,
                FeatureVectorHash    = r3e.FeatureVectorHash,
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
            return (false,
                $"no 4a row at any (ValidTime, Lead) found a matching 3e prediction across {rows3eByKey.Count} historical 3e cells " +
                "(4a today has rows but 3e tree has no shared keys — should not happen unless 3e tree is single-anchor)");

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
        return (true,
            $"wrote {joined.Count} new rows (matched {matchedCount}/{rows4a.Count} 4a cells to 3e across {rows3eByKey.Count} historical 3e cells); " +
            $"file now holds {merged.Count}; model_version={version4b}");
    }

    /// <summary>
    /// Read both phases via <see cref="ParquetReader"/>. Returns:
    ///   - rows4a: dict (ValidTime, Lead) → ProbWet, from TODAY's 4a
    ///     parquet only. Today's 4a is the canonical "what physical
    ///     times are we synthesising for now" driver — older 4a
    ///     parquets just produce duplicate keys that today's would win
    ///     anyway, so scanning them is wasted work.
    ///   - rows3eByKey: dict (ValidTime, Lead) → full PrecipPredictionRow
    ///     from the freshest 3e row across EVERY anchor's parquet for
    ///     the station. ROW_NUMBER over (ValidTime, Lead) ORDER BY
    ///     PredictionMadeAtUtc DESC keeps the latest cycle's prediction
    ///     when multiple anchors emit the same (V, L) cell.
    ///
    /// 3e is .NET-written with the full PrecipPredictionRow schema; 4a
    /// is Python-written and only carries ValidTimeUtc, LeadHours, ProbWet,
    /// PredictionMadeAtUtc, plus quantile bands we don't use here.
    /// Caller drives from rows4a and looks up the 3e row by (V, L).
    ///
    /// Each SELECT keeps its per-query ROW_NUMBER CTE inline — that is
    /// exactly what <see cref="ParquetReader"/> is designed to allow; it
    /// just lifts the open / read / dispose / "No files found" boilerplate.
    /// </summary>
    private (Dictionary<(DateTime ValidTime, int Lead), double> rows4a,
             Dictionary<(DateTime ValidTime, int Lead), PrecipPredictionRow> rows3eByKey)
        ReadBothPhases(string parquet4a, string parquet3eGlob, CancellationToken ct)
    {
        // 3e: glob across all anchor parquets for this station's
        // *_phase3e versions. ROW_NUMBER picks the freshest
        // PredictionMadeAtUtc per (ValidTime, Lead) — so an older
        // anchor only contributes a cell that today's anchor doesn't
        // (because the freshest wins per-cell). union_by_name = true
        // tolerates incidental schema drift across parquet versions.
        var sql3e = $@"
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours ORDER BY PredictionMadeAtUtc DESC) AS rn
  FROM read_parquet('{ParquetReader.Glob(parquet3eGlob)}', union_by_name = true)
)
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem, PrecipAifs, PrecipJma,
       PrecipMean, PrecipStd, PrecipMax, PrecipAgreementWet01,
       FeatureVectorHash
FROM ranked WHERE rn = 1";
        var rows3eList = ParquetReader.Query(
            sql3e,
            r => new PrecipPredictionRow
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
            },
            _log,
            $"3e prediction tree empty for glob {parquet3eGlob} — nothing to synthesise from",
            ct);
        var rows3e = rows3eList.ToDictionary(r => (r.ValidTimeUtc, r.LeadHours), r => r);

        // 4a: today's parquet only. Dedupe by freshest PMT per (V, L)
        // in case the file holds multiple intra-day cycles.
        var sql4a = $@"
WITH ranked AS (
  SELECT ValidTimeUtc, LeadHours, ProbWet, PredictionMadeAtUtc,
         ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, LeadHours ORDER BY PredictionMadeAtUtc DESC) AS rn
  FROM read_parquet('{ParquetReader.Glob(parquet4a)}', union_by_name = true)
)
SELECT ValidTimeUtc, LeadHours, ProbWet
FROM ranked WHERE rn = 1";
        var rows4aList = ParquetReader.Query(
            sql4a,
            r => (ValidTime: r.GetDateTime(0), Lead: (int)r.GetInt64(1), ProbWet: r.GetDouble(2)),
            _log,
            $"4a parquet {parquet4a} read as empty",
            ct);
        var rows4a = rows4aList.ToDictionary(t => (t.ValidTime, t.Lead), t => t.ProbWet);

        return (rows4a, rows3e);
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
