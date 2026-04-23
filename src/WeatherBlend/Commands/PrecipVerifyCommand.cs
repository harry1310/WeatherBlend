using System.Data;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Weekly rolling precipitation verification. Reads <see cref="PrecipPredictionRow"/>
/// parquet + EA 15-min rainfall parquet via DuckDB, aggregates truth to hourly (four
/// readings per hour required, matching training), loads training metadata for every
/// (station, version) that appears, then hands the pre-joined data to
/// <see cref="PrecipVerifier"/> for the stratification + drift logic.
///
/// Defaults tuned for wet-hour sparsity vs temperature's always-dense signal:
///   --window-days   30   (longer than temperature's 14 — wet hours are sparse)
///   --latency-days  5    (EA provisional-reading buffer)
///   --drift         1.5  (flag when rolling blend Brier &gt; 1.5× training test Brier)
/// </summary>
public sealed class PrecipVerifyCommand
{
    private readonly ILogger<PrecipVerifyCommand> _log;
    private readonly AppConfig _cfg;

    public PrecipVerifyCommand(ILogger<PrecipVerifyCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public async Task<int> RunAsync(
        string truthStation,
        DateOnly? asOf,
        int windowDays,
        int latencyDays,
        double driftThreshold,
        CancellationToken ct)
    {
        var modelsRoot = Path.Combine("data", "models");
        var allStations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (allStations.Count == 0)
        {
            _log.LogError("No precipitation blender artefacts found under {Dir}. Train first.",
                Path.Combine(modelsRoot, "precipitation"));
            return 2;
        }

        var stations = FilterStations(allStations, truthStation);
        if (stations.Count == 0)
        {
            _log.LogError("Station filter '{F}' matched no trained stations. Known: [{Known}]",
                truthStation, string.Join(", ", allStations));
            return 2;
        }

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-latencyDays);

        _log.LogInformation(
            "PrecipVerify: as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift {Drift:0.00}×, stations=[{Stations}]",
            asOfUtc, windowStart, windowEnd, latencyDays, driftThreshold, string.Join(", ", stations));

        var predictions = QueryPredictions(stations, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} prediction rows in window.", predictions.Count);
        if (predictions.Count == 0)
        {
            _log.LogWarning("No predictions in window — skipping report.");
            return 0;
        }

        // Truth lookup needs to cover the window + persistence lookback (up to 72h before windowStart).
        var truth = QueryTruth(stations, windowStart.AddHours(-72), windowEnd, ct);
        var truthPoints = truth.Sum(kv => kv.Value.Count);
        _log.LogInformation("Loaded {N} hourly rainfall truth points across {S} stations.",
            truthPoints, truth.Count);

        var metadata = LoadMetadataForKeys(modelsRoot,
            predictions.Select(p => (p.TruthStation, p.ModelVersion)).Distinct());
        _log.LogInformation("Loaded metadata for {N} (station, version) keys.", metadata.Count);

        var rows = PrecipVerifier.Compute(new PrecipVerifier.Inputs
        {
            Predictions = predictions,
            TruthByStationTime = truth,
            MetadataByKey = metadata,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = latencyDays,
            DriftThreshold = driftThreshold,
        });

        var md = PrecipVerifyReporter.BuildMarkdown(asOfUtc, windowDays, latencyDays, driftThreshold, rows);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var outPath = Path.Combine(_cfg.Storage.ReportsPath,
            $"verify_precipitation_{asOfUtc:yyyy-MM-dd}.md");
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Station} / {Version} lead {Lead}h — n={N} (wet={Wet}), Brier {Brier:0.0000} (clim {Clim:0.0000}, BSS {Bss:+0.000;-0.000;0.000}){Drift}",
                r.TruthStation, r.ModelVersion, r.LeadHours, r.N, r.WetN,
                r.BlendBrier, r.ClimBrier, r.Bss,
                r.DriftFlag ? "  [DRIFT]" : "");
        }

        var drifting = rows.Count(r => r.DriftFlag);
        return drifting > 0 ? 4 : 0;
    }

    private IReadOnlyList<string> FilterStations(IReadOnlyList<string> all, string filter)
    {
        if (string.Equals(filter, "all", StringComparison.OrdinalIgnoreCase))
            return all;

        var exact = all.FirstOrDefault(s => string.Equals(s, filter, StringComparison.OrdinalIgnoreCase));
        if (exact is not null) return new[] { exact };

        var derivedSlug = "ea_" + Slugify(filter);
        var byDerived = all.FirstOrDefault(s => string.Equals(s, derivedSlug, StringComparison.OrdinalIgnoreCase));
        return byDerived is null ? Array.Empty<string>() : new[] { byDerived };
    }

    private IReadOnlyList<PrecipPredictionRow> QueryPredictions(
        IReadOnlyList<string> stations,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Only read the station subtrees we're verifying. Passing a per-station glob
        // list into read_parquet() keeps cross-station noise out of the scan.
        var globs = string.Join(", ", stations.Select(s =>
            "'" + Path.Combine(_cfg.Storage.PredictionsPath, "precipitation", s, "**", "*.parquet")
                .Replace('\\', '/').Replace("'", "''") + "'"));

        var sql = $@"
SELECT LocationName, TruthStation, ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours,
       ProbWet, ClimatologyPWet,
       PrecipGfs, PrecipEcmwf, PrecipIcon, PrecipMf, PrecipUkmo, PrecipGem,
       PrecipAgreementWet01,
       FeatureVectorHash
FROM read_parquet([{globs}], hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY TruthStation, ModelVersion, LeadHours, ValidTimeUtc";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var rows = new List<PrecipPredictionRow>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                rows.Add(new PrecipPredictionRow
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
                    PrecipAgreementWet01 = NullableDouble(r, 14),
                    FeatureVectorHash   = r.IsDBNull(15) ? "" : r.GetString(15),
                });
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Predictions tree empty for requested stations — nothing to verify.");
            return Array.Empty<PrecipPredictionRow>();
        }
        return rows;
    }

    private IReadOnlyDictionary<string, IReadOnlyDictionary<DateTime, double>> QueryTruth(
        IReadOnlyList<string> stations,
        DateTime start, DateTime end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        // Map slug → StationName by matching against the config's rainfall stations.
        // Training uses `ea_<Slugify(StationConfig.Name)>`; reverse that here so DuckDB
        // can filter on the exact StationName stored in the truth parquet.
        var stationNamesBySlug = _cfg.Location.Rainfall.Stations.ToDictionary(
            s => "ea_" + Slugify(s.Name),
            s => s.Name,
            StringComparer.OrdinalIgnoreCase);

        var result = new Dictionary<string, IReadOnlyDictionary<DateTime, double>>(StringComparer.OrdinalIgnoreCase);
        foreach (var slug in stations)
        {
            if (!stationNamesBySlug.TryGetValue(slug, out var stationName))
            {
                _log.LogWarning("No config rainfall station maps to slug {Slug} — truth lookup will be empty.", slug);
                result[slug] = new Dictionary<DateTime, double>();
                continue;
            }
            result[slug] = QueryHourlyTruth(stationName, start, end, ct);
        }
        return result;
    }

    private IReadOnlyDictionary<DateTime, double> QueryHourlyTruth(
        string stationName, DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = Path.Combine(_cfg.Storage.RainfallPath, "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        // Same aggregation rule as training: four readings per hour required, else drop.
        // Otherwise a partial hour understates observed mm and can flip wet→dry at boundary.
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
       SUM(Value15MinMm) AS mm_hour
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{_cfg.Location.Name.Replace("'", "''")}'
  AND StationName  = '{stationName.Replace("'", "''")}'
  AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <= TIMESTAMP '{end.AddHours(1):yyyy-MM-dd HH:mm:ss}'
GROUP BY 1
HAVING COUNT(*) = 4
ORDER BY 1";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var truth = new Dictionary<DateTime, double>();
        try
        {
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                ct.ThrowIfCancellationRequested();
                truth[r.GetDateTime(0)] = r.GetDouble(1);
            }
        }
        catch (DuckDBException ex) when (ex.Message.Contains("No files found"))
        {
            _log.LogWarning("Rainfall tree empty for station {Station} — cannot verify.", stationName);
        }
        return truth;
    }

    private IReadOnlyDictionary<(string Station, string Version), ModelArtifact.TrainingMetadata>
        LoadMetadataForKeys(string modelsRoot, IEnumerable<(string Station, string Version)> keys)
    {
        var result = new Dictionary<(string, string), ModelArtifact.TrainingMetadata>();
        foreach (var (station, version) in keys)
        {
            try
            {
                var dir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, version);
                if (!Directory.Exists(dir))
                {
                    _log.LogWarning("Metadata missing for {Station} / {V} — drift column will be blank.", station, version);
                    continue;
                }
                result[(station, version)] = ModelArtifact.LoadTrainingMetadata(dir);
            }
            catch (Exception ex)
            {
                _log.LogWarning("Failed to load metadata for {Station} / {V}: {Msg}", station, version, ex.Message);
            }
        }
        return result;
    }

    private static double? NullableDouble(IDataReader r, int ord)
        => r.IsDBNull(ord) ? null : r.GetDouble(ord);

    private static string Slugify(string name)
    {
        var chars = name.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();
        var slug = new string(chars);
        while (slug.Contains("__")) slug = slug.Replace("__", "_");
        return slug.Trim('_');
    }
}
