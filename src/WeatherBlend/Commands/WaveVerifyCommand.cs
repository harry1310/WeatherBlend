using System.Data;
using System.Text.Json;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Waves;
using WeatherBlend.Models;
using WeatherBlend.Storage;
using WeatherBlend.Train;

namespace WeatherBlend.Commands;

/// <summary>
/// Rolling verification of the wave_height blend against the location's
/// primary wave buoy (Sevenstones for Sennen — verification truth; era5_ocean
/// is the TRAINING truth and is deliberately not scored here). Joins the
/// weekly verify cadence: markdown report + JSON history sidecar per
/// location, exit 4 on drift — same contract as the other verify commands.
///
/// Locations are driven from config: every location with a <c>marine:</c>
/// block that lists at least one buoy gets verified (no hardcoded location
/// list — adding a second marine location is a config.yaml edit). The first
/// configured buoy is the primary truth source.
///
/// Metrics per (version, 24h actual-lead bucket): Hs MAE (m) and the
/// empirical coverage of the calibrated 90% band. Drift = band coverage
/// well below the nominal 0.90 (floor 0.75) — COVERAGE-ONLY. There is no
/// MAE-threshold drift because the only training-time reference (cal_mae,
/// vs era5_ocean) sits under the buoy's ~0.3 m location-mismatch floor and
/// would alarm permanently; the CLI --drift multiplier is accepted but
/// ignored for this target. See <see cref="WaveVerifier"/> for the full
/// rationale.
/// </summary>
public sealed class WaveVerifyCommand
{
    /// <summary>Production drift-noise gate — same value the temp/element
    /// verifiers use (TempVerifyCommand.MinDriftNDefault).</summary>
    internal const int MinDriftNDefault = 10;

    /// <summary>Coverage "well below nominal 0.90" floor for the drift flag.</summary>
    internal const double CoverageDriftFloorDefault = 0.75;

    private readonly ILogger<WaveVerifyCommand> _log;
    private readonly AppConfig _cfg;
    private readonly TruthRepository _truth;
    private readonly ModelMetadataRepository _metadata;

    public WaveVerifyCommand(ILogger<WaveVerifyCommand> log, AppConfig cfg,
        TruthRepository truth, ModelMetadataRepository metadata)
    {
        _log = log;
        _cfg = cfg;
        _truth = truth;
        _metadata = metadata;
    }

    /// <summary><paramref name="driftThreshold"/> is accepted for CLI
    /// signature parity with the other verify targets but is NOT used —
    /// wave drift is coverage-only (see class summary).</summary>
    public async Task<int> RunAsync(
        DateOnly? asOf,
        int windowDays,
        int truthLatencyDays,
        double driftThreshold,
        string? locationOverride,
        CancellationToken ct)
    {
        _ = driftThreshold;
        if (_cfg.Locations.Count == 0)
            throw new InvalidOperationException("No locations configured.");

        // Marine locations from config — no hardcoded list. An explicit
        // --location must still be a marine location (we need its buoy).
        var marineLocs = _cfg.Locations
            .Where(l => l.Marine is { Buoys.Count: > 0 })
            .ToList();
        if (!string.IsNullOrWhiteSpace(locationOverride))
            marineLocs = marineLocs
                .Where(l => string.Equals(l.Name, locationOverride, StringComparison.OrdinalIgnoreCase))
                .ToList();

        if (marineLocs.Count == 0)
        {
            _log.LogWarning(
                "No marine locations with configured buoys{Scope} — nothing to verify.",
                string.IsNullOrWhiteSpace(locationOverride) ? "" : $" matching '{locationOverride}'");
            return 0;
        }

        var worst = 0;
        foreach (var loc in marineLocs)
        {
            var exit = await VerifyLocationAsync(loc, asOf, windowDays, truthLatencyDays, ct);
            worst = Math.Max(worst, exit);
        }
        return worst;
    }

    private async Task<int> VerifyLocationAsync(
        LocationConfig loc,
        DateOnly? asOf,
        int windowDays,
        int truthLatencyDays,
        CancellationToken ct)
    {
        var isPrimary = string.Equals(loc.Name, _cfg.Locations[0].Name, StringComparison.OrdinalIgnoreCase);
        var buoy = loc.Marine!.Buoys[0];

        var asOfUtc = asOf.HasValue
            ? asOf.Value.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc).AddDays(1)
            : DateTime.UtcNow;
        var windowStart = asOfUtc.AddDays(-windowDays);
        var windowEnd   = asOfUtc.AddDays(-truthLatencyDays);

        _log.LogInformation(
            "Verify wave_height: location={Loc}, buoy={Buoy}, as-of {AsOf:yyyy-MM-dd HH:mm}Z, window [{Start:yyyy-MM-dd}..{End:yyyy-MM-dd}], latency {Latency}d, drift = coverage < {Floor:0.00} (the --drift multiplier doesn't apply to waves)",
            loc.Name, buoy.Slug, asOfUtc, windowStart, windowEnd, truthLatencyDays, CoverageDriftFloorDefault);

        var predictions = QueryPredictions(loc.Name, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} wave prediction rows in window.", predictions.Count);

        if (predictions.Count == 0)
        {
            _log.LogWarning("No wave predictions in window for {Loc} — skipping report.", loc.Name);
            return 0;
        }

        var truth = _truth.GetBuoyWaveHeightHourly(loc.Name, buoy.Slug, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} buoy Hs truth hours (source={Src}).", truth.Count, buoy.Slug);

        // Period truth (buoy Tz) for the pass-through period check — same
        // tree, second column. Harry's ask 2026-06-12: score the unblended
        // best_match period so "do we need to blend period?" is a measured
        // call, not a hunch.
        var periodTruth = _truth.GetBuoyWavePeriodHourly(loc.Name, buoy.Slug, windowStart, windowEnd, ct);
        _log.LogInformation("Loaded {N} buoy period (Tz) truth hours.", periodTruth.Count);

        var versions = predictions.Select(p => p.ModelVersion).Distinct(StringComparer.Ordinal).ToList();
        var references = LoadCalibrationReferences(loc.Name, versions);
        var metadata = _metadata.GetTrainingMetadataForVersions("wave_height", loc.Name, versions);

        var rows = WaveVerifier.Compute(new WaveVerifier.Inputs
        {
            Predictions = predictions,
            TruthByTime = truth,
            PeriodTruthByTime = periodTruth,
            ReferenceByVersion = references,
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            TruthLatencyDays = truthLatencyDays,
            CoverageDriftFloor = CoverageDriftFloorDefault,
            MinDriftN = MinDriftNDefault,
        });

        var md = WaveVerifyReporter.BuildMarkdown(
            loc.Name, buoy.Slug, buoy.DisplayName, asOfUtc, windowDays, truthLatencyDays,
            CoverageDriftFloorDefault, rows, references);

        Directory.CreateDirectory(_cfg.Storage.ReportsPath);
        var mdName = isPrimary
            ? $"verify_wave_height_{asOfUtc:yyyy-MM-dd}.md"
            : $"verify_wave_height_{loc.Name}_{asOfUtc:yyyy-MM-dd}.md";
        var outPath = Path.Combine(_cfg.Storage.ReportsPath, mdName);
        await File.WriteAllTextAsync(outPath, md, ct);
        _log.LogInformation("Report written → {Path}", outPath);

        // Structured sidecar for the Sea-state Models page. Station carries
        // the location slug (the per-loc card key, like temp/element).
        // LeadHours carries the 24h actual-lead bucket's upper-edge label so
        // the page's lead columns line up with the Leads.Full convention.
        var history = new VerifyHistoryFile
        {
            Target = "wave_height",
            AsOfUtc = asOfUtc,
            WindowDays = windowDays,
            LatencyDays = truthLatencyDays,
            MetricLabel = "Hs MAE (m)",
            Rows = rows.Select(r => new VerifyHistoryRow
            {
                Station = loc.Name,
                ModelVersion = r.ModelVersion,
                Phase = metadata.TryGetValue(r.ModelVersion, out var meta) ? meta.Phase
                    : ModelArtifact.ExtractPhaseFromVersionName(r.ModelVersion)
                      ?? (r.ModelVersion.Contains("wave_height_lgb", StringComparison.Ordinal) ? "wave_height_lgb" : null),
                LeadHours = r.LeadBucketHours,
                WindowHours = null,
                N = r.N,
                BlendMetric = r.BlendMae,
                ClimMetric = null,
                MeanOfModelsMetric = null,
                BestSingleName = null,
                BestSingleMetric = null,
                ReferenceTrainingMetric = r.ReferenceCalMae,
                DriftFlag = r.DriftFlag,
                BandCoverage = double.IsNaN(r.BandCoverage) ? null : r.BandCoverage,
            }).ToList(),
        };
        await WeatherBlend.Evaluate.VerifyHistoryWriter.WriteAsync(
            _cfg.Storage.ReportsPath, history, locationName: isPrimary ? null : loc.Name, ct);

        // Second sidecar: the PASS-THROUGH period vs buoy Tz, as its own
        // target so the page machinery reads it like any other metric
        // series. Never drift-flagged — the model-mean-vs-Tz definitional
        // offset sits permanently in the bias; the signal is that bias
        // MOVING, which the chart shows.
        var periodRows = rows.Where(r => r.NPeriod > 0).ToList();
        if (periodRows.Count > 0)
        {
            var periodHistory = new VerifyHistoryFile
            {
                Target = "wave_period",
                AsOfUtc = asOfUtc,
                WindowDays = windowDays,
                LatencyDays = truthLatencyDays,
                MetricLabel = "period MAE vs Tz (s)",
                Rows = periodRows.Select(r => new VerifyHistoryRow
                {
                    Station = loc.Name,
                    ModelVersion = r.ModelVersion,
                    Phase = "best_match_passthrough",
                    LeadHours = r.LeadBucketHours,
                    WindowHours = null,
                    N = r.NPeriod,
                    BlendMetric = r.PeriodMae,
                    ClimMetric = null,
                    MeanOfModelsMetric = null,
                    BestSingleName = null,
                    BestSingleMetric = null,
                    ReferenceTrainingMetric = null,
                    DriftFlag = false,
                    BandCoverage = null,
                }).ToList(),
            };
            await WeatherBlend.Evaluate.VerifyHistoryWriter.WriteAsync(
                _cfg.Storage.ReportsPath, periodHistory, locationName: isPrimary ? null : loc.Name, ct);
        }

        foreach (var r in rows)
        {
            _log.LogInformation(
                "{Version} ≤{Bucket}h — n={N}, Hs MAE {Mae:0.000} m, bias {Bias:+0.000;-0.000;0.000}, band coverage {Cov} (n={NBand}){Drift}; period MAE {PMae} bias {PBias} (n={NPeriod}, vs Tz)",
                r.ModelVersion, r.LeadBucketHours, r.N, r.BlendMae, r.BlendBias,
                double.IsNaN(r.BandCoverage) ? "—" : r.BandCoverage.ToString("0.000"),
                r.NBand, r.DriftFlag ? "  [DRIFT]" : "",
                double.IsNaN(r.PeriodMae) ? "—" : r.PeriodMae.ToString("0.00") + " s",
                double.IsNaN(r.PeriodBias) ? "—" : r.PeriodBias.ToString("+0.00;-0.00;0.00") + " s",
                r.NPeriod);
        }

        return rows.Any(r => r.DriftFlag) ? 4 : 0;
    }

    /// <summary>
    /// Parse <c>calibration.json</c> from each scored version's bundle dir
    /// (<c>{models}/wave_height/{loc}/{version}/</c>) — the report/sidecar's
    /// cal-slice CONTEXT numbers (cal_mae / cal_band_coverage), never a
    /// drift trigger. Python-written snake_case; parsed tolerantly so a
    /// missing or partial file degrades the context columns rather than
    /// failing the run.
    /// </summary>
    private IReadOnlyDictionary<string, WaveVerifier.CalibrationReference> LoadCalibrationReferences(
        string locationName, IEnumerable<string> versions)
    {
        var result = new Dictionary<string, WaveVerifier.CalibrationReference>(StringComparer.Ordinal);
        foreach (var v in versions)
        {
            var path = Path.Combine(_cfg.Storage.ModelsPath, "wave_height", locationName, v, "calibration.json");
            if (!File.Exists(path))
            {
                _log.LogWarning("No calibration.json for {Version} — cal-slice context columns will be empty.", v);
                continue;
            }
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                var root = doc.RootElement;
                double? Num(string name) =>
                    root.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.Number
                        ? el.GetDouble() : null;
                string? Str(string name) =>
                    root.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.String
                        ? el.GetString() : null;
                result[v] = new WaveVerifier.CalibrationReference(
                    CalMae: Num("cal_mae"),
                    CalBandCoverage: Num("cal_band_coverage"),
                    NominalCoverage: Num("coverage"),
                    LeadMode: Str("lead_mode"),
                    Note: Str("note"));
            }
            catch (Exception ex)
            {
                _log.LogWarning("Failed to parse calibration.json for {Version}: {Msg}", v, ex.Message);
            }
        }
        return result;
    }

    private IReadOnlyList<WavePredictionRow> QueryPredictions(
        string locationName, DateTime start, DateTime end, CancellationToken ct)
    {
        var glob = ParquetReader.Glob(Path.Combine(_cfg.Storage.PredictionsPath, "wave_height", "**", "*.parquet"));

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        if (!ParquetReader.HasColumn(conn, glob, "BlendValue"))
        {
            _log.LogWarning("wave_height predictions tree empty — nothing to verify.");
            return Array.Empty<WavePredictionRow>();
        }
        // The Python writer stamps PredictionMadeAtUtc as TIMESTAMPTZ — pin
        // the session to UTC so the CAST yields UTC wall time on any runner
        // (same guard as RenderSiteCommand.QueryWavePredictions; without it
        // the made-at drifts an hour during BST).
        using (var tz = conn.CreateCommand())
        {
            tz.CommandText = "SET TimeZone = 'UTC'";
            tz.ExecuteNonQuery();
        }

        var sql = $@"
SELECT LocationName, ModelVersion,
       CAST(PredictionMadeAtUtc AS TIMESTAMP) AS PredictionMadeAtUtc,
       ValidTimeUtc, CAST(LeadHours AS INTEGER) AS LeadHours,
       BlendValue, BandLoM, BandHiM, WavePeriodS
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE Element = 'wave_height'
  AND LocationName = '{locationName.Replace("'", "''")}'
  AND BlendValue IS NOT NULL
  AND ValidTimeUtc >= TIMESTAMP '{start:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{end:yyyy-MM-dd HH:mm:ss}'
ORDER BY ModelVersion, LeadHours, ValidTimeUtc";

        return ParquetReader.Query(conn, sql, r => new WavePredictionRow
        {
            LocationName        = r.GetString(0),
            ModelVersion        = r.GetString(1),
            PredictionMadeAtUtc = r.GetDateTime(2),
            ValidTimeUtc        = r.GetDateTime(3),
            LeadHours           = r.GetInt32(4),
            BlendValue          = r.GetDouble(5),
            BandLoM             = r.IsDBNull(6) ? null : r.GetDouble(6),
            BandHiM             = r.IsDBNull(7) ? null : r.GetDouble(7),
            WavePeriodS         = r.IsDBNull(8) ? null : r.GetDouble(8),
        }, _log, "wave_height predictions tree empty — nothing to verify.", ct);
    }
}
