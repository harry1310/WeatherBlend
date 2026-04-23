using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Offline scorer that computes per-NWP-model test-set accuracy for every
/// active artefact and drops the result as <c>per_model_test.json</c> next to
/// the model file. The Models-tab render reads those JSONs to plot blend vs
/// each raw NWP model on the exact same held-out slice the blender was tested
/// against.
///
/// Inputs it needs per artefact:
///   - training_metadata.json → <c>PerLead[].DataRangeTest</c> (authoritative
///     bound for the test slice; we slice by this, not by re-splitting)
///   - forecasts parquet tree → per-model forecasts
///   - ERA5 parquet tree → temperature truth
///   - EA rainfall parquet tree → precipitation / dry-window truth
///
/// Metrics:
///   - Temperature: MAE of each model's Temperature2m vs ERA5 on the lead's
///     test slice.
///   - Precipitation: Brier score of deterministic (precip ≥ 0.1mm) per model
///     vs the 0/1 wet label, same slice. Treats a raw NWP precip forecast as
///     a probability of exactly 1.0 or 0.0 — the honest comparison when the
///     model doesn't publish a calibrated P(wet).
///   - Dry-window: Brier score of each model's own "does a length-N dry run
///     exist in this day's 24h" boolean (already computed as a feature in the
///     training row) vs the observed label.
///
/// Idempotent: re-running overwrites the JSON. Runs against whatever is in
/// MANIFEST.json at invocation time.
/// </summary>
public sealed class ScoreHistoricalCommand
{
    public const string PerModelTestFileName = "per_model_test.json";

    private readonly ILogger<ScoreHistoricalCommand> _log;
    private readonly AppConfig _cfg;

    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly int[] DryWindowWindows = { 3, 4, 6 };

    // Model order aligned with FeatureBuilder.ModelColumns / PrecipFeatureBuilder / DryWindowFeatureBuilder.
    private static readonly IReadOnlyList<string> ModelKeys = new[]
    {
        "gfs", "ecmwf", "icon", "mf", "ukmo", "gem",
    };

    public ScoreHistoricalCommand(ILogger<ScoreHistoricalCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    /// <param name="target">"temperature" | "precipitation" | "dry-window" | "all"</param>
    public async Task<int> RunAsync(string target, CancellationToken ct)
    {
        var normalised = target?.Trim().ToLowerInvariant() ?? "all";
        var modelsRoot = Path.Combine("data", "models");

        var totalArtefacts = 0;

        if (normalised is "all" or "temperature")
            totalArtefacts += await ScoreTemperatureAsync(modelsRoot, ct);

        if (normalised is "all" or "precipitation")
            totalArtefacts += await ScorePrecipitationAsync(modelsRoot, ct);

        if (normalised is "all" or "dry-window" or "dry_window")
            totalArtefacts += await ScoreDryWindowAsync(modelsRoot, ct);

        if (totalArtefacts == 0)
        {
            _log.LogWarning("No active artefacts found for target '{Target}'.", target);
            return 1;
        }

        _log.LogInformation("Wrote per_model_test.json for {N} artefact(s).", totalArtefacts);
        return 0;
    }

    // ---- Temperature --------------------------------------------------------

    private async Task<int> ScoreTemperatureAsync(string modelsRoot, CancellationToken ct)
    {
        var active = ModelArtifact.ResolveActive(modelsRoot, "temperature");
        if (active.Count == 0)
        {
            _log.LogWarning("No active temperature versions.");
            return 0;
        }

        // Build per-lead row sets once — shared across every temperature artefact
        // because all phases use the same (forecasts × ERA5) pairing.
        var byLead = new Dictionary<int, List<TrainingRow>>();
        foreach (var lead in Leads)
        {
            _log.LogInformation("Temperature: building rows for lead {Lead}h …", lead);
            byLead[lead] = FeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath,
                _cfg.Storage.Era5Path,
                _cfg.Location.Name,
                lead,
                ct);
        }

        var written = 0;
        foreach (var version in active)
        {
            ct.ThrowIfCancellationRequested();
            var versionDir = Path.Combine(modelsRoot, "temperature", version);
            var metaPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
            if (!File.Exists(metaPath))
            {
                _log.LogWarning("Skipping {Version}: no training_metadata.json", version);
                continue;
            }

            var meta = ModelArtifact.LoadTrainingMetadata(versionDir);
            var output = new PerModelTestScores
            {
                Version = version,
                Target = "temperature",
                Phase = meta.Phase,
                Station = null,
                WindowHours = null,
                ComputedAtUtc = DateTime.UtcNow,
                Metric = "mae",
            };

            foreach (var (leadKey, lead) in meta.PerLead)
            {
                if (!byLead.TryGetValue(lead.LeadHours, out var rows)) continue;
                var slice = SliceByValidTime(rows, lead.DataRangeTest, r => r.ValidTimeUtc).ToList();
                if (slice.Count == 0)
                {
                    _log.LogWarning("Temperature {V} lead {L}h: test slice empty.", version, lead.LeadHours);
                    continue;
                }

                var perModel = new Dictionary<string, ModelMetric>(StringComparer.Ordinal);
                var getters = new Func<TrainingRow, float>[]
                {
                    r => r.TempGfs, r => r.TempEcmwf, r => r.TempIcon,
                    r => r.TempMf,  r => r.TempUkmo, r => r.TempGem,
                };
                for (int i = 0; i < ModelKeys.Count; i++)
                {
                    var get = getters[i];
                    double sumAbs = 0, sumSigned = 0, sumSq = 0;
                    int n = 0;
                    foreach (var r in slice)
                    {
                        var p = get(r);
                        var a = r.Era5Temp;
                        if (float.IsNaN(p) || float.IsNaN(a)) continue;
                        var err = p - a;
                        sumAbs += Math.Abs(err);
                        sumSigned += err;
                        sumSq += err * err;
                        n++;
                    }
                    perModel[ModelKeys[i]] = n == 0
                        ? new ModelMetric(double.NaN, double.NaN, double.NaN, 0)
                        : new ModelMetric(sumAbs / n, Math.Sqrt(sumSq / n), sumSigned / n, n);
                }

                output.PerLead[leadKey] = new PerLeadScores
                {
                    N = slice.Count,
                    DataRange = lead.DataRangeTest,
                    PerModel = perModel,
                };
            }

            await WriteScoresAsync(versionDir, output, ct);
            written++;
        }
        return written;
    }

    // ---- Precipitation ------------------------------------------------------

    private async Task<int> ScorePrecipitationAsync(string modelsRoot, CancellationToken ct)
    {
        var stations = ModelArtifact.ListStations(modelsRoot, "precipitation");
        if (stations.Count == 0)
        {
            _log.LogWarning("No precipitation stations in manifest.");
            return 0;
        }

        var written = 0;
        foreach (var stationSlug in stations)
        {
            ct.ThrowIfCancellationRequested();
            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", stationSlug);
            if (active.Count == 0) continue;

            var friendly = ResolveFriendlyStationName(stationSlug);
            if (friendly is null)
            {
                _log.LogWarning("Precip: unknown station slug '{Slug}' — no friendly name in config. Skipping.", stationSlug);
                continue;
            }

            var byLead = new Dictionary<int, List<PrecipTrainingRow>>();
            foreach (var lead in Leads)
            {
                _log.LogInformation("Precip {Station}: building rows for lead {Lead}h …", stationSlug, lead);
                byLead[lead] = PrecipFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath,
                    _cfg.Storage.RainfallPath,
                    _cfg.Location.Name,
                    friendly,
                    lead,
                    ct);
            }

            foreach (var version in active)
            {
                var versionDir = Path.Combine(modelsRoot, "precipitation", stationSlug, version);
                var metaPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metaPath))
                {
                    _log.LogWarning("Skipping precip {S}/{V}: no training_metadata.json", stationSlug, version);
                    continue;
                }

                var meta = ModelArtifact.LoadTrainingMetadata(versionDir);
                var output = new PerModelTestScores
                {
                    Version = version,
                    Target = "precipitation",
                    Phase = meta.Phase,
                    Station = stationSlug,
                    WindowHours = null,
                    ComputedAtUtc = DateTime.UtcNow,
                    Metric = "brier",
                };

                foreach (var (leadKey, lead) in meta.PerLead)
                {
                    if (!byLead.TryGetValue(lead.LeadHours, out var rows)) continue;
                    var slice = SliceByValidTime(rows, lead.DataRangeTest, r => r.ValidTimeUtc).ToList();
                    if (slice.Count == 0) continue;

                    var perModel = new Dictionary<string, ModelMetric>(StringComparer.Ordinal);
                    var getters = new Func<PrecipTrainingRow, float>[]
                    {
                        r => r.PrecipGfs, r => r.PrecipEcmwf, r => r.PrecipIcon,
                        r => r.PrecipMf,  r => r.PrecipUkmo,  r => r.PrecipGem,
                    };
                    for (int i = 0; i < ModelKeys.Count; i++)
                    {
                        var get = getters[i];
                        double sumSq = 0, sumSigned = 0;
                        int n = 0, positives = 0;
                        foreach (var r in slice)
                        {
                            var p = get(r);
                            if (float.IsNaN(p)) continue;
                            var prob = p >= (float)PrecipFeatureBuilder.WetThresholdMm ? 1.0 : 0.0;
                            var actual = r.WetBinary ? 1.0 : 0.0;
                            var err = prob - actual;
                            sumSq += err * err;
                            sumSigned += err;
                            n++;
                            if (prob > 0) positives++;
                        }
                        perModel[ModelKeys[i]] = n == 0
                            ? new ModelMetric(double.NaN, double.NaN, double.NaN, 0)
                            : new ModelMetric(sumSq / n, Math.Sqrt(sumSq / n), sumSigned / n, n);
                    }

                    output.PerLead[leadKey] = new PerLeadScores
                    {
                        N = slice.Count,
                        DataRange = lead.DataRangeTest,
                        PerModel = perModel,
                    };
                }

                await WriteScoresAsync(versionDir, output, ct);
                written++;
            }
        }
        return written;
    }

    // ---- Dry window ---------------------------------------------------------

    private async Task<int> ScoreDryWindowAsync(string modelsRoot, CancellationToken ct)
    {
        var compositeKeys = ModelArtifact.ListStations(modelsRoot, "dry_window");
        if (compositeKeys.Count == 0)
        {
            _log.LogWarning("No dry_window composites in manifest.");
            return 0;
        }

        var written = 0;
        foreach (var key in compositeKeys)
        {
            ct.ThrowIfCancellationRequested();
            var (slug, windowHours) = ParseCompositeKey(key);
            if (slug is null || !DryWindowWindows.Contains(windowHours))
            {
                _log.LogWarning("Dry-window: composite key '{Key}' didn't parse; skipping.", key);
                continue;
            }

            var active = ModelArtifact.ResolveStationActive(modelsRoot, "dry_window", key);
            if (active.Count == 0) continue;

            var friendly = ResolveFriendlyStationName(slug);
            if (friendly is null)
            {
                _log.LogWarning("Dry-window: unknown slug '{Slug}' — skipping composite {Key}.", slug, key);
                continue;
            }

            var byLead = new Dictionary<int, List<DryWindowTrainingRow>>();
            foreach (var lead in Leads)
            {
                _log.LogInformation("Dry-window {Key} lead {Lead}h: building rows …", key, lead);
                byLead[lead] = DryWindowFeatureBuilder.BuildForLead(
                    _cfg.Storage.ForecastsPath,
                    _cfg.Storage.RainfallPath,
                    _cfg.Location.Name,
                    friendly,
                    lead,
                    windowHours,
                    ct);
            }

            foreach (var version in active)
            {
                var versionDir = Path.Combine(modelsRoot, "dry_window", key, version);
                var metaPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metaPath))
                {
                    _log.LogWarning("Skipping dry-window {K}/{V}: no training_metadata.json", key, version);
                    continue;
                }

                var meta = ModelArtifact.LoadTrainingMetadata(versionDir);
                var output = new PerModelTestScores
                {
                    Version = version,
                    Target = "dry_window",
                    Phase = meta.Phase,
                    Station = slug,
                    WindowHours = windowHours,
                    ComputedAtUtc = DateTime.UtcNow,
                    Metric = "brier",
                };

                foreach (var (leadKey, lead) in meta.PerLead)
                {
                    if (!byLead.TryGetValue(lead.LeadHours, out var rows)) continue;
                    // Dry-window uses date-only DataRangeTest like "2025-12-15Z → 2026-04-20Z".
                    var slice = SliceByValidTime(rows, lead.DataRangeTest, r => r.TargetDateUtc).ToList();
                    if (slice.Count == 0) continue;

                    var perModel = new Dictionary<string, ModelMetric>(StringComparer.Ordinal);
                    var getters = new Func<DryWindowTrainingRow, float>[]
                    {
                        r => r.HasDryWindowGfs, r => r.HasDryWindowEcmwf, r => r.HasDryWindowIcon,
                        r => r.HasDryWindowMf,  r => r.HasDryWindowUkmo,  r => r.HasDryWindowGem,
                    };
                    for (int i = 0; i < ModelKeys.Count; i++)
                    {
                        var get = getters[i];
                        double sumSq = 0, sumSigned = 0;
                        int n = 0;
                        foreach (var r in slice)
                        {
                            var p = get(r);
                            if (float.IsNaN(p)) continue;
                            // HasDryWindow<Model> is 0.0 or 1.0 — treat as deterministic forecast prob.
                            var prob = (double)p;
                            var actual = r.HasDryWindow ? 1.0 : 0.0;
                            var err = prob - actual;
                            sumSq += err * err;
                            sumSigned += err;
                            n++;
                        }
                        perModel[ModelKeys[i]] = n == 0
                            ? new ModelMetric(double.NaN, double.NaN, double.NaN, 0)
                            : new ModelMetric(sumSq / n, Math.Sqrt(sumSq / n), sumSigned / n, n);
                    }

                    output.PerLead[leadKey] = new PerLeadScores
                    {
                        N = slice.Count,
                        DataRange = lead.DataRangeTest,
                        PerModel = perModel,
                    };
                }

                await WriteScoresAsync(versionDir, output, ct);
                written++;
            }
        }
        return written;
    }

    // ---- Shared helpers -----------------------------------------------------

    /// <summary>
    /// Accepts "2026-01-11 13:00Z → 2026-04-13 23:00Z" (temp/precip) or
    /// "2025-12-15Z → 2026-04-20Z" (dry-window). Returns inclusive bounds; dry-window
    /// date-only strings widen to full days so the slice covers all matching rows.
    /// </summary>
    private static (DateTime Start, DateTime End)? ParseDataRange(string dataRange)
    {
        if (string.IsNullOrWhiteSpace(dataRange)) return null;
        var parts = dataRange.Split(new[] { "→", "→", "->" }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2) return null;
        var startStr = parts[0].Trim();
        var endStr   = parts[1].Trim();

        var start = TryParseBound(startStr, widenToDayStart: true);
        var end   = TryParseBound(endStr,   widenToDayStart: false);
        if (start is null || end is null) return null;
        return (start.Value, end.Value);
    }

    private static DateTime? TryParseBound(string s, bool widenToDayStart)
    {
        // Strip trailing Z so DateTime.TryParseExact with 'yyyy-MM-dd HH:mm' works without TZ fuss.
        var trimmed = s.TrimEnd('Z', 'z').Trim();
        var formats = new[]
        {
            "yyyy-MM-dd HH:mm",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd",
        };
        foreach (var f in formats)
        {
            if (DateTime.TryParseExact(trimmed, f, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt))
            {
                if (f == "yyyy-MM-dd" && !widenToDayStart)
                    dt = dt.AddDays(1).AddTicks(-1); // "widen to day end"
                return DateTime.SpecifyKind(dt, DateTimeKind.Utc);
            }
        }
        return null;
    }

    private static IEnumerable<T> SliceByValidTime<T>(IReadOnlyList<T> rows, string dataRange, Func<T, DateTime> timeSelector)
    {
        var range = ParseDataRange(dataRange);
        if (range is null) return Array.Empty<T>();
        var (start, end) = range.Value;
        return rows.Where(r =>
        {
            var t = timeSelector(r);
            return t >= start && t <= end;
        });
    }

    private static (string? Slug, int WindowHours) ParseCompositeKey(string key)
    {
        var m = System.Text.RegularExpressions.Regex.Match(key, @"^(?<slug>[^/]+)/window_(?<w>\d+)h$");
        if (!m.Success) return (null, 0);
        return (m.Groups["slug"].Value, int.Parse(m.Groups["w"].Value, CultureInfo.InvariantCulture));
    }

    private string? ResolveFriendlyStationName(string stationSlug)
    {
        foreach (var s in _cfg.Location.Rainfall.Stations)
        {
            var slug = "ea_" + Slugify(s.Name);
            if (string.Equals(slug, stationSlug, StringComparison.OrdinalIgnoreCase))
                return s.Name;
        }
        return null;
    }

    private static string Slugify(string name)
    {
        var chars = name.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();
        var slug = new string(chars);
        while (slug.Contains("__")) slug = slug.Replace("__", "_");
        return slug.Trim('_');
    }

    private static async Task WriteScoresAsync(string versionDir, PerModelTestScores scores, CancellationToken ct)
    {
        Directory.CreateDirectory(versionDir);
        var path = Path.Combine(versionDir, PerModelTestFileName);
        var json = JsonSerializer.Serialize(scores, JsonOpts);
        await File.WriteAllTextAsync(path, json, ct);
    }

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    // ---- Output DTOs --------------------------------------------------------

    public sealed class PerModelTestScores
    {
        public string Version { get; set; } = "";
        public string Target { get; set; } = "";
        public string Phase { get; set; } = "";
        public string? Station { get; set; }
        public int? WindowHours { get; set; }
        public DateTime ComputedAtUtc { get; set; }

        /// <summary>"mae" (temperature) or "brier" (precipitation / dry-window).</summary>
        public string Metric { get; set; } = "";

        public Dictionary<string, PerLeadScores> PerLead { get; set; } = new();
    }

    public sealed class PerLeadScores
    {
        public int N { get; set; }
        public string DataRange { get; set; } = "";
        public Dictionary<string, ModelMetric> PerModel { get; set; } = new();
    }

    public sealed record ModelMetric(double Value, double Rmse, double Bias, int N);
}
