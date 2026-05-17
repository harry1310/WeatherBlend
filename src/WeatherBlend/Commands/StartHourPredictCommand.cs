using System.Text.Json;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.DryWindow;

namespace WeatherBlend.Commands;

/// <summary>
/// Per-anchor "where in the daytime is the dry window?" curve, derived from
/// the SAME MC samples Phase 3g uses for the dry-window daily probability.
/// One parquet per (station, window) under
/// <c>data/predictions/dry_window_start_hour/{station}/window_{N}h/model_version=v2/date={anchor}/predictions.parquet</c>.
///
/// Was previously the analytical product π_s = ∏(1-q_h) / Σ same scaled by
/// the dry-window blender's daily marginal. Replaced 2026-05-04 with
/// <see cref="DryWindow3gPredictor.SampleStartHourCurve"/>: the same MC
/// pass that produces ProbHasDryWindow on the dry-window page also tallies
/// per-start-hour marginals, so the start-hour curve sums to the daily
/// probability shown alongside it on the site (no drift between
/// "P(any block today) = X" and "Σ start-hour probs = Y" with Y ≠ X). Under
/// independence the two methods are mathematically equivalent — this is a
/// numerical-consistency / single-source-of-truth fix, not a model change.
///
/// Pre-requisite: precip + dry-window predicts have already run for this
/// anchor — start-hour reads the live 3a champion's hourly P(wet) parquet
/// for the daytime window (same parquet 3g consumes for dry-window
/// predict). The combined predict-and-render workflow runs them in order
/// so the inputs land before this command reads them. Either upstream
/// missing → that composite skipped for this anchor, exit code stays 0
/// unless every composite is empty.
/// </summary>
public sealed class StartHourPredictCommand
{
    public const string PredictionsSubdir = "dry_window_start_hour";
    /// <summary>v1 was the analytical-product derivation (deleted 2026-05-04).
    /// v2 is the 3g-MC derivation (MC over the 3a champion's hourly P(wet)).
    /// Bumping the version on disk means old v1 rows + new v2 rows can coexist
    /// during the transition cycle. Consumed by Phase 3g's site table.</summary>
    public const string OutputModelVersion = "v2";

    /// <summary>The Phase 3s start-hour curve — same MC, but over the 3e
    /// champion's hourly P(wet) instead of 3a's. Written alongside <see
    /// cref="OutputModelVersion"/> every cycle; consumed by Phase 3s's site
    /// table. Distinct on-disk model_version so the two curves coexist.</summary>
    public const string OutputModelVersion3e = "v2-3e";
    /// <summary>MC sample count — matches DryWindow3gPredictor's default so
    /// the start-hour curve and the dry-window daily prob have identical
    /// sampling noise floors.</summary>
    public const int McSamples = DryWindow3gPredictor.DefaultMcSamples;
    /// <summary>Deterministic seed; matches the seed convention used in
    /// 3g predict + the dry-window train pipeline so results are
    /// reproducible across runs.</summary>
    public const int DefaultSeed = 42;

    private readonly ILogger<StartHourPredictCommand> _log;
    private readonly AppConfig _cfg;

    // --location resolves into _activeLocation at RunAsync entry; the
    // start-hour rows pin LocationName from it. Defaults to the primary
    // location (Phase B, commit 4).
    private Config.LocationConfig _activeLocation;

    public StartHourPredictCommand(ILogger<StartHourPredictCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
        _activeLocation = cfg.Location;
    }

    public Task<int> RunAsync(DateOnly? forDate, CancellationToken ct)
        => RunAsync(forDate, locationOverride: null, ct);

    public async Task<int> RunAsync(DateOnly? forDate, string? locationOverride, CancellationToken ct)
    {
        _activeLocation = _cfg.Location;
        if (!string.IsNullOrWhiteSpace(locationOverride))
        {
            var match = _cfg.Locations.FirstOrDefault(l =>
                l.Name.Equals(locationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Location '{Name}' not found in config.yaml's `locations:` list. Available: [{All}]",
                    locationOverride, string.Join(", ", _cfg.Locations.Select(l => l.Name)));
                return 2;
            }
            _activeLocation = match;
            _log.LogInformation("Predict location override → '{Loc}'.", _activeLocation.Name);
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var predictionMadeAt = DateTime.UtcNow;
        var anchor = PredictAnchor.Compute(predictionMadeAt, forDate);
        var anchorDate = anchor.Date;

        _log.LogInformation("Start-hour predict — anchor {Anchor:yyyy-MM-dd HH:mm}Z (for-date={ForDate})",
            anchor, forDate?.ToString("yyyy-MM-dd") ?? "live");

        var precipManifest = LoadManifest(modelsRoot, "precipitation");
        var dryManifest = LoadManifest(modelsRoot, "dry_window");
        if (precipManifest is null || dryManifest is null)
        {
            _log.LogError("Missing precipitation or dry_window manifest under {Root}.", modelsRoot);
            return 2;
        }

        var daytime = _cfg.DryWindow.BuildDaytimeWindow();

        // Two start-hour curves per cycle — same SampleStartHourCurve MC over
        // a precip champion's hourly P(wet), only the source differs:
        //   v2     ← 3a champion (manifest Current); Phase 3g's table consumes it.
        //   v2-3e  ← 3e champion (newest _phase3e in Active); Phase 3s's table
        //            consumes it. 3e's sharper hourly marginals localise the
        //            dry block better — see DryWindow3sPredictor + the
        //            2026-05-15 start-hour bake-off.
        var sources = new (string OutVersion, Func<ModelArtifact.StationEntry, string?> Resolve)[]
        {
            (OutputModelVersion,   e => string.IsNullOrEmpty(e.Current) ? null : e.Current),
            (OutputModelVersion3e, ResolveLatest3e),
        };

        var anyProduced = false;
        foreach (var (outVersion, resolve) in sources)
            anyProduced |= await ProcessSourceAsync(
                precipManifest, dryManifest, anchor, anchorDate, predictionMadeAt,
                daytime, resolve, outVersion, ct);
        return anyProduced ? 0 : 3;
    }

    /// <summary>Newest <c>_phase3e</c> version in a precip station's Active
    /// list, or null when the station has no 3e bundle. 3e is a precip
    /// challenger — never the manifest <c>Current</c> — so it's resolved the
    /// same way Phase 3s's train binds it.</summary>
    private static string? ResolveLatest3e(ModelArtifact.StationEntry e)
        => e.Active
            .Where(v => v.Contains("phase3e", StringComparison.Ordinal))
            .OrderBy(v => v, StringComparer.Ordinal)
            .LastOrDefault();

    /// <summary>
    /// Build + write the start-hour curve for one precip source. Walks the
    /// dry-window manifest composites, MCs the daytime q-vector at each lead,
    /// and emits one StartHourPredictionRow per candidate start hour.
    /// <paramref name="resolvePrecipVersion"/> picks which precip champion
    /// supplies the marginals; <paramref name="outModelVersion"/> tags the
    /// rows and their on-disk partition. Returns true if any rows were written.
    /// </summary>
    private async Task<bool> ProcessSourceAsync(
        ModelArtifact.Manifest precipManifest, ModelArtifact.Manifest dryManifest,
        DateTime anchor, DateTime anchorDate, DateTime predictionMadeAt,
        DaytimeWindow daytime,
        Func<ModelArtifact.StationEntry, string?> resolvePrecipVersion,
        string outModelVersion, CancellationToken ct)
    {
        var rng = new Random(DefaultSeed);

        // (station, windowHours) → all curve rows for that composite.
        var rowsByComposite = new Dictionary<(string Station, int Window), List<StartHourPredictionRow>>();
        var composites = 0;
        var skipped = 0;

        foreach (var (compositeKey, dryEntry) in dryManifest.Stations)
        {
            ct.ThrowIfCancellationRequested();
            composites++;

            var (station, windowHours) = ParseDryComposite(compositeKey);
            if (station is null)
            {
                _log.LogWarning("  skipping malformed dry-window composite key {Key}", compositeKey);
                skipped++; continue;
            }

            // Demoted dry-window stations (Current="") skip silently;
            // predict skipped them already.
            var dryVersion = dryEntry.Current;
            if (string.IsNullOrEmpty(dryVersion))
            {
                skipped++; continue;
            }

            if (!precipManifest.Stations.TryGetValue(station, out var precipEntry))
            {
                _log.LogWarning("  {Composite}: no precipitation manifest entry for {Station} — skip.",
                    compositeKey, station);
                skipped++; continue;
            }
            var precipVersion = resolvePrecipVersion(precipEntry);
            if (string.IsNullOrEmpty(precipVersion))
            {
                _log.LogWarning("  {Composite} [{V}]: no precip champion resolved for {Station} — skip.",
                    compositeKey, outModelVersion, station);
                skipped++; continue;
            }

            // Live precip prediction parquet for THIS anchor — the same file
            // the matching dry-window MC phase reads. Reading from the same
            // source keeps the start-hour curve consistent with that phase's
            // daily ProbHasDryWindow.
            Dictionary<DateTime, double> hourly;
            try
            {
                hourly = DryWindow3gPredictor.LoadLivePredictionsHourly(
                    _cfg.Storage.PredictionsPath, station, precipVersion, anchorDate);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex,
                    "  {Composite}: 3a prediction parquet missing for anchor {A:yyyy-MM-dd} (precip {V}); skip.",
                    compositeKey, anchorDate, precipVersion);
                skipped++; continue;
            }
            if (hourly.Count == 0)
            {
                _log.LogInformation("  {Composite}: 3a parquet empty at this anchor; skip.", compositeKey);
                skipped++; continue;
            }

            var composite = (station, windowHours);
            if (!rowsByComposite.TryGetValue(composite, out var bucket))
            {
                bucket = new List<StartHourPredictionRow>();
                rowsByComposite[composite] = bucket;
            }

            // For each lead, the target date is anchor + L days. Iterate the
            // standard 24/48/72h triple — matches what dry-window predict
            // wrote so the per-day cards line up.
            foreach (var lead in WeatherBlend.Train.Common.Leads.Short)
            {
                var targetDate = anchorDate.AddDays(lead / 24);
                var (startUtc, endUtc) = daytime.UtcHourRangeFor(DateOnly.FromDateTime(targetDate));
                var qDaytime = DryWindow3gPredictor.ExtractDaytimeQ(
                    hourly, targetDate, startUtc, endUtc);
                if (qDaytime is null)
                {
                    _log.LogWarning(
                        "  {Composite} lead {L}h ({D:yyyy-MM-dd}): 3a missing one or more daytime hours; skip.",
                        compositeKey, lead, targetDate);
                    continue;
                }

                var curve = DryWindow3gPredictor.SampleStartHourCurve(
                    qDaytime, windowHours, rng, McSamples);
                if (curve is null)
                {
                    // Window > daytime span — structurally impossible.
                    continue;
                }

                var marginal = curve.Value.MarginalByStart;
                var probAny = curve.Value.ProbAtLeast;
                var nStarts = marginal.Length;

                // π_s = marginal_s / Σ marginal — the conditional shape.
                // Falls back to uniform when Σ marginal ≈ 0 (every candidate
                // window has a near-certain wet hour) so downstream consumers
                // get a stable row count + ordering.
                double sumMarginal = 0;
                for (int i = 0; i < nStarts; i++) sumMarginal += marginal[i];
                var conditional = new double[nStarts];
                if (sumMarginal > 0)
                    for (int i = 0; i < nStarts; i++) conditional[i] = marginal[i] / sumMarginal;
                else
                    for (int i = 0; i < nStarts; i++) conditional[i] = 1.0 / nStarts;

                for (int i = 0; i < nStarts; i++)
                {
                    bucket.Add(new StartHourPredictionRow
                    {
                        LocationName = _activeLocation.Name,
                        TruthStation = station,
                        WindowHours = windowHours,
                        ModelVersion = outModelVersion,
                        PredictionMadeAtUtc = predictionMadeAt,
                        TargetDateUtc = DateTime.SpecifyKind(targetDate, DateTimeKind.Utc),
                        LeadHours = lead,
                        StartHourUtc = startUtc + i,
                        // RawProduct: per-start-hour marginal (same units as
                        // the analytical formula's p_s — kept for diagnostics
                        // / audit trail). Sum across starts ≈ E[# valid
                        // starts/day], not interpretable as a probability
                        // directly.
                        RawProduct = marginal[i],
                        // ConditionalProb: shape over candidate start hours,
                        // sums to 1. Same semantic the previous analytical
                        // method shipped.
                        ConditionalProb = conditional[i],
                        // CalibratedProb: π_s × ProbHasDryWindow (the daily
                        // prob 3g shows on the dry-window page). Sum across
                        // starts equals ProbHasDryWindow exactly — no drift.
                        CalibratedProb = conditional[i] * probAny,
                        DailyProbAnyBlock = probAny,
                        PrecipVersion = precipVersion,
                        DryWindowVersion = dryVersion,
                    });
                }
                _log.LogInformation(
                    "  {Composite} lead {L}h ({D:yyyy-MM-dd}) → {N} starts, ProbAny={P:0.000} (Σ calibrated = {Sum:0.000})",
                    compositeKey, lead, targetDate, nStarts, probAny,
                    bucket.TakeLast(nStarts).Sum(r => r.CalibratedProb));
            }
        }

        if (rowsByComposite.Count == 0)
        {
            _log.LogWarning(
                "Start-hour [{V}]: no curves produced — every composite skipped (see warnings above).",
                outModelVersion);
            return false;
        }

        var totalRows = 0;
        foreach (var ((station, windowHours), rows) in rowsByComposite)
        {
            await WriteAsync(station, windowHours, rows, anchorDate, outModelVersion, ct);
            totalRows += rows.Count;
        }

        _log.LogInformation(
            "Start-hour [{V}]: {Total} rows across {Composites} composites ({Skipped} skipped).",
            outModelVersion, totalRows, rowsByComposite.Count, skipped);
        return true;
    }

    private async Task WriteAsync(
        string station, int windowHours, List<StartHourPredictionRow> rows,
        DateTime anchorDate, string outModelVersion, CancellationToken ct)
    {
        var dateStr = anchorDate.ToString("yyyy-MM-dd");
        var outDir = Path.Combine(_cfg.Storage.PredictionsPath,
            PredictionsSubdir,
            station,
            $"window_{windowHours}h",
            $"model_version={outModelVersion}",
            $"date={dateStr}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "predictions.parquet");

        List<StartHourPredictionRow> existing = File.Exists(outPath)
            ? (await ParquetSerializer.DeserializeAsync<StartHourPredictionRow>(outPath, cancellationToken: ct)).ToList()
            : new List<StartHourPredictionRow>();

        var merged = MergeRows(existing, rows);

        await ParquetSerializer.SerializeAsync(merged, outPath, cancellationToken: ct);
        _log.LogInformation("  wrote {New} {Station}/{Window}h rows (file now holds {Total}) → {Path}",
            rows.Count, station, windowHours, merged.Count, outPath);
    }

    /// <summary>
    /// Concat existing + new curve rows and dedup on
    /// <c>(PredictionMadeAtUtc, TargetDateUtc, LeadHours, StartHourUtc)</c> —
    /// the row's natural identity. Same shape as the feels-like / precip
    /// merge pattern; ValidTime is the start hour for this surface.
    /// </summary>
    internal static List<StartHourPredictionRow> MergeRows(
        IEnumerable<StartHourPredictionRow> existing,
        IEnumerable<StartHourPredictionRow> incoming)
        => existing.Concat(incoming)
            .GroupBy(r => (r.PredictionMadeAtUtc, r.TargetDateUtc, r.LeadHours, r.StartHourUtc))
            .Select(g => g.MaxBy(r => r.PredictionMadeAtUtc)!)
            .OrderBy(r => r.TargetDateUtc)
            .ThenBy(r => r.LeadHours)
            .ThenBy(r => r.StartHourUtc)
            .ToList();

    private static ModelArtifact.Manifest? LoadManifest(string modelsRoot, string target)
    {
        var path = Path.Combine(modelsRoot, target, ModelArtifact.ManifestFileName);
        if (!File.Exists(path)) return null;
        try
        {
            return JsonSerializer.Deserialize<ModelArtifact.Manifest>(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Parse <c>"ea_bellever_dartmoor/window_6h"</c> → (station,
    /// windowHours). Returns (null, 0) on shape mismatch so the caller can
    /// log + skip rather than crash.
    /// </summary>
    internal static (string? Station, int WindowHours) ParseDryComposite(string compositeKey)
    {
        var slash = compositeKey.IndexOf('/');
        if (slash <= 0) return (null, 0);
        var station = compositeKey[..slash];
        var rest = compositeKey[(slash + 1)..];
        if (!rest.StartsWith("window_") || !rest.EndsWith("h")) return (null, 0);
        var hoursStr = rest["window_".Length..^1];
        return int.TryParse(hoursStr, out var hours) && hours > 0
            ? (station, hours)
            : (null, 0);
    }
}
