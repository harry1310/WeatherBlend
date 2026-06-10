using WeatherBlend.Models;
using WeatherBlend.Train;

namespace WeatherBlend.Predict.Coverage;

/// <summary>
/// Post-predict coverage assertion. Answers, per the active-phase registry
/// (<c>phases.yaml</c> + its <c>locations:</c> filter — NOT champion status):
/// "for every (target × location × phase) we ship here AND have a trained
/// bundle for, did we actually produce fresh predictions for this cycle's
/// anchor?" Any active+bundled cell that produced nothing → a <see cref="CoverageBreach"/>.
///
/// Motivation (2026-06-05 GEM outage): a stale upstream feed silently zeroed
/// Membury's temperature champion + every Bonehill element blender for days,
/// yet nothing went red — the predict commands returned 2/3/4 which
/// <c>predict-all</c> soft-skips (those codes ALSO mean "legitimately not
/// trained here"). There was no positive coverage assertion. This guard is
/// that assertion: a regression ("a cell that could produce, didn't") becomes
/// distinguishable from expected absence ("never shipped here"), because the
/// registry's <c>locations:</c> filter declares exactly where each phase
/// should run.
///
/// Granularity is CELL-LEVEL for v1: a breach fires only on TOTAL absence (no
/// <c>date=anchor</c> partition / zero rows). Partial-lead loss is tolerated
/// (exact-runtime 2d/3d legitimately publish a sparse-lead-but-non-empty
/// partition every cycle; MF caps ~72h; etc.) — that's a documented follow-up.
///
/// Pure + injectable: the manifest/metadata reads go through
/// <see cref="ModelArtifact"/> (cheap JSON), and the "did this cell produce
/// fresh rows?" probe is a delegate so the registry/manifest logic is
/// unit-testable without parquet I/O.
/// </summary>
public static class CoverageGuard
{
    /// <summary>How a target's predictions are laid out on disk — drives the
    /// partition path the freshness probe builds.</summary>
    public enum PredLayout
    {
        /// <summary>predictions/{target}/{stationKey}/model_version={v}/date={anchor}/ —
        /// the directory scopes the cell, so partition existence == produced.</summary>
        PerKeyDir,
        /// <summary>predictions/{target}/model_version={v}/date={anchor}/ — one parquet
        /// holds all locations' rows (the element blenders), so the probe must
        /// filter by LocationName.</summary>
        Flat,
    }

    /// <summary>Per-target station-key shape.</summary>
    private enum KeyKind { Location, EaStation, EaStationWindow }

    private sealed record Descriptor(string Target, KeyKind Key, PredLayout Layout);

    /// <summary>
    /// The targets the guard knows how to check, with their on-disk shape.
    /// Verified against the live R2 tree 2026-06-05:
    ///   - temperature / wind_direction: per-location dir.
    ///   - precipitation / rainfall_amount: per-EA-station dir.
    ///   - dry_window: per-EA-station/window composite dir (manifest keys are
    ///     "ea_x/window_Nh", which double as the path segment + metadata dir).
    ///   - wind / humidity / shortwave_radiation / cloud_cover / wind_gust:
    ///     FLAT predictions tree (LocationName in-column), per-location MODELS.
    /// A registry target with no descriptor here is surfaced as a warning
    /// (visible, not silently skipped).
    /// </summary>
    private static readonly Descriptor[] Descriptors =
    {
        new("temperature",         KeyKind.Location,        PredLayout.PerKeyDir),
        new("precipitation",       KeyKind.EaStation,       PredLayout.PerKeyDir),
        new("rainfall_amount",     KeyKind.EaStation,       PredLayout.PerKeyDir),
        new("dry_window",          KeyKind.EaStationWindow, PredLayout.PerKeyDir),
        new("wind",                KeyKind.Location,        PredLayout.Flat),
        new("humidity",            KeyKind.Location,        PredLayout.Flat),
        new("shortwave_radiation", KeyKind.Location,        PredLayout.Flat),
        new("cloud_cover",         KeyKind.Location,        PredLayout.Flat),
        new("wind_gust",           KeyKind.Location,        PredLayout.Flat),
        new("wind_direction",      KeyKind.Location,        PredLayout.PerKeyDir),
    };

    /// <summary>One location's identity for the guard: its name (the
    /// location-keyed manifest/station key) + its EA rainfall station slugs.</summary>
    public sealed record LocationSpec(string Name, IReadOnlyList<string> EaStationSlugs);

    /// <summary>A cell whose freshness the probe must answer. <see cref="Layout"/>
    /// + <see cref="StationKey"/> + <see cref="Version"/> are enough to build the
    /// partition path; <see cref="Location"/> is the LocationName filter for
    /// <see cref="PredLayout.Flat"/> trees.</summary>
    public sealed record CoverageCell(
        string Target, string StationKey, string Version, string Phase,
        string Location, PredLayout Layout);

    public sealed record CoverageBreach(
        string Target, string StationKey, string Phase, string Versions, string Reason);

    public sealed record CoverageWarning(
        string Target, string StationKey, string Phase, string Reason);

    public sealed record CoverageResult(
        IReadOnlyList<CoverageBreach> Breaches,
        IReadOnlyList<CoverageWarning> Warnings,
        int CellsChecked)
    {
        public bool Passed => Breaches.Count == 0;
    }

    /// <summary>
    /// Run the coverage check. <paramref name="producedFreshRows"/> answers
    /// "did this cell write ≥1 fresh row for the anchor?" (production: a
    /// filesystem + DuckDB probe keyed to the anchor; tests: a fake set).
    ///
    /// <paramref name="includePhase"/> filters which phases are checked
    /// (default: all). The predict / predict-and-render workflows pass an
    /// "impl == dotnet" filter, because those workflows only produce the
    /// dotnet phases (via predict-all + predict-tail). The python phases
    /// (4a / 3f / wind_mvn) are produced by SEPARATE workflows on their own
    /// cadences — predict-3f.yml even runs AFTER predict.yml — so checking
    /// them here would false-positive on the first cycle of a day before
    /// their partition exists. Guarding the python phases in their own
    /// workflows is a documented follow-up.
    /// </summary>
    public static CoverageResult Run(
        string modelsRoot,
        PhaseRegistry registry,
        IReadOnlyList<LocationSpec> locations,
        Func<CoverageCell, bool> producedFreshRows,
        Func<PhaseEntry, bool>? includePhase = null)
    {
        includePhase ??= _ => true;
        var breaches = new List<CoverageBreach>();
        var warnings = new List<CoverageWarning>();
        var cellsChecked = 0;

        var known = new HashSet<string>(Descriptors.Select(d => d.Target), StringComparer.Ordinal);
        foreach (var target in registry.ByTarget.Keys)
            if (!known.Contains(target))
                warnings.Add(new CoverageWarning(target, "", "",
                    "registry target has no coverage descriptor — NOT checked (add one to CoverageGuard.Descriptors)"));

        foreach (var loc in locations)
        {
            foreach (var desc in Descriptors)
            {
                // Expectation = active phases for THIS target that the registry's
                // locations: filter admits for THIS location. Champion/challenger
                // irrelevant — every shipped phase is expected.
                var expected = registry.AllPhases(desc.Target)
                    .Where(p => p.AppliesToLocation(loc.Name))
                    .Where(includePhase)
                    .ToList();
                if (expected.Count == 0) continue; // target not shipped at this location

                foreach (var stationKey in ResolveStationKeys(modelsRoot, desc, loc))
                {
                    var active = ModelArtifact.ResolveStationActive(modelsRoot, desc.Target, stationKey);
                    var phaseOf = new Dictionary<string, string>(StringComparer.Ordinal);
                    foreach (var v in active)
                        if (TryPhaseOf(modelsRoot, desc.Target, stationKey, v) is { } ph)
                            phaseOf[v] = ph;

                    foreach (var pe in expected)
                    {
                        // Composed-at-predict phases (retrain: none, e.g. wind_blend)
                        // have no trained bundle / manifest entry — the manifest-based
                        // check doesn't apply. Their inputs are guarded elsewhere
                        // (champion wind here; wind_speed_lgb in its own Python
                        // predict workflow since the 2026-06-10 cutover), so skip
                        // cleanly.
                        if (!pe.IsRetrained) continue;

                        var versionsOfPhase = active
                            .Where(v => phaseOf.TryGetValue(v, out var p) && p == pe.Id)
                            .ToList();

                        if (versionsOfPhase.Count == 0)
                        {
                            // Registry says ship it here, but no Active bundle exists.
                            // That's a training/rollout gap (RetrainGuard's job), not a
                            // predict regression — warn, don't fail.
                            warnings.Add(new CoverageWarning(desc.Target, stationKey, pe.Id,
                                "active in registry but no Active bundle for this cell (training/rollout gap)"));
                            continue;
                        }

                        cellsChecked++;
                        var anyProduced = versionsOfPhase.Any(v => producedFreshRows(
                            new CoverageCell(desc.Target, stationKey, v, pe.Id, loc.Name, desc.Layout)));
                        if (!anyProduced)
                            breaches.Add(new CoverageBreach(desc.Target, stationKey, pe.Id,
                                string.Join(",", versionsOfPhase),
                                "active bundle(s) present but produced NO fresh predictions for this anchor"));
                    }
                }
            }
        }

        return new CoverageResult(breaches, warnings, cellsChecked);
    }

    private static IReadOnlyList<string> ResolveStationKeys(string modelsRoot, Descriptor desc, LocationSpec loc)
        => desc.Key switch
        {
            KeyKind.Location => new[] { loc.Name },
            KeyKind.EaStation => loc.EaStationSlugs,
            // dry_window manifest keys are composite "ea_x/window_Nh"; keep those
            // whose station part is one of this location's gauges.
            KeyKind.EaStationWindow => ModelArtifact.ListStations(modelsRoot, desc.Target)
                .Where(k => loc.EaStationSlugs.Contains(StationPart(k), StringComparer.Ordinal))
                .ToList(),
            _ => Array.Empty<string>(),
        };

    private static string StationPart(string compositeKey)
    {
        var i = compositeKey.IndexOf('/');
        return i < 0 ? compositeKey : compositeKey[..i];
    }

    /// <summary>Phase of a version from its training metadata, or null when the
    /// bundle's metadata can't be read (treated as unknown → that version drops
    /// out of every phase group; a phase left with no resolvable version then
    /// warns as "no Active bundle" rather than silently passing).</summary>
    private static string? TryPhaseOf(string modelsRoot, string target, string stationKey, string version)
    {
        try
        {
            var dir = Path.Combine(modelsRoot, target, stationKey, version);
            var phase = ModelArtifact.LoadTrainingMetadata(dir).Phase;
            return string.IsNullOrWhiteSpace(phase) ? null : phase;
        }
        catch
        {
            return null;
        }
    }
}
