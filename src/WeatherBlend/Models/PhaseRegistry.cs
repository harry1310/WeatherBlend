using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace WeatherBlend.Models;

/// <summary>
/// Loads the canonical phase registry (config/phases.yaml) and exposes
/// the queries the rest of the codebase needs. <see cref="ActivePhasePolicy"/>
/// is a thin static wrapper over <see cref="Default"/> for back-compat
/// with code written before the YAML extraction.
///
/// Adding a new phase: start at phases.yaml, then follow the honest
/// touch-list in its header comment (retrain workflow step, sync script
/// case, predict handled-set; presentation now lives IN the yaml as the
/// per-phase `display:` block) — PhaseWiringConsistencyTests enforces the
/// wiring against this registry.
///
/// Default loader looks at <c>{AppContext.BaseDirectory}/config/phases.yaml</c>
/// (the file is copied to build output via the csproj). Tests should call
/// <see cref="LoadFromYaml"/> with their own content rather than touching
/// <see cref="Default"/> so they don't mutate the singleton.
/// </summary>
public sealed class PhaseRegistry
{
    private static readonly Lazy<PhaseRegistry> _default = new(LoadDefault);

    /// <summary>Process-wide singleton, loaded once from disk.</summary>
    public static PhaseRegistry Default => _default.Value;

    private readonly Dictionary<string, IReadOnlyList<PhaseEntry>> _byTarget;

    private PhaseRegistry(Dictionary<string, IReadOnlyList<PhaseEntry>> byTarget)
    {
        _byTarget = byTarget;
    }

    /// <summary>
    /// Champion-first ID lists per target. Drop-in replacement for the old
    /// hardcoded <c>ActivePhasePolicy.ByTarget</c> dict — anything filtering
    /// "what gets a Models card" or "what's a real prediction line" reads
    /// from here.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> ByTarget
    {
        get
        {
            var dict = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
            foreach (var (target, phases) in _byTarget)
            {
                dict[target] = phases.Select(p => p.Id).ToArray();
            }
            return dict;
        }
    }

    /// <summary>
    /// Champion-first phase-ID list for a (target, location) pair —
    /// <see cref="ByTarget"/> filtered to phases whose phases.yaml
    /// <c>locations:</c> filter admits <paramref name="location"/> (a
    /// phase with no filter applies everywhere). Phase B (commit 7):
    /// lets the per-location retrain fanout skip a phase that has no
    /// archive for a given location (e.g. 2d / 3d for membury_devon).
    /// </summary>
    public IReadOnlyList<string> ByTargetAndLocation(string target, string location)
    {
        if (!_byTarget.TryGetValue(target, out var phases)) return Array.Empty<string>();
        return phases
            .Where(p => p.AppliesToLocation(location))
            .Select(p => p.Id)
            .ToArray();
    }

    /// <summary>
    /// Targets (champion-first) that <paramref name="location"/> can render —
    /// any target whose phases.yaml lineup has at least one phase admitting
    /// <paramref name="location"/>. Used by site rendering to decide which
    /// sections to draw per location and by predict-all to derive its
    /// (location, target) matrix; once 3b lands, verify uses it too. Order
    /// is stable (yaml-target order). Returns empty list if the location has
    /// no applicable phase at all (would happen if every target's lineup is
    /// locations-filtered away from this loc — useful as a "should we even
    /// render a page for this loc?" gate).
    /// </summary>
    public IReadOnlyList<string> TargetsForLocation(string location)
    {
        var result = new List<string>(_byTarget.Count);
        foreach (var (target, phases) in _byTarget)
        {
            foreach (var p in phases)
            {
                if (!p.AppliesToLocation(location)) continue;
                result.Add(target);
                break;
            }
        }
        return result;
    }

    /// <summary>
    /// Returns true iff the (target, phase) pair is in the shipping
    /// lineup AS A PREDICTION LINE — i.e. champion or challenger.
    /// Empty / null phase strings are never active.
    /// Mirrors the pre-YAML <c>ActivePhasePolicy.IsActive</c> contract.
    /// </summary>
    public bool IsActive(string target, string? phase)
    {
        if (string.IsNullOrEmpty(phase)) return false;
        if (!_byTarget.TryGetValue(target, out var phases)) return false;
        return phases.Any(p => string.Equals(p.Id, phase, StringComparison.Ordinal));
    }

    /// <summary>
    /// Index of <paramref name="phase"/> in <paramref name="target"/>'s
    /// champion-first list. Lower = champion. Returns
    /// <see cref="int.MaxValue"/> for unknown / inactive phases so callers
    /// can sort with the rest of their data without special-casing nulls.
    /// </summary>
    public int Priority(string target, string phase)
    {
        if (!_byTarget.TryGetValue(target, out var phases)) return int.MaxValue;
        int idx = 0;
        foreach (var p in phases)
        {
            if (string.Equals(p.Id, phase, StringComparison.Ordinal)) return idx;
            idx++;
        }
        return int.MaxValue;
    }

    /// <summary>
    /// The champion phase ID for <paramref name="target"/> — the sole
    /// role=champion entry (<see cref="LoadFromYaml"/> asserts exactly one
    /// per target). This is the canonical "headline model": the site, the
    /// per-station champion-version resolver
    /// (<see cref="WeatherBlend.Train.ModelArtifact.ResolveStationChampionVersion"/>),
    /// and the dry-window MC source binding all derive from it — there is no
    /// mutable per-station champion pointer any more. Throws for an unknown
    /// target or a registry with no champion for it.
    /// </summary>
    public string ChampionPhase(string target)
    {
        if (!_byTarget.TryGetValue(target, out var phases))
            throw new ArgumentException(
                $"Unknown target '{target}' in the phase registry.", nameof(target));
        foreach (var p in phases)
            if (p.Role == PhaseRole.Champion) return p.Id;
        throw new InvalidOperationException(
            $"Target '{target}' has no champion phase in the registry.");
    }

    /// <summary>
    /// Full per-target phase list. Order is YAML order = champion-first.
    /// </summary>
    public IReadOnlyList<PhaseEntry> AllPhases(string target)
        => _byTarget.TryGetValue(target, out var phases)
            ? phases
            : Array.Empty<PhaseEntry>();

    /// <summary>Flattened (target, phase) pairs across all targets.</summary>
    public IEnumerable<(string Target, PhaseEntry Phase)> EnumerateAll()
    {
        foreach (var (target, phases) in _byTarget)
        foreach (var p in phases)
            yield return (target, p);
    }

    /// <summary>
    /// All phases (across all targets) with the given impl. Drives the
    /// retrain workflows: retrain-python filters to <see cref="PhaseImpl.Python"/>,
    /// retrain-blenders filters to <see cref="PhaseImpl.Dotnet"/>.
    /// </summary>
    public IEnumerable<(string Target, PhaseEntry Phase)> PhasesForImpl(PhaseImpl impl)
        => EnumerateAll().Where(t => t.Phase.Impl == impl);

    // ---- Loading -------------------------------------------------------

    /// <summary>
    /// Parse a YAML registry document and validate. Throws
    /// <see cref="InvalidOperationException"/> with a specific reason if
    /// the schema is broken — silent fallbacks here would re-introduce
    /// the "hardcoded list silently dropped a phase" failure mode this
    /// extraction is supposed to fix.
    /// </summary>
    public static PhaseRegistry LoadFromYaml(string yaml)
    {
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        var doc = deserializer.Deserialize<PhaseRegistryDocument>(yaml)
            ?? throw new InvalidOperationException("phases.yaml deserialised to null — file empty?");

        if (doc.Targets is null || doc.Targets.Count == 0)
            throw new InvalidOperationException("phases.yaml has no `targets` block, or it's empty.");

        var byTarget = new Dictionary<string, IReadOnlyList<PhaseEntry>>(StringComparer.Ordinal);
        foreach (var (target, def) in doc.Targets)
        {
            if (def?.Phases is null || def.Phases.Count == 0)
                throw new InvalidOperationException($"Target '{target}' has no phases.");

            var entries = new List<PhaseEntry>(def.Phases.Count);
            foreach (var raw in def.Phases)
            {
                if (string.IsNullOrWhiteSpace(raw.Id))
                    throw new InvalidOperationException($"Target '{target}' has a phase with no id.");
                if (string.IsNullOrWhiteSpace(raw.Role))
                    throw new InvalidOperationException($"Target '{target}' phase '{raw.Id}' has no role.");
                if (string.IsNullOrWhiteSpace(raw.Impl))
                    throw new InvalidOperationException($"Target '{target}' phase '{raw.Id}' has no impl.");

                var role = ParseRole(raw.Role, target, raw.Id);
                var impl = ParseImpl(raw.Impl, target, raw.Id);
                IReadOnlyList<string> locations = raw.Locations is null
                    ? Array.Empty<string>()
                    : raw.Locations
                        .Where(l => !string.IsNullOrWhiteSpace(l))
                        .Select(l => l.Trim())
                        .ToArray();
                var minValidTime = ParseMinValidTime(raw.MinValidTime, target, raw.Id);
                var retrain = NormalizeRetrain(raw.Retrain, target, raw.Id);
                var display = ParseDisplay(raw.Display, target, raw.Id);
                entries.Add(new PhaseEntry(raw.Id, role, impl, locations, minValidTime, retrain, display));
            }

            // Sanity: at most one champion per target. Multiple champions
            // would make Priority() ambiguous; enforce up front.
            var championCount = entries.Count(e => e.Role == PhaseRole.Champion);
            if (championCount > 1)
                throw new InvalidOperationException(
                    $"Target '{target}' has {championCount} champions; expected at most 1.");

            byTarget[target] = entries;
        }

        return new PhaseRegistry(byTarget);
    }

    /// <summary>
    /// Loads the singleton from <c>{AppContext.BaseDirectory}/Config/phases.yaml</c>.
    /// The csproj copies the file to output (PreserveNewest), so this just
    /// reads from disk relative to the running binary. Case matches the
    /// existing Config/ folder which already holds AppConfig / BlendersConfig
    /// — Linux CI is case-sensitive.
    /// </summary>
    private static PhaseRegistry LoadDefault()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Config", "phases.yaml");
        if (!File.Exists(path))
            throw new InvalidOperationException(
                $"phases.yaml not found at {path}. Check csproj copy-to-output for src/WeatherBlend/Config/phases.yaml.");
        return LoadFromYaml(File.ReadAllText(path));
    }

    private static PhaseRole ParseRole(string raw, string target, string phaseId)
        => raw.ToLowerInvariant() switch
        {
            "champion" => PhaseRole.Champion,
            "challenger" => PhaseRole.Challenger,
            _ => throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has unknown role '{raw}'. Expected: champion | challenger."),
        };

    private static PhaseImpl ParseImpl(string raw, string target, string phaseId)
        => raw.ToLowerInvariant() switch
        {
            "dotnet" => PhaseImpl.Dotnet,
            "python" => PhaseImpl.Python,
            _ => throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has unknown impl '{raw}'. Expected: dotnet | python."),
        };

    /// <summary>
    /// Parse the optional <c>min_valid_time</c> field on a phase entry.
    /// Absent / empty → null (no cutoff). String must be an ISO-8601 date
    /// or date-time the .NET DateTime parser accepts in invariant culture;
    /// returned as UTC. Throws with a useful message on parse failure
    /// rather than silently treating it as "no cutoff" — silent fallbacks
    /// here are the regression mode (drift creeps into the training slice
    /// without anyone noticing).
    ///
    /// Why this exists: the 2026-05-26 JMA-extension surfaced that several
    /// NWP archives (GFS / ICON / ECMWF / MF / GEM / AIFS) only have
    /// valid data from 2024 onward — pre-2024 rows are NULL-padded and
    /// dilute the training signal of 3a / 3c / 3b / 2b / 2c / 4a.
    /// 3o is excluded from the cutoff because its terrain features and
    /// LoadAuxNwpMeans non-precip-aux pull (Wind / Temp / DewPoint /
    /// SurfacePressure, which DO populate pre-2024) keep it informative
    /// across the JMA-extended range.
    /// </summary>
    private static DateTime? ParseMinValidTime(string? raw, string target, string phaseId)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        if (!DateTime.TryParse(
                raw,
                System.Globalization.CultureInfo.InvariantCulture,
                System.Globalization.DateTimeStyles.AssumeUniversal | System.Globalization.DateTimeStyles.AdjustToUniversal,
                out var dt))
        {
            throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has invalid min_valid_time '{raw}'. " +
                $"Expected an ISO-8601 date (e.g. '2024-01-01') or full timestamp.");
        }
        return DateTime.SpecifyKind(dt, DateTimeKind.Utc);
    }

    /// <summary>
    /// Parse the optional <c>display:</c> block (2026-06-10 — replaced the
    /// three hardcoded Site presentation tables, so a new phase's card/colour
    /// is one more field on its registry entry instead of touch-point 5 on
    /// the old list). When the block is present, <c>color</c> is mandatory —
    /// it's the one field every consumer reads; the rest default sensibly in
    /// the Site lookups. Absent block → null; the PhaseWiringConsistencyTests
    /// assert every SHIPPED phase carries one (the loader stays lenient so
    /// minimal test registries don't have to).
    ///
    /// Additive-only by design: WeatherProbabilistic's retrain-python.yml
    /// reads this file via yq selecting id/impl/locations/retrain — extra
    /// per-phase keys are invisible to it. Keep it that way (never rename
    /// the existing fields).
    /// </summary>
    private static PhaseDisplay? ParseDisplay(RawPhaseDisplay? raw, string target, string phaseId)
    {
        if (raw is null) return null;
        if (string.IsNullOrWhiteSpace(raw.Color))
            throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has a display block without a color. " +
                "color is the one display field every render path consumes — set it (e.g. \"#7c4dff\").");
        return new PhaseDisplay(
            Color: raw.Color.Trim(),
            Title: NullIfBlank(raw.Title),
            ShortTitle: NullIfBlank(raw.ShortTitle),
            Description: NullIfBlank(raw.Description),
            Label: NullIfBlank(raw.Label),
            StartHourCurveVersion: NullIfBlank(raw.StartHourCurveVersion));

        static string? NullIfBlank(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();
    }

    /// <summary>Normalise the optional <c>retrain:</c> field to lowercase and
    /// validate it. Absent → null (= trained by the phase's impl engine).
    /// Allowed explicit values: <c>blenders</c>, <c>python</c>, <c>none</c>.</summary>
    private static string? NormalizeRetrain(string? raw, string target, string phaseId)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        var v = raw.Trim().ToLowerInvariant();
        if (v is not ("blenders" or "python" or "none"))
            throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has invalid retrain '{raw}'. " +
                "Expected one of: blenders, python, none (or omit the field).");
        return v;
    }

    // ---- YAML POCOs (private; the public surface is PhaseEntry) -------

    private sealed class PhaseRegistryDocument
    {
        public Dictionary<string, TargetDefinition>? Targets { get; set; }
    }

    private sealed class TargetDefinition
    {
        public List<RawPhaseEntry>? Phases { get; set; }
    }

    private sealed class RawPhaseEntry
    {
        public string? Id { get; set; }
        public string? Role { get; set; }
        public string? Impl { get; set; }

        /// <summary>Optional per-phase location filter (Phase B, commit 7).
        /// Absent / empty = all configured locations.</summary>
        public List<string>? Locations { get; set; }

        /// <summary>Optional per-phase training-data cutoff (2026-05-26).
        /// ISO-8601 date or timestamp. Absent / empty = no cutoff.
        /// YAML field name follows CamelCaseNamingConvention →
        /// <c>minValidTime</c>. See <see cref="ParseMinValidTime"/> for why.
        /// </summary>
        public string? MinValidTime { get; set; }

        /// <summary>Optional retrain semantics (2026-05-31). Absent = trained
        /// by this phase's <see cref="Impl"/> engine (dotnet→retrain-blenders,
        /// python→retrain-python). <c>none</c> = composed at predict time, no
        /// retrain step (e.g. wind_blend's live sigmoid blend) — excluded from
        /// the train fanouts. Stops a predict-only phase being treated as
        /// trainable, which broke the 2026-05-31 retrain.</summary>
        public string? Retrain { get; set; }

        /// <summary>Optional presentation block (2026-06-10). YAML keys follow
        /// CamelCaseNamingConvention: <c>display: { title, shortTitle,
        /// description, label, color, startHourCurveVersion }</c>. See
        /// <see cref="ParseDisplay"/>.</summary>
        public RawPhaseDisplay? Display { get; set; }
    }

    private sealed class RawPhaseDisplay
    {
        public string? Color { get; set; }
        public string? Title { get; set; }
        public string? ShortTitle { get; set; }
        public string? Description { get; set; }
        public string? Label { get; set; }
        public string? StartHourCurveVersion { get; set; }
    }
}

/// <summary>
/// One row of <c>phases.yaml</c> after parsing. Public so test code and
/// future iterators (e.g. an <c>export-phases</c> CLI) can pattern-match
/// on the role/impl enums without re-reading the YAML.
/// </summary>
public sealed record PhaseEntry(
    string Id, PhaseRole Role, PhaseImpl Impl, IReadOnlyList<string> Locations,
    DateTime? MinValidTime = null, string? Retrain = null, PhaseDisplay? Display = null)
{
    /// <summary>
    /// True if this phase is produced by a retrain step (train or mint) and
    /// so must be wired into the retrain fanout + <c>sync_train_data.sh</c>.
    /// False only when <c>retrain: none</c> — a predict-time-only synthesised
    /// phase (e.g. wind_blend). The phases-consistency test asserts every
    /// retrained dotnet phase is fully wired; that invariant is what
    /// <see cref="Retrain"/> exists to make checkable.
    /// </summary>
    public bool IsRetrained
        => !string.Equals(Retrain, "none", StringComparison.OrdinalIgnoreCase);


    /// <summary>
    /// True if this phase should be trained / shipped for
    /// <paramref name="location"/>. An empty <see cref="Locations"/> list
    /// means "all configured locations" — the default for a phase whose
    /// phases.yaml entry has no <c>locations:</c> key. A non-empty list
    /// scopes the phase to exactly those locations (e.g. 2d / 3d are
    /// bonehill_rocks-only until a Membury exact-runtime S3 archive lands).
    /// Case-insensitive.
    /// </summary>
    public bool AppliesToLocation(string location)
        => Locations.Count == 0
           || Locations.Any(l => string.Equals(l, location, StringComparison.OrdinalIgnoreCase));
}

/// <summary>
/// Per-phase presentation metadata from phases.yaml's <c>display:</c> block —
/// what the three hardcoded Site presentation tables (PrecipPhases /
/// TempPhases / DryWindowPhases) used to carry. Those classes are now thin
/// lookups over this record, so adding a phase needs no Site-side edit.
/// Only <see cref="Color"/> is mandatory (every render path consumes it);
/// the title/label fields are used by targets that render per-phase cards
/// (precipitation, dry_window) and default to a "Phase {id}" gloss when
/// absent. <see cref="StartHourCurveVersion"/> is dry-window-MC-only: the
/// on-disk model_version of the phase's start-hour curve.
/// </summary>
public sealed record PhaseDisplay(
    string Color,
    string? Title = null,
    string? ShortTitle = null,
    string? Description = null,
    string? Label = null,
    string? StartHourCurveVersion = null);

/// <summary>
/// Phase rendering / lifecycle role per phases.yaml.
/// </summary>
public enum PhaseRole
{
    Champion,
    Challenger,
}

/// <summary>
/// Which trainer owns the phase. Drives which retrain workflow picks
/// it up: <see cref="Python"/> → retrain-python.yml,
/// <see cref="Dotnet"/> → retrain-blenders.yml (slice 4 of
/// AUTO_RETRAIN_PLAN.md).
/// </summary>
public enum PhaseImpl
{
    Dotnet,
    Python,
}
