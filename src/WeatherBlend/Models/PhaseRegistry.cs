using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace WeatherBlend.Models;

/// <summary>
/// Loads the canonical phase registry (config/phases.yaml) and exposes
/// the queries the rest of the codebase needs. <see cref="ActivePhasePolicy"/>
/// is a thin static wrapper over <see cref="Default"/> for back-compat
/// with code written before the YAML extraction.
///
/// Adding a new phase: edit phases.yaml. The Models page, the retrain
/// workflows, and predict/verify all see the change without code edits.
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
    /// Champion-first ID lists per target, EXCLUDING role=confidence
    /// phases. Drop-in replacement for the old hardcoded
    /// <c>ActivePhasePolicy.ByTarget</c> dict — anything filtering "what
    /// gets a Models card" or "what's a real prediction line" reads from
    /// here. Confidence-role phases (e.g. 5a) are reachable via
    /// <see cref="AllPhases"/> instead.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> ByTarget
    {
        get
        {
            var dict = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
            foreach (var (target, phases) in _byTarget)
            {
                dict[target] = phases
                    .Where(p => p.Role != PhaseRole.Confidence)
                    .Select(p => p.Id)
                    .ToArray();
            }
            return dict;
        }
    }

    /// <summary>
    /// Champion-first phase-ID list for a (target, location) pair —
    /// <see cref="ByTarget"/> filtered to phases whose phases.yaml
    /// <c>locations:</c> filter admits <paramref name="location"/> (a
    /// phase with no filter applies everywhere). Confidence-role phases
    /// are excluded, same as <see cref="ByTarget"/>. Phase B (commit 7):
    /// lets the per-location retrain fanout skip a phase that has no
    /// archive for a given location (e.g. 2d / 3d for membury_devon).
    /// </summary>
    public IReadOnlyList<string> ByTargetAndLocation(string target, string location)
    {
        if (!_byTarget.TryGetValue(target, out var phases)) return Array.Empty<string>();
        return phases
            .Where(p => p.Role != PhaseRole.Confidence && p.AppliesToLocation(location))
            .Select(p => p.Id)
            .ToArray();
    }

    /// <summary>
    /// Targets (champion-first) that <paramref name="location"/> can render —
    /// any target whose phases.yaml lineup has at least one champion-or-
    /// challenger phase admitting <paramref name="location"/>. Used by
    /// site rendering to decide which sections to draw per location and by
    /// predict-all to derive its (location, target) matrix; once 3b lands,
    /// verify uses it too. Confidence-role phases (e.g. 5a) are excluded
    /// because they don't render as a prediction line — same rule as
    /// <see cref="ByTarget"/> / <see cref="ByTargetAndLocation"/>. Order is
    /// stable (yaml-target order). Returns empty list if the location has
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
                if (p.Role == PhaseRole.Confidence) continue;
                if (!p.AppliesToLocation(location)) continue;
                result.Add(target);
                break;
            }
        }
        return result;
    }

    /// <summary>
    /// Returns true iff the (target, phase) pair is in the shipping
    /// lineup AS A PREDICTION LINE — i.e. champion or challenger, NOT
    /// confidence-role. Empty / null phase strings are never active.
    /// Mirrors the pre-YAML <c>ActivePhasePolicy.IsActive</c> contract.
    /// </summary>
    public bool IsActive(string target, string? phase)
    {
        if (string.IsNullOrEmpty(phase)) return false;
        if (!_byTarget.TryGetValue(target, out var phases)) return false;
        return phases.Any(p =>
            p.Role != PhaseRole.Confidence &&
            string.Equals(p.Id, phase, StringComparison.Ordinal));
    }

    /// <summary>
    /// Index of <paramref name="phase"/> in <paramref name="target"/>'s
    /// champion-first list (confidence-role excluded). Lower = champion.
    /// Returns <see cref="int.MaxValue"/> for unknown / inactive /
    /// confidence-role phases so callers can sort with the rest of their
    /// data without special-casing nulls.
    /// </summary>
    public int Priority(string target, string phase)
    {
        if (!_byTarget.TryGetValue(target, out var phases)) return int.MaxValue;
        int idx = 0;
        foreach (var p in phases)
        {
            if (p.Role == PhaseRole.Confidence) continue;
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
    /// Full per-target phase list INCLUDING confidence-role entries.
    /// Train workflows use this so role=confidence phases (5a) are still
    /// retrained on the Sunday sweep even though they don't render as
    /// prediction lines. Order is YAML order = champion-first.
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
                entries.Add(new PhaseEntry(raw.Id, role, impl, locations));
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
            "confidence" => PhaseRole.Confidence,
            _ => throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has unknown role '{raw}'. Expected: champion | challenger | confidence."),
        };

    private static PhaseImpl ParseImpl(string raw, string target, string phaseId)
        => raw.ToLowerInvariant() switch
        {
            "dotnet" => PhaseImpl.Dotnet,
            "python" => PhaseImpl.Python,
            _ => throw new InvalidOperationException(
                $"Target '{target}' phase '{phaseId}' has unknown impl '{raw}'. Expected: dotnet | python."),
        };

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
    }
}

/// <summary>
/// One row of <c>phases.yaml</c> after parsing. Public so test code and
/// future iterators (e.g. an <c>export-phases</c> CLI) can pattern-match
/// on the role/impl enums without re-reading the YAML.
/// </summary>
public sealed record PhaseEntry(
    string Id, PhaseRole Role, PhaseImpl Impl, IReadOnlyList<string> Locations)
{
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
/// Phase rendering / lifecycle role per phases.yaml.
/// <see cref="Confidence"/> phases are excluded from the Models page and
/// from <see cref="ActivePhasePolicy.IsActive"/> — they overlay another
/// phase's line (5a → 4a) rather than rendering as their own line.
/// </summary>
public enum PhaseRole
{
    Champion,
    Challenger,
    Confidence,
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
