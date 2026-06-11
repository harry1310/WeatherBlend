using WeatherBlend.Models;

namespace WeatherBlend.Site;

/// <summary>
/// Temperature phase → chart colour, single source of truth so the
/// rolling-MAE panel, the phase-comparison chart, and any future
/// per-phase render path all draw 2b in the same purple, 2c in the same
/// teal, etc. Since 2026-06-10 the colours live in phases.yaml's per-phase
/// <c>display:</c> blocks (loaded via <see cref="PhaseRegistry"/>) — this
/// class is a thin lookup, so adding a temperature phase is a registry
/// edit, not a code edit. The palette mirrors the precipitation side
/// (3a→2b purple, 3c→2c teal, 3d→2d red) — rationale comments sit on the
/// yaml entries.
/// </summary>
public static class TempPhases
{
    private const string Target = "temperature";

    /// <summary>Lookup a phase colour by key. Falls back to neutral grey
    /// for unknown / retired / display-less phases so a stale rolling-MAE
    /// row stays visible without claiming a known-phase colour.</summary>
    public static string ColorFor(string phaseKey)
    {
        foreach (var p in PhaseRegistry.Default.AllPhases(Target))
        {
            if (string.Equals(p.Id, phaseKey, StringComparison.Ordinal) && p.Display is not null)
                return p.Display.Color;
        }
        return "#9e9e9e";
    }
}
