namespace WeatherBlend.Config;

/// <summary>
/// Granite Force-Restore knobs for the rock surface temperature + condensation
/// module (docs/ROCK_SURFACE_TEMP_PLAN.md). One-to-one with the validated P0
/// spike's <c>PARAMS</c> (scripts/rock_temp_spike.py) plus the production-only
/// <see cref="SpinupHours"/> and <see cref="GreasyMarginC"/>. Config-driven so
/// P2 on-site calibration (and the greasy threshold) is a YAML edit, not a
/// recompile. Defaults here ARE the spike defaults, so the module is correct
/// even if the <c>rockSurface:</c> block is absent. YAML key: <c>rockSurface:</c>.
///
/// The global block holds the Bonehill-calibrated values; a location can
/// override any subset (and gate itself off) via its own <c>rockSurface:</c>
/// block — see <see cref="RockSurfaceOverrideConfig"/> and
/// <see cref="ResolveFor"/> (Sennen sea-cliff extension,
/// docs/SENNEN_ROCK_TEMP_PLAN.md S2).
/// </summary>
public sealed class RockSurfaceConfig
{
    /// <summary>Whether the rock-surface module runs at all for a location.
    /// Global default true; a per-location override of <c>false</c> keeps the
    /// location dark (no predictions, so nothing renders) while its physics /
    /// calibration is still being built — distinct from the missing-input
    /// soft-skip.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Granite shortwave albedo (fraction reflected).</summary>
    public double Albedo { get; set; } = 0.30;

    /// <summary>Longwave emissivity of the rock surface.</summary>
    public double EpsRock { get; set; } = 0.95;

    /// <summary>Thermal conductivity λ (W/m/K).</summary>
    public double Lambda { get; set; } = 3.0;

    /// <summary>Density ρ (kg/m³).</summary>
    public double Rho { get; set; } = 2650.0;

    /// <summary>Specific heat c (J/kg/K).</summary>
    public double CpRock { get; set; } = 790.0;

    /// <summary>Diurnal restore period τ (s) — one solar day.</summary>
    public double TauDaySeconds { get; set; } = 86400.0;

    /// <summary>Deep-reservoir drift timescale τ_long (s). Canonical
    /// force-restore ≈ one day; a multi-day value adds longer thermal memory.</summary>
    public double TauLongSeconds { get; set; } = 86400.0;

    /// <summary>Sky-view factor (1 = fully-exposed boulder top; &lt;1 = a boulder
    /// field where neighbours block part of the cold sky). Bonehill-specific knob.</summary>
    public double FSky { get; set; } = 1.0;

    /// <summary>Clear-sky emissivity scale (Brutsaert). GFS-calibrated ≈1.0.</summary>
    public double LwClearK { get; set; } = 1.0;

    /// <summary>Cloud-enhancement LW scale. Recalibrated 2026-06-04 from 1.0→0.54
    /// against GFS DLWRF (the original full-to-1.0 enhancement over-stated cloudy-sky
    /// LW by ~22 W/m² — a warm bias that under-predicted condensation). 1.0 = raw Brunt.</summary>
    public double LwCloudK { get; set; } = 0.54;

    /// <summary>Multiplier on the canonical force-restore skin heat capacity — the
    /// main granite knob. 0.3 (~3 cm radiating skin) is the P0 default where
    /// clear-calm nights cool below air; true value is a P2 calibration target.</summary>
    public double MuScale { get; set; } = 0.3;

    /// <summary>ODE sub-steps per hour (Euler stability).</summary>
    public int Substeps { get; set; } = 6;

    /// <summary>Hours of pre-anchor forcing to integrate before the first
    /// reported valid time, so Ts and the deep reservoir settle (the initial
    /// condition is discarded). Sourced from recent NWP best-estimate.</summary>
    public int SpinupHours { get; set; } = 48;

    /// <summary>Condensation margin (°C) within which the rock is flagged
    /// "potentially greasy": margin ≤ 0 = condensation, 0 &lt; margin ≤ this =
    /// potentially greasy, above = dry. Harry 2026-06-05 — flag the marginal
    /// regime before a strict Ts ≤ Td crossing.</summary>
    public double GreasyMarginC { get; set; } = 3.0;

    /// <summary>Crag faces to model (cliff-face mode, SENNEN_ROCK_TEMP_PLAN.md
    /// S3). Empty = whole-crag horizontal mode (Bonehill's all-aspect tor):
    /// the blend shortwave drives the budget unmodified and rows carry an
    /// empty <c>Face</c>. Non-empty = one Force-Restore integration per face,
    /// with the direct beam projected onto each face plane.</summary>
    public List<RockSurfaceFaceConfig> Faces { get; set; } = new();

    /// <summary>The effective config for one location: this global block with
    /// any fields the location's own <c>rockSurface:</c> block sets layered on
    /// top. Null/absent override (Bonehill, Membury) returns this instance
    /// unchanged, so existing behaviour is untouched.</summary>
    public RockSurfaceConfig ResolveFor(RockSurfaceOverrideConfig? o)
    {
        if (o is null) return this;
        return new RockSurfaceConfig
        {
            Enabled        = o.Enabled        ?? Enabled,
            Albedo         = o.Albedo         ?? Albedo,
            EpsRock        = o.EpsRock        ?? EpsRock,
            Lambda         = o.Lambda         ?? Lambda,
            Rho            = o.Rho            ?? Rho,
            CpRock         = o.CpRock         ?? CpRock,
            TauDaySeconds  = o.TauDaySeconds  ?? TauDaySeconds,
            TauLongSeconds = o.TauLongSeconds ?? TauLongSeconds,
            FSky           = o.FSky           ?? FSky,
            LwClearK       = o.LwClearK       ?? LwClearK,
            LwCloudK       = o.LwCloudK       ?? LwCloudK,
            MuScale        = o.MuScale        ?? MuScale,
            Substeps       = o.Substeps       ?? Substeps,
            SpinupHours    = o.SpinupHours    ?? SpinupHours,
            GreasyMarginC  = o.GreasyMarginC  ?? GreasyMarginC,
            Faces          = o.Faces          ?? Faces,
        };
    }
}

/// <summary>
/// One crag face for the cliff-face rock-temp mode: a plane defined by the
/// compass bearing it looks toward and its steepness. One (aspect, slope)
/// pair per face is the same order of simplification as a single sky-view
/// factor — real crags have corners and aspect spread.
/// </summary>
public sealed class RockSurfaceFaceConfig
{
    /// <summary>Stable identifier stamped on prediction rows (e.g. "west").</summary>
    public string Name { get; set; } = "";

    /// <summary>Compass bearing the face looks toward (deg, 0=N 90=E 180=S 270=W).</summary>
    public double AspectDeg { get; set; }

    /// <summary>Steepness: 0 = horizontal slab, 90 = vertical wall (default).</summary>
    public double SlopeDeg { get; set; } = 90.0;
}

/// <summary>
/// Per-location <c>rockSurface:</c> override block (Sennen sea-cliff extension,
/// docs/SENNEN_ROCK_TEMP_PLAN.md S2). Every field is optional — only the values
/// a location actually sets shadow the global <see cref="RockSurfaceConfig"/>;
/// everything else inherits. Field meanings are identical to the global block.
/// </summary>
public sealed class RockSurfaceOverrideConfig
{
    public bool? Enabled { get; set; }
    public double? Albedo { get; set; }
    public double? EpsRock { get; set; }
    public double? Lambda { get; set; }
    public double? Rho { get; set; }
    public double? CpRock { get; set; }
    public double? TauDaySeconds { get; set; }
    public double? TauLongSeconds { get; set; }
    public double? FSky { get; set; }
    public double? LwClearK { get; set; }
    public double? LwCloudK { get; set; }
    public double? MuScale { get; set; }
    public int? Substeps { get; set; }
    public int? SpinupHours { get; set; }
    public double? GreasyMarginC { get; set; }

    /// <summary>Replaces the global faces list WHOLESALE when set (faces are
    /// location geometry — merging two locations' lists makes no sense).</summary>
    public List<RockSurfaceFaceConfig>? Faces { get; set; }
}
