namespace WeatherBlend.Config;

/// <summary>
/// Granite Force-Restore knobs for the rock surface temperature + condensation
/// module (docs/ROCK_SURFACE_TEMP_PLAN.md). One-to-one with the validated P0
/// spike's <c>PARAMS</c> (scripts/rock_temp_spike.py) plus the production-only
/// <see cref="SpinupHours"/> and <see cref="GreasyMarginC"/>. Config-driven so
/// P2 on-site calibration (and the greasy threshold) is a YAML edit, not a
/// recompile. Defaults here ARE the spike defaults, so the module is correct
/// even if the <c>rockSurface:</c> block is absent. YAML key: <c>rockSurface:</c>.
/// </summary>
public sealed class RockSurfaceConfig
{
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
}
