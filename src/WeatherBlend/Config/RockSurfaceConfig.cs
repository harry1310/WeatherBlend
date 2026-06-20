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

    /// <summary>Representative fraction of the DIRECT solar beam the modelled
    /// surface actually catches, in whole-crag horizontal mode (no faces). The
    /// effective shortwave becomes diffuse + DirectBeamFactor·direct, with the
    /// direct/diffuse split taken from the NWP radiation fields. 1.0 (default) =
    /// today's full-horizontal behaviour; &lt;1 damps the clear-sky daytime peak
    /// to represent boulder self-shading and the steep climbing faces that rarely
    /// catch the full noon beam — WITHOUT assuming any compass aspect. A P2
    /// IR-gun calibration target (Bonehill). Ignored in cliff-face mode, where the
    /// per-face projection already handles the beam geometry.</summary>
    public double DirectBeamFactor { get; set; } = 1.0;

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

    // ---- Surface water / drying model (Phase A, 2026-06-16) ----
    /// <summary>Master switch for the surface-water / drying model. When false
    /// (default) the rock model still WRITES the wetness columns for inspection,
    /// but the climbing index IGNORES them — the existing greasy/condensation
    /// friction behaviour stands. Flip true (per location, via the override)
    /// once the evaporation knobs below are calibrated: then genuinely-wet RAIN-
    /// wetted rock gates the verdict Off and the verdict climbs back up the tiers
    /// as the film dries (Harry 2026-06-20 — a graded ladder, not the old Off→Prime
    /// snap). Overridable so it can be enabled location-by-location.</summary>
    public bool SurfaceWaterEnabled { get; set; } = false;

    /// <summary>Max surface-water film the near-impermeable granite holds (mm);
    /// water above this runs off. Caps the worst-case dry-out time. Literature
    /// anchor (2026-06-19): single-event interception/adhering water on a smooth
    /// surface is ~0.5 mm, reduced by gravity runoff on a steep climbing face →
    /// 0.4 mm. (Flat impervious depression storage is higher, ~1.5 mm.)</summary>
    public double MaxSurfaceWaterMm { get; set; } = 0.4;

    /// <summary>Film (mm) at/above which the surface counts as "wet" (heavy
    /// friction / the rain gate). Between this and ~0 is "damp". This is the one
    /// genuinely empirical knob — a climbing "too wet" judgement, not a physics
    /// value; 0.05 mm is a barely-visible damp sheen.</summary>
    public double WetThresholdMm { get; set; } = 0.05;

    /// <summary>VPD coefficient for the moisture flux — mm/hr per hPa of vapour-
    /// pressure deficit (e_sat(Ts) − e_air) per unit of the wind function. Drives
    /// BOTH evaporation (VPD &gt; 0) and dew deposition (VPD &lt; 0). Literature-
    /// grounded (2026-06-19) via the bulk-aerodynamic Dalton law E = ρ_a·C_E·U·Δq,
    /// Δq ≈ 0.622·VPD/P: with the water-vapour transfer coefficient C_E ≈ 1.3×10⁻³
    /// (eddy-covariance over water), ρ_a 1.2, P 1013 hPa → wind-driven term
    /// ≈ 0.0035 mm/hr per (hPa·m/s). Our wind term is EvapVpdCoeff·EvapWindSlope·V,
    /// so 0.010·0.35 = 0.0035 reproduces it. Field readings now VALIDATE the
    /// drying timescale rather than set this.</summary>
    public double EvapVpdCoeff { get; set; } = 0.010;

    /// <summary>Wind function for the moisture flux: still-air floor
    /// <see cref="EvapWindBase"/> + per-m/s ventilation gain <see cref="EvapWindSlope"/>·V.
    /// The classic Penman a + b·V form; the floor is the free-convection reference
    /// and the 0.35/m·s gain sits below open-water's ~0.5 (rough rock, partial wetting).</summary>
    public double EvapWindBase { get; set; } = 1.0;
    public double EvapWindSlope { get; set; } = 0.35;

    /// <summary>Radiation-driven evaporation — mm/hr per W/m² of ABSORBED short-
    /// wave, applied only while drying (sun bakes a wet slab dry). Anchored to the
    /// latent-heat constant (2026-06-19): E_rad = f·SW_abs / 680.6, where 680.6 is
    /// <see cref="LatentHeatWm2PerMmHr"/> and f is the fraction of absorbed sun that
    /// evaporates the film rather than conducting into the slab. f ≈ 0.4 (most
    /// absorbed SW heats the granite) → 0.4/680.6 = 0.0006.</summary>
    public double EvapRadCoeff { get; set; } = 0.0006;

    /// <summary>Latent-heat feedback (Phase B): W/m² of surface cooling per mm/hr
    /// of water evaporated (and warming per mm/hr condensed) — the slab loses
    /// the latent heat of vaporisation as its film dries, and gains it when dew
    /// forms. Default ≈ L_v / 3600 = 2.45×10⁶ J/kg ÷ 3600 s = 680.6 W/m² per
    /// mm/hr (1 mm/hr over 1 m² = 1 kg/m²/hr). Applied to the energy budget ONLY
    /// when <see cref="SurfaceWaterEnabled"/> is true, so a gate-off location's
    /// Ts is bit-for-bit unchanged from the pre-drying model. Set 0 to disable
    /// the feedback while still tracking the film.</summary>
    public double LatentHeatWm2PerMmHr { get; set; } = 680.6;

    // ---- Gravity drainage (vertical-face runoff, 2026-06-20) ----
    /// <summary>Retained surface-water film (mm) a sloped face holds AFTER gravity
    /// has drained the bulk — the damp residual gravity can't shed; below it the film
    /// only evaporates. A vertical wall sheds to this in minutes (Jeffreys/Nusselt
    /// drainage, <see cref="RockSurfacePhysics.Integrate"/>), then evaporates the
    /// residual — that evaporation IS the "drying, not yet climbable" period (you
    /// can't climb 10 min after a downpour). Set to 0.05 mm (2026-06-20): the literature
    /// post-drainage residual is microns (a few μm draining in seconds–minutes), but
    /// rough natural rock + the vertical-plate simplification hold more, and a damp
    /// residual that still needs to dry is the realistic climbing behaviour. The
    /// climbing index reads it as wet down to ~0.01 mm
    /// (<see cref="ClimbingConditions.RainWetThresholdMm"/>) — evaporating 0.05→0.01
    /// is ~20–60 min at 15 °C / 10 mph in normal humidity. Knob to validate against
    /// "stayed wet ~N h" field notes. (Was 0.1 mm, which sat above the old 0.05 wet
    /// line and doubled the dry-out — the bug behind the 5-hour traces.)</summary>
    public double RetainedFilmMm { get; set; } = 0.05;

    /// <summary>Characteristic vertical drainage length (m) for the gravity-runoff
    /// term — the height of the wet streak the film drains down. The Nusselt drainage
    /// rate ∝ 1/L, but the term is fast (minutes) across any plausible L, so this is
    /// not sensitive. Drainage scales with sin(face slope): a vertical wall sheds
    /// fast, a horizontal slab (slope 0) doesn't drain at all — it ponds (correct).</summary>
    public double DrainagePathLengthM { get; set; } = 2.5;

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
            DirectBeamFactor = o.DirectBeamFactor ?? DirectBeamFactor,
            LwClearK       = o.LwClearK       ?? LwClearK,
            LwCloudK       = o.LwCloudK       ?? LwCloudK,
            MuScale        = o.MuScale        ?? MuScale,
            Substeps       = o.Substeps       ?? Substeps,
            SpinupHours    = o.SpinupHours    ?? SpinupHours,
            GreasyMarginC  = o.GreasyMarginC  ?? GreasyMarginC,
            // Surface-water model: the enable switch is per-location overridable;
            // the physics coefficients are global (copied through so a global
            // config.yaml change still reaches a location that has an override).
            SurfaceWaterEnabled = o.SurfaceWaterEnabled ?? SurfaceWaterEnabled,
            MaxSurfaceWaterMm = MaxSurfaceWaterMm,
            WetThresholdMm = WetThresholdMm,
            EvapVpdCoeff   = EvapVpdCoeff,
            EvapWindBase   = EvapWindBase,
            EvapWindSlope  = EvapWindSlope,
            EvapRadCoeff   = EvapRadCoeff,
            LatentHeatWm2PerMmHr = LatentHeatWm2PerMmHr,
            RetainedFilmMm = RetainedFilmMm,
            DrainagePathLengthM = DrainagePathLengthM,
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
    public double? DirectBeamFactor { get; set; }
    public double? LwClearK { get; set; }
    public double? LwCloudK { get; set; }
    public double? MuScale { get; set; }
    public int? Substeps { get; set; }
    public int? SpinupHours { get; set; }
    public double? GreasyMarginC { get; set; }

    /// <summary>Per-location enable of the surface-water/drying model (the
    /// physics coefficients stay global). Lets Bonehill and Sennen turn it on
    /// independently once each is calibrated.</summary>
    public bool? SurfaceWaterEnabled { get; set; }

    /// <summary>Replaces the global faces list WHOLESALE when set (faces are
    /// location geometry — merging two locations' lists makes no sense).</summary>
    public List<RockSurfaceFaceConfig>? Faces { get; set; }
}
