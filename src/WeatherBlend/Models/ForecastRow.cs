namespace WeatherBlend.Models;

/// <summary>
/// One hourly forecast row for one model, one run, one valid-time.
/// Nullable numeric fields so the schema is stable even if a model
/// doesn't expose a given variable.
/// </summary>
public sealed class ForecastRow
{
    public required string LocationName { get; init; }
    public required string Model { get; init; }
    public required DateTime RunTimeUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>
    /// How the RunTime/LeadHours fields were derived. Nullable for back-compat with pre-phase-3 files.
    /// "exact" — real cycle stamp from a direct-source GRIB archive (GFS S3, ECMWF Open Data, DWD, WB2).
    /// "reported" — cycle stamp reported by a model-metadata endpoint (e.g. Open-Meteo's
    ///              last_run_initialisation_time) — approximate, may not match what the forecast endpoint served.
    /// "synthesised" — fabricated for backward compatibility (e.g. midnight-of-valid-day, fetch-time-floored).
    ///                 Do not use for per-lead-time training.
    /// "offset_day"  — Open-Meteo Previous Runs API row. RunTime = ValidTime − 24·N h and
    ///                 LeadHours = 24·N is the lower edge of the [24N..24N+23] bucket.
    /// null — legacy file; treat as "synthesised" for training purposes.
    /// </summary>
    public string? RunTimeSource { get; init; }

    public double? Temperature2m { get; init; }
    public double? DewPoint2m { get; init; }
    public double? RelativeHumidity2m { get; init; }
    public double? Precipitation { get; init; }
    public double? PrecipitationProbability { get; init; }
    public double? Rain { get; init; }
    public double? Showers { get; init; }
    public double? Snowfall { get; init; }
    public double? CloudCover { get; init; }
    public double? CloudCoverLow { get; init; }
    public double? CloudCoverMid { get; init; }
    public double? CloudCoverHigh { get; init; }
    public double? WindSpeed10m { get; init; }
    public double? WindDirection10m { get; init; }
    public double? WindGusts10m { get; init; }
    public double? SurfacePressure { get; init; }
    /// <summary>
    /// Mean sea level pressure (hPa). Added 2026-06-03. Synoptic-pattern
    /// predictor — distinct from <see cref="SurfacePressure"/> (station-level,
    /// elevation-dependent). Populated from OM (pressure_msl), GFS (PRMSL),
    /// GEFS (PRMSL) and ECMWF (msl); nullable for back-compat with older rows.
    /// </summary>
    public double? PressureMsl { get; init; }
    public double? Cape { get; init; }
    /// <summary>Precipitable water / total column water vapour (kg/m² ≈ mm) — the
    /// convective "fuel" (CAPE is the trigger). Added 2026-06-19. GFS
    /// (PWAT:entire atmosphere), GEFS (PWAT), ECMWF IFS (tcwv). Null for sources
    /// that don't publish it (Met Office, AIFS, OM offset_day). Nullable for back-compat.</summary>
    public double? PrecipitableWater { get; init; }
    /// <summary>Convective inhibition (J/kg) — the "lid" that pairs with CAPE.
    /// Added 2026-06-19. GFS (CIN:surface), GEFS (CIN:180-0 mb above ground),
    /// Met Office global (CIN_surface). Null for ECMWF/AIFS/UKV/OM (not published).
    /// Sign follows each source (typically ≤ 0). Nullable for back-compat.</summary>
    public double? ConvectiveInhibition { get; init; }
    public double? Visibility { get; init; }
    public double? ShortwaveRadiation { get; init; }
    public double? DirectRadiation { get; init; }
    public double? DiffuseRadiation { get; init; }
    /// <summary>
    /// Surface downwelling longwave (thermal) radiation, W/m² (interval-mean).
    /// Added 2026-06-03 to feed the net-longwave term of the rock surface-temp
    /// Force-Restore budget directly, replacing the cloud-driven Brunt
    /// parameterisation (ROCK_SURFACE_TEMP_PLAN §3/§9). ONLY the GFS exact
    /// archive supplies it (DLWRF:surface) — Open-Meteo exposes no longwave
    /// variable, GEFS pgrb2a doesn't carry it, and the ECMWF oper stream we
    /// pull maps only ssrd. Null on every non-GFS source.
    /// </summary>
    public double? DownwardLongwaveRadiation { get; init; }

    /// <summary>
    /// Modelled cloud base height in metres above ground at the forecast point.
    /// Currently populated from the GFS S3 backfill (<c>HGT:cloud ceiling</c>);
    /// nullable for back-compat with NWP rows that don't expose it. Compared
    /// against the location's elevation to derive the "is the tor in cloud?"
    /// signal directly, replacing the Espy (T-Td) proxy where available.
    /// Sentinels at very large values (NCEP often emits ~20000m for "no cloud
    /// detected") are stored as-is — callers should treat values above the
    /// model top as "no cloud base" rather than literally that height.
    /// </summary>
    public double? CloudBaseHeightM { get; init; }

    // ---- Multi-level (pressure-level) fields — added 2026-05-29 ----
    // Open-Meteo per-model exposure varies; absent levels arrive as NULL and
    // are safe to ignore at training time. Not yet consumed by any blender
    // spec — purely accumulating a backfill window for future upper-air work.
    public double? Temperature850hPa { get; init; }
    public double? Temperature700hPa { get; init; }
    public double? Temperature500hPa { get; init; }
    public double? GeopotentialHeight850hPa { get; init; }
    public double? GeopotentialHeight500hPa { get; init; }
    public double? WindSpeed850hPa { get; init; }
    public double? WindSpeed500hPa { get; init; }
    public double? WindDirection850hPa { get; init; }
    public double? WindDirection500hPa { get; init; }
    public double? RelativeHumidity850hPa { get; init; }
}
