using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One rock surface temperature + condensation prediction (Phase P1 — derived,
/// not trained). <c>Ts</c> from the Force-Restore integration over the blended
/// hourly forcing trajectory; <c>Td</c> from NWP dew point; the condensation
/// margin <c>m = Ts − Td</c> and a greasiness tier
/// (<c>dry | potentially_greasy | condensation</c>) are the decision outputs.
/// Inputs + source model versions are captured per row so the chain is
/// auditable. Stored under
/// <c>data/predictions/rock_surface/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>.
/// </summary>
public sealed class RockSurfacePredictionRow
{
    [SetsRequiredMembers]
    public RockSurfacePredictionRow()
    {
        LocationName = "";
        ModelVersion = "";
        Face = "";
        GreasinessStatus = "";
        TempModelVersion = "";
        WindModelVersion = "";
        RadiationModelVersion = "";
        CloudModelVersion = "";
        DewPointSource = "";
    }

    public required string LocationName { get; init; }
    public required string ModelVersion { get; init; }

    /// <summary>Crag face this row models (cliff-face mode, e.g. "west") —
    /// each configured face is its own Force-Restore integration with the
    /// direct beam projected onto its plane. Empty = whole-crag horizontal
    /// mode (Bonehill). NOT <c>required</c>: rows written before this column
    /// existed deserialize with the constructor's empty default.</summary>
    public string Face { get; init; }
    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }

    /// <summary>Lead of the BLEND forcing that drove this hour (smallest-lead
    /// per valid time, e.g. 24/48/72). Spin-up hours are not emitted.</summary>
    public required int LeadHours { get; init; }

    /// <summary>Rock surface temperature Ts (°C) — the model output.</summary>
    public required double RockSurfaceTempC { get; init; }

    /// <summary>2 m air temperature Ta (°C) used as forcing this hour.</summary>
    public required double AirTempC { get; init; }

    /// <summary>Dew point Td (°C) — NWP, drives the margin + clear-sky LW.</summary>
    public required double DewPointC { get; init; }

    /// <summary>Condensation margin m = Ts − Td (°C). ≤ 0 ⇒ condensation.</summary>
    public required double CondensationMarginC { get; init; }

    /// <summary>Greasiness tier: <c>condensation</c> (m ≤ 0),
    /// <c>potentially_greasy</c> (0 &lt; m ≤ greasyMargin), or <c>dry</c>.
    /// String (not enum) for portable parquet readers.</summary>
    public required string GreasinessStatus { get; init; }

    /// <summary>Total surface-water film (mm) — rain plus condensed dew — from
    /// the Phase A drying model (VPD-driven evaporation + dew, radiation drying,
    /// runoff cap). NOT <c>required</c>: rows written before the drying model
    /// deserialize with the default 0. The climbing index only acts on this when
    /// <c>rockSurface.surfaceWaterEnabled</c> is true for the location.</summary>
    public double SurfaceWaterMm { get; init; }

    /// <summary>The rain-derived part of <see cref="SurfaceWaterMm"/> (mm). Split
    /// out so a consumer can tell "wet from rain" (a heavy, continuous friction
    /// penalty that eases as the film dries) from "wet from dew" (the greasiness
    /// penalty). NOT <c>required</c>: defaults 0.</summary>
    public double RainWaterMm { get; init; }

    /// <summary>Most recent hour (UTC) at which a meaningful shower last wet the
    /// rock — INCLUDING the model's hidden spin-up window, so the site can say
    /// "wet from rain since HHZ" even when the wetting happened before the
    /// reported forecast hours. Null = no rain on this trajectory. NOT
    /// <c>required</c>: rows written before this column deserialize as null.</summary>
    public DateTime? LastRainAtUtc { get; init; }

    // ---- forcing provenance ----
    public required double ShortwaveDownWm2 { get; init; }
    public required double CloudCoverPct { get; init; }
    public required double WindSpeed10mMs { get; init; }
    public required double LongwaveDownWm2 { get; init; }
    /// <summary>Deep-reservoir temperature Td_deep (°C) — the model's thermal-memory state.</summary>
    public required double DeepTempC { get; init; }

    /// <summary>Sea surface temperature (°C) feeding the S4 sea-longwave term
    /// (sea-cliff locations; best_match marine forecast). Null = no marine
    /// block / no SST coverage — neutral surroundings. NOT <c>required</c>:
    /// pre-S4 rows deserialize as null.</summary>
    public double? SeaSurfaceTempC { get; init; }

    // ---- source model versions ----
    public required string TempModelVersion { get; init; }
    public required string WindModelVersion { get; init; }
    public required string RadiationModelVersion { get; init; }
    public required string CloudModelVersion { get; init; }
    /// <summary>How dew point was sourced (e.g. "nwp_mean") — Td is taken from
    /// the NWP forecast tree directly, not a blender.</summary>
    public required string DewPointSource { get; init; }
}
