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

    // ---- forcing provenance ----
    public required double ShortwaveDownWm2 { get; init; }
    public required double CloudCoverPct { get; init; }
    public required double WindSpeed10mMs { get; init; }
    public required double LongwaveDownWm2 { get; init; }
    /// <summary>Deep-reservoir temperature Td_deep (°C) — the model's thermal-memory state.</summary>
    public required double DeepTempC { get; init; }

    // ---- source model versions ----
    public required string TempModelVersion { get; init; }
    public required string WindModelVersion { get; init; }
    public required string RadiationModelVersion { get; init; }
    public required string CloudModelVersion { get; init; }
    /// <summary>How dew point was sourced (e.g. "nwp_mean") — Td is taken from
    /// the NWP forecast tree directly, not a blender.</summary>
    public required string DewPointSource { get; init; }
}
