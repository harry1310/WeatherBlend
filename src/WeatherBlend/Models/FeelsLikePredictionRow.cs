using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One "feels like" prediction with full input provenance. Carries both the
/// rigorous UTCI (Bröde 2012 biothermal index) and the friendlier Steadman
/// 1994 apparent temperature alongside every input that fed them. Derived
/// from the temperature blender (lean 2b) plus the four element blenders
/// (humidity / wind / shortwave-radiation / cloud-cover) at a shared
/// (valid_time, lead) row. Stored under
/// <c>data/predictions/feels_like/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>.
///
/// Each input model version is captured per row so the chain is auditable
/// — "which radiation champion produced the SW input that drove this Tmrt?".
/// </summary>
public sealed class FeelsLikePredictionRow
{
    [SetsRequiredMembers]
    public FeelsLikePredictionRow()
    {
        LocationName = "";
        ModelVersion = "";
        Band = "";
        TempModelVersion = "";
        HumidityModelVersion = "";
        WindModelVersion = "";
        RadiationModelVersion = "";
        CloudModelVersion = "";
    }

    public required string LocationName { get; init; }
    public required string ModelVersion { get; init; }
    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    public required double TemperatureC { get; init; }
    public required double RelativeHumidityPct { get; init; }
    public required double WindSpeed10mMs { get; init; }
    public required double WindSpeed1m1Ms { get; init; }
    public required double ShortwaveDownWm2 { get; init; }
    public required double CloudCoverPct { get; init; }

    public required double VapourPressureHpa { get; init; }
    public required double LongwaveDownWm2 { get; init; }
    public required double LongwaveUpWm2 { get; init; }
    public required double MeanRadiantTemperatureC { get; init; }
    public required double UtciC { get; init; }

    /// <summary>UTCI thermal-stress band name (Bröde 2012 Table 1) — string
    /// rather than enum for portable parquet readers.</summary>
    public required string Band { get; init; }

    /// <summary>
    /// Steadman 1994 outdoor "apparent temperature" (°C) — the BBC/BoM-style
    /// shade-form companion to UTCI. Derived from the same Ta + e + ws inputs
    /// captured above, so persistence here means the home-card chip (and any
    /// future verify pipeline) reads it directly from the row instead of
    /// recomputing in the renderer. <see cref="UtciC"/> stays the rigorous
    /// biothermal index; this one is the public-facing "feels like".
    /// </summary>
    public required double ApparentTemperatureC { get; init; }

    public required string TempModelVersion { get; init; }
    public required string HumidityModelVersion { get; init; }
    public required string WindModelVersion { get; init; }
    public required string RadiationModelVersion { get; init; }
    public required string CloudModelVersion { get; init; }
}
