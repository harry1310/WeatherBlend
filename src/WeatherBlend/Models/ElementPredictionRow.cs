using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One blended-element prediction (wind / humidity / shortwave-radiation /
/// cloud-cover) with full provenance. Stored under
/// <c>data/predictions/{element}/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>
/// — same per-element subtree pattern as <see cref="PredictionRow"/> uses for
/// temperature.
///
/// Per-model fields are nullable: wind sets <see cref="ModelMf"/> to null because
/// MétéoFrance has no wind data on Open-Meteo Previous Runs; radiation may set
/// <see cref="ModelUkmo"/> to null at lead ≥ 48h. Verification baselines
/// (best-single, mean-of-models) compute over the non-null entries.
/// </summary>
public sealed class ElementPredictionRow
{
    [SetsRequiredMembers]
    public ElementPredictionRow()
    {
        LocationName = "";
        Element = "";
        ModelVersion = "";
        FeatureVectorHash = "";
    }

    public required string LocationName { get; init; }

    /// <summary>Element CLI name: "wind" | "humidity" | "shortwave-radiation" | "cloud-cover".</summary>
    public required string Element { get; init; }

    public required string ModelVersion { get; init; }

    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>The blender's output (units depend on element — see <see cref="Element"/>).</summary>
    public required double BlendValue { get; init; }

    public double? ModelGfs { get; init; }
    public double? ModelEcmwf { get; init; }
    public double? ModelIcon { get; init; }
    public double? ModelMf { get; init; }
    public double? ModelUkmo { get; init; }
    public double? ModelGem { get; init; }

    public DateTime? RunTimeGfs { get; init; }
    public DateTime? RunTimeEcmwf { get; init; }
    public DateTime? RunTimeIcon { get; init; }
    public DateTime? RunTimeMf { get; init; }
    public DateTime? RunTimeUkmo { get; init; }
    public DateTime? RunTimeGem { get; init; }

    public double? Mean { get; init; }
    public double? Std { get; init; }
    public double? Range { get; init; }

    public required string FeatureVectorHash { get; init; }
}
