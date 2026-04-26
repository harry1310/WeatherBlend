using Microsoft.ML.Data;

namespace WeatherBlend.Train.Element.Cloud;

/// <summary>
/// One training row for the cloud-cover blender.
/// Truth: ERA5 <c>cloud_cover</c> (total) at the Bonehill grid cell (%).
///
/// Features (13): per-model total cloud cover (6), inter-model spread (3),
/// cyclical hour + doy (4). The brief originally proposed adding low/mid/high
/// per-model layered cloud — audit 2026-04-25 confirmed Open-Meteo Previous Runs
/// returns 100% null for those layered columns across every model and ERA5, so
/// the lean shape collapses to total only. Adding layered cloud is tracked as a
/// future scope (would require collector changes and a 3-year backfill).
/// </summary>
public sealed class CloudRow
{
    public DateTime ValidTimeUtc { get; set; }

    public float CcGfs { get; set; }
    public float CcEcmwf { get; set; }
    public float CcIcon { get; set; }
    public float CcMf { get; set; }
    public float CcUkmo { get; set; }
    public float CcGem { get; set; }

    public float CcMean { get; set; }
    public float CcStd { get; set; }
    public float CcRange { get; set; }

    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    [ColumnName("Label")]
    public float Era5Cc { get; set; }
}
