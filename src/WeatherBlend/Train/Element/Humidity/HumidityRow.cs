using Microsoft.ML.Data;

namespace WeatherBlend.Train.Element.Humidity;

/// <summary>
/// One training row for the humidity blender.
/// Truth: ERA5 <c>relative_humidity_2m</c> at the Bonehill grid cell (%).
/// Features (19): per-model RH (6), per-model dew point (6), inter-model spread on RH
/// (3), cyclical hour + doy (4). Dewpoint goes in alongside RH because the two are
/// related but parameterised differently across models — keeping both lets the blender
/// learn which encoding generalises better at this site.
/// </summary>
public sealed class HumidityRow
{
    public DateTime ValidTimeUtc { get; set; }

    public float RhGfs { get; set; }
    public float RhEcmwf { get; set; }
    public float RhIcon { get; set; }
    public float RhMf { get; set; }
    public float RhUkmo { get; set; }
    public float RhGem { get; set; }

    public float DpGfs { get; set; }
    public float DpEcmwf { get; set; }
    public float DpIcon { get; set; }
    public float DpMf { get; set; }
    public float DpUkmo { get; set; }
    public float DpGem { get; set; }

    public float RhMean { get; set; }
    public float RhStd { get; set; }
    public float RhRange { get; set; }

    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    [ColumnName("Label")]
    public float Era5Rh { get; set; }
}
