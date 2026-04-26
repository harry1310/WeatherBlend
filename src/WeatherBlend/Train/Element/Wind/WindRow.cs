using Microsoft.ML.Data;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// One training row for the wind blender.
///
/// Truth: ERA5 <c>wind_speed_10m</c> at the Bonehill grid cell (m/s).
///
/// Features (25 total): per-model wind speed (5 models — MétéoFrance is excluded
/// because Open-Meteo Previous Runs ships no MF wind), per-model wind direction
/// encoded as sin/cos (5 × 2), inter-model spread on speed (mean / std / range),
/// hour & doy cyclical encodings.
///
/// Why direction is in here even though we're predicting speed: synoptic flow
/// direction often correlates with gust regimes and orographic enhancement at
/// Bonehill. The blender can learn to weight per-model speeds differently when
/// the ensemble agrees on a southwesterly vs an easterly. LightGBM is happy
/// to ignore the direction features if they prove uninformative.
/// </summary>
public sealed class WindRow
{
    public DateTime ValidTimeUtc { get; set; }

    // Per-model 10m wind speed (m/s). 5 models — MF excluded.
    public float SpdGfs { get; set; }
    public float SpdEcmwf { get; set; }
    public float SpdIcon { get; set; }
    public float SpdUkmo { get; set; }
    public float SpdGem { get; set; }

    // Per-model 10m wind direction encoded as sin/cos of bearing (degrees).
    public float DirSinGfs { get; set; }
    public float DirCosGfs { get; set; }
    public float DirSinEcmwf { get; set; }
    public float DirCosEcmwf { get; set; }
    public float DirSinIcon { get; set; }
    public float DirCosIcon { get; set; }
    public float DirSinUkmo { get; set; }
    public float DirCosUkmo { get; set; }
    public float DirSinGem { get; set; }
    public float DirCosGem { get; set; }

    // Inter-model spread of wind speed across the 5 present models.
    public float SpdMean { get; set; }
    public float SpdStd { get; set; }
    public float SpdRange { get; set; }

    // Cyclical calendar.
    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    [ColumnName("Label")]
    public float Era5WindSpeed { get; set; }
}
