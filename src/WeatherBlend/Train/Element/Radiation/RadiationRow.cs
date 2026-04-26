using Microsoft.ML.Data;

namespace WeatherBlend.Train.Element.Radiation;

/// <summary>
/// One training row for the shortwave-radiation blender.
/// Truth: ERA5 <c>shortwave_radiation</c> at the Bonehill grid cell (W/m²).
/// Features (25): per-model SW (6), per-model direct + diffuse (12), inter-model spread
/// on SW (3), cyclical hour + doy (4). UKMO is included; per the audit it is non-null
/// at lead 24h (~74% coverage) but always-null at 48h+ — those rows carry NaN and
/// LightGBM treats it as "missing" natively.
/// </summary>
public sealed class RadiationRow
{
    public DateTime ValidTimeUtc { get; set; }

    public float SwGfs { get; set; }
    public float SwEcmwf { get; set; }
    public float SwIcon { get; set; }
    public float SwMf { get; set; }
    public float SwUkmo { get; set; }
    public float SwGem { get; set; }

    public float DirectGfs { get; set; }
    public float DirectEcmwf { get; set; }
    public float DirectIcon { get; set; }
    public float DirectMf { get; set; }
    public float DirectUkmo { get; set; }
    public float DirectGem { get; set; }

    public float DiffuseGfs { get; set; }
    public float DiffuseEcmwf { get; set; }
    public float DiffuseIcon { get; set; }
    public float DiffuseMf { get; set; }
    public float DiffuseUkmo { get; set; }
    public float DiffuseGem { get; set; }

    public float SwMean { get; set; }
    public float SwStd { get; set; }
    public float SwRange { get; set; }

    public float HourSin { get; set; }
    public float HourCos { get; set; }
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    [ColumnName("Label")]
    public float Era5Sw { get; set; }
}
