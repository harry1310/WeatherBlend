using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One wave_height prediction row as read by the wave verify rail — the
/// scoring subset of the Python predict_wave_pi.py parquet (the site's
/// pass-through extras like tide / swell / SST are not read here; see
/// <c>RenderSiteCommand.QueryWavePredictions</c> for the display-side shape).
/// </summary>
public sealed class WavePredictionRow
{
    [SetsRequiredMembers]
    public WavePredictionRow()
    {
        LocationName = "";
        ModelVersion = "";
    }

    public required string LocationName { get; init; }
    public required string ModelVersion { get; init; }
    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }

    /// <summary>Actual forecast lead in hours. wave_height_lgb is ANY-LEAD v1:
    /// each predict cycle emits one continuous hourly forecast (leads 0..~146),
    /// so this is a true hourly lead, not a trained-bucket label.</summary>
    public required int LeadHours { get; init; }

    /// <summary>Blend significant wave height, m (the q50 point forecast).</summary>
    public required double BlendValue { get; init; }

    /// <summary>Calibrated 90% band lower bound, m. Null on rows written
    /// before the CQR band columns existed.</summary>
    public double? BandLoM { get; init; }

    /// <summary>Calibrated 90% band upper bound, m.</summary>
    public double? BandHiM { get; init; }

    /// <summary>Total wave period pass-through, s — NOT a blended output:
    /// it comes straight from the best_match wave model. Verified against
    /// the buoy's Tz with a known definitional offset (model mean period
    /// reads ~1-2 s longer); bias is the load-bearing metric.</summary>
    public double? WavePeriodS { get; init; }
}
