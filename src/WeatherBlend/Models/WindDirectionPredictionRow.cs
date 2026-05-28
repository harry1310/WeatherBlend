using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One wind-direction prediction row written by WeatherProbabilistic's
/// <c>predict_wind_mvn.py</c> (Phase 2 of WIND_BLENDER_PLAN). Stored under
/// <c>data/predictions/wind_direction/{location}/model_version={v}_wind_mvn/date={d}/predictions.parquet</c>.
///
/// The MLP outputs a bivariate normal over (u, v); from those parameters
/// we derive direction (atan2(-μ_u, -μ_v)) + speed magnitude
/// (sqrt(μ_u² + μ_v²)), and from MC samples scaled by per-lead α'
/// calibration scalars we derive 95% CIs for both. The C# side mostly
/// CONSUMES this row at render-time (UTCI pop-out's direction wedge,
/// once site work ships) and during the wind_blend mint step, which
/// reads BlendSpeedMagnitude to feed the sigmoid composition with
/// wind_speed_lgb's BlendValue.
///
/// Per-NWP <c>RunTime*</c> columns are filled by the predictor where the
/// underlying forecast row was present; nulls are tolerated (some NWPs
/// don't contribute at every valid time).
/// </summary>
public sealed class WindDirectionPredictionRow
{
    [SetsRequiredMembers]
    public WindDirectionPredictionRow()
    {
        LocationName = "";
        ModelVersion = "";
    }

    public required string LocationName { get; init; }
    public required string ModelVersion { get; init; }
    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>Mean μ_u of the MVN over (u, v) — east-component of the wind
    /// vector. Internal — site renders <see cref="BlendDirection"/> + speed.</summary>
    public required double MuU { get; init; }
    /// <summary>Mean μ_v of the MVN over (u, v) — north-component.</summary>
    public required double MuV { get; init; }
    public required double SigmaU { get; init; }
    public required double SigmaV { get; init; }
    public required double Rho { get; init; }

    /// <summary>Direction in degrees from-direction (meteorological
    /// convention, 0=North, 90=East), 0..360.</summary>
    public required double BlendDirection { get; init; }
    /// <summary>95% credible interval bounds for direction (degrees).
    /// Calibrated via per-lead α'_dir scaling of σ_u, σ_v before MC.</summary>
    public required double BlendDirectionCi95Lo { get; init; }
    public required double BlendDirectionCi95Hi { get; init; }

    /// <summary>80% credible interval bounds for direction (degrees).
    /// Same MC samples as the CI95 pair, just tighter percentiles
    /// (10/90 of the highest-density arc). Added 2026-05-28; nullable
    /// so parquets written before that date deserialize cleanly.</summary>
    public double? BlendDirectionCi80Lo { get; init; }
    public double? BlendDirectionCi80Hi { get; init; }

    /// <summary>Speed magnitude from the MVN parameters,
    /// sqrt(μ_u² + μ_v²) (m/s). NOT used as the canonical wind_speed —
    /// wind_blend in predict-tail composes this with wind_speed_lgb via
    /// the sigmoid blend.</summary>
    public required double BlendSpeedMagnitude { get; init; }
    /// <summary>95% credible interval bounds for speed (m/s). Calibrated
    /// via per-lead α'_spd scaling of σ_u, σ_v before MC.</summary>
    public required double BlendSpeedCi95Lo { get; init; }
    public required double BlendSpeedCi95Hi { get; init; }

    /// <summary>80% credible interval bounds for speed (m/s). Same MC
    /// samples as the CI95 pair, percentiles 10/90 instead of 2.5/97.5.
    /// The wind page's chart ribbon defaults to this band — 95% was
    /// visually overwhelming given the bundle's σ on 30-day training.
    /// Nullable for back-compat with pre-2026-05-28 parquets.</summary>
    public double? BlendSpeedCi80Lo { get; init; }
    public double? BlendSpeedCi80Hi { get; init; }

    public DateTime? RunTimeUtcGfs   { get; init; }
    public DateTime? RunTimeUtcEcmwf { get; init; }
    public DateTime? RunTimeUtcIcon  { get; init; }
    public DateTime? RunTimeUtcMf    { get; init; }
    public DateTime? RunTimeUtcUkmo  { get; init; }
    public DateTime? RunTimeUtcGem   { get; init; }
    public DateTime? RunTimeUtcAifs  { get; init; }
    public DateTime? RunTimeUtcJma   { get; init; }
}
