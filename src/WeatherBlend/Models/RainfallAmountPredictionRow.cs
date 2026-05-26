using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One Phase 3f rainfall_amount prediction row — the mixed
/// dry/wet predictive distribution for a single (valid_time, lead)
/// at one EA rainfall station.
///
/// 3f is a two-stage distributional forecast. Stage 1 (Phase 3a)
/// supplies π = P(hour is wet) hourly. Stage 2 (NGBoost-LogNormal,
/// trained on the wet-only rows) supplies (μ_log, σ_log) for the
/// conditional LogNormal of mm/h given the hour is wet. The full
/// predictive distribution is:
///
///   F(x) = (1 − π) · δ_0(x) + π · LogNormal(μ_log, σ_log)(x)
///
/// Written by WeatherProbabilistic's <c>predict_3f.py</c> to
/// <c>data/predictions/rainfall_amount/{station_slug}/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>
/// on each predict cycle, then synced to R2 via the standard
/// per-station-tree convention. The .NET render path reads it via
/// DuckDB to populate the rainfall_amount predictions card + the
/// per-station skill widgets (CRPS rolling, PIT histogram, coverage,
/// exceedance reliability).
///
/// Raw (μ_log, σ_log) are preserved alongside the derived quantiles +
/// exceedance probabilities so future consumers can compute additional
/// quantiles or threshold probabilities without retraining. Provenance
/// columns (<see cref="Precip3aVersion"/>, <see cref="ModelVersion"/>)
/// let verify reconstruct exactly which stage-1 / stage-2 version
/// produced each row.
///
/// Schema deliberately mirrors the plan-doc table at
/// <c>docs/RAINFALL_AMOUNT_3F_PLAN.md</c> §2.4 — the Python writer
/// MUST emit the same column names + types, otherwise the .NET
/// DuckDB read trips on missing columns.
/// </summary>
public sealed class RainfallAmountPredictionRow
{
    [SetsRequiredMembers]
    public RainfallAmountPredictionRow()
    {
        LocationName = "";
        TruthStation = "";
        ModelVersion = "";
        Precip3aVersion = "";
    }

    /// <summary>Location slug from config.yaml — e.g. "membury_devon".</summary>
    public required string LocationName { get; init; }

    /// <summary>EA rainfall station slug — e.g. "ea_chards_snowdon_hill".</summary>
    public required string TruthStation { get; init; }

    /// <summary>Phase 3f bundle version that produced this row —
    /// e.g. "v2026-05-26_104230_phase3f". Resolves via
    /// <see cref="WeatherBlend.Train.ModelArtifact.ResolveStationVersionDir"/>
    /// in the rainfall_amount tree.</summary>
    public required string ModelVersion { get; init; }

    /// <summary>Predict cycle anchor — the wall-clock time
    /// predict_3f.py was invoked. Used for de-duping + freshness
    /// pick (multiple cycles can write the same valid_time).</summary>
    public required DateTime PredictionMadeAtUtc { get; init; }

    /// <summary>Target valid time for this hourly forecast.</summary>
    public required DateTime ValidTimeUtc { get; init; }

    /// <summary>Forecast lead in hours (ValidTime − PredictionMadeAt).
    /// Plan's initial scope: {24, 48, 72, 96, 120}.</summary>
    public required int LeadHours { get; init; }

    // ---- Stage-1 marginal (from Phase 3a) ----
    /// <summary>π — P(hour is wet) from the bound Phase 3a champion.
    /// Mixed into the predictive distribution as the wet-branch
    /// probability.</summary>
    public required double Pi { get; init; }

    // ---- Stage-2 NGBoost-LogNormal raw parameters ----
    /// <summary>μ_log — location parameter of the conditional LogNormal
    /// over mm/h | wet. Mean of log(mm/h) in NGBoost's output space.</summary>
    public required double MuLog { get; init; }

    /// <summary>σ_log — scale parameter of the conditional LogNormal.
    /// Standard deviation of log(mm/h) | wet.</summary>
    public required double SigmaLog { get; init; }

    // ---- Derived headline values ----
    /// <summary>Mixed mean: π · exp(μ_log + σ_log² / 2). The
    /// distribution's expectation. Skew-sensitive on LogNormal —
    /// median is the safer headline for site display.</summary>
    public required double MeanMmPerHr { get; init; }

    /// <summary>Mixed median: π · exp(μ_log). Robust to LogNormal
    /// skew — this is what the rainfall_amount card's headline
    /// strip displays.</summary>
    public required double MedianMmPerHr { get; init; }

    // ---- Quantile fan (configured alphas) ----
    // These are the mixed quantiles: each α maps via
    // scipy.stats.lognorm.ppf with weight π plus a δ_0 mass with
    // weight (1-π). For α < 1-π the quantile is 0; for α ≥ 1-π
    // the quantile is the conditional LogNormal's
    // ((α - (1-π)) / π)-quantile.
    /// <summary>2.5th percentile of the mixed predictive distribution.</summary>
    public required double P2_5MmPerHr { get; init; }

    /// <summary>10th percentile. Lower bound of the 80% credible band.</summary>
    public required double P10MmPerHr { get; init; }

    /// <summary>Median of the mixed distribution. Same as
    /// <see cref="MedianMmPerHr"/> when π · 0.5 + (1-π) · 0 ≥ 0.5
    /// (i.e. when the mixed CDF reaches 0.5 inside the LogNormal
    /// branch). For very-low-π hours both can be 0.</summary>
    public required double P50MmPerHr { get; init; }

    /// <summary>90th percentile. Upper bound of the 80% band.</summary>
    public required double P90MmPerHr { get; init; }

    /// <summary>97.5th percentile.</summary>
    public required double P97_5MmPerHr { get; init; }

    // ---- Exceedance probabilities (configured thresholds, mm) ----
    /// <summary>P(mm/h ≥ 0.1) — π · (1 − CDF_LogNormal(0.1)).</summary>
    public required double PExceed0_1 { get; init; }

    /// <summary>P(mm/h ≥ 1.0).</summary>
    public required double PExceed1 { get; init; }

    /// <summary>P(mm/h ≥ 5.0).</summary>
    public required double PExceed5 { get; init; }

    /// <summary>P(mm/h ≥ 10.0).</summary>
    public required double PExceed10 { get; init; }

    // ---- Provenance ----
    /// <summary>The exact Phase 3a champion version that supplied π
    /// for this row. Stamped at train time onto the 3f bundle's
    /// metadata, then carried per row so the audit chain is honest
    /// even if a different 3a champion is current at read time.</summary>
    public required string Precip3aVersion { get; init; }
}
