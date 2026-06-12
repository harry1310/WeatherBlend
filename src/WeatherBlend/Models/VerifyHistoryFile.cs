using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// JSON sidecar written next to every weekly verify markdown report. Same
/// per-(version, lead) data the markdown holds, but in a structured shape
/// so the Models page can render a "Verify history" table without parsing
/// markdown. One file per (target, asOf) — see
/// <c>cloudflare/scheduler-worker/README.md</c> for the cron that produces
/// them weekly.
///
/// Sibling filename pattern matches the markdown that already exists, with
/// the extension swapped:
///   verify_temperature_{yyyy-MM-dd}.json
///   verify_precipitation_{yyyy-MM-dd}.json
///   verify_dry_window_{yyyy-MM-dd}.json
///   verify_element_{element}_{yyyy-MM-dd}.json
///   verify_wave_height_{location}_{yyyy-MM-dd}.json
///
/// Start-hour verify is intentionally NOT in this schema — its metric set
/// (top-1 / Brier / log-loss skill) doesn't fit a single-blend-metric row,
/// and the curve is a derivation rather than a trained model so it doesn't
/// surface on the Models page.
/// </summary>
public sealed class VerifyHistoryFile
{
    [SetsRequiredMembers]
    public VerifyHistoryFile()
    {
        Target = "";
        MetricLabel = "";
        Rows = new List<VerifyHistoryRow>();
    }

    /// <summary>Target identifier — <c>temperature</c>, <c>precipitation</c>,
    /// <c>dry_window</c>, or <c>element_{name}</c>. Maps to the same composite
    /// keys the Models page already uses for grouping.</summary>
    public required string Target { get; init; }

    /// <summary>Asof timestamp the verify was run against — UTC, hour-level
    /// resolution. Same value the markdown report's filename encodes.</summary>
    public required DateTime AsOfUtc { get; init; }

    public required int WindowDays { get; init; }
    public required int LatencyDays { get; init; }

    /// <summary>Display label for the blend metric — <c>"MAE (°C)"</c>,
    /// <c>"Brier"</c>, etc. Lets the Models page render units correctly
    /// without target-specific switches.</summary>
    public required string MetricLabel { get; init; }

    public required List<VerifyHistoryRow> Rows { get; init; }
}

/// <summary>One per (station?, version, lead [+ optional window]) cell.
/// Optional fields are <c>null</c> when not applicable to the target — temp
/// has no station, dry-window has window-hours, etc. The Models page
/// renderer treats nulls as "—".</summary>
public sealed class VerifyHistoryRow
{
    [SetsRequiredMembers]
    public VerifyHistoryRow()
    {
        ModelVersion = "";
    }

    /// <summary>Per-target scope key. Polymorphic by <see cref="VerifyHistoryFile.Target"/>:
    ///   * precipitation / dry-window → EA rainfall gauge slug
    ///     (<c>ea_bellever_dartmoor</c>) — loc-unique by config convention.
    ///   * temperature / element_* → location slug (<c>bonehill_rocks</c>,
    ///     <c>membury_devon</c>) — mirrors the per-station temperature
    ///     manifest (2026-05-14 layout) where the manifest's "station" key
    ///     IS the location slug. Distinguishes Bonehill vs Membury cards
    ///     on the Models page (Phase C commit 4, 2026-05-20). Old sidecars
    ///     written before commit 4 have <c>null</c> for temp/element — the
    ///     Models-page filter treats null as a primary-location fallback so
    ///     the Bonehill card's history doesn't disappear at the rollout.
    /// </summary>
    public string? Station { get; init; }

    public required string ModelVersion { get; init; }

    /// <summary>training_metadata.Phase for <see cref="ModelVersion"/> at the
    /// time of the run (e.g. <c>"2b"</c>, <c>"2c"</c>, <c>"3a"</c>,
    /// <c>"3c"</c>, <c>"3b"</c>). The Models-page renderer matches per-card
    /// history on this field rather than on <see cref="ModelVersion"/>: the
    /// card represents the current champion/challenger of a phase, but verify
    /// inevitably scores older versions (5-day truth latency means the freshly
    /// retrained Active version has zero paired rows). Matching by phase
    /// surfaces the lineage's verify history on the same card across retrains.
    /// Optional for backward compat with sidecars written before this field
    /// existed; the renderer treats null/empty as "won't match".</summary>
    public string? Phase { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>Dry-window only — the window length in hours (3 / 4 / 6).</summary>
    public int? WindowHours { get; init; }

    /// <summary>Pair count behind the metric — readers gauge stability of
    /// the headline number from this.</summary>
    public required int N { get; init; }

    public required double BlendMetric { get; init; }
    public double? ClimMetric { get; init; }
    public double? MeanOfModelsMetric { get; init; }

    /// <summary>"GFS" / "ECMWF" / etc. — the per-NWP best-single baseline
    /// the blend was up against on this slice. Null when the target has no
    /// per-NWP baseline set up (start-hour, but it's not in this file).</summary>
    public string? BestSingleName { get; init; }
    public double? BestSingleMetric { get; init; }

    /// <summary>The metric this blender posted on its training-time test
    /// partition — what the model said it could do at training time.
    /// Drift flag fires when the live <see cref="BlendMetric"/> exceeds
    /// <c>driftThreshold × ReferenceTrainingMetric</c>.</summary>
    public double? ReferenceTrainingMetric { get; init; }

    public required bool DriftFlag { get; init; }

    /// <summary>
    /// View discriminator. <c>"trained"</c> = the row aggregates predictions
    /// by their trained-lead bucket (the existing behaviour, where
    /// <see cref="LeadHours"/> equals the bucket label, e.g. 24 / 48 / 72).
    /// <c>"actual_6h"</c> = the row aggregates predictions by their ACTUAL
    /// lead at prediction time (<c>ValidTimeUtc − PredictionMadeAtUtc</c>),
    /// in 6-hour buckets — needed once predict started emitting 24 hourly
    /// rows per cycle (2026-05-04), which spreads each trained-lead bucket
    /// across actual leads L .. L+23. Defaults to <c>"trained"</c> for
    /// backward compat with sidecars written before this field existed.
    /// </summary>
    public string? ViewKind { get; init; }

    /// <summary>
    /// Actual-lead bucket lower bound (inclusive, hours). Set only on
    /// <see cref="ViewKind"/>=<c>"actual_6h"</c> rows. The bucket spans
    /// <c>[ActualLeadBucketLowH, ActualLeadBucketLowH + 6)</c>.
    /// </summary>
    public int? ActualLeadBucketLowH { get; init; }

    // ---- Phase 3f (rainfall_amount) distributional metrics -------------
    //
    // Populated only on rainfall_amount target rows. The mixed
    // distribution F(x) = (1-π)·δ_0(x) + π·LogNormal(μ_log, σ_log)(x)
    // produced by 3f doesn't fit a single Brier/MAE headline, so verify
    // emits an expanded metric set. <see cref="BlendMetric"/> carries
    // mean CRPS (the primary skill metric); the fields below carry the
    // calibration + reliability companions the site's skill widgets
    // surface alongside it. All nullable so existing producers (temp /
    // precip / dry-window / element) keep emitting clean JSON.

    /// <summary>Mean absolute error on the WET observed rows only —
    /// secondary skill metric. <c>|median_mm − observed_mm|</c> averaged
    /// over rows where <c>observed_mm > 0</c>. Null on non-rainfall_amount
    /// rows.</summary>
    public double? MaeWet { get; init; }

    /// <summary>Empirical coverage of the announced 80% predictive
    /// interval — fraction of observed values in [P10, P90]. Should be
    /// ~0.80 for a well-calibrated forecast. Departures flag
    /// under/over-spread independent of CRPS.</summary>
    public double? Coverage80 { get; init; }

    /// <summary>Mean of the per-row PIT values
    /// <c>(1-π) + π·LogNormalCDF(y; μ, σ)</c> for wet observations, or
    /// the per-row mass at zero for dry obs (uniform sample from
    /// <c>[0, 1-π]</c>). Should be ~0.5 if the distribution is well-
    /// calibrated. Departures flag bias.</summary>
    public double? PitMean { get; init; }

    /// <summary>PIT histogram bin counts (10 equal-width bins on
    /// <c>[0, 1]</c>) — flat for a well-calibrated forecast; bumps at
    /// 0 / 1 flag over/under-spread; tilt flags mean bias. The skill
    /// page renders this as a bar chart so the reader sees the shape
    /// directly. Sums to <see cref="N"/>.</summary>
    public List<int>? PitBins { get; init; }

    // ---- Wave-height (wave_height) band metric ------------------------

    /// <summary>Empirical coverage of the wave blend's calibrated 90%
    /// prediction band — fraction of matched buoy hours with observed Hs ∈
    /// [BandLoM, BandHiM]. Nominal 0.90; the wave verify flags drift when
    /// this lands well below nominal (floor 0.75 by default). Populated only
    /// on <c>wave_height</c> target rows; null everywhere else. (Distinct
    /// from <see cref="Coverage80"/>, which is 3f's 80% interval.)</summary>
    public double? BandCoverage { get; init; }

    /// <summary>Per-threshold Brier scores for the exceedance forecasts
    /// (P(mm/h ≥ threshold) vs the binary observed exceedance). Keys are
    /// stringified thresholds (e.g. <c>"0.1"</c>, <c>"1"</c>, <c>"5"</c>,
    /// <c>"10"</c>); the configured threshold list lives in
    /// <c>BlendersConfig.RainfallAmount["3f"].PredictOutput.ExceedanceMm</c>.
    /// Lower is better, same direction as <see cref="BlendMetric"/>.</summary>
    public Dictionary<string, double>? ExceedanceBriers { get; init; }
}
