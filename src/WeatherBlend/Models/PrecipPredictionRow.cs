using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One Phase 3a P(wet) prediction with full provenance. Stored under
/// <c>data/predictions/precipitation/{truth_station}/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>
/// — one subtree per truth station because each station has its own blender and
/// its own skill profile.
///
/// Per-model inputs + run-times are carried through so verification can compute
/// baselines (mean-of-models, best-single) from a prediction row alone, without
/// re-reading the forecast tree for that valid-time.
/// </summary>
public sealed class PrecipPredictionRow
{
    /// <summary>
    /// Number of named per-model precip slots (PrecipGfs..PrecipJma).
    /// See <see cref="TempPredictionRow.PerModelFieldCount"/> for the rationale.
    /// </summary>
    public const int PerModelFieldCount = 8;

    [SetsRequiredMembers]
    public PrecipPredictionRow()
    {
        LocationName = "";
        TruthStation = "";
        ModelVersion = "";
        FeatureVectorHash = "";
    }

    public required string LocationName { get; init; }

    /// <summary>Truth source this blender was trained against, e.g. <c>ea_bellever_dartmoor</c>.</summary>
    public required string TruthStation { get; init; }

    public required string ModelVersion { get; init; }

    public required DateTime PredictionMadeAtUtc { get; init; }
    public required DateTime ValidTimeUtc { get; init; }
    public required int LeadHours { get; init; }

    /// <summary>Blender-calibrated P(hour ≥ 0.1mm). Primary deliverable.</summary>
    public required double ProbWet { get; init; }

    /// <summary>Training-derived climatology P(wet) for this (month, hour-of-day). Skill-score reference.</summary>
    public required double ClimatologyPWet { get; init; }

    // Per-model inputs — nullable because a model may not publish precip for this lead.
    public double? PrecipGfs { get; init; }
    public double? PrecipEcmwf { get; init; }
    public double? PrecipIcon { get; init; }
    public double? PrecipMf { get; init; }
    public double? PrecipUkmo { get; init; }
    public double? PrecipGem { get; init; }
    public double? PrecipAifs { get; init; }
    public double? PrecipJma { get; init; }

    // ProbXxxModel fields removed 2026-04-28 — every prob_* training feature
    // had 0.000 gain at every lead (Open-Meteo's precipitation_probability adds
    // nothing the trees can't infer from the raw precipitation rate). Old
    // prediction parquets that still contain these columns are read by the new
    // verify/site SQL via union_by_name + explicit column list, so the now-
    // unselected columns are simply ignored.

    public DateTime? RunTimeGfs { get; init; }
    public DateTime? RunTimeEcmwf { get; init; }
    public DateTime? RunTimeIcon { get; init; }
    public DateTime? RunTimeMf { get; init; }
    public DateTime? RunTimeUkmo { get; init; }
    public DateTime? RunTimeGem { get; init; }
    public DateTime? RunTimeAifs { get; init; }
    public DateTime? RunTimeJma { get; init; }

    // Exact-runtime per-model slots (Phase 3d, 2026-05-05+). Populated by the
    // exact-runtime predict path only — 3a/3c rows leave these null. Distinct
    // from the offset_day Precip* slots above so a single parquet can carry
    // both phase shapes side-by-side. Model identities don't fully overlap
    // (IFS oper vs IFS025, MO Global vs ukmo_seamless, plus UKV which has no
    // offset_day twin), so the per-NWP picker dispatches by phase rather
    // than sharing slots — same architecture as TempPredictionRow's
    // *Exact slots.
    public double? PrecipGfsExact { get; init; }
    public double? PrecipIfsOperExact { get; init; }
    public double? PrecipAifsOperExact { get; init; }
    public double? PrecipMoGlobalExact { get; init; }
    public double? PrecipUkvExact { get; init; }

    public DateTime? RunTimeGfsExact { get; init; }
    public DateTime? RunTimeIfsOperExact { get; init; }
    public DateTime? RunTimeAifsOperExact { get; init; }
    public DateTime? RunTimeMoGlobalExact { get; init; }
    public DateTime? RunTimeUkvExact { get; init; }

    // Ensemble spread aggregates — same definitions as the training feature row.
    public double? PrecipMean { get; init; }
    public double? PrecipStd { get; init; }
    public double? PrecipMax { get; init; }
    public double? PrecipAgreementWet01 { get; init; }

    /// <summary>SHA-256 hex of the 27 feature floats (little-endian, in OccurrenceFeatureNames order).</summary>
    public required string FeatureVectorHash { get; init; }

    /// <summary>
    /// Conformal-prediction set tag {"Dry", "Wet", "Ambiguous"} when the
    /// version dir has a fitted conformal calibrator (precip-conformal-fit).
    /// "Ambiguous" flags rows where the model's P(wet) sits inside the
    /// 90%-coverage ambiguity zone — calibrated alternative to the per-NWP
    /// agreement chip. Null on legacy rows / versions without a calibrator.
    /// </summary>
    public string? ConformalSetTag { get; init; }

    // ---- Bayesian-uncertainty columns (added 2026-05-09) -------------------
    // Persisted by phases that compute a posterior distribution per row:
    // 4a (dbarts BART) and 5a (Bayesian logreg). Null on 3a/3c/3d/2x rows
    // and on pre-2026-05-09 4a rows. Renderers should treat null as
    // "uncertainty signal not available" and skip the band.
    public double? ProbWetStd { get; init; }
    public double? ProbWetQ05 { get; init; }
    public double? ProbWetQ95 { get; init; }
    public double? Ci80Width  { get; init; }
    public double? Ci90Width  { get; init; }
}
