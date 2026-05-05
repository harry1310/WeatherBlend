using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One blended-temperature prediction with full provenance. Stored under
/// <c>data/predictions/temperature/model_version={v}/date={yyyy-MM-dd}/predictions.parquet</c>
/// so verification can later stratify rolling metrics by model version.
///
/// Every field here is a deliberate provenance record — "our forecast missed" is
/// only investigable if we can point at exactly which cycle of which model fed the
/// blender and what feature vector it saw.
/// </summary>
public sealed class TempPredictionRow
{
    /// <summary>
    /// Number of named per-model output slots on this row type
    /// (TempGfs, TempEcmwf, TempIcon, TempMf, TempUkmo, TempGem, TempAifs).
    /// Predict pipelines size their per-model output arrays from this and
    /// use the same constant for the "skip ci ≥ N" guard so future model
    /// additions to <see cref="WeatherBlend.Train.TempFeatureBuilder.CanonicalModelOrder"/>
    /// don't silently drop data — failing the build via missing-field is a
    /// clearer signal than an out-of-range silent skip.
    /// </summary>
    public const int PerModelFieldCount = 7;

    // ParquetSerializer.DeserializeAsync<T>() needs new() — see CLAUDE.md gotcha.
    // Seeding required strings matches the Era5Row/ObservationRow pattern and keeps
    // the nullable analyzer quiet.
    [SetsRequiredMembers]
    public TempPredictionRow()
    {
        LocationName = "";
        ModelVersion = "";
        FeatureVectorHash = "";
    }

    public required string LocationName { get; init; }
    public required string ModelVersion { get; init; }

    /// <summary>Wall-clock UTC when the predict job ran. Distinct from per-model cycle times.</summary>
    public required DateTime PredictionMadeAtUtc { get; init; }

    /// <summary>What valid-time this forecast predicts.</summary>
    public required DateTime ValidTimeUtc { get; init; }

    /// <summary>Nominal lead (24, 48, 72). Matches the per-lead blender that produced this row.</summary>
    public required int LeadHours { get; init; }

    /// <summary>The blender's output — primary deliverable.</summary>
    public required double BlendTemperature { get; init; }

    // Per-model inputs. Verification computes baselines (best-single, mean-of-models)
    // from these directly without needing to re-read the forecast tree.
    public double? TempGfs { get; init; }
    public double? TempEcmwf { get; init; }
    public double? TempIcon { get; init; }
    public double? TempMf { get; init; }
    public double? TempUkmo { get; init; }
    public double? TempGem { get; init; }
    public double? TempAifs { get; init; }

    // Which cycle each per-model temperature came from. Different NWPs post at
    // different cadences (ECMWF 2x/day, GFS/ICON/GEM 4x/day, ...) and latencies
    // differ, so at any moment "latest available" spans multiple cycles. Recording
    // each avoids future wonder about why a Tuesday prediction used Monday's 12z
    // ECMWF alongside Tuesday's 06z GFS.
    public DateTime? RunTimeGfs { get; init; }
    public DateTime? RunTimeEcmwf { get; init; }
    public DateTime? RunTimeIcon { get; init; }
    public DateTime? RunTimeMf { get; init; }
    public DateTime? RunTimeUkmo { get; init; }
    public DateTime? RunTimeGem { get; init; }
    public DateTime? RunTimeAifs { get; init; }

    public double? TempMean { get; init; }
    public double? TempStd { get; init; }
    public double? TempRange { get; init; }

    // Exact-runtime per-model slots (Phase 2d). Populated by the exact-runtime
    // predict path only — 2b/2c rows leave these null. Distinct from the
    // offset_day TempGfs / TempEcmwf / ... slots above so a single parquet can
    // carry both phase shapes side-by-side without overwriting each other.
    // Model identities don't fully overlap (IFS oper vs IFS025, MO Global vs
    // ukmo_seamless), so the per-NWP picker dispatches by phase rather than
    // sharing slots. UKV is exact-runtime-only and hence has no offset_day twin.
    public double? TempGfsExact { get; init; }
    public double? TempIfsOperExact { get; init; }
    public double? TempAifsOperExact { get; init; }
    public double? TempMoGlobalExact { get; init; }
    public double? TempUkvExact { get; init; }

    public DateTime? RunTimeGfsExact { get; init; }
    public DateTime? RunTimeIfsOperExact { get; init; }
    public DateTime? RunTimeAifsOperExact { get; init; }
    public DateTime? RunTimeMoGlobalExact { get; init; }
    public DateTime? RunTimeUkvExact { get; init; }

    /// <summary>SHA-256 hex of the 13 feature floats (little-endian, in FeatureNames order).
    /// Lets us prove, post-hoc, that a given prediction corresponds to a specific input.</summary>
    public required string FeatureVectorHash { get; init; }
}
