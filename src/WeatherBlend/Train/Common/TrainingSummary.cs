namespace WeatherBlend.Train.Common;

/// <summary>
/// Pre-train data sanity sidecar — captured at fit time and persisted next
/// to <c>training_metadata.json</c> as <c>training_summary.json</c>. Read at
/// the start of the next retrain by <c>RetrainGuard</c> (Phase 1b of
/// AUTO_RETRAIN_PLAN.md, not yet shipped) to detect upstream data shifts
/// before a bad model gets written.
///
/// Ships a flat dictionary of per-feature stats (NaN%, mean, std, p01, p99)
/// + per-station label rates for binary targets, so the guard's tolerance
/// bands (rows ±30%, NaN% absolute 0.20, label-rate delta absolute 0.10) can
/// fire on any column shifting outside its accepted range. Single summary
/// per version (aggregated across leads where the trainer loops); per-lead
/// breakdown stays in <c>training_metadata.json</c>'s PerLead.
/// </summary>
public sealed class TrainingSummary
{
    /// <summary>Schema version for forwards-compat. Bump when adding required fields.</summary>
    public string SchemaVersion { get; set; } = "1";

    /// <summary>Composite this summary belongs to (e.g. "temperature",
    /// "precipitation/ea_bellever_dartmoor", "dry_window/ea_bellever_dartmoor/3h").
    /// Mirrors the key used by ModelSummary on the site so the guard can
    /// resolve the previous summary deterministically.</summary>
    public string Composite { get; set; } = "";

    /// <summary>Phase tag, mirrors training_metadata.Phase ("2b", "3a", etc).</summary>
    public string Phase { get; set; } = "";

    /// <summary>Version string, mirrors training_metadata.Version.</summary>
    public string Version { get; set; } = "";

    /// <summary>UTC timestamp the summary was computed (typically same as
    /// training_metadata.TrainedAtUtc; tracked here to detect summary/metadata
    /// drift if a partial write left them out of sync).</summary>
    public DateTime ComputedAtUtc { get; set; }

    /// <summary>Row counts after split. Sum across leads if the trainer loops.</summary>
    public int RowsTrain { get; set; }
    public int RowsVal { get; set; }
    public int RowsTest { get; set; }

    /// <summary>Number of feature columns the model actually saw — after
    /// dropping all-NaN columns at fit time. Guard tolerance is 0 (any
    /// change = abort) since a feature appearing or disappearing always
    /// signals an upstream schema shift worth investigating.</summary>
    public int FeaturesEffective { get; set; }

    /// <summary>Per-feature stats, keyed by feature name. Computed on the
    /// TRAIN slice only (not val/test) since that's the data the model
    /// actually fit on; val/test are evaluation slices and shifting them
    /// means a different problem. NaN% is the fraction of rows where the
    /// column was missing/NaN before any imputation.</summary>
    public Dictionary<string, FeatureStats> PerFeature { get; set; } = new();

    /// <summary>Per-station label rate for binary targets (precipitation,
    /// dry-window). Empty for regression targets (temperature, element
    /// blenders). Key is the station slug (e.g. "ea_bellever_dartmoor").
    /// For dry-window, also keyed by window length when relevant — see
    /// the builder for the convention used at the call site.</summary>
    public Dictionary<string, double> LabelRates { get; set; } = new();
}

/// <summary>
/// Per-feature train-slice descriptive stats. NaN-aware: <see cref="Mean"/>
/// and <see cref="Std"/> are computed over the non-NaN subset; <see cref="NanPct"/>
/// is the fraction that WAS NaN. Quantiles (<see cref="P01"/>, <see cref="P99"/>)
/// likewise use the non-NaN subset.
/// </summary>
public sealed class FeatureStats
{
    public double NanPct { get; set; }
    public double Mean { get; set; }
    public double Std { get; set; }
    public double P01 { get; set; }
    public double P99 { get; set; }
}
