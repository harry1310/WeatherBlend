namespace WeatherBlend.Train.Common;

/// <summary>
/// Runtime descriptor for one (target, feature-set, lead) blender. Single source
/// of truth for: which models contribute as required vs optional, and the
/// ordered feature schema fed to LightGBM.
///
/// Three-bucket model membership:
///   * <see cref="RequiredModels"/> — must have data per row; row dropped otherwise.
///   * <see cref="OptionalModels"/> — included as a feature, allowed to be NaN.
///   * Anything else — excluded entirely (no slot, no NaN sentinel).
///
/// <see cref="Models"/> = required ∪ optional in canonical order. That's the set
/// of columns the SQL pivots and the per-model entries that lead the feature vector.
///
/// Built from <see cref="WeatherBlend.Config.BlendersConfig"/> at training/predict
/// time and persisted (per-lead) into <c>feature_schema.json</c> so predict can
/// rebuild the same vector layout.
/// </summary>
public sealed class BlenderSpec
{
    public string Target { get; init; } = "";
    public string FeatureSet { get; init; } = "";
    public int LeadHours { get; init; }

    /// <summary>Models whose values MUST be NOT NULL for the row to survive.</summary>
    public IReadOnlyList<string> RequiredModels { get; init; } = Array.Empty<string>();

    /// <summary>Models whose values are included as features but may be NaN.</summary>
    public IReadOnlyList<string> OptionalModels { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Required ∪ Optional, in canonical order. The SQL pivots one column per
    /// entry; the feature vector packs per-model values in this order before
    /// spread/covariates/calendar.
    /// </summary>
    public IReadOnlyList<string> Models { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Ordered feature names in the float[] vector. Layout is builder-specific
    /// (per-model values first, then spread, then covariates, then calendar).
    /// Persisted in feature_schema.json so train/predict stay lockstep.
    /// </summary>
    public IReadOnlyList<string> FeatureNames { get; init; } = Array.Empty<string>();

    public int FeatureCount => FeatureNames.Count;

    /// <summary>Index of named feature in the vector. Throws if not found.</summary>
    public int IndexOf(string featureName)
    {
        for (int i = 0; i < FeatureNames.Count; i++)
            if (FeatureNames[i] == featureName) return i;
        throw new ArgumentException(
            $"Feature '{featureName}' not in BlenderSpec({Target}/{FeatureSet}/lead={LeadHours}h). " +
            $"Available: [{string.Join(", ", FeatureNames)}]");
    }

    public override string ToString()
        => $"BlenderSpec({Target}/{FeatureSet}/lead={LeadHours}h, " +
           $"required=[{string.Join(",", RequiredModels)}], " +
           $"optional=[{string.Join(",", OptionalModels)}], " +
           $"features={FeatureCount})";
}
