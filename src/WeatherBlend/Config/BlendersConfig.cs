namespace WeatherBlend.Config;

/// <summary>
/// Per-blender model membership configuration. One entry per (target,
/// feature-set) blender family. Each entry expresses three buckets per lead:
///
///   * <see cref="BlenderConfig.RequiredModels"/> — must have data for the row
///     to survive (post-pivot WHERE: every entry IS NOT NULL).
///   * <see cref="BlenderConfig.OptionalModels"/> — included as a feature, but
///     allowed to be NaN at the row level (LightGBM handles NaN natively).
///   * Anything else — excluded entirely from the feature vector. No slot,
///     no NaN sentinel.
///
/// A single implicit safety check still applies: at least one of
/// <c>RequiredModels ∪ OptionalModels</c> must be non-null per row. Otherwise
/// the feature vector would be all-NaN, contributing nothing to training.
///
/// <see cref="BlenderConfig.PerLeadOverrides"/> replaces both lists for a
/// specific lead — used where data availability differs (MF caps at 72h;
/// UKMO data starts 2024-09; etc.).
/// </summary>
public sealed class BlendersConfig
{
    public List<BlenderConfig> Items { get; set; } = new();

    public BlenderConfig Get(string target, string featureSet)
    {
        var match = Items.FirstOrDefault(b =>
            string.Equals(b.Target, target, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(b.FeatureSet, featureSet, StringComparison.OrdinalIgnoreCase));
        if (match is null)
            throw new InvalidOperationException(
                $"No blender config for target='{target}' featureSet='{featureSet}'. " +
                $"Available: [{string.Join(", ", Items.Select(b => $"{b.Target}/{b.FeatureSet}"))}]");
        return match;
    }
}

public sealed class BlenderConfig
{
    public string Target { get; set; } = "";
    public string FeatureSet { get; set; } = "";

    /// <summary>Required at every lead unless overridden. Row dropped if any one is NULL.</summary>
    public List<string> RequiredModels { get; set; } = new();

    /// <summary>Optional at every lead unless overridden. Allowed to be NaN at the row level.</summary>
    public List<string> OptionalModels { get; set; } = new();

    /// <summary>Per-lead overrides — replace both lists for a specific lead.</summary>
    public List<LeadOverrideConfig> PerLeadOverrides { get; set; } = new();

    /// <summary>Resolved required-model list for a lead (override beats default).</summary>
    public IReadOnlyList<string> RequiredForLead(int leadHours)
    {
        var ovr = PerLeadOverrides.FirstOrDefault(o => o.Lead == leadHours);
        return ovr?.RequiredModels ?? RequiredModels;
    }

    /// <summary>Resolved optional-model list for a lead (override beats default).</summary>
    public IReadOnlyList<string> OptionalForLead(int leadHours)
    {
        var ovr = PerLeadOverrides.FirstOrDefault(o => o.Lead == leadHours);
        return ovr?.OptionalModels ?? OptionalModels;
    }
}

public sealed class LeadOverrideConfig
{
    public int Lead { get; set; }
    public List<string> RequiredModels { get; set; } = new();
    public List<string> OptionalModels { get; set; } = new();
}
