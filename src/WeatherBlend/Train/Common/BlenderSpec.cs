using WeatherBlend.Train.Exact12h;

namespace WeatherBlend.Train.Common;

/// <summary>
/// What kind of forecast pull feeds the blender — one of two canonical
/// data-source pipelines. Persisted as a string field on
/// <see cref="BlenderSpec.DataSource"/> so the renderer (and any
/// future consumer) can read the choice as data rather than guessing
/// from a FeatureSet tag prefix. Strings deliberately stay stable;
/// adding a new source means adding a new constant here, not parsing
/// a new tag.
/// </summary>
public static class BlenderDataSource
{
    /// <summary>Open-Meteo previous_runs API, one row per cycle anchor;
    /// the offset_day RunTimeSource. Used by 2b/2c/3a/3c blenders.</summary>
    public const string OpenMeteoPreviousRuns = "open_meteo_previous_runs";

    /// <summary>Raw S3 cycles from NOAA / ECMWF Open Data / Met Office /
    /// JMA, hourly out to ~T+168h, joined by exact RunTimeUtc. Used by
    /// 2d / 3d exact-runtime blenders.</summary>
    public const string ExactRuntimeS3 = "exact_runtime_s3";
}

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

    /// <summary>
    /// Which forecast pull feeds this blender. One of the constants on
    /// <see cref="BlenderDataSource"/>. Empty string on legacy schemas
    /// trained before this field was added (2026-05-07) — readers should
    /// treat empty as "infer from FeatureSet prefix" for back-compat.
    /// </summary>
    public string DataSource { get; init; } = "";

    /// <summary>
    /// Tier label as named on disk (e.g. "lean", "rich", "T1", "T2",
    /// "P1", "P2"). Mirrors the trailing tier segment in
    /// <see cref="FeatureSet"/> but is the structured-data version —
    /// readers don't have to parse the FeatureSet string. Empty on
    /// legacy schemas.
    /// </summary>
    public string Tier { get; init; } = "";

    /// <summary>
    /// UKV picker strategy when <c>temp_ukv</c> / <c>precip_ukv</c> is in
    /// <see cref="FeatureNames"/>; <c>null</c> when UKV is not used.
    /// Strict (cycles 0/6/12/18Z, lead = target) vs Averaging (cycles
    /// 3/15Z, two leads bracketing target). Decided per-target by
    /// 2026-05-06 bake-off: temp uses Strict, precip uses Averaging.
    /// Persisted as the enum's string name via JsonStringEnumConverter.
    /// </summary>
    public Exact12hFeatureBuilder.UkvPickStrategy? UkvStrategy { get; init; }

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

    /// <summary>
    /// Shared BuildSpec body for every blenders-config-driven builder
    /// (Temp / TempRich / Precip / PrecipRich / DryWindow / Wind / WindGust /
    /// Humidity / Radiation / Cloud). The ten per-builder BuildSpec methods
    /// used to repeat this verbatim; each now keeps only its genuinely
    /// different part — the feature-name composition — as
    /// <paramref name="featureNamer"/>.
    ///
    /// What the factory owns:
    ///   * model membership from <c>config.yaml</c>'s <c>blenders</c> section
    ///     (required/optional per lead, via <see cref="BlendersConfig.Get"/>);
    ///   * the required∩optional overlap guard — a model listed in both is a
    ///     config bug that previously only half the builders caught; it now
    ///     fires uniformly (config-error path only, no data-path change);
    ///   * canonical ordering via <see cref="Nwp.BlenderModelIds"/> and the
    ///     "no models active" guard;
    ///   * the structured-spec defaults shared by all ten: every config-driven
    ///     builder reads the Open-Meteo previous_runs tree
    ///     (<see cref="BlenderDataSource.OpenMeteoPreviousRuns"/>), stamps
    ///     <see cref="Tier"/> = base feature-set, and never uses UKV
    ///     (UKV is exact-runtime only).
    ///
    /// <paramref name="featureSetLabel"/> overrides the stamped
    /// <see cref="FeatureSet"/> only — the precip builders stamp
    /// "lean-ua"/"rich-ua" for the experimental upper-air variants while
    /// <see cref="Tier"/> stays the base set.
    ///
    /// Deliberate non-users of this factory:
    ///   * Exact12hFeatureBuilder / PrecipExactFeatureBuilder — TierSpec-driven
    ///     (not BlendersConfig), DataSource=ExactRuntimeS3, UkvStrategy set;
    ///   * PrecipRichOroFeatureBuilder — derives its spec from PrecipRich's
    ///     rather than from config.
    /// </summary>
    public static BlenderSpec Build(
        WeatherBlend.Config.BlendersConfig blendersCfg,
        string target,
        string featureSet,
        int leadHours,
        Func<IReadOnlyList<string>, List<string>> featureNamer,
        string? featureSetLabel = null)
    {
        var blender = blendersCfg.Get(target, featureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var overlap = requiredSet.Intersect(optionalSet, StringComparer.OrdinalIgnoreCase).ToArray();
        if (overlap.Length > 0)
            throw new InvalidOperationException(
                $"BlenderConfig {target}/{featureSet} at lead {leadHours}h has model(s) " +
                $"listed as both required and optional: [{string.Join(",", overlap)}].");

        var orderedRequired = Nwp.BlenderModelIds.Where(requiredSet.Contains).ToList();
        var orderedOptional = Nwp.BlenderModelIds.Where(optionalSet.Contains).ToList();
        var orderedModels = Nwp.BlenderModelIds
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {target}/{featureSet} at lead {leadHours}h.");

        return new BlenderSpec
        {
            Target = target,
            FeatureSet = featureSetLabel ?? featureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = featureNamer(orderedModels),
            DataSource = BlenderDataSource.OpenMeteoPreviousRuns,
            Tier = featureSet,
            UkvStrategy = null,
        };
    }
}
