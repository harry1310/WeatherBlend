using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Oro;

/// <summary>
/// v4 feature builder: rich (59) + terrain (9) + N interaction features.
///
/// Wraps <see cref="PrecipRichOroFeatureBuilder"/> and, after the standard 68
/// features are packed, appends interaction features computed as products
/// (or other simple functions) of two existing feature columns by index.
///
/// The interaction list is data-driven from the SHAP analysis of the v3
/// pooled-oro model — see <see cref="Interactions"/>. The list is filled in
/// post-SHAP; until then it's empty and the builder produces the same
/// 68-feature vector as PrecipRichOroFeatureBuilder.
///
/// Restricting interactions to feature-feature products (rather than arbitrary
/// non-linear forms) means LightGBM gets exactly what its tree splits CAN'T
/// represent directly — a single feature that captures "feature A high AND
/// feature B high" as a continuous gradient rather than via a deep tree path.
/// </summary>
public static class PrecipRichOroInteractFeatureBuilder
{
    public const string SpecFeatureSet = "rich-oro-interact";

    /// <summary>
    /// Interaction definitions — populated from SHAP findings on the v3
    /// pooled-oro bundle. Each entry: (featureA_name, featureB_name, op, output_name).
    /// Names are matched against the rich-oro feature vector's <see cref="BlenderSpec.FeatureNames"/>.
    /// </summary>
    public sealed record InteractionDef(string A, string B, InteractionOp Op, string OutputName);

    public enum InteractionOp
    {
        Product,     // a * b
        AbsProduct,  // |a * b|
        ProductPos,  // max(0, a * b)
    }

    /// <summary>
    /// SHAP-derived interaction list, populated 2026-05-25 from v3 bake-off
    /// SHAP analysis on the pooled-oro model at lead 24h. Top SHAP NWP
    /// features were precip_mean / precip_agreement_wet_01 / wind_speed_mean
    /// / dew_depression_mean / precip_aifs / cape_mean; top SHAP terrain
    /// features were oro_relief_5km_m / oro_wind_cos / oro_wind_sin (the
    /// other 6 terrain features dropped out of the top 20).
    ///
    /// Interactions pair top NWP × top terrain (4 features) and add two
    /// physics-motivated pairs (uplift × rh, upwind_gain × wind_speed) that
    /// would be hard for tree splits to discover from their marginal effects
    /// alone. 6 interactions → 74-feature vector for the v4 arm.
    /// </summary>
    public static readonly IReadOnlyList<InteractionDef> Interactions = new List<InteractionDef>
    {
        new("precip_mean",                  "oro_relief_5km_m",              InteractionOp.Product, "ix_precip_x_relief"),
        new("precip_mean",                  "oro_wind_cos",                  InteractionOp.Product, "ix_precip_x_wind_cos"),
        new("precip_agreement_wet_01",      "oro_relief_5km_m",              InteractionOp.Product, "ix_agree_x_relief"),
        new("wind_speed_mean",              "oro_relief_5km_m",              InteractionOp.Product, "ix_windspd_x_relief"),
        new("oro_upwind_gain_per_wind_5km_m","wind_speed_mean",              InteractionOp.Product, "ix_upwindgain_x_wind"),
        new("oro_uplift_m_per_s",           "rh_mean",                       InteractionOp.Product, "ix_uplift_x_rh"),
    };

    public static int InteractionCount => Interactions.Count;

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var oro = PrecipRichOroFeatureBuilder.BuildSpec(blendersCfg, leadHours);
        var names = oro.FeatureNames.ToList();
        names.AddRange(Interactions.Select(ix => ix.OutputName));
        return new BlenderSpec
        {
            Target = oro.Target,
            FeatureSet = SpecFeatureSet,
            LeadHours = oro.LeadHours,
            RequiredModels = oro.RequiredModels,
            OptionalModels = oro.OptionalModels,
            Models = oro.Models,
            FeatureNames = names,
            DataSource = oro.DataSource,
            Tier = SpecFeatureSet,
            UkvStrategy = oro.UkvStrategy,
        };
    }

    /// <summary>
    /// Build rich-oro-interact training rows for one (station, lead). Per-row
    /// feature layout = rich vector (59) || terrain (9) || interactions (N).
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        OroStaticFeatures oro,
        int stationIndex,
        BlenderSpec richOroInteractSpec,
        CancellationToken ct = default)
    {
        // Materialise a matching rich-oro spec by trimming the trailing
        // interaction names off the interact spec — pass that down.
        var richOroSpec = new BlenderSpec
        {
            Target = richOroInteractSpec.Target,
            FeatureSet = PrecipRichOroFeatureBuilder.SpecFeatureSet,
            LeadHours = richOroInteractSpec.LeadHours,
            RequiredModels = richOroInteractSpec.RequiredModels,
            OptionalModels = richOroInteractSpec.OptionalModels,
            Models = richOroInteractSpec.Models,
            FeatureNames = richOroInteractSpec.FeatureNames
                .Take(richOroInteractSpec.FeatureNames.Count - InteractionCount).ToList(),
            DataSource = richOroInteractSpec.DataSource,
            Tier = PrecipRichOroFeatureBuilder.SpecFeatureSet,
            UkvStrategy = richOroInteractSpec.UkvStrategy,
        };

        var oroRows = PrecipRichOroFeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName, oro, stationIndex, richOroSpec, ct);
        if (oroRows.Count == 0 || InteractionCount == 0)
        {
            // Pad rows up to the larger feature length if needed.
            if (InteractionCount == 0) return oroRows;
            var padded = new List<BinaryTrainingRow>(oroRows.Count);
            foreach (var rr in oroRows)
            {
                var f = new float[richOroInteractSpec.FeatureCount];
                Array.Copy(rr.Features, f, rr.Features.Length);
                padded.Add(new BinaryTrainingRow
                {
                    ValidTimeUtc = rr.ValidTimeUtc, Features = f, Label = rr.Label, TruthMmHour = rr.TruthMmHour,
                });
            }
            return padded;
        }

        // Resolve interaction-feature indices once (against the rich-oro spec).
        var indices = new (int IdxA, int IdxB, InteractionOp Op)[InteractionCount];
        for (int i = 0; i < InteractionCount; i++)
        {
            var ix = Interactions[i];
            indices[i] = (richOroSpec.IndexOf(ix.A), richOroSpec.IndexOf(ix.B), ix.Op);
        }

        var rows = new List<BinaryTrainingRow>(oroRows.Count);
        var outDim = richOroInteractSpec.FeatureCount;
        var oroDim = richOroSpec.FeatureCount;
        foreach (var rr in oroRows)
        {
            ct.ThrowIfCancellationRequested();
            var f = new float[outDim];
            Array.Copy(rr.Features, f, oroDim);
            for (int i = 0; i < InteractionCount; i++)
            {
                var (ia, ib, op) = indices[i];
                var a = rr.Features[ia];
                var b = rr.Features[ib];
                var prod = a * b;
                f[oroDim + i] = op switch
                {
                    InteractionOp.AbsProduct => Math.Abs(prod),
                    InteractionOp.ProductPos => Math.Max(0f, prod),
                    _                         => prod,
                };
            }
            rows.Add(new BinaryTrainingRow
            {
                ValidTimeUtc = rr.ValidTimeUtc, Features = f, Label = rr.Label, TruthMmHour = rr.TruthMmHour,
            });
        }
        return rows;
    }
}
