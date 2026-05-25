using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Oro;

/// <summary>
/// Phase 3c-oro v2 feature builder: rich (59) + v1 terrain (9) + v2 DEM
/// aggregations (14) = 82 features per row.
///
/// v2 strict-superset of v1 — keeps every v1 terrain feature in place at
/// the same indices, then appends 14 purely-static aggregations driven from
/// the SRTM patch around each site:
///
///   TPI at 4 scales (200 m / 1 km / 5 km / 25 km)        — 4 features
///   Per-sector lee obstruction in 10 km (8 compass dirs) — 8 features
///   mean_slope_5km                                       — 1 feature
///   aspect_dominance_5km                                 — 1 feature
///
/// All 14 are STATIC per station (no NWP-state dependency). They give the
/// model richer topographic context — multi-scale TPI distinguishes
/// "tor on a plateau" from "valley below plateau"; lee obstruction
/// disambiguates the v1 upwind_gain ambiguity ("continuous upslope" vs
/// "ridge then drop"); slope/aspect dominance characterise the regional
/// terrain regime around the site.
///
/// Data flow mirrors <see cref="PrecipRichOroFeatureBuilder"/> — call the
/// v1 builder, then append the new static block per row.
/// </summary>
public static class PrecipRichOroV2FeatureBuilder
{
    public const string SpecFeatureSet = "rich-oro-v2";

    /// <summary>Names of the 14 v2-DEM features, appended after the v1 terrain block.</summary>
    public static readonly string[] V2TerrainFeatureNames =
    {
        "oro_tpi_200m",
        "oro_tpi_1000m",
        "oro_tpi_5000m",
        "oro_tpi_25000m",
        "oro_lee_obstr_n",   "oro_lee_obstr_ne",
        "oro_lee_obstr_e",   "oro_lee_obstr_se",
        "oro_lee_obstr_s",   "oro_lee_obstr_sw",
        "oro_lee_obstr_w",   "oro_lee_obstr_nw",
        "oro_mean_slope_5km",
        "oro_aspect_dominance_5km",
    };

    public const int V2TerrainFeatureCount = 14;

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var v1 = PrecipRichOroFeatureBuilder.BuildSpec(blendersCfg, leadHours);
        var names = v1.FeatureNames.Concat(V2TerrainFeatureNames).ToList();
        return new BlenderSpec
        {
            Target = v1.Target,
            FeatureSet = SpecFeatureSet,
            LeadHours = v1.LeadHours,
            RequiredModels = v1.RequiredModels,
            OptionalModels = v1.OptionalModels,
            Models = v1.Models,
            FeatureNames = names,
            DataSource = v1.DataSource,
            Tier = SpecFeatureSet,
            UkvStrategy = v1.UkvStrategy,
        };
    }

    /// <summary>
    /// Build rich-oro-v2 training rows for one (station, lead). Per-row layout:
    /// rich (59) || v1-terrain (9) || v2-DEM (14) = 82 features.
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        OroStaticFeatures oro,
        int stationIndex,
        BlenderSpec v2Spec,
        CancellationToken ct = default)
    {
        // Trim v2 features off the spec to get the embedded v1 spec.
        var v1Spec = new BlenderSpec
        {
            Target = v2Spec.Target,
            FeatureSet = PrecipRichOroFeatureBuilder.SpecFeatureSet,
            LeadHours = v2Spec.LeadHours,
            RequiredModels = v2Spec.RequiredModels,
            OptionalModels = v2Spec.OptionalModels,
            Models = v2Spec.Models,
            FeatureNames = v2Spec.FeatureNames
                .Take(v2Spec.FeatureNames.Count - V2TerrainFeatureCount).ToList(),
            DataSource = v2Spec.DataSource,
            Tier = PrecipRichOroFeatureBuilder.SpecFeatureSet,
            UkvStrategy = v2Spec.UkvStrategy,
        };

        var v1Rows = PrecipRichOroFeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName, oro, stationIndex, v1Spec, ct);
        if (v1Rows.Count == 0) return v1Rows;

        // Pre-compute the 14 v2 static features once for this site — they don't
        // vary by row. Then copy into every row's vector.
        var v2Block = ComposeV2TerrainBlock(oro);

        var v1Dim = v1Spec.FeatureCount;
        var outDim = v2Spec.FeatureCount;
        if (outDim != v1Dim + V2TerrainFeatureCount)
            throw new InvalidOperationException(
                $"v2 spec dim mismatch: {outDim} != {v1Dim} + {V2TerrainFeatureCount}");

        var rows = new List<BinaryTrainingRow>(v1Rows.Count);
        foreach (var rr in v1Rows)
        {
            ct.ThrowIfCancellationRequested();
            var f = new float[outDim];
            Array.Copy(rr.Features, f, v1Dim);
            Array.Copy(v2Block, 0, f, v1Dim, V2TerrainFeatureCount);
            rows.Add(new BinaryTrainingRow
            {
                ValidTimeUtc = rr.ValidTimeUtc, Features = f, Label = rr.Label, TruthMmHour = rr.TruthMmHour,
            });
        }
        return rows;
    }

    /// <summary>
    /// Pack the 14-feature v2 terrain block from one site's static record.
    /// All features are constants for the site (no NWP-state dependency).
    /// </summary>
    public static float[] ComposeV2TerrainBlock(OroStaticFeatures oro)
    {
        var b = new float[V2TerrainFeatureCount];
        int i = 0;
        // TPI at 4 scales
        b[i++] = (float)Get(oro.TpiByRadiusM, "200");
        b[i++] = (float)Get(oro.TpiByRadiusM, "1000");
        b[i++] = (float)Get(oro.TpiByRadiusM, "5000");
        b[i++] = (float)Get(oro.TpiByRadiusM, "25000");
        // Lee obstruction per sector — match V2TerrainFeatureNames order N..NW.
        foreach (var s in OroStaticFeatures.Sectors)
            b[i++] = (float)Get(oro.LeeObstruction10km, s);
        b[i++] = (float)oro.MeanSlope5km;
        b[i++] = (float)oro.AspectDominance5km;
        return b;
    }

    private static double Get(IReadOnlyDictionary<string, double> map, string key)
        => map.TryGetValue(key, out var v) ? v : 0.0;
}
