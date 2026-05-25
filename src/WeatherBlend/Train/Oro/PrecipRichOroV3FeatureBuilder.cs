using WeatherBlend.Config;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Oro;

/// <summary>
/// Phase 3c-oro v3 feature builder: rich (59) + v1 terrain (9) + v2 DEM
/// aggregations (14) + atmospheric climatology lookup (6) = 88 features
/// per row.
///
/// Strict superset of v2. The 6 new features are per-(wind sector × month)
/// climatology values looked up from the static record:
///
///   climo_lapse_850_500   — typical T_500 − T_850 in this regime
///   climo_lapse_700_500   — typical T_500 − T_700
///   climo_q_850           — typical specific humidity at 850 hPa
///   climo_wind_500_speed  — typical mid-trop wind speed
///   climo_shear_850_500   — typical vertical wind shear
///   climo_thickness_proxy — typical 850 hPa temperature
///
/// Computed offline by `build_atm_climatology.py` from 4 years of GFS
/// pressure-level history. At predict time we look up the (sector, month)
/// bin using the NWP-mean wind direction (already in the v1 terrain block
/// at indices `oro_wind_sin` / `oro_wind_cos`) and the valid-time month.
///
/// Captures upper-air atmospheric structure as a per-regime prior, without
/// needing live multi-level NWP — see overnight report for the design
/// rationale and limitations.
/// </summary>
public static class PrecipRichOroV3FeatureBuilder
{
    public const string SpecFeatureSet = "rich-oro-v3";

    public static readonly string[] ClimatologyFeatureNames =
        OroStaticFeatures.ClimatologyFeatureNames;

    public const int ClimatologyFeatureCount = OroStaticFeatures.ClimatologyFeatureCount;

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var v2 = PrecipRichOroV2FeatureBuilder.BuildSpec(blendersCfg, leadHours);
        var names = v2.FeatureNames.Concat(ClimatologyFeatureNames).ToList();
        return new BlenderSpec
        {
            Target = v2.Target,
            FeatureSet = SpecFeatureSet,
            LeadHours = v2.LeadHours,
            RequiredModels = v2.RequiredModels,
            OptionalModels = v2.OptionalModels,
            Models = v2.Models,
            FeatureNames = names,
            DataSource = v2.DataSource,
            Tier = SpecFeatureSet,
            UkvStrategy = v2.UkvStrategy,
        };
    }

    /// <summary>
    /// Build rich-oro-v3 training rows for one (station, lead). Per-row layout:
    /// rich (59) || v1-terrain (9) || v2-DEM (14) || climatology (6) = 88 features.
    /// </summary>
    public static List<BinaryTrainingRow> BuildForLead(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        OroStaticFeatures oro,
        int stationIndex,
        BlenderSpec v3Spec,
        CancellationToken ct = default)
    {
        // Trim climatology features off the spec to get the embedded v2 spec.
        var v2Spec = new BlenderSpec
        {
            Target = v3Spec.Target,
            FeatureSet = PrecipRichOroV2FeatureBuilder.SpecFeatureSet,
            LeadHours = v3Spec.LeadHours,
            RequiredModels = v3Spec.RequiredModels,
            OptionalModels = v3Spec.OptionalModels,
            Models = v3Spec.Models,
            FeatureNames = v3Spec.FeatureNames
                .Take(v3Spec.FeatureNames.Count - ClimatologyFeatureCount).ToList(),
            DataSource = v3Spec.DataSource,
            Tier = PrecipRichOroV2FeatureBuilder.SpecFeatureSet,
            UkvStrategy = v3Spec.UkvStrategy,
        };

        var v2Rows = PrecipRichOroV2FeatureBuilder.BuildForLead(
            forecastsPath, rainfallPath, locationName, stationName, oro, stationIndex, v2Spec, ct);
        if (v2Rows.Count == 0) return v2Rows;

        // Resolve indices of wind sin/cos features in the v2 vector so we can
        // recover wind direction without re-doing the aux SQL pass.
        var idxSin = v2Spec.IndexOf("oro_wind_sin");
        var idxCos = v2Spec.IndexOf("oro_wind_cos");

        var v2Dim = v2Spec.FeatureCount;
        var outDim = v3Spec.FeatureCount;
        if (outDim != v2Dim + ClimatologyFeatureCount)
            throw new InvalidOperationException(
                $"v3 spec dim mismatch: {outDim} != {v2Dim} + {ClimatologyFeatureCount}");

        var rows = new List<BinaryTrainingRow>(v2Rows.Count);
        foreach (var rr in v2Rows)
        {
            ct.ThrowIfCancellationRequested();

            // Recover wind direction from sin/cos. atan2(sin, cos) — same
            // convention as the original NwpMeanRow.
            var sin = rr.Features[idxSin];
            var cos = rr.Features[idxCos];
            double windDirRad;
            if (float.IsNaN(sin) || float.IsNaN(cos))
                windDirRad = double.NaN;
            else
                windDirRad = Math.Atan2(sin, cos);

            var month = rr.ValidTimeUtc.Month;
            var bin = oro.ClimatologyAt(windDirRad, month);

            var f = new float[outDim];
            Array.Copy(rr.Features, f, v2Dim);
            f[v2Dim + 0] = (float)bin.Lapse850_500;
            f[v2Dim + 1] = (float)bin.Lapse700_500;
            f[v2Dim + 2] = (float)bin.Q850;
            f[v2Dim + 3] = (float)bin.Wind500Speed;
            f[v2Dim + 4] = (float)bin.Shear850_500;
            f[v2Dim + 5] = (float)bin.ThicknessProxy;

            rows.Add(new BinaryTrainingRow
            {
                ValidTimeUtc = rr.ValidTimeUtc, Features = f, Label = rr.Label, TruthMmHour = rr.TruthMmHour,
            });
        }
        return rows;
    }
}
