using DuckDB.NET.Data;
using WeatherBlend.Config;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;
using WeatherBlend.Train.Truth;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// SQL pivot + row composition for the <c>wind_speed_lgb</c> blender —
/// Phase 3 of WIND_BLENDER_PLAN (locked 2026-05-27). Same physical target
/// as <see cref="WindFeatureBuilder"/> (10 m wind speed) but with two
/// substantive differences:
///
///   1. Truth is **Dunkeswell SYNOP** (MIDAS Open station 01383), not
///      ERA5. Loaded via <see cref="MidasOpenLoader"/> and joined to the
///      pivoted forecast tree in-memory (no SQL join — MIDAS isn't a
///      parquet tree).
///   2. Features are the **29-feature lean+ORO+spread** set the
///      production-scope bake-off (<c>scripts/wind_speed_production_scope_bakeoff.py</c>,
///      2026-05-27) found wins at 0.9591 m/s MAE on 1,163 Dunkeswell
///      test rows (vs 1.1454 m/s NWP-mean, −16.3%).
///
/// Feature layout per N active models (N=6 in production scope):
///   N per-NWP <c>WindSpeed10m</c>
///   N per-NWP <c>WindDirection10m</c> (RAW degrees, not sin/cos — the
///       bake-off used raw degrees; ORO features below carry the
///       directional information via the NWP-mean direction)
///   5 ORO_LEAN dynamic terrain features (computed per-row from
///       NWP-mean wind direction + dewpoint + pressure + the static
///       <see cref="OroStaticFeatures"/> for the location):
///       oro_wind_sin, oro_wind_cos, oro_upwind_gain, oro_uplift, oro_uplift_x_q
///   12 SPREAD features (mean + std over each of 6 vars):
///       wsp_spd_{mean,std}, gust_spd_{mean,std}, t_spd_{mean,std},
///       td_spd_{mean,std}, p_spd_{mean,std}, cc_spd_{mean,std}
///
/// Total = 2N + 5 + 12. For N=6 → 29 features.
///
/// Model membership comes from <c>blenders.wind_speed_lgb</c> in
/// config.yaml. AIFS is excluded entirely (no 2024 rows for the
/// Dunkeswell training window); MétéoFrance excluded everywhere (no MF
/// wind on Open-Meteo Previous Runs). UKMO sits in <c>optionalModels</c>
/// under the same <see cref="TrainingWindow.UkmoCleanWindowStart"/>
/// floor the existing wind blender uses.
/// </summary>
public static class WindSpeedLgbFeatureBuilder
{
    public const string SpecTarget = "wind_speed_lgb";
    public const string SpecFeatureSet = "default";

    /// <summary>Dunkeswell aerodrome MIDAS Open station id. Hard-coded here
    /// because the wind_speed_lgb blender's truth source is fixed — there
    /// is no per-location MIDAS station selector (production blender is
    /// Bonehill-only and Dunkeswell is the only SYNOP within Devon with
    /// hourly archive coverage 2022–2024).</summary>
    public const string DunkeswellStationId = "01383";

    /// <summary>MIDAS years to pull when loading truth. Matches the
    /// hard-coded <c>DUNKESWELL_YEARS</c> in <c>train_wind_mvn.py</c>.</summary>
    public static readonly int[] DunkeswellYears = new[] { 2022, 2023, 2024 };

    /// <summary>The 6 cross-NWP variables that drive the SPREAD block,
    /// in canonical order. The Open-Meteo column name on the left is what
    /// the SQL pivot extracts; the short label on the right names the
    /// resulting <c>{label}_mean</c> / <c>{label}_std</c> feature pair.</summary>
    private static readonly (string Column, string Label)[] SpreadVars = new[]
    {
        ("WindSpeed10m",    "wsp_spd"),
        ("WindGusts10m",    "gust_spd"),
        ("Temperature2m",   "t_spd"),
        ("DewPoint2m",      "td_spd"),
        ("SurfacePressure", "p_spd"),
        ("CloudCover",      "cc_spd"),
    };

    /// <summary>ORO_LEAN feature names in canonical order. Mirrors the
    /// Python bake-off's <c>ORO_LEAN</c> list (same names, same order)
    /// so the bake-off's importance / SHAP output is portable.</summary>
    public static readonly string[] OroLeanNames = new[]
    {
        "oro_wind_sin", "oro_wind_cos", "oro_upwind_gain", "oro_uplift", "oro_uplift_x_q",
    };

    public static BlenderSpec BuildSpec(BlendersConfig blendersCfg, int leadHours)
    {
        var blender = blendersCfg.Get(SpecTarget, SpecFeatureSet);
        var requiredSet = blender.RequiredForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = blender.OptionalForLead(leadHours).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var orderedRequired = TempFeatureBuilder.CanonicalModelOrder.Where(m => requiredSet.Contains(m)).ToList();
        var orderedOptional = TempFeatureBuilder.CanonicalModelOrder.Where(m => optionalSet.Contains(m)).ToList();
        var orderedModels = TempFeatureBuilder.CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m)).ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"No models active for {SpecTarget}/{SpecFeatureSet} at lead {leadHours}h.");

        var names = new List<string>();
        foreach (var m in orderedModels) names.Add($"wsp_{TempFeatureBuilder.ShortName(m)}");
        foreach (var m in orderedModels) names.Add($"wdir_{TempFeatureBuilder.ShortName(m)}");
        names.AddRange(OroLeanNames);
        foreach (var (_, label) in SpreadVars)
        {
            names.Add($"{label}_mean");
            names.Add($"{label}_std");
        }

        return new BlenderSpec
        {
            Target = SpecTarget,
            FeatureSet = SpecFeatureSet,
            LeadHours = leadHours,
            RequiredModels = orderedRequired,
            OptionalModels = orderedOptional,
            Models = orderedModels,
            FeatureNames = names,
            DataSource = BlenderDataSource.OpenMeteoPreviousRuns,
            Tier = SpecFeatureSet,
            UkvStrategy = null,
        };
    }

    public static List<RegressionTrainingRow> BuildForLead(
        string forecastsPath,
        string midasRoot,
        string locationName,
        OroStaticFeatures oro,
        BlenderSpec spec,
        CancellationToken ct = default,
        string? floorOverride = null)
    {
        // Load Dunkeswell truth into a {ValidTimeUtc → wsp_ms} dictionary
        // BEFORE running the forecast pivot. Cheaper to do the inner join
        // in-memory than to push MIDAS into a DuckDB temp table — the
        // archive is ~30k SYNOP rows × 3 years.
        var midas = MidasOpenLoader.LoadStationWindByYear(midasRoot, DunkeswellStationId, DunkeswellYears);

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var n = spec.Models.Count;

        // No training-window floor by default: floor bake-off (2026-06-09) + the earlier
        // no-floor validation showed dropping the old 2024-09 floor is fully usable and
        // gives ~2.5× the data (Feb–Dec 2024). UKMO is optional + NaN-tolerated.
        // floorOverride re-imposes a floor if ever needed.
        var floorClause = floorOverride is null ? "" : $"AND ValidTimeUtc >= TIMESTAMP '{floorOverride}'";

        // 7 per-NWP variables pulled in one pivot pass: wsp + wdir feed
        // features directly, gust + t + td + p + cc feed the SPREAD block
        // (their NWP-mean/std end up in features but the per-NWP slots
        // don't). td + p also flow into the ORO uplift_x_q feature via
        // the NWP-mean.
        var pivotWsp  = PivotClause(spec.Models, "WindSpeed10m",     "wsp");
        var pivotWdir = PivotClause(spec.Models, "WindDirection10m", "wdir");
        var pivotGust = PivotClause(spec.Models, "WindGusts10m",     "gust");
        var pivotT    = PivotClause(spec.Models, "Temperature2m",    "t");
        var pivotTd   = PivotClause(spec.Models, "DewPoint2m",       "td");
        var pivotP    = PivotClause(spec.Models, "SurfacePressure",  "p");
        var pivotCc   = PivotClause(spec.Models, "CloudCover",       "cc");

        string Project(string prefix) => string.Join(", ",
            spec.Models.Select(m => $"p.{prefix}_{TempFeatureBuilder.ShortName(m)}"));

        // Required: per-NWP wsp + wdir for every required model (same
        // policy as WindFeatureBuilder — these two are load-bearing).
        // Optional NWPs may be NaN for both. Gust/t/td/p/cc are NaN-tolerant
        // anywhere because they only feed SPREAD aggregates (and oro's
        // td/p NWP-mean).
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m =>
                $"p.wsp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL " +
                $"AND p.wdir_{TempFeatureBuilder.ShortName(m)} IS NOT NULL"))
            : "TRUE";
        var anyWspNotNull = "(" + string.Join(" OR ", spec.Models.Select(m =>
            $"p.wsp_{TempFeatureBuilder.ShortName(m)} IS NOT NULL")) + ")";

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           WindSpeed10m, WindDirection10m, WindGusts10m,
           Temperature2m, DewPoint2m, SurfacePressure, CloudCover,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND RunTimeSource = 'offset_day'
      AND LeadHours = {spec.LeadHours}
      AND WindSpeed10m IS NOT NULL
      AND WindDirection10m IS NOT NULL
      {floorClause}
      AND Model IN {modelInClause}
),
pivoted AS (
    SELECT ValidTimeUtc,
        {pivotWsp},
        {pivotWdir},
        {pivotGust},
        {pivotT},
        {pivotTd},
        {pivotP},
        {pivotCc}
    FROM latest WHERE rn = 1 GROUP BY ValidTimeUtc
)
SELECT p.ValidTimeUtc, {Project("wsp")}, {Project("wdir")},
       {Project("gust")}, {Project("t")}, {Project("td")},
       {Project("p")}, {Project("cc")}
FROM pivoted p
WHERE ({requiredNotNull})
  AND {anyWspNotNull}
ORDER BY p.ValidTimeUtc;";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var wsp  = new double[n];
        var wdir = new double[n];
        var gust = new double[n];
        var t    = new double[n];
        var td   = new double[n];
        var p    = new double[n];
        var cc   = new double[n];
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            // Inner-join against MIDAS truth in-memory. Rows without a
            // truth observation get dropped — matches the Python pipe.
            if (!midas.TryGetValue(valid, out var truth)) continue;

            int col = 1;
            ReadBlock(r, ref col, wsp,  n);
            ReadBlock(r, ref col, wdir, n);
            ReadBlock(r, ref col, gust, n);
            ReadBlock(r, ref col, t,    n);
            ReadBlock(r, ref col, td,   n);
            ReadBlock(r, ref col, p,    n);
            ReadBlock(r, ref col, cc,   n);

            rows.Add(ComposeRow(spec, valid, wsp, wdir, gust, t, td, p, cc, oro, truth.WindSpeedMs));
        }
        return rows;
    }

    /// <summary>Test-friendly composition: given the pivoted per-NWP
    /// values for one ValidTime, build a 29-feature row. The truth Wsp
    /// becomes the regression label.</summary>
    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> wspMs,
        IReadOnlyList<double> wdirDeg,
        IReadOnlyList<double> gustMs,
        IReadOnlyList<double> tempC,
        IReadOnlyList<double> dewC,
        IReadOnlyList<double> pressureHpa,
        IReadOnlyList<double> cloudPct,
        OroStaticFeatures oro,
        double dunkeswellWspMs)
    {
        var n = spec.Models.Count;
        if (wspMs.Count != n)         throw new ArgumentException($"Expected {n} wsp", nameof(wspMs));
        if (wdirDeg.Count != n)       throw new ArgumentException($"Expected {n} wdir", nameof(wdirDeg));
        if (gustMs.Count != n)        throw new ArgumentException($"Expected {n} gust", nameof(gustMs));
        if (tempC.Count != n)         throw new ArgumentException($"Expected {n} temp", nameof(tempC));
        if (dewC.Count != n)          throw new ArgumentException($"Expected {n} dew", nameof(dewC));
        if (pressureHpa.Count != n)   throw new ArgumentException($"Expected {n} pressure", nameof(pressureHpa));
        if (cloudPct.Count != n)      throw new ArgumentException($"Expected {n} cloud", nameof(cloudPct));

        // NWP-mean wind components for the ORO computation. Sin/cos are
        // averaged first then atan2'd, otherwise wrap at 0/360 would skew
        // the mean toward 180° around northerly winds.
        double wspMeanSum = 0, sinSum = 0, cosSum = 0, tdSum = 0, pSum = 0;
        int wspPresent = 0, dirPresent = 0, tdPresent = 0, pPresent = 0;
        for (int i = 0; i < n; i++)
        {
            if (!double.IsNaN(wspMs[i])) { wspMeanSum += wspMs[i]; wspPresent++; }
            if (!double.IsNaN(wdirDeg[i]))
            {
                var rad = wdirDeg[i] * Math.PI / 180.0;
                sinSum += Math.Sin(rad);
                cosSum += Math.Cos(rad);
                dirPresent++;
            }
            if (!double.IsNaN(dewC[i]))       { tdSum += dewC[i]; tdPresent++; }
            if (!double.IsNaN(pressureHpa[i])){ pSum += pressureHpa[i]; pPresent++; }
        }
        var wspMean = wspPresent == 0 ? double.NaN : wspMeanSum / wspPresent;
        var sinMean = dirPresent == 0 ? double.NaN : sinSum / dirPresent;
        var cosMean = dirPresent == 0 ? double.NaN : cosSum / dirPresent;
        var tdMean  = tdPresent  == 0 ? double.NaN : tdSum / tdPresent;
        var pMean   = pPresent   == 0 ? double.NaN : pSum / pPresent;

        var oroLean = OroDynamic(wspMean, sinMean, cosMean, tdMean, pMean, oro);

        // SPREAD over 6 vars, in the order locked by SpreadVars.
        var spreadFeatures = new float[2 * SpreadVars.Length];
        for (int v = 0; v < SpreadVars.Length; v++)
        {
            var src = SpreadVars[v].Column switch
            {
                "WindSpeed10m"     => wspMs,
                "WindGusts10m"     => gustMs,
                "Temperature2m"    => tempC,
                "DewPoint2m"       => dewC,
                "SurfacePressure"  => pressureHpa,
                "CloudCover"       => cloudPct,
                _ => throw new InvalidOperationException(SpreadVars[v].Column),
            };
            var (mean, std) = NanSafeMeanStd(src);
            spreadFeatures[2 * v + 0] = (float)mean;
            spreadFeatures[2 * v + 1] = (float)std;
        }

        var validUtc = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);

        var features = new float[spec.FeatureCount];
        // Layout: [0..n) wsp, [n..2n) wdir, [2n..2n+5) ORO_LEAN, [2n+5..2n+5+12) SPREAD
        for (int i = 0; i < n; i++)  features[i]         = (float)wspMs[i];
        for (int i = 0; i < n; i++)  features[n + i]     = (float)wdirDeg[i];
        for (int i = 0; i < 5; i++)  features[2 * n + i] = (float)oroLean[i];
        for (int i = 0; i < 12; i++) features[2 * n + 5 + i] = spreadFeatures[i];

        return new RegressionTrainingRow
        {
            ValidTimeUtc = validUtc,
            Features = features,
            Label = (float)dunkeswellWspMs,
        };
    }

    // ---- ORO_LEAN per-row computation (mirror of train_wind_mvn.oro_dynamic) ----

    private static readonly Dictionary<string, double> SectorRad = new(StringComparer.Ordinal)
    {
        ["N"] = 0.0,                ["NE"] = Math.PI / 4,
        ["E"] = Math.PI / 2,        ["SE"] = 3 * Math.PI / 4,
        ["S"] = Math.PI,            ["SW"] = 5 * Math.PI / 4,
        ["W"] = 3 * Math.PI / 2,    ["NW"] = 7 * Math.PI / 4,
    };

    private static double UpwindGainSector(double rad, IReadOnlyDictionary<string, double> upwindGain)
    {
        if (double.IsNaN(rad)) return 0.0;
        var d = rad % (2 * Math.PI);
        if (d < 0) d += 2 * Math.PI;
        string best = "N";
        double bestDiff = Math.PI * 3;
        foreach (var (s, c) in SectorRad)
        {
            var diff = Math.Min(Math.Abs(d - c), 2 * Math.PI - Math.Abs(d - c));
            if (diff < bestDiff) { bestDiff = diff; best = s; }
        }
        return upwindGain[best];
    }

    /// <summary>Compute the 5 ORO_LEAN features for one row.
    /// Mirrors <c>oro_dynamic</c> in <c>scripts/wind_speed_production_scope_bakeoff.py</c>
    /// and <c>scripts/train_wind_mvn.py</c> — same Tetens vapour-pressure
    /// formula, same uplift sign convention (wind FROM direction → u = −wsp·sin,
    /// v = −wsp·cos).</summary>
    private static double[] OroDynamic(
        double wspMean, double sinMean, double cosMean,
        double tdMean, double pMean,
        OroStaticFeatures oro)
    {
        var gradDx = oro.TerrainGradientDx;
        var gradDy = oro.TerrainGradientDy;
        var rad = (!double.IsNaN(sinMean) && !double.IsNaN(cosMean))
            ? Math.Atan2(sinMean, cosMean)
            : double.NaN;
        var gain = UpwindGainSector(rad, oro.UpwindGain5km);

        double uplift;
        if (double.IsNaN(wspMean) || double.IsNaN(sinMean) || double.IsNaN(cosMean))
            uplift = 0.0;
        else
            uplift = Math.Max(0.0,
                (-wspMean * sinMean) * gradDx + (-wspMean * cosMean) * gradDy);

        double q;
        if (double.IsNaN(tdMean) || double.IsNaN(pMean) || pMean <= 0)
            q = 0.0;
        else
        {
            // Tetens vapour-pressure formula (Murphy & Koop 2005 form, hPa).
            var e = 6.112 * Math.Exp(17.62 * tdMean / (tdMean + 243.12));
            q = Math.Max(0.0, 0.622 * e / (pMean - 0.378 * e) * 1000.0);
        }

        return new[]
        {
            double.IsNaN(sinMean) ? 0.0 : sinMean,
            double.IsNaN(cosMean) ? 0.0 : cosMean,
            gain,
            uplift,
            uplift * q,
        };
    }

    private static (double Mean, double Std) NanSafeMeanStd(IReadOnlyList<double> src)
    {
        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var x in src)
        {
            if (double.IsNaN(x)) continue;
            sum += x; sumSq += x * x; n++;
        }
        if (n == 0) return (double.NaN, double.NaN);
        var mean = sum / n;
        var var0 = Math.Max(0.0, (sumSq / n) - (mean * mean));
        return (mean, Math.Sqrt(var0));
    }

    // ---- SQL helpers ----

    private static string PivotClause(IReadOnlyList<string> models, string column, string prefix)
        => string.Join(",\n        ", models.Select(m =>
            $"MAX(CASE WHEN Model = '{m}' THEN {column} END) AS {prefix}_{TempFeatureBuilder.ShortName(m)}"));

    private static void ReadBlock(DuckDB.NET.Data.DuckDBDataReader r, ref int col, double[] dest, int n)
    {
        for (int i = 0; i < n; i++)
        {
            dest[i] = r.IsDBNull(col) ? double.NaN : r.GetDouble(col);
            col++;
        }
    }

    private static string NormaliseGlob(string path) => path.Replace('\\', '/').Replace("'", "''");
}
