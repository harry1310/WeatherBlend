using DuckDB.NET.Data;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;

namespace WeatherBlend.Train.PrecipExact;

/// <summary>
/// Sister of <c>Exact12hFeatureBuilder</c> for precipitation. Same exact-runtime
/// philosophy: <c>RunTimeSource = 'exact'</c> rows only, strict lead, ValidTime
/// grid {00, 06, 12, 18 UTC}. Differences:
///
///   - Reads <c>Precipitation</c> column instead of <c>Temperature2m</c>
///   - Joins to EA Hydrology gauge truth at the requested station, NOT ERA5.
///     Matches <see cref="PrecipFeatureBuilder.BuildForLead"/> exactly: 15-min
///     readings aggregated to hourly with the strict 4-of-4 rule (partial
///     hours dropped). This is THE point — head-to-head Brier comparison
///     against 3a/3c only makes sense if both score against the same per-
///     station gauge truth. Using ERA5 (~25km grid) was a transient mistake
///     2026-05-05 morning; ERA5 smooths out the localised orographic /
///     convective signal that makes Dartmoor precip hard to forecast, so
///     ERA5-truth Brier numbers are spuriously good and not comparable to
///     gauge-truth 3a/3c numbers.
///
/// Per-model precip semantics differ — known caveat for the bake-off:
///   GFS APCP        = accumulation in some interval ending at ValidTime (mm)
///   IFS  tp         = TOTAL accumulation since cycle start (mm, after × 1000 conversion)
///   AIFS tp         = ditto IFS; ÷1000 fix landed 2026-05-05 (units bug)
///   MO Global       = instantaneous rate (mm/h, after × 3.6e6 from m/s)
///   UKV (optional)  = same as MO Global (Met Office NetCDF schema)
/// LightGBM splits handle these cleanly since each input is on its own
/// column. Wet/dry decision boundary at 0.1mm/h is what matters.
/// </summary>
public static class PrecipExactFeatureBuilder
{
    public const int DefaultTargetLead = 12;

    /// <summary>Canonical model order. AIFS reinstated 2026-05-05 after the
    /// EcmwfClient tp units bug was fixed and the AIFS chunks re-backfilled
    /// against the corrected parser.</summary>
    public static readonly IReadOnlyList<string> CanonicalModelOrder = new[]
    {
        "gfs_ncep",
        "ecmwf_ifs_oper",
        "ecmwf_aifs_oper",
        "met_office_global",
    };

    public static string ShortName(string modelId) => modelId switch
    {
        "gfs_ncep"          => "gfs",
        "ecmwf_ifs_oper"    => "ifs",
        "ecmwf_aifs_oper"   => "aifs",
        "met_office_global" => "moglobal",
        _ => throw new ArgumentException($"Unknown modelId '{modelId}'", nameof(modelId)),
    };

    public sealed record TierSpec(
        string Name,
        IReadOnlyList<string> Required,
        IReadOnlyList<string> Optional,
        DateOnly StartDate,
        string Description);

    /// <summary>Two precip tiers, used as champion/challenger:
    ///   * <b>P1</b> — GFS + IFS + AIFS required, MO Global optional. Original
    ///     1-tier definition, mirrors temp 2d T2.
    ///   * <b>P2</b> — GFS + AIFS required, MO Global optional. Drops IFS
    ///     entirely (and is paired with includeUkv=false for the bake-off
    ///     against P1). Added 2026-05-07 after the IFS scda backfill
    ///     produced a uniform regression at every (station, lead) cell when
    ///     the 3d retrain ingested it — raw 0.25° IFS precip carries an
    ///     elevation/grid bias at Bonehill (393m tor) that the blender
    ///     learns to weight more as IFS coverage densifies, and that
    ///     hurts test Brier. The temp 2d retrain on the SAME data
    ///     improved (-4.5% MAE), so the bias is precip-specific, not a
    ///     general scda problem. P2 lets us isolate whether removing
    ///     IFS recovers the previous Brier and, if so, whether re-adding
    ///     UKV (2km grid — least elevation bias of the raw S3 sources)
    ///     buys back any of the dropped IFS contribution.</summary>
    public static readonly IReadOnlyList<TierSpec> AllTiers = new[]
    {
        new TierSpec(
            Name: "P1",
            Required: new[] { "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper" },
            Optional: new[] { "met_office_global" },
            StartDate: new DateOnly(2024, 5, 4),
            Description: "GFS + IFS + AIFS required, MO Global optional. From 2024-05-04 (MO Global archive start). Mirrors temp 2d T2."),

        new TierSpec(
            Name: "P2",
            Required: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
            Optional: new[] { "met_office_global" },
            StartDate: new DateOnly(2024, 5, 4),
            Description: "GFS + AIFS required, MO Global optional. IFS dropped — bake-off challenger to P1 after the 2026-05-07 IFS scda regression."),
    };

    public static TierSpec GetTier(string name) =>
        AllTiers.FirstOrDefault(t => t.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException($"Unknown tier '{name}'", nameof(name));

    /// <summary>
    /// Mirrors <c>Exact12hFeatureBuilder.BuildSpec(..., includeUkv: bool)</c>.
    /// When <paramref name="includeUkv"/> is true the feature vector gets one
    /// extra always-optional column "precip_ukv" pulled from UKV's 03Z + 15Z
    /// cycles via the same per-V-hour (cycle, lead) conditional rule the
    /// temperature builder uses. UKV stays optional regardless so a missing
    /// pull doesn't drop the row.
    /// </summary>
    public static BlenderSpec BuildSpec(
        TierSpec tier,
        int targetLead = DefaultTargetLead,
        bool includeUkv = false)
    {
        var requiredSet = tier.Required.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = tier.Optional.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var orderedModels = CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m))
            .ToList();

        var ukvExtra = includeUkv ? 1 : 0;
        var featureNames = new List<string>(orderedModels.Count + ukvExtra + 7);
        foreach (var m in orderedModels) featureNames.Add($"precip_{ShortName(m)}");
        if (includeUkv) featureNames.Add("precip_ukv");
        featureNames.Add("precip_mean");
        featureNames.Add("precip_std");
        featureNames.Add("precip_range");
        featureNames.Add("hour_sin");
        featureNames.Add("hour_cos");
        featureNames.Add("doy_sin");
        featureNames.Add("doy_cos");

        return new BlenderSpec
        {
            Target = "precipitation",
            FeatureSet = $"exact-l{targetLead:00}-{tier.Name}{(includeUkv ? "-ukv" : "")}",
            LeadHours = targetLead,
            RequiredModels = CanonicalModelOrder.Where(requiredSet.Contains).ToList(),
            OptionalModels = CanonicalModelOrder.Where(optionalSet.Contains).ToList(),
            Models = orderedModels,
            FeatureNames = featureNames,
        };
    }

    /// <summary>Wet-threshold matching production 3a: ≥ 0.1 mm/h is "wet".
    /// Same threshold ERA5 / EA Hydrology gauge labels use across the
    /// project, so the exact-runtime Brier blender is directly comparable
    /// to 3a/3c.</summary>
    public const double WetThresholdMm = 0.1;

    public static List<BinaryTrainingRow> Build(
        string forecastsPath,
        string rainfallPath,
        string locationName,
        string stationName,
        TierSpec tier,
        BlenderSpec spec,
        int targetLead = DefaultTargetLead,
        bool includeUkv = false,
        CancellationToken ct = default)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var rnGlob = NormaliseGlob(Path.Combine(rainfallPath, "**", "*.parquet"));
        var escStation = stationName.Replace("'", "''");
        var escLocation = locationName.Replace("'", "''");
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";

        var pivotCols = string.Join(",\n        ",
            spec.Models.Select(m => $"MAX(CASE WHEN Model = '{m}' THEN Precipitation END) AS precip_{ShortName(m)}"));
        var selectCols = string.Join(", ", spec.Models.Select(m => $"p.precip_{ShortName(m)}"));

        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.precip_{ShortName(m)} IS NOT NULL"))
            : "TRUE";

        var fcWhere =
            $"LocationName = '{escLocation}' " +
            $"AND RunTimeSource = 'exact' " +
            $"AND LeadHours = {targetLead} " +
            $"AND HOUR(ValidTimeUtc) IN (0, 6, 12, 18) " +
            $"AND Precipitation IS NOT NULL " +
            $"AND CAST(ValidTimeUtc AS DATE) >= DATE '{tier.StartDate:yyyy-MM-dd}' " +
            $"AND Model IN {modelInClause}";

        // Optional UKV CTE — shares Exact12hFeatureBuilder.UkvPerVOrClause
        // so the per-V-hour (cycle, lead) rule is target-lead-aware (avg
        // UKV effective-lead matches targetLead, mirroring the other
        // models' strict lead). Reads Precipitation instead of
        // Temperature2m. Always-optional → LEFT JOIN, NaN at any V where
        // the pull failed.
        var ukvCte = !includeUkv ? "" : $@",
ukv_per_v AS (
    SELECT ValidTimeUtc, Precipitation AS ukv_precip
    FROM (
        SELECT ValidTimeUtc, Precipitation,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{escLocation}'
          AND Model = 'met_office_ukv'
          AND RunTimeSource = 'exact'
          AND Precipitation IS NOT NULL
          AND CAST(ValidTimeUtc AS DATE) >= DATE '{tier.StartDate:yyyy-MM-dd}'
          AND HOUR(ValidTimeUtc) IN (0, 6, 12, 18)
          AND ({Exact12hFeatureBuilder.UkvPerVOrClause(targetLead, Exact12hFeatureBuilder.UkvPickStrategy.Averaging)})
    )
    WHERE rn = 1
)";
        var ukvSelectCol = includeUkv ? ", u.ukv_precip" : "";
        var ukvJoin = includeUkv ? "LEFT JOIN ukv_per_v u USING (ValidTimeUtc)" : "";

        // Truth = EA Hydrology gauge for the requested station, aggregated
        // 15-min readings up to hourly totals with the strict 4-of-4 rule
        // (matches PrecipFeatureBuilder.BuildForLead — partial hours dropped
        // so they can't flip wet↔dry). Joining to gauge truth (NOT ERA5) is
        // the whole point of 3d being directly comparable to 3a/3c on the
        // same per-station target.
        var sql = $@"
WITH hourly_truth AS (
    SELECT
        date_trunc('hour', ObservedTimeUtc) AS valid_time,
        SUM(Value15MinMm) AS precip_mm_hour
    FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{escLocation}'
      AND StationName  = '{escStation}'
      AND Value15MinMm IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) = 4
),
latest AS (
    SELECT ValidTimeUtc, Model, Precipitation,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT ValidTimeUtc, {pivotCols}
    FROM latest WHERE rn = 1
    GROUP BY ValidTimeUtc
){ukvCte}
SELECT p.ValidTimeUtc, {selectCols}, t.precip_mm_hour{ukvSelectCol}
FROM pivoted p
JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
{ukvJoin}
WHERE {requiredNotNull}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<BinaryTrainingRow>();
        var modelCount = spec.Models.Count;
        var pcps = new double[modelCount];
        // SELECT order: ValidTimeUtc, perModelPrecip..., truthMm [, ukv_precip]
        var ukvIdx = includeUkv ? 2 + modelCount : -1;
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < modelCount; i++)
                pcps[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var truthMm = r.GetDouble(1 + modelCount);
            var ukvPrecip = ukvIdx >= 0
                ? (r.IsDBNull(ukvIdx) ? double.NaN : r.GetDouble(ukvIdx))
                : double.NaN;
            rows.Add(ComposeRow(spec, valid, pcps, truthMm, ukvPrecip));
        }
        return rows;
    }

    public static BinaryTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelPrecip,
        double truthMmHour,
        double ukvPrecip = double.NaN)
    {
        // Spread features computed across the per-model precip values only —
        // UKV is excluded so spread retains "model disagreement" semantics
        // unchanged when UKV toggles on.
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int n = 0;
        for (int i = 0; i < perModelPrecip.Count; i++)
        {
            var x = perModelPrecip[i];
            if (double.IsNaN(x)) continue;
            sum += x; sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            n++;
        }
        var mean = n == 0 ? double.NaN : sum / n;
        var variance = n == 0 ? double.NaN : Math.Max(0.0, (sumSq / n) - (mean * mean));
        var std = double.IsNaN(variance) ? double.NaN : Math.Sqrt(variance);
        var range = n == 0 ? double.NaN : max - min;

        var v = validTimeUtc.Kind == DateTimeKind.Utc
            ? validTimeUtc
            : DateTime.SpecifyKind(validTimeUtc, DateTimeKind.Utc);
        var hourFrac = v.Hour / 24.0;
        var doyFrac  = (v.DayOfYear - 1) / 365.0;

        // Mirror Exact12hFeatureBuilder: derive includeUkv from the spec
        // rather than a parameter so callers can't get the pair out of sync.
        var includeUkv = spec.FeatureNames.Contains("precip_ukv");
        var expectedCount = perModelPrecip.Count + (includeUkv ? 1 : 0) + 7;
        if (expectedCount != spec.FeatureCount)
            throw new InvalidOperationException(
                $"perModelPrecip count {perModelPrecip.Count} (+ ukv:{includeUkv}) + 7 stats != spec.FeatureCount {spec.FeatureCount}.");

        var features = new float[spec.FeatureCount];
        int idx = 0;
        for (int i = 0; i < perModelPrecip.Count; i++) features[idx++] = (float)perModelPrecip[i];
        if (includeUkv) features[idx++] = (float)ukvPrecip;
        features[idx++] = (float)mean;
        features[idx++] = (float)std;
        features[idx++] = (float)range;
        features[idx++] = (float)Math.Sin(2 * Math.PI * hourFrac);
        features[idx++] = (float)Math.Cos(2 * Math.PI * hourFrac);
        features[idx++] = (float)Math.Sin(2 * Math.PI * doyFrac);
        features[idx++] = (float)Math.Cos(2 * Math.PI * doyFrac);

        return new BinaryTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = truthMmHour >= WetThresholdMm,
            TruthMmHour = (float)truthMmHour,
        };
    }

    private static string NormaliseGlob(string p) => p.Replace('\\', '/');
}
