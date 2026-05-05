using DuckDB.NET.Data;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Exact12h;

/// <summary>
/// Builds the training dataset for the exact-runtime 12h temperature blender.
/// Strict design (Option 1 from the 2026-05-05 scoping discussion):
///
///   * Source: <c>RunTimeSource = 'exact'</c> rows ONLY (the AWS-archive
///     backfills, not Open-Meteo's offset_day Previous Runs).
///   * Lead: <c>LeadHours = 12</c> EXACTLY. No tolerance, no fallback.
///   * ValidTime grid: hour ∈ {0, 6, 12, 18 UTC} — the synoptic cycle phase
///     that GFS / IFS / AIFS / Global naturally produce at lead 12. UKV runs
///     at 03/15Z so its lead-12 forecasts target {03, 15} ValidTimes which
///     don't overlap; UKV is excluded from this builder by design (revisit
///     once we decide how to honestly fold in a different cycle phase).
///   * Models considered: <c>gfs_ncep, ecmwf_ifs_oper, ecmwf_aifs_oper,
///     met_office_global</c>. ICON / MeteoFrance / GEM / JMA have no
///     exact-runtime AWS source; they're absent from this blender entirely.
///
/// Per-cycle reality (verified against R2 2026-05-05):
///   GFS                cycles 00/06/12/18Z → ValidTimes 12/18/00/06
///   AIFS               cycles 00/06/12/18Z → ValidTimes 12/18/00/06
///   IFS                cycles 00/12Z only  → ValidTimes 12/00 (06/18Z don't publish)
///   Global             cycles 00/12Z only  → ValidTimes 12/00 (06/18Z cap at 66h)
/// So at ValidTime 06/18 only GFS + AIFS report; IFS/Global are NULL there
/// regardless of policy. Tier T1 (all required) is therefore restricted to
/// ValidTimes {00, 12} by data; T2/T3 keep all four ValidTimes by making
/// IFS/Global optional (LightGBM splits handle the NaN slots natively).
/// </summary>
public static class Exact12hFeatureBuilder
{
    /// <summary>Default target lead for the experiment (12h ahead).
    /// Bake-off can override via <c>--lead N</c>. Single-lead runs use this
    /// for both the SQL filter and the BlenderSpec.LeadHours metadata.
    /// Multi-lead runs use this as the *canonical* lead — the one used to
    /// compute spread features and define the per-model baseline.</summary>
    public const int DefaultTargetLead = 12;

    /// <summary>Models in canonical column order. Adding a new exact-runtime
    /// model = append here + add to <see cref="ShortName"/>; existing tier
    /// definitions stay valid because they reference models by id.</summary>
    public static readonly IReadOnlyList<string> CanonicalModelOrder = new[]
    {
        "gfs_ncep",
        "ecmwf_ifs_oper",
        "ecmwf_aifs_oper",
        "met_office_global",
    };

    /// <summary>Stable short suffix for feature column names (temp_gfs, ...).</summary>
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

    /// <summary>The three tiers from the user-approved scoping
    /// (2026-05-05). Required + optional must be disjoint and together must
    /// cover every model that should appear as a column. Models not listed
    /// in either set are excluded from the column vector entirely.</summary>
    public static readonly IReadOnlyList<TierSpec> AllTiers = new[]
    {
        new TierSpec(
            Name: "T1",
            Required: new[] { "gfs_ncep", "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global" },
            Optional: Array.Empty<string>(),
            StartDate: new DateOnly(2024, 5, 4),
            Description: "All four models REQUIRED. Restricted to ValidTimes {00, 12} by IFS/Global cycle structure. Smallest, strictest tier."),

        new TierSpec(
            Name: "T2",
            Required: new[] { "gfs_ncep", "ecmwf_aifs_oper" },
            Optional: new[] { "ecmwf_ifs_oper", "met_office_global" },
            StartDate: new DateOnly(2024, 2, 29),
            Description: "GFS + AIFS required (the 4-cycle-publishing pair). IFS + Global optional. Captures all four ValidTimes; IFS/Global NaN at 06/18."),

        new TierSpec(
            Name: "T3",
            Required: new[] { "gfs_ncep" },
            Optional: new[] { "ecmwf_ifs_oper", "ecmwf_aifs_oper", "met_office_global" },
            StartDate: new DateOnly(2023, 1, 18),
            Description: "Only GFS required (its archive goes back furthest). Everything else optional, NaN before its bucket-start date."),
    };

    public static TierSpec GetTier(string name) =>
        AllTiers.FirstOrDefault(t => t.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException($"Unknown tier '{name}'. Expected: {string.Join(",", AllTiers.Select(t => t.Name))}.", nameof(name));

    /// <summary>
    /// Build a <see cref="BlenderSpec"/> for one tier.
    ///
    /// Single-lead mode (<paramref name="inputLeads"/> contains one entry):
    ///   feature columns are <c>temp_{model}</c> in canonical order.
    ///
    /// Multi-lead mode (more than one input lead — the 2026-05-05 experiment
    /// asking "does adding leads 6 and 18 alongside lead 12 hurt the blend?"):
    ///   feature columns are <c>temp_{model}_l{lead:00}</c> for every (model,
    ///   lead) pair, in canonical model order × ascending lead order. Spread
    ///   features (mean/std/range) are still computed across just the
    ///   canonical lead's per-model values so they mean the same thing as in
    ///   single-lead mode — adding more leads doesn't artificially inflate
    ///   model disagreement.
    /// </summary>
    /// <summary>
    /// UKV-included flag — when true, the feature vector gets one extra
    /// column "temp_ukv" whose value per ValidTime is pulled from the (cycle,
    /// lead) tuple that lands closest to a 12h-ahead UKV forecast for that
    /// ValidTime hour. Per-V mapping (UKV runs at 03Z and 15Z only):
    ///   V=00 → 15Z prev day + lead 9  (9h-ahead)
    ///   V=06 → 15Z prev day + lead 15 (15h-ahead)
    ///   V=12 → 03Z same day + lead 9  (9h-ahead)
    ///   V=18 → 03Z same day + lead 15 (15h-ahead)
    /// Average UKV-effective-lead = 12h, matching the other models'
    /// strict lead-12. UKV is always optional (NaN at any V where the
    /// pull failed) so blender training drops nothing.
    /// </summary>
    public static BlenderSpec BuildSpec(
        TierSpec tier,
        int targetLead = DefaultTargetLead,
        IReadOnlyList<int>? inputLeads = null,
        bool includeUkv = false)
    {
        inputLeads ??= new[] { targetLead };
        if (inputLeads.Count == 0)
            throw new ArgumentException("inputLeads must have at least one entry.", nameof(inputLeads));
        if (!inputLeads.Contains(targetLead))
            throw new ArgumentException(
                $"inputLeads {string.Join(",", inputLeads)} must contain the targetLead {targetLead} " +
                "(it's the canonical lead used for required-model gating + spread features).",
                nameof(inputLeads));

        var requiredSet = tier.Required.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var optionalSet = tier.Optional.ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (requiredSet.Overlaps(optionalSet))
            throw new InvalidOperationException(
                $"Tier {tier.Name} has model(s) in both required and optional: " +
                string.Join(",", requiredSet.Intersect(optionalSet)));

        var orderedModels = CanonicalModelOrder
            .Where(m => requiredSet.Contains(m) || optionalSet.Contains(m))
            .ToList();
        if (orderedModels.Count == 0)
            throw new InvalidOperationException($"Tier {tier.Name} has no models active.");

        var leadsSorted = inputLeads.OrderBy(l => l).ToList();
        var isMulti = leadsSorted.Count > 1;

        var ukvExtra = includeUkv ? 1 : 0;
        var featureNames = new List<string>(orderedModels.Count * leadsSorted.Count + ukvExtra + 7);
        foreach (var m in orderedModels)
            foreach (var l in leadsSorted)
                featureNames.Add(isMulti ? $"temp_{ShortName(m)}_l{l:00}" : $"temp_{ShortName(m)}");
        if (includeUkv)
            featureNames.Add("temp_ukv"); // UKV slot — always optional, see BuildSpec docstring
        featureNames.Add("temp_mean");
        featureNames.Add("temp_std");
        featureNames.Add("temp_range");
        featureNames.Add("hour_sin");
        featureNames.Add("hour_cos");
        featureNames.Add("doy_sin");
        featureNames.Add("doy_cos");

        var leadTag = isMulti
            ? "leads" + string.Concat(leadsSorted.Select(l => l.ToString("00")))
            : $"l{targetLead:00}";

        return new BlenderSpec
        {
            Target = "temperature",
            FeatureSet = $"exact-{leadTag}-{tier.Name}",
            LeadHours = targetLead,
            RequiredModels = CanonicalModelOrder.Where(requiredSet.Contains).ToList(),
            OptionalModels = CanonicalModelOrder.Where(optionalSet.Contains).ToList(),
            Models = orderedModels,
            FeatureNames = featureNames,
        };
    }

    /// <summary>
    /// Read the exact-runtime forecast tree, pivot to (ValidTime × per-model
    /// × per-lead temperature), join to ERA5 truth by ValidTime, drop rows
    /// where any REQUIRED model is missing AT THE TARGET LEAD, return as
    /// <see cref="RegressionTrainingRow"/>.
    ///
    /// Required gating uses <paramref name="targetLead"/> only — non-canonical
    /// leads are always optional and may be NaN. Spread features (mean/std/
    /// range) are computed across the per-model values at the target lead so
    /// they retain "model disagreement" semantics in both single- and
    /// multi-lead modes (otherwise wider lead spread would inflate them).
    /// </summary>
    public static List<RegressionTrainingRow> Build(
        string forecastsPath,
        string era5Path,
        string locationName,
        TierSpec tier,
        BlenderSpec spec,
        int targetLead = DefaultTargetLead,
        IReadOnlyList<int>? inputLeads = null,
        bool includeUkv = false,
        CancellationToken ct = default)
    {
        inputLeads ??= new[] { targetLead };
        var leadsSorted = inputLeads.OrderBy(l => l).ToList();
        if (!leadsSorted.Contains(targetLead))
            throw new ArgumentException(
                $"inputLeads must contain targetLead {targetLead}.", nameof(inputLeads));

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        var fcGlob = NormaliseGlob(Path.Combine(forecastsPath, "**", "*.parquet"));
        var eraGlob = NormaliseGlob(Path.Combine(era5Path, "**", "*.parquet"));
        var modelInClause = "(" + string.Join(",", spec.Models.Select(m => $"'{m}'")) + ")";
        var leadInClause  = "(" + string.Join(",", leadsSorted) + ")";

        // Pivot per (Model × Lead). Column name format MUST match
        // BuildSpec's FeatureNames so the SELECT order downstream lines up.
        // Single-lead mode collapses to one column per model (no _l suffix).
        var isMulti = leadsSorted.Count > 1;
        string ColName(string m, int l) => isMulti
            ? $"temp_{ShortName(m)}_l{l:00}"
            : $"temp_{ShortName(m)}";

        var pivotCols = new List<string>();
        foreach (var m in spec.Models)
            foreach (var l in leadsSorted)
                pivotCols.Add($"MAX(CASE WHEN Model = '{m}' AND LeadHours = {l} THEN Temperature2m END) AS {ColName(m, l)}");
        var pivotSql = string.Join(",\n        ", pivotCols);

        // Order the SELECT to match the BlenderSpec.FeatureNames order
        // exactly (model-major, lead-minor).
        var orderedFeatureCols = new List<string>();
        foreach (var m in spec.Models)
            foreach (var l in leadsSorted)
                orderedFeatureCols.Add($"p.{ColName(m, l)}");
        var selectCols = string.Join(", ", orderedFeatureCols);

        // Strict required NOT NULL filter — gated on the TARGET LEAD only.
        // Non-target leads are always optional, even for "required" models.
        var requiredNotNull = spec.RequiredModels.Count > 0
            ? string.Join("\n  AND ", spec.RequiredModels.Select(m => $"p.{ColName(m, targetLead)} IS NOT NULL"))
            : "TRUE";

        // Restrict to ValidTime hour ∈ {0,6,12,18}: lead 6/12/18 from cycles
        // 00/06/12/18Z all land on this 6-hour grid. Lead 24 likewise. Other
        // leads (1, 3) would land on different hours and need a richer grid
        // — explicitly disallowed here so we don't silently expand the
        // ValidTime universe and break per-tier comparisons.
        foreach (var l in leadsSorted)
            if (l % 6 != 0)
                throw new ArgumentException(
                    $"Lead {l}h is not on the 6-hour cycle grid. Only multiples of 6 are supported here " +
                    "(otherwise ValidTime hour ∈ {0,6,12,18} is invalid). Got: " +
                    string.Join(",", leadsSorted), nameof(inputLeads));

        var fcWhere =
            $"LocationName = '{locationName}' " +
            $"AND RunTimeSource = 'exact' " +
            $"AND LeadHours IN {leadInClause} " +
            $"AND HOUR(ValidTimeUtc) IN (0, 6, 12, 18) " +
            $"AND Temperature2m IS NOT NULL " +
            $"AND CAST(ValidTimeUtc AS DATE) >= DATE '{tier.StartDate:yyyy-MM-dd}' " +
            $"AND Model IN {modelInClause}";

        // Optional UKV CTE — picks one UKV value per ValidTime from a
        // hour-conditional (cycle, lead) tuple chosen to land at ~12h-ahead
        // (see BuildSpec docstring for the rules). LEFT JOIN so a missing
        // UKV row doesn't drop the whole training row.
        var ukvCte = !includeUkv ? "" : $@",
ukv_per_v AS (
    -- For each ValidTime hour, pick the (UKV cycle, lead) tuple that
    -- gives a ~12h-ahead forecast for that V. UKV runs at 03Z + 15Z only,
    -- so V hours map to specific (run_hr, run_date_offset, lead) triples.
    SELECT ValidTimeUtc, Temperature2m AS ukv_temp
    FROM (
        SELECT ValidTimeUtc, Temperature2m,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{locationName}'
          AND Model = 'met_office_ukv'
          AND RunTimeSource = 'exact'
          AND Temperature2m IS NOT NULL
          AND CAST(ValidTimeUtc AS DATE) >= DATE '{tier.StartDate:yyyy-MM-dd}'
          AND HOUR(ValidTimeUtc) IN (0, 6, 12, 18)
          AND (
            -- V=00: prev day 15Z run + lead 9
            (HOUR(ValidTimeUtc) = 0  AND HOUR(RunTimeUtc) = 15
             AND CAST(RunTimeUtc AS DATE) = CAST(ValidTimeUtc AS DATE) - INTERVAL 1 DAY
             AND LeadHours = 9)
            -- V=06: prev day 15Z run + lead 15
         OR (HOUR(ValidTimeUtc) = 6  AND HOUR(RunTimeUtc) = 15
             AND CAST(RunTimeUtc AS DATE) = CAST(ValidTimeUtc AS DATE) - INTERVAL 1 DAY
             AND LeadHours = 15)
            -- V=12: same day 03Z run + lead 9
         OR (HOUR(ValidTimeUtc) = 12 AND HOUR(RunTimeUtc) = 3
             AND CAST(RunTimeUtc AS DATE) = CAST(ValidTimeUtc AS DATE)
             AND LeadHours = 9)
            -- V=18: same day 03Z run + lead 15
         OR (HOUR(ValidTimeUtc) = 18 AND HOUR(RunTimeUtc) = 3
             AND CAST(RunTimeUtc AS DATE) = CAST(ValidTimeUtc AS DATE)
             AND LeadHours = 15)
          )
    )
    WHERE rn = 1
)";

        var ukvSelectCol = includeUkv ? ", u.ukv_temp" : "";
        var ukvJoin      = includeUkv ? "LEFT JOIN ukv_per_v u USING (ValidTimeUtc)" : "";

        // Latest forecast per (ValidTime, Model, Lead) — defensive against
        // any case where two cycles published the same (ValidTime, Lead)
        // for one model (shouldn't happen at lead 12+ from the cycles we
        // pulled, but free correctness here).
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, LeadHours, Temperature2m, WindDirection10m,
           ROW_NUMBER() OVER (
               PARTITION BY ValidTimeUtc, Model, LeadHours
               ORDER BY RunTimeUtc DESC
           ) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {fcWhere}
),
pivoted AS (
    SELECT
        ValidTimeUtc,
        {pivotSql},
        AVG(WindDirection10m) AS wind_dir_mean
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
),
era5 AS (
    SELECT ValidTimeUtc, Temperature2m AS era5_temp
    FROM read_parquet('{eraGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locationName}'
      AND Temperature2m IS NOT NULL
){ukvCte}
SELECT
    p.ValidTimeUtc,
    {selectCols},
    p.wind_dir_mean,
    e.era5_temp{ukvSelectCol}
FROM pivoted p
JOIN era5 e USING (ValidTimeUtc)
{ukvJoin}
WHERE {requiredNotNull}
ORDER BY p.ValidTimeUtc;
";

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();

        var rows = new List<RegressionTrainingRow>();
        var perModelLeadCount = spec.Models.Count * leadsSorted.Count;
        // Index of each (model, lead) pair within the flat feature vector,
        // matching the SELECT order — model-major, lead-minor.
        var targetLeadIdx = leadsSorted.IndexOf(targetLead);
        var perModelLeadValues = new double[perModelLeadCount];
        var canonicalPerModel  = new double[spec.Models.Count];
        // SELECT order: ValidTimeUtc, perModelLeads..., wind_dir_mean, era5_temp [, ukv_temp]
        var ukvIdx = includeUkv ? 3 + perModelLeadCount : -1; // -1 = absent
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            for (int i = 0; i < perModelLeadCount; i++)
                perModelLeadValues[i] = r.IsDBNull(1 + i) ? double.NaN : r.GetDouble(1 + i);
            var windDirMean = r.IsDBNull(1 + perModelLeadCount) ? double.NaN : r.GetDouble(1 + perModelLeadCount);
            var era5Temp = r.GetDouble(2 + perModelLeadCount);
            var ukvTemp = ukvIdx >= 0
                ? (r.IsDBNull(ukvIdx) ? double.NaN : r.GetDouble(ukvIdx))
                : double.NaN;

            // Extract canonical-lead per-model values for the spread features.
            for (int m = 0; m < spec.Models.Count; m++)
                canonicalPerModel[m] = perModelLeadValues[m * leadsSorted.Count + targetLeadIdx];

            rows.Add(ComposeRow(spec, valid, perModelLeadValues, canonicalPerModel, windDirMean, era5Temp, ukvTemp));
        }
        return rows;
    }

    /// <summary>
    /// Pack per-(model, lead) temps + spread + calendar features into a
    /// <see cref="RegressionTrainingRow"/>. <paramref name="perModelLeadValues"/>
    /// is the full feature vector in spec.FeatureNames order;
    /// <paramref name="canonicalPerModel"/> contains just the target-lead
    /// values used to compute the spread features (so multi-lead runs don't
    /// inflate model disagreement with cross-lead variance).
    /// </summary>
    public static RegressionTrainingRow ComposeRow(
        BlenderSpec spec,
        DateTime validTimeUtc,
        IReadOnlyList<double> perModelLeadValues,
        IReadOnlyList<double> canonicalPerModel,
        double windDirMeanDeg,
        double era5Temp,
        double ukvTemp = double.NaN)
    {
        // Spread features: NaN-safe across present-only canonical-lead values.
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int n = 0;
        for (int i = 0; i < canonicalPerModel.Count; i++)
        {
            var x = canonicalPerModel[i];
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
        var hourSin = Math.Sin(2 * Math.PI * hourFrac);
        var hourCos = Math.Cos(2 * Math.PI * hourFrac);
        var doySin  = Math.Sin(2 * Math.PI * doyFrac);
        var doyCos  = Math.Cos(2 * Math.PI * doyFrac);

        // FeatureCount = perModelLeadValues + 7 stats [+ 1 if UKV included].
        // The UKV column's presence is encoded in spec.FeatureNames containing
        // "temp_ukv" — derive includeUkv from the spec rather than passing
        // another arg, so callers can't get the pair out of sync.
        var includeUkv = spec.FeatureNames.Contains("temp_ukv");
        var expectedCount = perModelLeadValues.Count + (includeUkv ? 1 : 0) + 7;
        if (expectedCount != spec.FeatureCount)
            throw new InvalidOperationException(
                $"perModelLeadValues count {perModelLeadValues.Count} (+ ukv:{includeUkv}) + 7 stats != spec.FeatureCount {spec.FeatureCount}.");

        var features = new float[spec.FeatureCount];
        int idx = 0;
        for (int i = 0; i < perModelLeadValues.Count; i++)
            features[idx++] = (float)perModelLeadValues[i];
        if (includeUkv)
            features[idx++] = (float)ukvTemp; // NaN-safe — LightGBM handles missing
        features[idx++] = (float)mean;
        features[idx++] = (float)std;
        features[idx++] = (float)range;
        features[idx++] = (float)hourSin;
        features[idx++] = (float)hourCos;
        features[idx++] = (float)doySin;
        features[idx++] = (float)doyCos;

        return new RegressionTrainingRow
        {
            ValidTimeUtc = v,
            Features = features,
            Label = (float)era5Temp,
            WindDirMean = (float)windDirMeanDeg,
        };
    }

    private static string NormaliseGlob(string p) => p.Replace('\\', '/');
}
