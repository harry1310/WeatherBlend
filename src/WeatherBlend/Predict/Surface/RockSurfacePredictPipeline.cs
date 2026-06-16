using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Predict.Surface;

/// <summary>
/// Builds the per-anchor rock surface temperature + condensation forecast
/// (Phase P1). Unlike the element blenders (independent per-(lead,valid)
/// samples), Force-Restore needs a CONTIGUOUS hourly forcing trajectory, so
/// this pipeline assembles one and marches the ODE over it
/// (<see cref="RockSurfacePhysics.Integrate"/>).
///
/// Forcing (decision 2026-06-05 — blends, 3-day horizon):
///  - FORWARD window (the reported hours): Ta / SW / cloud / wind from the
///    BLENDS — temperature champion (2b) + the wind / shortwave_radiation /
///    cloud_cover element champions — collapsed to smallest-lead-per-valid so
///    they tile a contiguous hourly series across +24h..+72h (element blenders
///    only train to lead 72).
///  - SPIN-UP window (the <see cref="RockSurfaceConfig.SpinupHours"/> before the
///    first blend hour, incl. the sub-24h gap): Ta / SW / cloud / wind from
///    recent NWP best-estimate (smallest-lead-per-valid mean across models).
///    Discarded after integration — it only settles Ts + the deep reservoir.
///  - DEW POINT (Td): NWP DewPoint2m directly throughout (mean across models),
///    per §4 — drives both the condensation margin and the clear-sky LW↓.
///
/// CLIFF-FACE MODE (docs/SENNEN_ROCK_TEMP_PLAN.md S3): when the location's
/// resolved <see cref="RockSurfaceConfig.Faces"/> is non-empty, the ODE runs
/// once per face with the direct beam re-projected onto that face's plane
/// (<see cref="SolarGeometry"/>; direct share = the NWP-mean direct fraction
/// applied to the blended/NWP horizontal SW) and rows are stamped with the
/// face name. Empty faces = whole-crag horizontal mode (Bonehill), SW
/// unmodified, Face = "".
///
/// SEA LONGWAVE (S4): locations with a marine block also feed the integrator
/// the best_match sea surface temperature (smallest-lead per valid, held
/// forward across gaps — SST moves slowly), which fills the (1−Fsky) view
/// fraction as a warm radiator (<see cref="RockSurfacePhysics"/>). No marine
/// block / no SST rows → null → neutral surroundings, pre-S4 behaviour.
///
/// Derived, not trained (no bundle / manifest). Output version "v1".
/// </summary>
public static class RockSurfacePredictPipeline
{
    public const string OutputModelVersion = "v1";

    private const string DewPointSourceTag = "nwp_mean";

    public static async Task<List<RockSurfacePredictionRow>> ComposeForAnchorAsync(
        ILogger log,
        LocationConfig location,
        string predictionsRoot,
        string modelsRoot,
        string forecastsPath,
        string marinePath,
        RockSurfaceConfig cfg,
        DateTime anchor,
        DateTime predictionMadeAtUtc,
        CancellationToken ct)
    {
        var loc = location.Name;

        // --- Champion versions of each blend input (per-location station key) ---
        var tempV = ModelArtifact.ResolveStationChampionVersion(modelsRoot, "temperature", loc);
        var windV = ModelArtifact.ResolveStationChampionVersion(modelsRoot, ElementTargets.Wind.ModelDirName, loc);
        var radV = ModelArtifact.ResolveStationChampionVersion(modelsRoot, ElementTargets.ShortwaveRadiation.ModelDirName, loc);
        var cloudV = ModelArtifact.ResolveStationChampionVersion(modelsRoot, ElementTargets.CloudCover.ModelDirName, loc);
        if (string.IsNullOrEmpty(tempV) || string.IsNullOrEmpty(windV) || string.IsNullOrEmpty(radV) || string.IsNullOrEmpty(cloudV))
        {
            log.LogError("Rock-surface: missing a champion blend version for {Loc} (temp={T}, wind={W}, rad={R}, cloud={C}) — skipping until all four are trained for this location.",
                loc, tempV, windV, radV, cloudV);
            return new List<RockSurfacePredictionRow>();
        }

        // --- Blend forcing, smallest-lead-per-valid (freshest cycle wins ties) ---
        var temp = await LoadTempByValidAsync(predictionsRoot, loc, tempV, anchor, ct);
        var wind = await LoadElementByValidAsync(predictionsRoot, ElementTargets.Wind, loc, windV, anchor, ct);
        var rad = await LoadElementByValidAsync(predictionsRoot, ElementTargets.ShortwaveRadiation, loc, radV, anchor, ct);
        var cloud = await LoadElementByValidAsync(predictionsRoot, ElementTargets.CloudCover, loc, cloudV, anchor, ct);

        // Gust champion (wind_gust_lgb) — the convective-cooling wind forcing uses
        // mean(champion wind, gust), not the mean wind alone. Field validation
        // (Bonehill IR-gun 2026-06-07) showed the rock model UNDER-cools because it
        // saw only mean wind; the gust-blended effective wind closed ~2°C of the
        // warm bias. Optional: if there's no gust champion / no gust for a valid,
        // fall back to champion wind alone (EffWind below).
        var gustV = ModelArtifact.ResolveStationChampionVersion(modelsRoot, ElementTargets.WindGust.ModelDirName, loc);
        var gust = string.IsNullOrEmpty(gustV)
            ? new Dictionary<DateTime, double>()
            : await LoadElementByValidAsync(predictionsRoot, ElementTargets.WindGust, loc, gustV, anchor, ct);
        // Effective wind for convective cooling = mean(champion wind, gust) when gust
        // is present at this valid, else champion wind alone.
        double EffWind(DateTime t) => gust.TryGetValue(t, out var g) ? 0.5 * (wind[t] + g) : wind[t];

        // Forward window = valids where ALL four blends exist (the +24..+72 tile).
        var forwardValids = temp.Keys
            .Where(v => wind.ContainsKey(v) && rad.ContainsKey(v) && cloud.ContainsKey(v))
            .OrderBy(v => v)
            .ToList();
        if (forwardValids.Count == 0)
        {
            log.LogError("Rock-surface: no valid times where temperature + all element blends overlap for {Loc} (temp={T} wind={W} rad={R} cloud={C} rows). Run predict for those targets first.",
                loc, temp.Count, wind.Count, rad.Count, cloud.Count);
            return new List<RockSurfacePredictionRow>();
        }

        var firstBlend = forwardValids[0];
        var lastBlend = forwardValids[^1];
        var spinupStart = firstBlend.AddHours(-Math.Max(0, cfg.SpinupHours));

        // --- NWP best-estimate (mean across models) over [spinupStart, lastBlend] ---
        // Td for the whole reported window + all forcing for the spin-up hours.
        var nwp = QueryNwpMeanByValid(forecastsPath, loc, spinupStart, lastBlend, anchor, ct);
        if (nwp.Count == 0)
        {
            log.LogError("Rock-surface: no NWP forcing rows for {Loc} in [{A:yyyy-MM-dd HH}..{B:yyyy-MM-dd HH}] — cannot source dew point / spin-up.",
                loc, spinupStart, lastBlend);
            return new List<RockSurfacePredictionRow>();
        }

        // --- Sea surface temperature (S4) — sea-cliff locations only. ---
        var sst = location.Marine is null
            ? new Dictionary<DateTime, double>()
            : QuerySeaTempByValid(marinePath, loc, spinupStart, lastBlend, anchor, ct);
        if (location.Marine is not null && sst.Count == 0)
            log.LogWarning("Rock-surface: no best_match SST rows for {Loc} — sea longwave term disabled this run (neutral surroundings fallback).", loc);

        // --- Assemble the contiguous hourly BASE trajectory: spin-up (NWP)
        //     then forward (blends). Horizontal SW and the NWP direct fraction
        //     are kept separate so cliff-face mode can re-project the beam per
        //     face below. Td always NWP. Hold-forward across the rare NWP gap
        //     so the Euler 1h cadence stays intact. ---
        var forwardSet = new HashSet<DateTime>(forwardValids);
        var baseHours = new List<BaseHour>();
        NwpMean? lastNwp = null;
        double? lastSst = null;
        var gaps = 0;
        for (var t = spinupStart; t <= lastBlend; t = t.AddHours(1))
        {
            ct.ThrowIfCancellationRequested();
            if (nwp.TryGetValue(t, out var hourNwp)) lastNwp = hourNwp;
            if (sst.TryGetValue(t, out var hourSst)) lastSst = hourSst;
            var useNwp = lastNwp;
            if (useNwp is null) { gaps++; continue; }

            baseHours.Add(t >= firstBlend && forwardSet.Contains(t)
                // Blend-driven reported hour. Td (+ direct fraction) from NWP.
                ? new BaseHour(t, temp[t].Value, useNwp.Td, cloud[t] / 100.0, EffWind(t), rad[t], useNwp.DirectFrac, lastSst)
                // Spin-up hour (incl. the sub-24h gap before the first blend hour).
                : new BaseHour(t, useNwp.Ta, useNwp.Td, useNwp.CloudPct / 100.0, useNwp.WindMs, useNwp.ShortwaveWm2, useNwp.DirectFrac, lastSst));
        }
        if (gaps > 0)
            log.LogWarning("Rock-surface: {Gaps} hour(s) had no NWP forcing and were skipped — spin-up may be shorter than requested.", gaps);
        if (baseHours.Count == 0)
        {
            log.LogError("Rock-surface: forcing trajectory empty after assembly for {Loc}.", loc);
            return new List<RockSurfacePredictionRow>();
        }

        // --- One integration per modelled surface: whole-crag horizontal when
        //     no faces are configured (Bonehill — SW unmodified, Face = ""),
        //     else one per configured face with the direct beam projected onto
        //     its plane (S3). Solar position is evaluated at the half-hour
        //     midpoint because the hourly SW value is the mean over the
        //     PRECEDING hour (Open-Meteo convention). Emit ONLY the
        //     blend-driven reported hours. ---
        var surfaces = cfg.Faces.Count == 0
            ? new List<RockSurfaceFaceConfig?> { null }
            : cfg.Faces.Select(f => (RockSurfaceFaceConfig?)f).ToList();

        var sstByValid = baseHours.ToDictionary(b => b.ValidTimeUtc, b => b.SeaTempC);
        var rows = new List<RockSurfacePredictionRow>();
        foreach (var face in surfaces)
        {
            var swByValid = new Dictionary<DateTime, double>(baseHours.Count);
            var forcing = new List<RockSurfacePhysics.ForcingHour>(baseHours.Count);
            foreach (var b in baseHours)
            {
                double sw;
                if (face is not null)
                {
                    var (elev, az) = SolarGeometry.SolarPosition(
                        b.ValidTimeUtc.AddMinutes(-30), location.Latitude, location.Longitude);
                    sw = SolarGeometry.FaceIncidentSw(
                        b.SwHorizontalWm2, b.DirectFrac, elev, az, face.AspectDeg, face.SlopeDeg, cfg.FSky);
                }
                else
                {
                    // Whole-crag horizontal mode: damp the direct beam by the
                    // representative direct-beam factor (boulder self-shading +
                    // steep climbing faces rarely catch the full noon beam).
                    // Diffuse untouched, no aspect assumed. factor 1.0 (default)
                    // leaves the horizontal SW unmodified.
                    sw = RockSurfacePhysics.DampedHorizontalSw(
                        b.SwHorizontalWm2, b.DirectFrac, cfg.DirectBeamFactor);
                }
                swByValid[b.ValidTimeUtc] = sw;
                forcing.Add(new RockSurfacePhysics.ForcingHour(
                    b.ValidTimeUtc, b.AirTempC, b.DewPointC, b.CloudFrac, b.WindMs, sw, b.SeaTempC));
            }

            var integrated = RockSurfacePhysics.Integrate(forcing, cfg);
            foreach (var h in integrated)
            {
                if (h.ValidTimeUtc < firstBlend || !forwardSet.Contains(h.ValidTimeUtc)) continue; // spin-up
                var lead = temp[h.ValidTimeUtc].Lead;
                var margin = h.MarginC;
                rows.Add(new RockSurfacePredictionRow
                {
                    LocationName = loc,
                    ModelVersion = OutputModelVersion,
                    Face = face?.Name ?? "",
                    PredictionMadeAtUtc = predictionMadeAtUtc,
                    ValidTimeUtc = h.ValidTimeUtc,
                    LeadHours = lead,
                    RockSurfaceTempC = h.RockTempC,
                    AirTempC = h.AirTempC,
                    DewPointC = h.DewPointC,
                    CondensationMarginC = margin,
                    GreasinessStatus = RockSurfacePhysics.Greasiness(margin, cfg.GreasyMarginC),
                    // Face mode: the FACE-INCIDENT shortwave that actually
                    // drove this row's budget, not the horizontal blend value.
                    ShortwaveDownWm2 = swByValid[h.ValidTimeUtc],
                    CloudCoverPct = cloud[h.ValidTimeUtc],
                    WindSpeed10mMs = EffWind(h.ValidTimeUtc),
                    LongwaveDownWm2 = h.LongwaveDownWm2,
                    DeepTempC = h.DeepTempC,
                    SeaSurfaceTempC = sstByValid[h.ValidTimeUtc],
                    TempModelVersion = tempV,
                    WindModelVersion = windV,
                    RadiationModelVersion = radV,
                    CloudModelVersion = cloudV,
                    DewPointSource = DewPointSourceTag,
                });
            }
        }
        if (surfaces.Count > 1)
            log.LogInformation("Rock-surface: {N} faces integrated for {Loc}.", surfaces.Count, loc);

        return rows.OrderBy(r => r.ValidTimeUtc).ThenBy(r => r.Face, StringComparer.Ordinal).ToList();
    }

    /// <summary>One hour of the shared base trajectory: everything the
    /// integrator needs except the per-face shortwave projection.
    /// <paramref name="SeaTempC"/> null = no marine block / no SST coverage
    /// yet at this hour (neutral surroundings).</summary>
    private readonly record struct BaseHour(
        DateTime ValidTimeUtc, double AirTempC, double DewPointC,
        double CloudFrac, double WindMs, double SwHorizontalWm2, double DirectFrac,
        double? SeaTempC);

    private readonly record struct LeadValue(double Value, int Lead);

    /// <summary>Temperature blend per valid time — smallest lead, freshest made on ties.</summary>
    private static async Task<Dictionary<DateTime, LeadValue>> LoadTempByValidAsync(
        string predictionsRoot, string loc, string version, DateTime anchor, CancellationToken ct)
    {
        var path = Path.Combine(predictionsRoot, "temperature", loc,
            $"model_version={version}", $"date={anchor:yyyy-MM-dd}", "predictions.parquet");
        if (!File.Exists(path)) return new();
        var raw = (await ParquetSerializer.DeserializeAsync<TempPredictionRow>(path, cancellationToken: ct)).ToList();
        return raw
            .GroupBy(r => r.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g =>
                {
                    var best = g.OrderBy(r => r.LeadHours).ThenByDescending(r => r.PredictionMadeAtUtc).First();
                    return new LeadValue(best.BlendTemperature, best.LeadHours);
                });
    }

    /// <summary>Element blend value per valid time — smallest lead, freshest made.
    /// Flat tree (LocationName in-column) → filter to this location.</summary>
    private static async Task<Dictionary<DateTime, double>> LoadElementByValidAsync(
        string predictionsRoot, ElementTarget target, string loc, string version, DateTime anchor, CancellationToken ct)
    {
        var path = Path.Combine(predictionsRoot, target.ModelDirName,
            $"model_version={version}", $"date={anchor:yyyy-MM-dd}", "predictions.parquet");
        if (!File.Exists(path)) return new();
        var raw = (await ParquetSerializer.DeserializeAsync<ElementPredictionRow>(path, cancellationToken: ct))
            .Where(r => string.Equals(r.LocationName, loc, StringComparison.Ordinal))
            .ToList();
        return raw
            .GroupBy(r => r.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(r => r.LeadHours).ThenByDescending(r => r.PredictionMadeAtUtc).First().BlendValue);
    }

    /// <summary>
    /// best_match sea surface temperature per valid time (S4 sea-longwave
    /// forcing) — smallest lead, freshest run, anchored like the NWP query.
    /// offset_day rows excluded; hist_forecast rows admitted (lead-unlabelled
    /// archive — SST moves slowly enough that best-available is fine).
    /// </summary>
    private static Dictionary<DateTime, double> QuerySeaTempByValid(
        string marinePath, string loc, DateTime earliestValid, DateTime latestValid, DateTime asOf, CancellationToken ct)
    {
        var glob = Path.Combine(marinePath, $"location={loc}", "model=best_match", "**", "*.parquet")
            .Replace('\\', '/').Replace("'", "''");

        var sql = $@"
WITH ranked AS (
    SELECT ValidTimeUtc, SeaSurfaceTemperature,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc ORDER BY LeadHours ASC, RunTimeUtc DESC) AS rn
    FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
    WHERE SeaSurfaceTemperature IS NOT NULL
      AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
      AND LeadHours >= 0
      AND RunTimeUtc <= TIMESTAMP '{asOf:yyyy-MM-dd HH:mm:ss}'
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)
SELECT ValidTimeUtc, SeaSurfaceTemperature FROM ranked WHERE rn = 1 ORDER BY ValidTimeUtc;";

        var result = new Dictionary<DateTime, double>();
        try
        {
            using var conn = new DuckDBConnection("DataSource=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = sql;
            using var reader = cmd.ExecuteReader();
            while (reader.Read())
            {
                ct.ThrowIfCancellationRequested();
                if (reader.IsDBNull(0) || reader.IsDBNull(1)) continue;
                result[reader.GetDateTime(0)] = reader.GetDouble(1);
            }
        }
        catch (DuckDBException)
        {
            // Empty/missing marine tree (glob matches nothing) — caller logs
            // the neutral-surroundings fallback.
        }
        return result;
    }

    /// <summary><paramref name="DirectFrac"/> = direct share of horizontal SW
    /// (0..1, from the NWP-mean direct/diffuse split) — cliff-face mode applies
    /// it to the blended GHI to recover the beam for the face projection.
    /// 0 when the radiation split is unavailable (all-diffuse: the face then
    /// receives only fSky-scaled diffuse, the conservative fallback).</summary>
    private sealed record NwpMean(double Ta, double Td, double CloudPct, double WindMs, double ShortwaveWm2, double DirectFrac);

    /// <summary>
    /// Per-valid NWP best-estimate forcing, averaged across the canonical models.
    /// For each (valid, model) the shortest non-negative lead with RunTime ≤ anchor
    /// wins (freshest/closest-to-analysis), then we average across models. Rows
    /// missing dew point are excluded (Td is the load-bearing field). offset_day
    /// (historical-training) rows are excluded.
    /// </summary>
    private static Dictionary<DateTime, NwpMean> QueryNwpMeanByValid(
        string forecastsPath, string loc, DateTime earliestValid, DateTime latestValid, DateTime asOf, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var locSql = loc.Replace("'", "''");
        var modelList = string.Join(", ",
            TempFeatureBuilder.CanonicalModelOrder.Select(m => $"'{m.Replace("'", "''")}'"));

        var sql = $@"
WITH ranked AS (
    SELECT ValidTimeUtc, Model, Temperature2m, DewPoint2m, CloudCover, WindSpeed10m, ShortwaveRadiation,
           DirectRadiation, DiffuseRadiation,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY LeadHours ASC, RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{locSql}'
      AND Model IN ({modelList})
      AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
      AND LeadHours >= 0
      AND RunTimeUtc <= TIMESTAMP '{asOf:yyyy-MM-dd HH:mm:ss}'
      AND DewPoint2m IS NOT NULL
      AND ValidTimeUtc BETWEEN TIMESTAMP '{earliestValid:yyyy-MM-dd HH:mm:ss}'
                           AND TIMESTAMP '{latestValid:yyyy-MM-dd HH:mm:ss}'
)
SELECT ValidTimeUtc,
       avg(Temperature2m)     AS ta,
       avg(DewPoint2m)        AS td,
       avg(CloudCover)        AS cc,
       avg(WindSpeed10m)      AS v,
       avg(ShortwaveRadiation) AS sw,
       avg(DirectRadiation)   AS dr,
       avg(DiffuseRadiation)  AS df
FROM ranked
WHERE rn = 1
GROUP BY ValidTimeUtc
ORDER BY ValidTimeUtc;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var reader = cmd.ExecuteReader();

        var result = new Dictionary<DateTime, NwpMean>();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            if (reader.IsDBNull(0)) continue;
            var valid = reader.GetDateTime(0);
            // Td (col 2) is guaranteed non-null by the WHERE; the others can be
            // null if a model omits them — fall back to 0 / NaN-safe defaults.
            double Col(int i) => reader.IsDBNull(i) ? double.NaN : reader.GetDouble(i);
            var ta = Col(1);
            var td = Col(2);
            if (double.IsNaN(td)) continue;
            var cc = Col(3); if (double.IsNaN(cc)) cc = 0;
            var v = Col(4); if (double.IsNaN(v)) v = 0;
            var sw = Col(5); if (double.IsNaN(sw)) sw = 0;
            if (double.IsNaN(ta)) continue; // air temp is load-bearing for spin-up
            // Direct fraction from the direct/diffuse split (face mode). NaN
            // (no model carried the split) or zero-sum → 0 = all-diffuse.
            var dr = Col(6); var df = Col(7);
            var directFrac = !double.IsNaN(dr) && !double.IsNaN(df) && dr + df > 1e-9
                ? Math.Clamp(dr / (dr + df), 0.0, 1.0)
                : 0.0;
            result[valid] = new NwpMean(ta, td, cc, v, sw, directFrac);
        }
        return result;
    }
}
