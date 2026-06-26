using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Predict;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Commands;

/// <summary>
/// Per-lead policy producers (docs/PRECIP_LEAD_POLICY_PLAN.md). Three CLI
/// commands fit + emit the per-(target, location) <c>LEAD_POLICY_&lt;loc&gt;.json</c>
/// that <see cref="PrecipPredictCommand"/> / <see cref="TempPredictCommand"/>
/// consult to pick, per 6h band, which lead model (or equal-weight blend) serves
/// that band — incl. the lead-0 nowcast at the &lt;24hr bands:
///   • <see cref="RunPolicyRetrainAsync"/> — mints the no-UA, cutoff-trained 3c
///     (+ 3o where the Bonehill oro pool exists) walk-forward study bundles.
///   • <see cref="RunFitLeadPolicyAsync"/> — precip producer: scores every
///     candidate + equal-weight blend at each band on LIVE inputs vs EA truth,
///     SELECT/SCORE split, margin/hysteresis/truth gates → emits the policy.
///   • <see cref="RunFitLeadPolicyTempAsync"/> — temperature twin (2c study
///     models trained in-process, MAE vs ERA5).
///
/// Candidate scoring reuses the production predict pivot
/// (<c>RunTimeUtc &lt;= ValidTimeUtc - leadHoursLowerBound</c>, freshest cycle
/// ≥ L h stale) + the production feature builders (rich / rich-oro, persistence,
/// upper-air, terrain). LIVE forecasts only (offset_day + hist_forecast excluded)
/// so candidates score on what predict sees. Per-location via <c>--location</c>.
///
/// (Class name retains the historical "Bakeoff" prefix; the one-off bake-off /
/// crossover diagnostics it once hosted were removed 2026-06-15 — git history
/// has them. A rename is deferred to avoid churning the DI/CLI wiring.)
/// </summary>
public sealed class PrecipCrossLeadBakeoffCommand
{
    private const double WetThresholdMm = 0.1;
    private const int ModelLead = 24;
    private static readonly int[] InputLeads = { 12, 18, 24 };

    // Mirror PrecipPredictCommand.Phase3oStationIndex (train-time station order).
    private static readonly IReadOnlyDictionary<string, int> Phase3oStationIndex =
        new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            ["ea_bellever_dartmoor"]     = 0,
            ["ea_bovey_tracey"]          = 1,
            ["ea_dartmoor_nr_hexworthy"] = 2,
            ["ea_princetown"]            = 3,
        };

    private readonly ILogger<PrecipCrossLeadBakeoffCommand> _log;
    private readonly AppConfig _cfg;

    public PrecipCrossLeadBakeoffCommand(ILogger<PrecipCrossLeadBakeoffCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    // ---- 3c rich predict (mirror of PrecipPredictCommand.RunStationAsync rich path) ----
    private static double? Predict3c(
        MLContext ml, ITransformer model, BlenderSpec spec, List<string> canonOrder,
        DateTime valid, int inputLead, PivotedRow pivot,
        Dictionary<DateTime, double> hourlyRain,
        IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)> ua,
        double[]? overrideUa = null)
    {
        int N = spec.Models.Count;
        var specPrecip = new double[N];
        var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var dew = new double[N]; var rh = new double[N]; var dewDep = new double[N]; var pres = new double[N];
        for (int i = 0; i < N; i++)
        {
            var ci = canonOrder.IndexOf(spec.Models[i]);
            var pv = pivot.Precip[ci];
            specPrecip[i] = pv ?? double.NaN;
            if (!pv.HasValue && requiredSet.Contains(spec.Models[i])) return null;
            var t = pivot.Temp2m[ci]; var dd = pivot.Dew[ci];
            dew[i] = dd ?? double.NaN;
            rh[i] = pivot.Rh[ci] ?? double.NaN;
            dewDep[i] = (t.HasValue && dd.HasValue) ? t.Value - dd.Value : double.NaN;
            pres[i] = pivot.Pressure[ci] ?? double.NaN;
        }

        // Leak-safe anchor: identity for ≥24h leads (so the policy eval is
        // unchanged), but at lead 0 it back-shifts off valid so persistence
        // can't read the predicted hour's own gauge value. Mirrors training
        // (PrecipRichFeatureBuilder.BuildForLead → NowcastSource).
        var runTime = valid.AddHours(-NowcastSource.PersistenceAnchorHours(inputLead));
        var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain, runTime);
        double[]? uaValues = overrideUa ?? (spec.FeatureNames.Contains("t850_mean")
            ? PrecipFeatureBuilder.UpperAirValuesFor(ua, valid) : null);

        var row = PrecipRichFeatureBuilder.ComposeRow(
            spec, valid, specPrecip, dew, rh, dewDep, pres,
            rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
            cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
            cloudHighMean: pivot.CloudHighMean,
            capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
            eaRainPrev24hMm: persist.Prev24hMm,
            eaRainPrev72hMm: persist.Prev72hMm,
            eaWetHoursLast24h: persist.WetHoursLast24h,
            eaDryHoursTrailing: persist.DryHoursTrailing,
            truthMmHour: 0.0,
            upperAir: uaValues);

        return PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, new[] { row })[0];
    }

    // ---- 3o rich-oro predict (mirror of PrecipPredictCommand.RunStationAsOroAsync) ----
    private static double? Predict3o(
        MLContext ml, ITransformer model, BlenderSpec spec, List<string> canonOrder,
        DateTime valid, int inputLead, PivotedRow pivot,
        Dictionary<DateTime, double> hourlyRain,
        IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)> ua,
        OroStaticFeatures oro,
        Dictionary<DateTime, PrecipRichOroFeatureBuilder.NwpMeanRow> nwpMeanByValid,
        int stationIndex)
    {
        if (!nwpMeanByValid.TryGetValue(valid, out var nwpMean)) return null;

        int richDim = spec.FeatureCount - PrecipRichOroFeatureBuilder.TerrainFeatureCount;
        var richSpec = new BlenderSpec
        {
            Target = spec.Target,
            FeatureSet = PrecipRichFeatureBuilder.SpecFeatureSet,
            LeadHours = spec.LeadHours,
            RequiredModels = spec.RequiredModels,
            OptionalModels = spec.OptionalModels,
            Models = spec.Models,
            FeatureNames = spec.FeatureNames.Take(richDim).ToList(),
            DataSource = spec.DataSource,
            Tier = PrecipRichFeatureBuilder.SpecFeatureSet,
            UkvStrategy = spec.UkvStrategy,
        };

        int N = spec.Models.Count;
        var specPrecip = new double[N];
        var requiredSet = spec.RequiredModels.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var dew = new double[N]; var rh = new double[N]; var dewDep = new double[N]; var pres = new double[N];
        for (int i = 0; i < N; i++)
        {
            var ci = canonOrder.IndexOf(spec.Models[i]);
            var pv = pivot.Precip[ci];
            specPrecip[i] = pv ?? double.NaN;
            if (!pv.HasValue && requiredSet.Contains(spec.Models[i])) return null;
            var t = pivot.Temp2m[ci]; var dd = pivot.Dew[ci];
            dew[i] = dd ?? double.NaN;
            rh[i] = pivot.Rh[ci] ?? double.NaN;
            dewDep[i] = (t.HasValue && dd.HasValue) ? t.Value - dd.Value : double.NaN;
            pres[i] = pivot.Pressure[ci] ?? double.NaN;
        }

        // Leak-safe anchor (identity for ≥24h leads; back-shifts off valid at
        // lead 0 so persistence can't read the predicted hour) — mirrors
        // Predict3c + training (NowcastSource).
        var runTime = valid.AddHours(-NowcastSource.PersistenceAnchorHours(inputLead));
        var persist = PrecipRichFeatureBuilder.ComputePersistence(hourlyRain, runTime);
        double[]? uaValues = richSpec.FeatureNames.Contains("t850_mean")
            ? PrecipFeatureBuilder.UpperAirValuesFor(ua, valid) : null;

        var richRow = PrecipRichFeatureBuilder.ComposeRow(
            richSpec, valid, specPrecip, dew, rh, dewDep, pres,
            rhMean: pivot.RhMean, dewDepressionMean: pivot.DewDepressionMean,
            cloudLowMean: pivot.CloudLowMean, cloudMidMean: pivot.CloudMidMean,
            cloudHighMean: pivot.CloudHighMean,
            capeMean: pivot.CapeMean, windSpeedMean: pivot.WindSpeedMean,
            eaRainPrev24hMm: persist.Prev24hMm,
            eaRainPrev72hMm: persist.Prev72hMm,
            eaWetHoursLast24h: persist.WetHoursLast24h,
            eaDryHoursTrailing: persist.DryHoursTrailing,
            truthMmHour: 0.0,
            upperAir: uaValues);

        var terrain = PrecipRichOroFeatureBuilder.ComposeTerrainBlock(oro, nwpMean, stationIndex);
        var features = new float[spec.FeatureCount];
        Array.Copy(richRow.Features, features, richDim);
        for (int i = 0; i < PrecipRichOroFeatureBuilder.TerrainFeatureCount; i++)
            features[richDim + i] = terrain[i];
        var fullRow = new BinaryTrainingRow { ValidTimeUtc = valid, Features = features, Label = false, TruthMmHour = 0.0f };

        return PrecipOccurrenceTrainer.PredictVectorProbability(ml, model, spec, new[] { fullRow })[0];
    }

    // ===================== STUDY retrain (walk-forward, no-UA, cutoff) =====================
    //
    // Mints local study bundles for the per-lead policy study: 3c (per-gauge) +
    // 3o (pooled), Bonehill, leads {24,48,72,96,120}, trained ONLY on offset_day
    // data ≤ cutoff (so the live scoring window stays OOS), with upper-air OFF.
    // Writes to data/models_study/ (NOT the production tree, no manifest promote).
    // Mirrors PrecipTrainCommand's 3c/3o train loops but isolated + parameterised.

    private static readonly string[] BonehillOrder3o =
        { "Bellever Dartmoor", "Bovey Tracey", "Dartmoor nr Hexworthy", "Princetown", "Manaton" };

    /// <summary>
    /// Resolve a <c>--location</c> override to its <see cref="Config.LocationConfig"/>,
    /// mirroring <c>TrainCommand</c>. Blank = the primary location (Bonehill).
    /// Returns null (and logs) for an unknown name so the policy producers can
    /// exit-2 — the same pattern the trainers use.
    /// </summary>
    internal Config.LocationConfig? ResolveLocation(string? locationOverride)
    {
        if (string.IsNullOrWhiteSpace(locationOverride)) return _cfg.Location;
        var loc = _cfg.Locations.FirstOrDefault(l =>
            l.Name.Equals(locationOverride, StringComparison.OrdinalIgnoreCase));
        if (loc is null)
            _log.LogError("Location '{Name}' not found in config.yaml's `locations:` list. Available: [{All}]",
                locationOverride, string.Join(", ", _cfg.Locations.Select(l => l.Name)));
        return loc;
    }

    public async Task<int> RunPolicyRetrainAsync(string? asOfStr, string? locationOverride, CancellationToken ct)
    {
        await Task.Yield();
        var location = ResolveLocation(locationOverride);
        if (location is null) return 2;
        var cutoff = DateOnly.TryParse(asOfStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var min3c = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc); // matches phases.yaml 3c minValidTime
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
        var ml = new MLContext(seed: 42);
        // Include lead 0 (the nowcast model, sourced from hist_forecast by the
        // builders) so the fit can score it as a candidate at the short bands.
        var leads = new[] { NowcastSource.LeadHours }.Concat(Leads.Full).ToArray();

        // Scan-once cache: the per-(phase,gauge,lead) BuildForLead each re-globs the whole
        // forecast tree (thousands of partition files) and only differ by the in-file LeadHours
        // filter — so a 40-call loop would scan the same data 40×. Consolidate Bonehill
        // forecasts (≤cutoff) + rainfall into single parquets ONCE; point every build at them
        // → ~40 one-file reads instead of ~40 full-tree globs.
        var scratch = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_study");
        var fcPart = Path.Combine(scratch, "fc", "p"); Directory.CreateDirectory(fcPart);
        var rnPart = Path.Combine(scratch, "rn", "p"); Directory.CreateDirectory(rnPart);
        var fcPath = Path.Combine(scratch, "fc");
        var rnPath = Path.Combine(scratch, "rn");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=" + location.Name, "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
                $"WHERE ValidTimeUtc <= TIMESTAMP '{cutoff:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(fcPart, "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.RainfallPath, "location=" + location.Name, "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true)) " +
                $"TO '{Esc(Path.Combine(rnPart, "rn.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }
        _log.LogInformation("Scan-once cache materialised → {Scratch} (forecasts ≤{Cut:yyyy-MM-dd} + rainfall, {Loc})", scratch, cutoff, location.Name);

        _log.LogInformation("STUDY retrain — loc={Loc}, train ValidTime ≤ {Cut:yyyy-MM-dd} (live window stays OOS), UA OFF; out={Root}",
            location.Name, cutoff, studyRoot);

        // ---- 3c: per-gauge, rich (no-UA). WeatherLink gauges (e.g. Lands End)
        // flow through exactly like EA gauges — they are just a different truth
        // source: slug is wl_* (s.Slug), and the wet label is read from
        // data/truth/weatherlink instead of the EA tree. ----
        foreach (var s in location.ProductRainfallStations)   // 3c study: product gauges only
        {
            ct.ThrowIfCancellationRequested();
            var slug = s.Slug;
            var (wlPath, wlKey) = s.WeatherLinkTruth(_cfg.Storage);
            var versionDir = Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3c_noua");
            Directory.CreateDirectory(versionDir);
            var specs = new Dictionary<int, BlenderSpec>();
            foreach (var lead in leads)
            {
                ct.ThrowIfCancellationRequested();
                var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: false);
                specs[lead] = spec;
                var rows = PrecipRichFeatureBuilder.BuildForLead(
                    fcPath, rnPath, location.Name, s.Name, spec,
                    minValidTime: min3c, ct: ct, maxValidTime: cutoff,
                    weatherLinkTruthPath: wlPath,
                    weatherLinkTruthLocation: wlKey);
                if (rows.Count < 500) { _log.LogWarning("  3c {Slug} L{Lead}h: only {N} rows ≤cutoff — skipping.", slug, lead, rows.Count); continue; }
                var ds = BinaryDataset.Split(rows);
                var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
                ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, versionDir, lead);
                _log.LogInformation("  3c {Slug} L{Lead}h: rows={N} train={T} (last valid {E:yyyy-MM-dd}) → saved.",
                    slug, lead, rows.Count, ds.Train.Count, rows[^1].ValidTimeUtc);
            }
            ModelArtifact.SaveBlenderSpecs(versionDir, specs);
        }

        // ---- 3o: pooled across the 4 Bonehill gauges, rich-oro (no-UA) ----
        // 3o is the Bonehill Dartmoor product (4-gauge orographic pool). It does
        // not exist for single-gauge lowland/coastal locations (Membury, Sennen),
        // whose precip champion falls back to 3c — and the policy fit already
        // skips 3o for stations without an oro record. So when this location
        // doesn't carry the full Bonehill pool, mint 3c study bundles only and
        // skip 3o rather than erroring (the old behaviour returned exit-2, which
        // is why precip-policy-retrain was Bonehill-only).
        var hasOroPool = BonehillOrder3o.All(name =>
            location.Rainfall.Stations.Any(x => x.Name.Equals(name, StringComparison.OrdinalIgnoreCase)));
        if (!hasOroPool)
        {
            _log.LogInformation("STUDY retrain DONE — 3c-only (no 3o oro pool for {Loc}) under {Root}", location.Name, studyRoot);
            return 0;
        }
        var oroRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        var pool = new List<(string Name, string Slug, OroStaticFeatures Oro, int Index, string? WlPath, string? WlLoc)>();
        for (int i = 0; i < BonehillOrder3o.Length; i++)
        {
            var match = location.Rainfall.Stations.FirstOrDefault(x => x.Name.Equals(BonehillOrder3o[i], StringComparison.OrdinalIgnoreCase));
            if (match is null) { _log.LogWarning("3o pool: station '{N}' not in config — skipping from study pool.", BonehillOrder3o[i]); continue; }
            var slug = match.Slug;   // wl_manaton for the WeatherLink pool member; ea_* otherwise
            if (!oroBySlug.TryGetValue(slug, out var oro)) { _log.LogError("3o pool: no oro record for {S}.", slug); return 2; }
            var (wlPath, wlLoc) = match.WeatherLinkTruth(_cfg.Storage);
            pool.Add((match.Name, slug, oro, i, wlPath, wlLoc));
        }
        var stationDirs = pool.ToDictionary(p => p.Slug, p => Path.Combine(studyRoot, "precipitation", p.Slug, "vstudy_phase3o_noua"));
        foreach (var dir in stationDirs.Values) Directory.CreateDirectory(dir);
        var specs3o = new Dictionary<int, BlenderSpec>();
        foreach (var lead in leads)
        {
            ct.ThrowIfCancellationRequested();
            var spec = PrecipRichOroFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: false);
            specs3o[lead] = spec;
            var perStation = new List<(string Slug, BinaryDataset Ds)>();
            foreach (var (name, slug, oro, idx, wlPath, wlLoc) in pool)
            {
                ct.ThrowIfCancellationRequested();
                var rows = PrecipRichOroFeatureBuilder.BuildForLead(
                    fcPath, rnPath, location.Name, name, oro, idx, spec,
                    ct: ct, maxValidTime: cutoff,
                    weatherLinkTruthPath: wlPath, weatherLinkTruthLocation: wlLoc);
                if (rows.Count < 200) { _log.LogWarning("  3o {Slug} L{Lead}h: only {N} rows — skipping station.", slug, lead, rows.Count); continue; }
                perStation.Add((slug, BinaryDataset.Split(rows)));
            }
            if (perStation.Count < 2) { _log.LogError("  3o L{Lead}h: <2 usable stations — skipping lead.", lead); continue; }
            var pooledTrain = perStation.SelectMany(p => p.Ds.Train).ToList();
            var pooledVal = perStation.SelectMany(p => p.Ds.Val).ToList();
            var trained = PrecipOccurrenceTrainer.TrainVector(pooledTrain, pooledVal, spec, hp);
            foreach (var dir in stationDirs.Values)
                ModelArtifact.SaveLeadModel(trained.Ml, trained.Model, trained.InputSchema, dir, lead);
            _log.LogInformation("  3o L{Lead}h: pooled train={T} (wet {W:P1}) across {S} gauges → saved to all.",
                lead, pooledTrain.Count, pooledTrain.Count(r => r.Label) / (double)Math.Max(1, pooledTrain.Count), perStation.Count);
        }
        foreach (var dir in stationDirs.Values) ModelArtifact.SaveBlenderSpecs(dir, specs3o);

        _log.LogInformation("STUDY retrain DONE — 3c+3o no-UA bundles (cutoff {Cut:yyyy-MM-dd}) under {Root}", cutoff, studyRoot);
        return 0;
    }

    // ===================== temperature pivot + predict helpers =====================
    // Shared by RunFitLeadPolicyTempAsync: the per-model temp pivot (freshest
    // live cycle ≥ τh stale, non-offset_day), ERA5 truth load, and the 2c-rich
    // predict. Feature layout is lead-independent so a model trained at one lead
    // scores any τ's pivot.
    internal sealed record TempPivot(
        double?[] Temp, double?[] Dew, double?[] Rh, double?[] Cloud,
        double?[] CloudLow, double?[] CloudMid, double?[] CloudHigh,
        double?[] WindSpeed, double?[] WindDir, double?[] WindGust, double?[] Pressure);

    // ERA5 hourly 2m-temperature truth (valid→°C) for one location + window.
    internal static IReadOnlyList<(DateTime Valid, double TempC)> LoadEra5Temp(
        string era5Path, string locationName, DateTime earliest, DateTime latest, CancellationToken ct)
    {
        var glob = Path.Combine(era5Path, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var esc = locationName.Replace("'", "''");
        var sql = $@"
SELECT ValidTimeUtc, Temperature2m
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{esc}' AND Temperature2m IS NOT NULL
  AND ValidTimeUtc >  TIMESTAMP '{earliest:yyyy-MM-dd HH:mm:ss}'
  AND ValidTimeUtc <= TIMESTAMP '{latest:yyyy-MM-dd HH:mm:ss}'
ORDER BY ValidTimeUtc;";
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<(DateTime, double)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            rows.Add((r.GetDateTime(0), r.GetDouble(1)));
        }
        return rows;
    }

    // Freshest live cycle (RunTime ≤ valid − τ, non-offset_day) per (valid,
    // model) — the temp-rich per-model pivot. Indexed by canonical model slot.
    internal static IReadOnlyDictionary<DateTime, TempPivot> QueryLatestTempRows(
        string forecastsPath, string locationName,
        DateTime earliestValid, DateTime latestValid, DateTime asOfRunTime, int leadHoursLowerBound,
        IReadOnlyList<string> canonOrder, CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var filter = PredictForecastFilters.LiveCycleAsOf(
            locationName, asOfRunTime, earliestValid, latestValid, leadHoursLowerBound);
        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc,
           Temperature2m, DewPoint2m, RelativeHumidity2m, CloudCover,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh,
           WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {filter}
)
SELECT ValidTimeUtc, Model,
       Temperature2m, DewPoint2m, RelativeHumidity2m, CloudCover,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh,
       WindSpeed10m, WindDirection10m, WindGusts10m, SurfacePressure
FROM latest WHERE rn = 1;";

        var slot = canonOrder.Select((id, i) => (id, i)).ToDictionary(x => x.id, x => x.i);
        int n = canonOrder.Count;
        var acc = new Dictionary<DateTime, TempPivot>();
        TempPivot New() => new(new double?[n], new double?[n], new double?[n], new double?[n],
            new double?[n], new double?[n], new double?[n], new double?[n], new double?[n], new double?[n], new double?[n]);

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();
        double? G(int i) => r.IsDBNull(i) ? (double?)null : r.GetDouble(i);
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = r.GetDateTime(0);
            if (!slot.TryGetValue(r.GetString(1), out var si)) continue;
            if (!acc.TryGetValue(valid, out var p)) { p = New(); acc[valid] = p; }
            p.Temp[si] = G(2); p.Dew[si] = G(3); p.Rh[si] = G(4); p.Cloud[si] = G(5);
            p.CloudLow[si] = G(6); p.CloudMid[si] = G(7); p.CloudHigh[si] = G(8);
            p.WindSpeed[si] = G(9); p.WindDir[si] = G(10); p.WindGust[si] = G(11); p.Pressure[si] = G(12);
        }
        return acc;
    }

    // Build the temp-rich row from a pivot (canonical-slot indexed) in spec.Models
    // order and predict. Null when no model has a temperature for this valid.
    private static double? PredictTemp(
        TempTrainer.TrainedBlender m, BlenderSpec spec, List<string> canonOrder,
        DateTime valid, TempPivot pv)
    {
        int n = spec.Models.Count;
        double[] Col(Func<TempPivot, double?[]> sel)
        {
            var src = sel(pv);
            var outv = new double[n];
            for (int i = 0; i < n; i++)
            {
                var ci = canonOrder.IndexOf(spec.Models[i]);
                outv[i] = ci >= 0 && src[ci].HasValue ? src[ci]!.Value : double.NaN;
            }
            return outv;
        }
        var temps = Col(p => p.Temp);
        if (temps.All(double.IsNaN)) return null;
        var row = TempRichFeatureBuilder.ComposeRow(
            spec, valid, temps,
            dewPoints: Col(p => p.Dew), rhs: Col(p => p.Rh), clouds: Col(p => p.Cloud),
            cloudLows: Col(p => p.CloudLow), cloudMids: Col(p => p.CloudMid), cloudHighs: Col(p => p.CloudHigh),
            windSpeeds: Col(p => p.WindSpeed), windDirsDeg: Col(p => p.WindDir),
            windGusts: Col(p => p.WindGust), pressures: Col(p => p.Pressure),
            windDirMeanDeg: 0, era5Temp: 0);
        return TempTrainer.PredictVector(m.Ml, m.Model, spec, new[] { row })[0];
    }

    // ===================== Phase 1 producer: fit + emit LEAD_POLICY.json =====================
    //
    // docs/PRECIP_LEAD_POLICY_PLAN.md Phase 1. Combines the band-eval
    // methodology (RunPolicyBandAsync) with the SELECT/SCORE date split
    // (RunPolicyEvalSplitAsync), then applies the locked governance gates and
    // emits data/models/precipitation/LEAD_POLICY.json:
    //   * margin gate 0.75%: a NEW deviation from the production bucket model
    //     enters only when its held-out SCORE Brier beats the bucket baseline
    //     by ≥ MarginPct;
    //   * hysteresis 0.5%: an INCUMBENT band entry (from the existing
    //     LEAD_POLICY.json) is only displaced — by a different candidate or by
    //     reversion to baseline — when the challenger beats it by ≥
    //     HysteresisPct on SCORE. Equivalently an incumbent re-qualifies at
    //     (MarginPct − HysteresisPct); a fresh entry needs the full margin.
    //   * truth-settled guard: if the SCORE window's pooled EA truth coverage
    //     is below 70% of expected hours, SKIP the update entirely and keep
    //     the last-good policy (the 2026-06 EA outage failure mode).
    //   * 3c is SINGLES-ONLY (locked 2026-06-09: its blend winners flip pair
    //     per band — overfit noise); 3o may use equal-weight pairs.
    // Choices are made on the SELECT slice and graded on SCORE, so no
    // candidate is gated on its own selection data. Absent bands in the
    // artifact mean "production bucket" — an empty policy is a no-op.

    public async Task<int> RunFitLeadPolicyAsync(
        string? startDateStr, string? cutoffStr, string? locationOverride, CancellationToken ct)
    {
        await Task.Yield();
        var location = ResolveLocation(locationOverride);
        if (location is null) return 2;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var studyRoot = Path.Combine(Path.GetDirectoryName(modelsRoot)!, "models_study");
        var thresholds = new PrecipLeadPolicy.ThresholdsBlock();   // locked defaults

        var windowStart = DateOnly.TryParse(startDateStr, out var d0)
            ? d0.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var studyCutoff = DateOnly.TryParse(cutoffStr, out var dc)
            ? dc.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        var splitDate = windowEnd.Date.AddDays(-thresholds.HoldoutDays);
        if (splitDate <= windowStart.AddDays(7))
        {
            _log.LogError("fit-lead-policy: SELECT slice would be <7 days ({S:yyyy-MM-dd}..{Split:yyyy-MM-dd}) — widen --start.",
                windowStart, splitDate);
            return 2;
        }

        // Lead 0 (nowcast) is a candidate; taus run from 0 (3-spaced) so the
        // 0-6h band exists. Per-band eligibility is the ±24h window applied
        // below (|model lead − band| ≤ 24), NOT "any model anywhere".
        int[] cands = { 0, 24, 48, 72, 96, 120 };
        var taus = Enumerable.Range(0, 117 / 3 + 1).Select(i => 3 * i).ToArray();
        const int CandWindowHours = 24;   // a model lead L competes at band lo iff |L − lo| ≤ 24
        var pairs = new List<(int Lo, int Hi)>();
        for (int i = 0; i < cands.Length; i++)
            for (int j = i + 1; j < cands.Length; j++) pairs.Add((cands[i], cands[j]));

        _log.LogInformation(
            "fit-lead-policy — window {S:yyyy-MM-dd}..{E:yyyy-MM-dd}, SELECT<{Split:yyyy-MM-dd}≤SCORE ({Hold}d holdout), " +
            "study cutoff {Cut:yyyy-MM-dd}, margin {M}%, hysteresis {H}%",
            windowStart, windowEnd, splitDate, thresholds.HoldoutDays, studyCutoff,
            thresholds.MarginPct, thresholds.HysteresisPct);

        // ---- scan-once live forecast cache (same shape as the band eval) ----
        var fcPart = Path.Combine(Path.GetDirectoryName(modelsRoot)!, "scratch", "policy_fit", "fc", "p");
        Directory.CreateDirectory(fcPart);
        var fcPath = Path.Combine(Path.GetDirectoryName(modelsRoot)!, "scratch", "policy_fit", "fc");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=" + location.Name, "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
                // LIVE FORECASTS ONLY — exclude both offset_day (historical
                // backfill) AND hist_forecast (the lead-0 archive ≈ analysis).
                // The policy must score every candidate, ESPECIALLY the 0h model
                // at the 0-2h bucket, on exactly what predict sees live: the
                // freshest real cycle. The hist_forecast archive (RunTime=valid)
                // would out-rank live cycles at τ=0 and isn't available at live
                // predict time — scoring on it would falsely qualify the 0h model
                // (e.g. precip's unbankable −4.2% τ=0). This only matters since
                // hist_forecast was backfilled into the scoring window (2026-06-14).
                $"WHERE (RunTimeSource IS NULL OR RunTimeSource NOT IN ('offset_day', 'hist_forecast')) AND ValidTimeUtc BETWEEN TIMESTAMP '{windowStart:yyyy-MM-dd HH:mm:ss}' AND TIMESTAMP '{windowEnd:yyyy-MM-dd HH:mm:ss}') " +
                $"TO '{Esc(Path.Combine(fcPart, "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }

        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var emptyUa = System.Array.Empty<(DateTime, double[])>();
        var oroBySlug = OroStaticFeatures.LoadAll(Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic"));
        // WeatherLink gauges (e.g. Lands End) flow through like EA gauges — slug is
        // wl_* and truth + persistence read from data/truth/weatherlink. Carry the
        // station config and read the truth source off it per gauge.
        var gauges = location.ProductRainfallStations;   // policy scored on product gauges only
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var s in gauges)
        {
            var slug = s.Slug;
            var (wlPath, wlKey) = s.WeatherLinkTruth(_cfg.Storage);
            truth[slug] = wlKey is { } wlLoc
                ? LoadHourlyTruthWeatherLink(wlPath!, wlLoc, windowStart, windowEnd, ct)
                : LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, s.Name, windowStart, windowEnd, ct);
            rain[slug] = PrecipTruthLoader.LoadHourlyRain(s, _cfg.Storage, location.Name, null, ct);
            foreach (var (dir, dst) in new[] {
                (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3c_noua"), m3c),
                (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3o_noua"), m3o) })
            {
                if (!Directory.Exists(dir)) continue;
                var specs = ModelArtifact.LoadBlenderSpecs(dir);
                var byLead = new Dictionary<int, (ITransformer, BlenderSpec)>();
                foreach (var lead in cands)
                    if (specs.TryGetValue(lead, out var sp))
                        byLead[lead] = (ModelArtifact.LoadLeadModel(ml, dir, lead, out _), sp);
                dst[slug] = byLead;
            }
        }
        if (m3c.Count == 0 && m3o.Count == 0)
        {
            _log.LogError("fit-lead-policy: no study bundles under {Root} — run precip-policy-retrain first.", studyRoot);
            return 2;
        }
        // Effective candidate set: drop the lead-0 nowcast when no study bundle
        // carries it (a location with no hist_forecast surface archive yet). The
        // per-station gate below requires cands.All(byLead.ContainsKey); with 0
        // in cands but no lead-0 model that gate would skip EVERY station, empty
        // the policy, and wipe the location's existing ≥24h bands. The ≥24h
        // study leads are always present, so this only ever removes 0. Rebuild
        // the blend pairs from the effective set so the scoring + band loops agree.
        if (!m3c.Values.Concat(m3o.Values).Any(bl => bl.ContainsKey(NowcastSource.LeadHours)))
        {
            cands = cands.Where(c => c != NowcastSource.LeadHours).ToArray();
            pairs.Clear();
            for (int i = 0; i < cands.Length; i++)
                for (int j = i + 1; j < cands.Length; j++) pairs.Add((cands[i], cands[j]));
            _log.LogInformation("fit-lead-policy: no lead-0 study model for {Loc} — nowcast candidate dropped; cands now [{C}].",
                location.Name, string.Join(",", cands));
        }
        var anySpec3o = m3o.Values.FirstOrDefault()?.Values.FirstOrDefault().Spec;
        var aux = anySpec3o is null ? new() : PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(fcPath, location.Name, anySpec3o, windowStart, windowEnd, ct);

        // ---- truth-settled guard (BEFORE any scoring) ----
        // Pooled EA hourly truth in the SCORE window must cover ≥70% of the
        // expected hours, else this run keeps the last-good policy. Guards the
        // EA-outage failure mode: scoring a half-empty SCORE slice would gate
        // policy changes on noise.
        var expectedScoreHours = (windowEnd - splitDate).TotalHours * gauges.Count;
        var actualScoreHours = gauges.Sum(g => truth[g.Slug].Count(t => t.Item1 >= splitDate));
        var coverage = expectedScoreHours <= 0 ? 0.0 : actualScoreHours / expectedScoreHours;
        if (coverage < 0.70)
        {
            _log.LogWarning(
                "fit-lead-policy: SCORE-window truth coverage {Cov:P0} < 70% (EA latency/outage?) — " +
                "SKIPPING the policy update, last-good LEAD_POLICY.json stays.", coverage);
            return 0;
        }
        _log.LogInformation("fit-lead-policy: SCORE-window truth coverage {Cov:P0} — proceeding.", coverage);

        // ---- score every candidate + pair at every τ, split SELECT/SCORE ----
        var selSq = new Dictionary<string, double>(); var scoSq = new Dictionary<string, double>();
        var selN = new Dictionary<string, int>(); var scoN = new Dictionary<string, int>();
        void Add(Dictionary<string, double> sq, Dictionary<string, int> n, string ptk, string series, double err2)
            => sq[$"{ptk}|{series}"] = sq.GetValueOrDefault($"{ptk}|{series}") + err2;

        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestForecastRows(fcPath, location.Name, windowStart, windowEnd, asOf, tau, ct);
            foreach (var s in gauges)
            {
                var slug = s.Slug;
                if (!Phase3oStationIndex.TryGetValue(slug, out var stIdx)) stIdx = -1;
                oroBySlug.TryGetValue(slug, out var oro);
                foreach (var (phase, store) in new[] { ("3c", m3c), ("3o", m3o) })
                {
                    if (!store.TryGetValue(slug, out var byLead) || !cands.All(byLead.ContainsKey)) continue;
                    if (phase == "3o" && (oro is null || stIdx < 0)) continue;
                    var ptk = $"{phase}|{tau}";
                    foreach (var (V, mm) in truth[slug])
                    {
                        if (!pivot.TryGetValue(V, out var pv) || !pv.Precip.Any(p => p.HasValue)) continue;
                        var preds = new double[cands.Length];
                        var ok = true;
                        for (int i = 0; i < cands.Length; i++)
                        {
                            var ms = byLead[cands[i]];
                            double? p = phase == "3c"
                                ? Predict3c(ml, ms.Model, ms.Spec, canon, V, tau, pv, rain[slug], emptyUa)
                                : Predict3o(ml, ms.Model, ms.Spec, canon, V, tau, pv, rain[slug], emptyUa, oro!, aux, stIdx);
                            if (p is null) { ok = false; break; }
                            preds[i] = p.Value;
                        }
                        if (!ok) continue;
                        var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                        var (sq, n) = V < splitDate ? (selSq, selN) : (scoSq, scoN);
                        n[ptk] = n.GetValueOrDefault(ptk) + 1;
                        for (int i = 0; i < cands.Length; i++)
                            Add(sq, n, ptk, $"s{cands[i]}", (preds[i] - y) * (preds[i] - y));
                        foreach (var (lo, hi) in pairs)
                        {
                            var bl = 0.5 * (preds[Array.IndexOf(cands, lo)] + preds[Array.IndexOf(cands, hi)]);
                            Add(sq, n, ptk, $"b{lo}x{hi}", (bl - y) * (bl - y));
                        }
                    }
                }
            }
        }

        // ---- per-band decisions + gates ----
        var incumbent = PrecipLeadPolicy.TryLoad(modelsRoot, "precipitation", location.Name);
        var policy = new PrecipLeadPolicy
        {
            FittedAtUtc = asOf,
            Location = location.Name,
            WindowStartUtc = windowStart,
            WindowEndUtc = windowEnd,
            StudyCutoffUtc = studyCutoff,
            SelectScoreSplitUtc = splitDate,
            Thresholds = thresholds,
        };
        const int MinScoreN = 300;   // a band decided on fewer held-out rows is noise — stay on baseline

        Console.WriteLine();
        Console.WriteLine("=== fit-lead-policy: 6h-band decisions (SELECT picks, SCORE grades; [base] *final) ===");
        foreach (var phase in new[] { "3c", "3o" })
        {
            // ONE asymmetry between the phases (Harry, 2026-06-10): both run
            // the same gates ("the fit chooses, the gates protect" — singles
            // CAN deviate for either phase), but blends are 3o-only. 3c's
            // winning pair flips in nearly every band across refits — that's
            // selection noise, not signal — while its single-model deviations
            // pass the same SELECT/SCORE discipline as everything else.
            var allowBlend = phase == "3o";
            var entries = new List<PrecipLeadPolicy.BandEntry>();
            Console.WriteLine();
            Console.WriteLine($"### {phase}{(allowBlend ? "" : "   (singles only — locked)")}");
            Console.WriteLine($"  {"band",9} {"Nsco",6} {"baseline",13} {"SELECT pick",16} {"SCORE",8} {"decision",18}");
            for (int lo = 0; lo < 120; lo += 6)
            {
                int hi = lo + 6;
                var tin = taus.Where(t => t >= lo && t < hi).ToArray();
                double SelB(string s) { var n = tin.Sum(t => selN.GetValueOrDefault($"{phase}|{t}")); return n == 0 ? double.NaN : tin.Sum(t => selSq.GetValueOrDefault($"{phase}|{t}|{s}")) / n; }
                double ScoB(string s) { var n = tin.Sum(t => scoN.GetValueOrDefault($"{phase}|{t}")); return n == 0 ? double.NaN : tin.Sum(t => scoSq.GetValueOrDefault($"{phase}|{t}|{s}")) / n; }
                int nSco = tin.Sum(t => scoN.GetValueOrDefault($"{phase}|{t}"));

                var baseLead = PrecipLeadPolicy.BucketModelFor(lo);
                var baseSco = ScoB($"s{baseLead}");
                if (nSco < MinScoreN || double.IsNaN(baseSco))
                {
                    Console.WriteLine($"  {lo + "-" + hi + "h",9} {nSco,6} (insufficient SCORE rows — baseline)");
                    continue;
                }

                // Candidate set: only models whose training lead is within ±24h
                // of this band (|L − lo| ≤ 24) — so e.g. the 0h model competes at
                // bands up to ~24h but not beyond, and m120 never competes near
                // the nowcast end. Singles always; pairs (both legs in-window)
                // when the phase allows.
                // L − W ≤ band < L + W (exclusive upper, per the agreed rule):
                // 0h competes at bands lo ∈ {0,6,12,18} (not 24); 120h at
                // lo ∈ {96..114}; etc.
                bool InWindow(int m) => lo >= m - CandWindowHours && lo < m + CandWindowHours;
                var candidates = new List<(string Series, string Kind, List<int> Leads)>();
                foreach (var m in cands.Where(InWindow)) candidates.Add(($"s{m}", "single", new List<int> { m }));
                if (allowBlend)
                    foreach (var (l, h) in pairs.Where(p => InWindow(p.Lo) && InWindow(p.Hi)))
                        candidates.Add(($"b{l}x{h}", "blend", new List<int> { l, h }));
                if (candidates.Count == 0) continue;   // no in-window model (shouldn't happen — baseline covers it)

                var inc = incumbent?.Lookup(phase, lo);
                var pick = DecideBand(candidates, SelB, ScoB, baseLead, baseSco, inc, thresholds);
                var decision = pick.Passes
                    ? (pick.Kind == "blend" ? $"blend {pick.Leads[0]}+{pick.Leads[1]}" : $"m{pick.Leads[0]}")
                    : $"baseline m{baseLead}";
                Console.WriteLine($"  {lo + "-" + hi + "h",9} {nSco,6} {$"m{baseLead} {baseSco:F4}",13} {$"{pick.Series} sel",16} {pick.ScoreBrier,8:F4} {decision,18}{(pick.Passes ? $"  (+{pick.DeltaPct:F2}%)" : "")}");

                if (pick.Passes)
                    entries.Add(new PrecipLeadPolicy.BandEntry
                    {
                        LeadLo = lo, LeadHi = hi, Kind = pick.Kind, Leads = pick.Leads,
                        BaselineBrier = baseSco, PolicyBrier = pick.ScoreBrier,
                        DeltaPct = pick.DeltaPct, ScoreN = nSco,
                    });
            }
            if (entries.Count > 0) policy.Phases[phase] = entries;
        }

        policy.Save(modelsRoot, "precipitation", location.Name);
        Console.WriteLine();
        Console.WriteLine($"LEAD_POLICY written → {PrecipLeadPolicy.PathFor(modelsRoot, "precipitation", location.Name)} " +
                          $"({policy.Phases.Sum(p => p.Value.Count)} deviation band(s); absent bands = production buckets)");
        return 0;
    }

    /// <summary>
    /// One band's model choice for a lead policy: SELECT-pick → hysteresis vs the
    /// live incumbent → margin gate vs the production bucket baseline. Shared by
    /// the precipitation (3c/3o) and temperature (2c) producers so the two can't
    /// drift apart — which is exactly how the 2026-06-15 crash arose: the precip
    /// copy kept the incumbent with a throwing <c>candidates.First(...)</c>, so an
    /// incumbent band whose model had fallen OUTSIDE the ±24h candidate window
    /// (e.g. a global-file 3c@84-90h = lead-48, with only {72,96} in window) threw
    /// "Sequence contains no matching element" and killed the whole fit before any
    /// policy was saved. An out-of-window incumbent isn't on the menu, so it can't
    /// be kept — FirstOrDefault falls back to the SELECT pick.
    /// </summary>
    /// <param name="candidates">In-window models for this band (singles, plus
    /// blends when the phase allows). Must be non-empty.</param>
    /// <param name="selScore">Series → SELECT-slice error (lower better).</param>
    /// <param name="scoScore">Series → SCORE-slice error (lower better); returns
    /// NaN for a series with no held-out rows.</param>
    /// <param name="baseLead">Production bucket model lead for this band.</param>
    /// <param name="baseSco">SCORE-slice error of the baseline bucket model.</param>
    /// <param name="incumbent">The live policy's entry for this band, or null.</param>
    internal static BandPick DecideBand(
        IReadOnlyList<(string Series, string Kind, List<int> Leads)> candidates,
        Func<string, double> selScore,
        Func<string, double> scoScore,
        int baseLead,
        double baseSco,
        PrecipLeadPolicy.BandEntry? incumbent,
        PrecipLeadPolicy.ThresholdsBlock thresholds)
    {
        static string IncSeries(PrecipLeadPolicy.BandEntry e) =>
            e.Kind == "blend" ? $"b{e.Leads[0]}x{e.Leads[1]}" : $"s{e.Leads[0]}";

        // 1. Pick on SELECT (lowest error on the slice no candidate trained on).
        var pick = candidates.OrderBy(c => selScore(c.Series)).First();

        // 2. Hysteresis: a DIFFERENT challenger must beat the incumbent on SCORE by
        //    HysteresisPct, else the incumbent stays — but only when the incumbent
        //    is STILL an in-window candidate (FirstOrDefault falls back to `pick`).
        if (incumbent is not null)
        {
            var incSco = scoScore(IncSeries(incumbent));
            if (!double.IsNaN(incSco)
                && IncSeries(incumbent) != pick.Series
                && !(scoScore(pick.Series) <= incSco * (1 - thresholds.HysteresisPct / 100.0)))
                pick = candidates.FirstOrDefault(c => c.Series == IncSeries(incumbent), pick);
        }

        // 3. Margin gate vs baseline on SCORE. Incumbents re-qualify at
        //    (margin − hysteresis) so a sub-threshold wobble can't churn them out.
        var pickSco = scoScore(pick.Series);
        var isIncumbentPick = incumbent is not null && IncSeries(incumbent) == pick.Series;
        var requiredPct = isIncumbentPick
            ? Math.Max(0.0, thresholds.MarginPct - thresholds.HysteresisPct)
            : thresholds.MarginPct;
        var isBaselinePick = pick.Kind == "single" && pick.Leads[0] == baseLead;
        var passes = !isBaselinePick && pickSco <= baseSco * (1 - requiredPct / 100.0);
        var deltaPct = 100.0 * (baseSco - pickSco) / baseSco;

        return new BandPick(pick.Series, pick.Kind, pick.Leads, pickSco, deltaPct, passes);
    }

    /// <summary>The resolved choice for one 6h band — see <see cref="DecideBand"/>.
    /// <c>Passes</c> false means "stay on the baseline bucket model" (no entry).</summary>
    internal readonly record struct BandPick(
        string Series, string Kind, List<int> Leads, double ScoreBrier, double DeltaPct, bool Passes);

    // ===================== TEMPERATURE per-lead policy producer (target-generic) =====================
    //
    // Temp twin of RunFitLeadPolicyAsync. Same band machinery (±24h candidate
    // window, SELECT/SCORE split, margin + hysteresis gates, lead-0 candidate)
    // but: single phase 2c, single-location ERA5 truth (MAE, not gauge Brier),
    // models trained in-process (TempRichFeatureBuilder + TempTrainer), and the
    // τ pivot read via QueryLatestTempRows. Writes the per-location temperature
    // LEAD_POLICY. Scoring is LIVE-ONLY (the score cache excludes offset_day AND
    // hist_forecast) so the 0h model is judged on what predict sees at runtime.
    public async Task<int> RunFitLeadPolicyTempAsync(string? startDateStr, string? cutoffStr, string? locationOverride, CancellationToken ct)
    {
        await Task.Yield();
        var location = ResolveLocation(locationOverride);
        if (location is null) return 2;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var thresholds = new PrecipLeadPolicy.ThresholdsBlock();
        var windowStart = DateOnly.TryParse(startDateStr, out var d0)
            ? d0.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var studyCutoff = DateOnly.TryParse(cutoffStr, out var dc)
            ? dc.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var minValid = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        var splitDate = windowEnd.Date.AddDays(-thresholds.HoldoutDays);
        if (splitDate <= windowStart.AddDays(7))
        {
            _log.LogError("temp fit-lead-policy: SELECT slice <7 days — widen --start."); return 2;
        }
        int[] cands = { 0, 24, 48, 72, 96, 120 };
        var taus = Enumerable.Range(0, 117 / 3 + 1).Select(i => 3 * i).ToArray();
        const int CandWindowHours = 24;
        var pairs = new List<(int Lo, int Hi)>();
        for (int i = 0; i < cands.Length; i++)
            for (int j = i + 1; j < cands.Length; j++) pairs.Add((cands[i], cands[j]));

        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var hp = TempTrainer.Hyperparameters.Default();
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        var root = Path.GetDirectoryName(modelsRoot)!;
        var fcGlobAll = Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=" + location.Name, "**", "*.parquet"));
        var eraGlob = Esc(Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet"));
        var escLoc = location.Name.Replace("'", "''");
        // TRAIN cache: ≤cutoff, all sources (offset_day for ≥24h, hist_forecast
        // for lead 0). SCORE cache: in-window, LIVE ONLY (no offset_day, no
        // hist_forecast). ERA cache: truth.
        var trainFc = Path.Combine(root, "scratch", "temp_policy_fit", "trfc"); Directory.CreateDirectory(Path.Combine(trainFc, "p"));
        var scoreFc = Path.Combine(root, "scratch", "temp_policy_fit", "scfc"); Directory.CreateDirectory(Path.Combine(scoreFc, "p"));
        var eraCache = Path.Combine(root, "scratch", "temp_policy_fit", "era"); Directory.CreateDirectory(Path.Combine(eraCache, "p"));
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcGlobAll}', hive_partitioning=false, union_by_name=true) WHERE ValidTimeUtc <= TIMESTAMP '{studyCutoff:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(trainFc, "p", "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            // Live forecasts only (see RunFitLeadPolicyAsync's cache note) — score on what predict sees.
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcGlobAll}', hive_partitioning=false, union_by_name=true) WHERE (RunTimeSource IS NULL OR RunTimeSource NOT IN ('offset_day', 'hist_forecast')) AND ValidTimeUtc BETWEEN TIMESTAMP '{windowStart:yyyy-MM-dd HH:mm:ss}' AND TIMESTAMP '{windowEnd:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(scoreFc, "p", "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{eraGlob}', hive_partitioning=false, union_by_name=true) WHERE LocationName = '{escLoc}') TO '{Esc(Path.Combine(eraCache, "p", "era.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }

        // Train one 2c study model per lead on ≤cutoff (trainFc is already bounded).
        var specByLead = cands.ToDictionary(l => l, l => TempRichFeatureBuilder.BuildSpec(_cfg.Blenders, l));
        var models = new Dictionary<int, TempTrainer.TrainedBlender>();
        foreach (var l in cands)
        {
            ct.ThrowIfCancellationRequested();
            var rows = TempRichFeatureBuilder.BuildForLead(trainFc, eraCache, location.Name, specByLead[l], minValid, ct);
            if (rows.Count < 300) { _log.LogWarning("  temp study lead {L}h: {N} rows ≤cutoff — skip.", l, rows.Count); continue; }
            var ds = RegressionDataset.Split(rows);
            models[l] = TempTrainer.TrainVector(ds.Train, ds.Val, specByLead[l], hp);
            _log.LogInformation("  temp study lead {L}h trained (n={N}).", l, ds.Train.Count);
        }
        if (!models.ContainsKey(24)) { _log.LogError("temp fit: no lead-24 study model — abort."); return 2; }

        // ERA5 truth over the score window, then per-τ MAE accumulation split SELECT/SCORE.
        var truth = LoadEra5Temp(eraCache, location.Name, windowStart, windowEnd, ct);
        if (truth.Count == 0) { _log.LogWarning("temp fit: no ERA5 truth in window — skip update."); return 0; }
        var selAbs = new Dictionary<string, double>(); var scoAbs = new Dictionary<string, double>();
        var selN = new Dictionary<string, int>(); var scoN = new Dictionary<string, int>();
        void Add(Dictionary<string, double> a, string tk, string ser, double e) => a[$"{tk}|{ser}"] = a.GetValueOrDefault($"{tk}|{ser}") + e;
        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestTempRows(scoreFc, location.Name, windowStart, windowEnd, asOf, tau, canon, ct);
            var tk = $"2c|{tau}";
            foreach (var (V, t) in truth)
            {
                if (!pivot.TryGetValue(V, out var pv)) continue;
                var preds = new Dictionary<int, double>();
                foreach (var m in cands)
                {
                    if (!models.TryGetValue(m, out var mdl)) continue;
                    var p = PredictTemp(mdl, specByLead[m], canon, V, pv);
                    if (p is double x) preds[m] = x;
                }
                if (!preds.ContainsKey(24)) continue;   // need the baseline for a comparable row
                var (a, n) = V < splitDate ? (selAbs, selN) : (scoAbs, scoN);
                n[tk] = n.GetValueOrDefault(tk) + 1;
                foreach (var (m, p) in preds) Add(a, tk, $"s{m}", Math.Abs(p - t));
                foreach (var (lo, hi) in pairs)
                    if (preds.TryGetValue(lo, out var pl) && preds.TryGetValue(hi, out var ph))
                        Add(a, tk, $"b{lo}x{hi}", Math.Abs(0.5 * (pl + ph) - t));
            }
        }

        // Band decisions (mirror the precip producer; MAE in place of Brier).
        var incumbent = PrecipLeadPolicy.TryLoad(modelsRoot, "temperature", location.Name);
        var policy = new PrecipLeadPolicy
        {
            FittedAtUtc = asOf, Location = location.Name,
            WindowStartUtc = windowStart, WindowEndUtc = windowEnd,
            StudyCutoffUtc = studyCutoff, SelectScoreSplitUtc = splitDate, Thresholds = thresholds,
        };
        const int MinScoreN = 300;
        const string phase = "2c";
        var entries = new List<PrecipLeadPolicy.BandEntry>();
        Console.WriteLine();
        Console.WriteLine("=== temp fit-lead-policy (2c): 6h-band decisions, MAE °C vs ERA5 (SELECT picks, SCORE grades) ===");
        Console.WriteLine($"  {"band",9} {"Nsco",6} {"baseline",13} {"SELECT pick",14} {"SCORE",8} {"decision",16}");
        for (int lo = 0; lo < 120; lo += 6)
        {
            var tin = taus.Where(t => t >= lo && t < lo + 6).ToArray();
            double SelB(string s) { var n = tin.Sum(t => selN.GetValueOrDefault($"{phase}|{t}")); return n == 0 ? double.NaN : tin.Sum(t => selAbs.GetValueOrDefault($"{phase}|{t}|{s}")) / n; }
            double ScoB(string s) { var n = tin.Sum(t => scoN.GetValueOrDefault($"{phase}|{t}")); return n == 0 ? double.NaN : tin.Sum(t => scoAbs.GetValueOrDefault($"{phase}|{t}|{s}")) / n; }
            int nSco = tin.Sum(t => scoN.GetValueOrDefault($"{phase}|{t}"));
            var baseLead = PrecipLeadPolicy.BucketModelFor(lo);
            var baseSco = ScoB($"s{baseLead}");
            if (nSco < MinScoreN || double.IsNaN(baseSco)) { Console.WriteLine($"  {lo + "-" + (lo + 6) + "h",9} {nSco,6} (insufficient SCORE rows — baseline)"); continue; }
            bool InWindow(int m) => lo >= m - CandWindowHours && lo < m + CandWindowHours;
            var candidates = new List<(string Series, string Kind, List<int> Leads)>();
            foreach (var m in cands.Where(m => InWindow(m) && models.ContainsKey(m))) candidates.Add(($"s{m}", "single", new List<int> { m }));
            foreach (var (l, h) in pairs.Where(p => InWindow(p.Lo) && InWindow(p.Hi) && models.ContainsKey(p.Lo) && models.ContainsKey(p.Hi)))
                candidates.Add(($"b{l}x{h}", "blend", new List<int> { l, h }));
            if (candidates.Count == 0) continue;
            var inc = incumbent?.Lookup(phase, lo);
            var pick = DecideBand(candidates, SelB, ScoB, baseLead, baseSco, inc, thresholds);
            var decision = pick.Passes ? (pick.Kind == "blend" ? $"blend {pick.Leads[0]}+{pick.Leads[1]}" : $"m{pick.Leads[0]}") : $"baseline m{baseLead}";
            Console.WriteLine($"  {lo + "-" + (lo + 6) + "h",9} {nSco,6} {$"m{baseLead} {baseSco:F3}",13} {$"{pick.Series}",14} {pick.ScoreBrier,8:F3} {decision,16}{(pick.Passes ? $"  (+{pick.DeltaPct:F2}%)" : "")}");
            if (pick.Passes)
                entries.Add(new PrecipLeadPolicy.BandEntry { LeadLo = lo, LeadHi = lo + 6, Kind = pick.Kind, Leads = pick.Leads, BaselineBrier = baseSco, PolicyBrier = pick.ScoreBrier, DeltaPct = pick.DeltaPct, ScoreN = nSco });
        }
        if (entries.Count > 0) policy.Phases[phase] = entries;
        policy.Save(modelsRoot, "temperature", location.Name);
        Console.WriteLine();
        Console.WriteLine($"temperature LEAD_POLICY written → {PrecipLeadPolicy.PathFor(modelsRoot, "temperature", location.Name)} ({entries.Count} deviation band(s))");
        return 0;
    }

    // ===================== forecast pivot (copied from PrecipPredictCommand) =====================

    private sealed record PivotedRow(
        double?[] Precip, DateTime?[] RunTime, double?[] Dew, double?[] Rh, double?[] Temp2m, double?[] Pressure,
        double RhMean, double DewDepressionMean, double CloudLowMean, double CloudMidMean, double CloudHighMean,
        double CapeMean, double WindSpeedMean);

    private sealed class Scratch
    {
        private static int N => TempFeatureBuilder.CanonicalModelOrder.Count;
        public double?[] Precip { get; } = new double?[N];
        public DateTime?[] RunTime { get; } = new DateTime?[N];
        public double?[] Dew { get; } = new double?[N];
        public double?[] Rh { get; } = new double?[N];
        public double?[] Temp2m { get; } = new double?[N];
        public double?[] CloudLow { get; } = new double?[N];
        public double?[] CloudMid { get; } = new double?[N];
        public double?[] CloudHigh { get; } = new double?[N];
        public double?[] Cape { get; } = new double?[N];
        public double?[] WindSpeed { get; } = new double?[N];
        public double?[] Pressure { get; } = new double?[N];
    }

    private static double MeanOfSlots(double?[] slots)
    {
        double sum = 0; int n = 0;
        foreach (var v in slots) if (v.HasValue) { sum += v.Value; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private static double MeanOfDepressions(double?[] temps, double?[] dews)
    {
        double sum = 0; int n = 0;
        for (int i = 0; i < temps.Length; i++)
            if (temps[i].HasValue && dews[i].HasValue) { sum += temps[i]!.Value - dews[i]!.Value; n++; }
        return n == 0 ? double.NaN : sum / n;
    }

    private static IReadOnlyDictionary<DateTime, PivotedRow> QueryLatestForecastRows(
        string forecastsPath, string locationName,
        DateTime earliestValid, DateTime latestValid, DateTime asOfRunTime, int leadHoursLowerBound,
        CancellationToken ct)
    {
        var fcGlob = Path.Combine(forecastsPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var filter = PredictForecastFilters.LiveCycleAsOf(
            locationName, asOfRunTime, earliestValid, latestValid, leadHoursLowerBound);

        var sql = $@"
WITH latest AS (
    SELECT ValidTimeUtc, Model, RunTimeUtc, Precipitation,
           RelativeHumidity2m, Temperature2m, DewPoint2m,
           CloudCoverLow, CloudCoverMid, CloudCoverHigh, Cape, WindSpeed10m, SurfacePressure,
           ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
    FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
    WHERE {filter}
)
SELECT ValidTimeUtc, Model, RunTimeUtc, Precipitation,
       RelativeHumidity2m, Temperature2m, DewPoint2m,
       CloudCoverLow, CloudCoverMid, CloudCoverHigh, Cape, WindSpeed10m, SurfacePressure
FROM latest WHERE rn = 1 ORDER BY ValidTimeUtc, Model;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;

        var modelSlot = TempFeatureBuilder.CanonicalModelOrder
            .Select((id, i) => (id, Index: i)).ToDictionary(x => x.id, x => x.Index);
        var scratch = new Dictionary<DateTime, Scratch>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ct.ThrowIfCancellationRequested();
            var valid = reader.GetDateTime(0);
            var modelName = reader.GetString(1);
            if (!modelSlot.TryGetValue(modelName, out var slot)) continue;
            var runTime = reader.GetDateTime(2);
            double? G(int i) => reader.IsDBNull(i) ? (double?)null : reader.GetDouble(i);

            if (!scratch.TryGetValue(valid, out var s)) { s = new Scratch(); scratch[valid] = s; }
            s.Precip[slot] = G(3); s.Rh[slot] = G(4); s.Temp2m[slot] = G(5); s.Dew[slot] = G(6);
            s.CloudLow[slot] = G(7); s.CloudMid[slot] = G(8); s.CloudHigh[slot] = G(9);
            s.Cape[slot] = G(10); s.WindSpeed[slot] = G(11); s.Pressure[slot] = G(12);
            s.RunTime[slot] = runTime;
        }

        return scratch.ToDictionary(kv => kv.Key, kv => new PivotedRow(
            kv.Value.Precip, kv.Value.RunTime, kv.Value.Dew, kv.Value.Rh, kv.Value.Temp2m, kv.Value.Pressure,
            MeanOfSlots(kv.Value.Rh), MeanOfDepressions(kv.Value.Temp2m, kv.Value.Dew),
            MeanOfSlots(kv.Value.CloudLow), MeanOfSlots(kv.Value.CloudMid), MeanOfSlots(kv.Value.CloudHigh),
            MeanOfSlots(kv.Value.Cape), MeanOfSlots(kv.Value.WindSpeed)));
    }

    // Hourly EA truth — identical aggregation to PrecipFeatureBuilder: complete
    // hours only (HAVING COUNT(*) = 4), hourly total mm. Returns valid→mm.
    internal static IReadOnlyList<(DateTime Valid, double Mm)> LoadHourlyTruth(
        string rainfallPath, string locationName, string stationName,
        DateTime earliest, DateTime latest, CancellationToken ct)
    {
        var rnGlob = Path.Combine(rainfallPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var escLoc = locationName.Replace("'", "''");
        var escSt = stationName.Replace("'", "''");
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time, SUM(Value15MinMm) AS mm
FROM read_parquet('{rnGlob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLoc}' AND StationName = '{escSt}' AND Value15MinMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{earliest:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <  TIMESTAMP '{latest:yyyy-MM-dd HH:mm:ss}'
GROUP BY 1 HAVING COUNT(*) = 4 ORDER BY 1;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<(DateTime, double)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            rows.Add((r.GetDateTime(0), r.IsDBNull(1) ? 0.0 : r.GetDouble(1)));
        }
        return rows;
    }

    /// <summary>
    /// Hourly rainfall truth (mm) for a WeatherLink cove gauge (e.g. Lands End for
    /// Sennen), keyed by hour. WeatherLink rows are already HOURLY <c>RainfallMm</c>
    /// under their own <c>location=</c> partition, so SUM-by-hour with no 4-of-4
    /// completeness rule — the WeatherLink counterpart of <see cref="LoadHourlyTruth"/>,
    /// so a WeatherLink-sourced gauge flows through the lead-policy study/fit exactly
    /// like an EA gauge.
    /// </summary>
    internal static IReadOnlyList<(DateTime Valid, double Mm)> LoadHourlyTruthWeatherLink(
        string weatherLinkPath, string weatherLinkLocation,
        DateTime earliest, DateTime latest, CancellationToken ct)
    {
        var glob = Path.Combine(weatherLinkPath, "**", "*.parquet").Replace('\\', '/').Replace("'", "''");
        var escLoc = weatherLinkLocation.Replace("'", "''");
        var sql = $@"
SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time, SUM(RainfallMm) AS mm
FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)
WHERE LocationName = '{escLoc}' AND RainfallMm IS NOT NULL
  AND ObservedTimeUtc >= TIMESTAMP '{earliest:yyyy-MM-dd HH:mm:ss}'
  AND ObservedTimeUtc <  TIMESTAMP '{latest:yyyy-MM-dd HH:mm:ss}'
GROUP BY 1 ORDER BY 1;";

        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var rows = new List<(DateTime, double)>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            ct.ThrowIfCancellationRequested();
            rows.Add((r.GetDateTime(0), r.IsDBNull(1) ? 0.0 : r.GetDouble(1)));
        }
        return rows;
    }
}
