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
/// EXPERIMENT (uncommitted bake-off): take a 24h-lead-trained P(wet) blender
/// (3c rich, or 3o rich-oro) and score it against EA gauge truth using LIVE
/// forecast inputs taken at SHORTER leads than it was trained on. The
/// hypothesis: fresher NWP inputs (lead 12/18) may yield better predictions
/// than the lead-24 inputs the model was trained on, even though the model
/// itself is fixed at the lead-24 weights.
///
/// Mechanism: the production predict path (<see cref="PrecipPredictCommand"/>)
/// already selects forecast rows with
/// <c>RunTimeUtc &lt;= ValidTimeUtc - leadHoursLowerBound</c> (freshest cycle
/// at least L hours stale). We reuse that exact pivot + the exact production
/// feature builders (rich / rich-oro, persistence, upper-air, terrain), but
/// DECOUPLE the input-lead bound from the model: the model + spec are always
/// the bundle's lead-24 artefacts, while the input pivot is rebuilt at each
/// of {12, 18, 24} h.
///
/// Truth + Brier match production exactly: hourly EA total ≥ 0.1 mm (complete
/// hours only, HAVING COUNT(*) = 4), Brier = mean (p − y)².
///
/// Scope: Bonehill primary location, its configured rainfall gauges.
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

    private sealed record ResultRow(
        string Phase, string Station, int InputLead,
        int N, int WetN, double Brier, double ClimBrier);

    public async Task<int> RunAsync(string? startDateStr, bool useUpperAir, CancellationToken ct)
    {
        var location = _cfg.Location; // primary = bonehill_rocks
        var modelsRoot = _cfg.Storage.ModelsPath;

        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 4, 15, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;

        _log.LogInformation("Cross-lead bake-off — location={Loc} window {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} model-lead={ML}h input-leads=[{IL}]",
            location.Name, windowStart, windowEnd, ModelLead, string.Join(",", InputLeads));

        // --- Per-input-lead live pivots + upper-air (per-location; reused across stations + bundles) ---
        var pivotByLead = new Dictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>>();
        var uaByLead = new Dictionary<int, IReadOnlyList<(DateTime ValidTimeUa, double[] PerModelCol)>>();
        foreach (var il in InputLeads)
        {
            pivotByLead[il] = QueryLatestForecastRows(
                _cfg.Storage.ForecastsPath, location.Name, windowStart, windowEnd, asOf, il, ct);
            uaByLead[il] = useUpperAir
                ? PrecipFeatureBuilder.LoadUpperAirLive(
                    _cfg.Storage.ForecastsPath, location.Name, il, windowStart.AddDays(-2), windowEnd, ct)
                : System.Array.Empty<(DateTime, double[])>();
            _log.LogInformation("Input lead {IL}h: {P} valid-time pivots, {U} upper-air rows (live tree){Note}.",
                il, pivotByLead[il].Count, uaByLead[il].Count, useUpperAir ? "" : " [UA DISABLED]");
        }

        var canonOrder = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var results = new List<ResultRow>();

        var stations = location.Rainfall.Stations
            .Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name))
            .ToList();

        foreach (var (station, friendly) in stations)
        {
            ct.ThrowIfCancellationRequested();

            var truth = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            if (truth.Count == 0)
            {
                _log.LogWarning("Station {Station}: no complete-hour truth in window — skipping.", station);
                continue;
            }
            var hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(
                _cfg.Storage.RainfallPath, location.Name, friendly, minValidTime: null, ct);

            // Resolve which active bundles are 3c / 3o for this station.
            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
            foreach (var phase in new[] { "3c", "3o" })
            {
                var version = active.FirstOrDefault(v =>
                    string.Equals(ModelArtifact.ExtractPhaseFromVersionName(v), phase, StringComparison.Ordinal));
                if (version is null)
                {
                    _log.LogInformation("Station {Station}: no active {Phase} bundle — skipping.", station, phase);
                    continue;
                }

                var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, version);
                var metadata = ModelArtifact.LoadTrainingMetadata(versionDir);
                var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
                if (!specs.TryGetValue(ModelLead, out var spec))
                {
                    _log.LogWarning("Station {Station} {Phase} {V}: no lead-{ML}h spec — skipping.", station, phase, version, ModelLead);
                    continue;
                }
                var model = ModelArtifact.LoadLeadModel(ml, versionDir, ModelLead, out _);
                var is3o = string.Equals(phase, "3o", StringComparison.Ordinal);

                // 3o-only: static orographic record + aux NWP means (live, lead-independent — matches production 3o predict).
                OroStaticFeatures? oro = null;
                int stationIndex = -1;
                Dictionary<DateTime, PrecipRichOroFeatureBuilder.NwpMeanRow>? nwpMeanByValid = null;
                if (is3o)
                {
                    if (!Phase3oStationIndex.TryGetValue(station, out stationIndex))
                    {
                        _log.LogWarning("Station {Station}: 3o bundle present but slug not in station-index map — skipping 3o.", station);
                        continue;
                    }
                    var oroDir = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
                    var oroBySlug = OroStaticFeatures.LoadAll(oroDir);
                    if (!oroBySlug.TryGetValue(station, out oro))
                    {
                        _log.LogWarning("Station {Station}: no orographic record under {Dir} — skipping 3o.", station, oroDir);
                        continue;
                    }
                    nwpMeanByValid = PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(
                        _cfg.Storage.ForecastsPath, location.Name, spec, windowStart, windowEnd, ct);
                }

                _log.LogInformation("Station {Station} {Phase} {V}: scoring lead-{ML}h model on input leads [{IL}] over {T} truth hours.",
                    station, phase, metadata.Version, ModelLead, string.Join(",", InputLeads), truth.Count);

                foreach (var il in InputLeads)
                {
                    var perValid = pivotByLead[il];
                    var ua = uaByLead[il];
                    int n = 0, wetN = 0;
                    double sumSq = 0, climWet = 0;

                    foreach (var (valid, mm) in truth)
                    {
                        if (!perValid.TryGetValue(valid, out var pivot)) continue;
                        if (!pivot.Precip.Any(p => p.HasValue)) continue;

                        // UA control is entirely via `ua` being empty (loader gate above):
                        // an empty asof list → UpperAirValuesFor returns a NaN-filled block of
                        // the correct length, so every lead gets identical (absent) UA and the
                        // feature-vector length still matches the bundle schema. Passing null
                        // would instead trigger ComposeRow's legacy-length path and mismatch.
                        var pWet = is3o
                            ? Predict3o(ml, model, spec, canonOrder, valid, il, pivot, hourlyRain, ua, oro!, nwpMeanByValid!, stationIndex)
                            : Predict3c(ml, model, spec, canonOrder, valid, il, pivot, hourlyRain, ua);
                        if (pWet is null) continue; // missing required model

                        var label = mm >= WetThresholdMm ? 1.0 : 0.0;
                        sumSq += (pWet.Value - label) * (pWet.Value - label);
                        climWet += label;
                        n++;
                        if (label > 0) wetN++;
                    }

                    if (n == 0) { _log.LogWarning("Station {Station} {Phase} lead {IL}h: 0 scored rows.", station, phase, il); continue; }
                    var brier = sumSq / n;
                    var baseRate = climWet / n;
                    var climBrier = baseRate * (1 - baseRate); // Brier of constant base-rate prediction
                    results.Add(new ResultRow(phase, station, il, n, wetN, brier, climBrier));
                }
            }
        }

        PrintReport(results);
        return results.Count == 0 ? 3 : 0;
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

        var runTime = valid.AddHours(-inputLead);
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

    private void PrintReport(List<ResultRow> results)
    {
        Console.WriteLine();
        Console.WriteLine("=========================================================================");
        Console.WriteLine(" Cross-lead bake-off: 24h-trained P(wet) model on live inputs @ 12/18/24h");
        Console.WriteLine(" Brier (lower=better) — wet ≥ 0.1 mm/h, hourly EA truth (complete hours)");
        Console.WriteLine("=========================================================================");

        foreach (var phase in results.Select(r => r.Phase).Distinct())
        {
            Console.WriteLine();
            Console.WriteLine($"### Phase {phase}");
            Console.WriteLine($"  {"station",-26} {"inLead",6} {"N",6} {"wet%",6} {"Brier",9} {"BSS",8}");
            foreach (var st in results.Where(r => r.Phase == phase).Select(r => r.Station).Distinct())
            {
                foreach (var r in results.Where(x => x.Phase == phase && x.Station == st).OrderBy(x => x.InputLead))
                {
                    var bss = (r.ClimBrier - r.Brier) / r.ClimBrier;
                    Console.WriteLine($"  {r.Station,-26} {r.InputLead + "h",6} {r.N,6} {(double)r.WetN / r.N,6:P0} {r.Brier,9:F4} {bss,8:+0.000;-0.000;0.000}");
                }
            }

            // Aggregate across stations per input lead (pooled rows).
            Console.WriteLine($"  {"— AGG (pooled) —",-26}");
            foreach (var il in InputLeads)
            {
                var rs = results.Where(r => r.Phase == phase && r.InputLead == il).ToList();
                if (rs.Count == 0) continue;
                int n = rs.Sum(r => r.N), wet = rs.Sum(r => r.WetN);
                // Pool Brier weighted by N (each ResultRow already a per-row mean).
                double brier = rs.Sum(r => r.Brier * r.N) / n;
                double baseRate = (double)wet / n;
                double climBrier = baseRate * (1 - baseRate);
                double bss = (climBrier - brier) / climBrier;
                Console.WriteLine($"  {"all stations",-26} {il + "h",6} {n,6} {baseRate,6:P0} {brier,9:F4} {bss,8:+0.000;-0.000;0.000}");
            }
        }
        Console.WriteLine();
    }

    // ===================== UA-construction A/B test (staged step 2, predict-only) =====================
    //
    // A (current production): strict lead-24 UA, forward-filled. Under realistic
    // morning-predict availability only the anchor-day-00Z cycle is in hand, so the
    // single lead-24 snapshot valid D 00:00 is used for ALL of day D → up to ~23h stale.
    // B (proposed): from the SAME anchor-00Z cycle, the nearest available lead among
    // {24,36,48} so the UA valid-time lands near the target → near valid-exact.
    // Precip + everything else identical between arms, so the Brier delta is the UA effect.

    private sealed record UaResult(string Station, int N, int WetN, double BrierA, double BrierB, double StaleAHrs, double StaleBHrs);

    public async Task<int> RunUaConstructionAsync(string? startDateStr, CancellationToken ct)
    {
        var location = _cfg.Location;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 4, 15, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;

        _log.LogInformation("UA-construction A/B — loc={Loc} window {S:yyyy-MM-dd}..{E:yyyy-MM-dd} (3c lead-24; A=strict-lead24 fwd-fill, B=nearest-lead, morning-predict availability)",
            location.Name, windowStart, windowEnd);

        // Precip pivot at lead 24 — IDENTICAL for both arms (isolates UA).
        var pivot24 = QueryLatestForecastRows(_cfg.Storage.ForecastsPath, location.Name, windowStart, windowEnd, asOf, 24, ct);

        // Exact UA per valid at leads 24/36/48. For the valids we look up, the freshest
        // cycle is uniquely the anchor-00Z cycle (lead L valid = cycle + L), i.e. exactly
        // what the morning predict has in hand.
        Dictionary<DateTime, double[]> UaDict(int lead) =>
            PrecipFeatureBuilder.LoadUpperAirLive(_cfg.Storage.ForecastsPath, location.Name, lead, windowStart.AddDays(-2), windowEnd, ct)
                .GroupBy(x => x.ValidTimeUa).ToDictionary(g => g.Key, g => g.First().PerModelCol);
        var ua24 = UaDict(24); var ua36 = UaDict(36); var ua48 = UaDict(48);
        _log.LogInformation("Exact UA rows: lead24={A} lead36={B} lead48={C}", ua24.Count, ua36.Count, ua48.Count);

        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var results = new List<UaResult>();

        foreach (var s in location.Rainfall.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var station = StationSlug.WithEaPrefix(s.Name);
            var truth = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, s.Name, windowStart, windowEnd, ct);
            if (truth.Count == 0) continue;
            var hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, s.Name, null, ct);

            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
            var version = active.FirstOrDefault(v => ModelArtifact.ExtractPhaseFromVersionName(v) == "3c");
            if (version is null) { _log.LogInformation("Station {S}: no 3c bundle", station); continue; }
            var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, version);
            var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
            if (!specs.TryGetValue(24, out var spec)) continue;
            var model = ModelArtifact.LoadLeadModel(ml, versionDir, 24, out _);

            int n = 0, wet = 0; double sqA = 0, sqB = 0, staleA = 0, staleB = 0;
            foreach (var (V, mm) in truth)
            {
                if (!pivot24.TryGetValue(V, out var pivot)) continue;
                if (!pivot.Precip.Any(p => p.HasValue)) continue;
                var D = new DateTime(V.Year, V.Month, V.Day, 0, 0, 0, DateTimeKind.Utc);
                var anchor00 = D.AddDays(-1);
                // A needs the lead-24 snapshot valid D 00:00 (from anchor00).
                if (!ua24.TryGetValue(anchor00.AddHours(24), out var vecA)) continue;
                var uaA = AssembleUaBlock(vecA);
                // B: nearest available snapshot among anchor00's leads 24/36/48.
                var cands = new (DateTime valid, double[]? vec)[]
                {
                    (anchor00.AddHours(24), vecA),
                    (anchor00.AddHours(36), ua36.GetValueOrDefault(anchor00.AddHours(36))),
                    (anchor00.AddHours(48), ua48.GetValueOrDefault(anchor00.AddHours(48))),
                };
                var best = cands.Where(c => c.vec != null).OrderBy(c => Math.Abs((c.valid - V).TotalHours)).First();
                var uaB = AssembleUaBlock(best.vec!);

                var pA = Predict3c(ml, model, spec, canon, V, 24, pivot, hourlyRain, System.Array.Empty<(DateTime, double[])>(), overrideUa: uaA);
                var pB = Predict3c(ml, model, spec, canon, V, 24, pivot, hourlyRain, System.Array.Empty<(DateTime, double[])>(), overrideUa: uaB);
                if (pA is null || pB is null) continue;
                var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                sqA += (pA.Value - y) * (pA.Value - y);
                sqB += (pB.Value - y) * (pB.Value - y);
                staleA += Math.Abs((anchor00.AddHours(24) - V).TotalHours);
                staleB += Math.Abs((best.valid - V).TotalHours);
                n++; if (y > 0) wet++;
            }
            if (n > 0) results.Add(new UaResult(station, n, wet, sqA / n, sqB / n, staleA / n, staleB / n));
        }

        Console.WriteLine();
        Console.WriteLine("=================================================================================");
        Console.WriteLine(" UA-construction A/B (3c lead-24, predict-only) — A=strict-lead24 fwd-fill, B=nearest-lead");
        Console.WriteLine(" Brier lower=better; stale = mean |UA valid − target valid| (h). NOTE: model trained on A.");
        Console.WriteLine("=================================================================================");
        Console.WriteLine($"  {"station",-26} {"N",6} {"wet%",6} {"BrierA",9} {"BrierB",9} {"Δ%",7} {"staleA",7} {"staleB",7}");
        foreach (var r in results)
            Console.WriteLine($"  {r.Station,-26} {r.N,6} {(double)r.WetN / r.N,6:P0} {r.BrierA,9:F4} {r.BrierB,9:F4} {(r.BrierB - r.BrierA) / r.BrierA * 100,7:+0.0;-0.0;0.0} {r.StaleAHrs,7:F1} {r.StaleBHrs,7:F1}");
        if (results.Count > 0)
        {
            int N = results.Sum(r => r.N), W = results.Sum(r => r.WetN);
            double bA = results.Sum(r => r.BrierA * r.N) / N, bB = results.Sum(r => r.BrierB * r.N) / N;
            double sA = results.Sum(r => r.StaleAHrs * r.N) / N, sB = results.Sum(r => r.StaleBHrs * r.N) / N;
            Console.WriteLine($"  {"POOLED",-26} {N,6} {(double)W / N,6:P0} {bA,9:F4} {bB,9:F4} {(bB - bA) / bA * 100,7:+0.0;-0.0;0.0} {sA,7:F1} {sB,7:F1}");
        }
        Console.WriteLine();
        return results.Count == 0 ? 3 : 0;
    }

    // ===================== model-crossover analysis (24-model vs 48-model over the 24-47h window) =====================
    //
    // The lead-24 model currently serves the whole 24-47h valid window. Question: at what
    // actual input-lead does the lead-48-trained model start beating it on the SAME input?
    // For each lead bucket τ we build one input pivot (freshest cycle ≥ τ stale) and run
    // BOTH models on it. UA disabled (NaN, both models) so the comparison is pure model×lead
    // — exact-pressure UA only exists at 24/36/48 anyway, and freshness was shown a non-lever.

    private sealed class CrossAcc { public int N, Wet; public double SqA, SqB; }

    public async Task<int> RunModelCrossoverAsync(string? startDateStr, IReadOnlyList<int>? tausOverride, int leadA, int leadB, CancellationToken ct)
    {
        var location = _cfg.Location;
        var modelsRoot = _cfg.Storage.ModelsPath;
        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 4, 15, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        int[] taus = (tausOverride is { Count: > 0 } ? tausOverride.ToArray() : new[] { 24, 30, 36, 42, 48, 54 });

        _log.LogInformation("Model-crossover — loc={Loc} window {S:yyyy-MM-dd}..{E:yyyy-MM-dd} (3c lead-{A} model vs lead-{B} model, input-lead τ ∈ [{T}], UA disabled)",
            location.Name, windowStart, windowEnd, leadA, leadB, string.Join(",", taus));

        var pivotByTau = new Dictionary<int, IReadOnlyDictionary<DateTime, PivotedRow>>();
        foreach (var t in taus)
        {
            pivotByTau[t] = QueryLatestForecastRows(_cfg.Storage.ForecastsPath, location.Name, windowStart, windowEnd, asOf, t, ct);
            _log.LogInformation("  input-lead τ={T}h: {N} pivots", t, pivotByTau[t].Count);
        }

        var emptyUa = System.Array.Empty<(DateTime, double[])>();
        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var acc = taus.ToDictionary(t => t, _ => new CrossAcc());

        foreach (var s in location.Rainfall.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var station = StationSlug.WithEaPrefix(s.Name);
            var truth = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, s.Name, windowStart, windowEnd, ct);
            if (truth.Count == 0) continue;
            var hourlyRain = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, s.Name, null, ct);

            var active = ModelArtifact.ResolveStationActive(modelsRoot, "precipitation", station);
            var version = active.FirstOrDefault(v => ModelArtifact.ExtractPhaseFromVersionName(v) == "3c");
            if (version is null) continue;
            var versionDir = ModelArtifact.ResolveStationVersionDir(modelsRoot, "precipitation", station, version);
            var specs = ModelArtifact.LoadBlenderSpecs(versionDir);
            if (!specs.TryGetValue(leadA, out var specA) || !specs.TryGetValue(leadB, out var specB)) continue;
            var modelA = ModelArtifact.LoadLeadModel(ml, versionDir, leadA, out _);
            var modelB = ModelArtifact.LoadLeadModel(ml, versionDir, leadB, out _);
            _log.LogInformation("Station {S}: 3c {V} — scoring lead-{A} vs lead-{B} model over {N} truth hours × {T} input leads",
                station, version, leadA, leadB, truth.Count, taus.Length);

            foreach (var t in taus)
            {
                var pv = pivotByTau[t];
                var a = acc[t];
                foreach (var (V, mm) in truth)
                {
                    if (!pv.TryGetValue(V, out var pivot)) continue;
                    if (!pivot.Precip.Any(p => p.HasValue)) continue;
                    var pA = Predict3c(ml, modelA, specA, canon, V, t, pivot, hourlyRain, emptyUa);
                    var pB = Predict3c(ml, modelB, specB, canon, V, t, pivot, hourlyRain, emptyUa);
                    if (pA is null || pB is null) continue;
                    var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                    a.SqA += (pA.Value - y) * (pA.Value - y);
                    a.SqB += (pB.Value - y) * (pB.Value - y);
                    a.N++; if (y > 0) a.Wet++;
                }
            }
        }

        Console.WriteLine();
        Console.WriteLine("=================================================================================");
        Console.WriteLine($" Model crossover: lead-{leadA} model vs lead-{leadB} model on the SAME input (Bonehill, pooled)");
        Console.WriteLine(" Brier lower=better. τ = input-lead lower bound (actual lead ≈ τ..τ+6h). UA disabled.");
        Console.WriteLine("=================================================================================");
        Console.WriteLine($"  {"τ (input lead)",16} {"N",7} {"wet%",6} {$"Brier[{leadA}-mdl]",14} {$"Brier[{leadB}-mdl]",14} {$"Δ({leadB}−{leadA})",10} {"winner",8}");
        foreach (var t in taus)
        {
            var a = acc[t];
            if (a.N == 0) { Console.WriteLine($"  {t + "h",16} {0,7}  (no rows)"); continue; }
            double bA = a.SqA / a.N, bB = a.SqB / a.N;
            var dpct = (bB - bA) / bA * 100;
            var winner = bB < bA ? $"{leadB}-mdl" : $"{leadA}-mdl";
            Console.WriteLine($"  {t + "h",16} {a.N,7} {(double)a.Wet / a.N,6:P0} {bA,14:F4} {bB,14:F4} {dpct,9:+0.0;-0.0;0.0}% {winner,8}");
        }
        Console.WriteLine();
        return acc.Values.Any(a => a.N > 0) ? 0 : 3;
    }

    // ===================== STUDY retrain (walk-forward, no-UA, cutoff) =====================
    //
    // Mints local study bundles for the per-lead policy study: 3c (per-gauge) +
    // 3o (pooled), Bonehill, leads {24,48,72,96,120}, trained ONLY on offset_day
    // data ≤ cutoff (so the live scoring window stays OOS), with upper-air OFF.
    // Writes to data/models_study/ (NOT the production tree, no manifest promote).
    // Mirrors PrecipTrainCommand's 3c/3o train loops but isolated + parameterised.

    private static readonly string[] BonehillOrder3o =
        { "Bellever Dartmoor", "Bovey Tracey", "Dartmoor nr Hexworthy", "Princetown" };

    public async Task<int> RunPolicyRetrainAsync(string? asOfStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var cutoff = DateOnly.TryParse(asOfStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var min3c = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc); // matches phases.yaml 3c minValidTime
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
        var ml = new MLContext(seed: 42);
        var leads = Leads.Full;

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
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
                $"WHERE ValidTimeUtc <= TIMESTAMP '{cutoff:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(fcPart, "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.RainfallPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true)) " +
                $"TO '{Esc(Path.Combine(rnPart, "rn.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }
        _log.LogInformation("Scan-once cache materialised → {Scratch} (forecasts ≤{Cut:yyyy-MM-dd} + rainfall, Bonehill)", scratch, cutoff);

        _log.LogInformation("STUDY retrain — loc={Loc}, train ValidTime ≤ {Cut:yyyy-MM-dd} (live window stays OOS), UA OFF; out={Root}",
            location.Name, cutoff, studyRoot);

        // ---- 3c: per-gauge, rich (no-UA) ----
        foreach (var s in location.Rainfall.Stations)
        {
            ct.ThrowIfCancellationRequested();
            var slug = StationSlug.WithEaPrefix(s.Name);
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
                    minValidTime: min3c, ct: ct, maxValidTime: cutoff);
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
        var oroRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        var pool = new List<(string Name, string Slug, OroStaticFeatures Oro, int Index)>();
        for (int i = 0; i < BonehillOrder3o.Length; i++)
        {
            var match = location.Rainfall.Stations.FirstOrDefault(x => x.Name.Equals(BonehillOrder3o[i], StringComparison.OrdinalIgnoreCase));
            if (match is null) { _log.LogError("3o pool: station '{N}' missing from config.", BonehillOrder3o[i]); return 2; }
            var slug = StationSlug.WithEaPrefix(match.Name);
            if (!oroBySlug.TryGetValue(slug, out var oro)) { _log.LogError("3o pool: no oro record for {S}.", slug); return 2; }
            pool.Add((match.Name, slug, oro, i));
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
            foreach (var (name, slug, oro, idx) in pool)
            {
                ct.ThrowIfCancellationRequested();
                var rows = PrecipRichOroFeatureBuilder.BuildForLead(
                    fcPath, rnPath, location.Name, name, oro, idx, spec,
                    ct: ct, maxValidTime: cutoff);
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

    // ===================== per-lead policy EVAL (study bundles, live OOS inputs) =====================
    //
    // For each target lead τ, score every candidate study model (nominal lead M ≤ τ — no model
    // longer than the lead) on the SAME live input at lead τ, vs EA truth. Study bundles are the
    // no-UA, ≤2026-03-15 cutoff ones in models_study (so the live window is OOS). Phase 1: per-τ
    // best single model + the production baseline (model whose bucket contains τ). Blend + SELECT/
    // SCORE split come next. Scan-once live cache so the per-τ pivots are cheap.

    private sealed class EvalAcc { public int N, Wet; public double Sq; }

    public async Task<int> RunPolicyEvalAsync(string? startDateStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        int[] candidateLeads = { 24, 48, 72, 96, 120 };
        int[] taus = { 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 108, 120 };
        static int Bucket(int tau) => tau >= 120 ? 120 : tau >= 96 ? 96 : tau >= 72 ? 72 : tau >= 48 ? 48 : 24;

        // Scan-once live cache: Bonehill live rows (non-offset_day) in the window → one parquet.
        var fcPart = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc", "p");
        Directory.CreateDirectory(fcPart);
        var fcPath = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
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
        _log.LogInformation("Policy eval — live cache materialised; window {S:yyyy-MM-dd}..{E:yyyy-MM-dd}, τ=[{T}]", windowStart, windowEnd, string.Join(",", taus));

        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var emptyUa = System.Array.Empty<(DateTime, double[])>();
        var oroBySlug = OroStaticFeatures.LoadAll(Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic"));

        // Per-gauge: truth, hourlyRain, study bundles (3c + 3o), oro+stationIndex.
        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var (slug, friendly) in gauges)
        {
            truth[slug] = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            rain[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, friendly, null, ct);
            foreach (var (phase, dir, dst) in new[] {
                ("3c", Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3c_noua"), m3c),
                ("3o", Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3o_noua"), m3o) })
            {
                if (!Directory.Exists(dir)) continue;
                var specs = ModelArtifact.LoadBlenderSpecs(dir);
                var byLead = new Dictionary<int, (ITransformer, BlenderSpec)>();
                foreach (var lead in candidateLeads)
                    if (specs.TryGetValue(lead, out var sp))
                        byLead[lead] = (ModelArtifact.LoadLeadModel(ml, dir, lead, out _), sp);
                dst[slug] = byLead;
            }
        }
        // 3o aux NWP means (live, lead-independent) over the window — one load.
        var anySpec3o = m3o.Values.FirstOrDefault()?.Values.FirstOrDefault().Spec;
        var aux = anySpec3o is null ? new() : PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(fcPath, location.Name, anySpec3o, windowStart, windowEnd, ct);

        var acc = new Dictionary<string, EvalAcc>();
        EvalAcc Acc(string k) { if (!acc.TryGetValue(k, out var a)) { a = new EvalAcc(); acc[k] = a; } return a; }

        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestForecastRows(fcPath, location.Name, windowStart, windowEnd, asOf, tau, ct);
            foreach (var (slug, friendly) in gauges)
            {
                if (!Phase3oStationIndex.TryGetValue(slug, out var stIdx)) stIdx = -1;
                oroBySlug.TryGetValue(slug, out var oro);
                foreach (var cand in candidateLeads.Where(m => m <= tau))
                {
                    foreach (var (phase, store) in new[] { ("3c", m3c), ("3o", m3o) })
                    {
                        if (!store.TryGetValue(slug, out var byLead) || !byLead.TryGetValue(cand, out var ms)) continue;
                        if (phase == "3o" && (oro is null || stIdx < 0)) continue;
                        var a = Acc($"{phase}|{tau}|{cand}");
                        foreach (var (V, mm) in truth[slug])
                        {
                            if (!pivot.TryGetValue(V, out var pv) || !pv.Precip.Any(p => p.HasValue)) continue;
                            double? p = phase == "3c"
                                ? Predict3c(ml, ms.Model, ms.Spec, canon, V, tau, pv, rain[slug], emptyUa)
                                : Predict3o(ml, ms.Model, ms.Spec, canon, V, tau, pv, rain[slug], emptyUa, oro!, aux, stIdx);
                            if (p is null) continue;
                            var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                            a.Sq += (p.Value - y) * (p.Value - y); a.N++; if (y > 0) a.Wet++;
                        }
                    }
                }
            }
            _log.LogInformation("  τ={Tau}h scored.", tau);
        }

        // Report per phase: τ × candidate Brier (pooled over gauges), best + baseline.
        foreach (var phase in new[] { "3c", "3o" })
        {
            Console.WriteLine();
            Console.WriteLine($"=== Policy eval {phase}: Brier by target-lead τ × candidate model (Bonehill pooled, live OOS) ===");
            Console.WriteLine($"  {"τ",4} {"N",6} {"wet%",5} " + string.Join(" ", candidateLeads.Select(m => $"{("m" + m),8}")) + "   best  base");
            foreach (var tau in taus)
            {
                var cells = candidateLeads.Select(m => acc.TryGetValue($"{phase}|{tau}|{m}", out var a) && a.N > 0 ? (double?)(a.Sq / a.N) : null).ToArray();
                var anyA = candidateLeads.Select(m => acc.GetValueOrDefault($"{phase}|{tau}|{m}")).FirstOrDefault(x => x is { N: > 0 });
                if (anyA is null) continue;
                var bestIdx = -1; double bestV = double.MaxValue;
                for (int i = 0; i < cells.Length; i++) if (cells[i] is double v && v < bestV) { bestV = v; bestIdx = i; }
                var baseLead = Bucket(tau);
                var s = $"  {tau + "h",4} {anyA.N,6} {(double)anyA.Wet / anyA.N,5:P0} ";
                for (int i = 0; i < cells.Length; i++)
                    s += cells[i] is double v ? $"{(i == bestIdx ? "*" : " ")}{v,7:F4}" : $"{"-",8}";
                s += $"   m{candidateLeads[bestIdx]}  m{baseLead}";
                Console.WriteLine(s);
            }
        }
        Console.WriteLine();
        return acc.Count == 0 ? 3 : 0;
    }

    // ===================== NOWCAST (lead-0) bake-off =====================
    //
    // Does a lead-0 model — trained on the hist_forecast archive (≈analysis at
    // lead 0) — beat the lead-24 model over the short-range ≤12h window? Walk-
    // forward OOS: train 3c {0, 24} per Bonehill gauge on data ≤ cutoff, then
    // score BOTH at target leads τ ∈ {0,3,6,9,12} on the held-out window
    // (cutoff, now], fed the freshest cycle ≥τh stale, vs EA gauge truth. The
    // headline cell is τ=12: nowcast model vs the 24h model the ≤12h tab uses.
    // 3c only for this first read (3o is a follow-up if 3c looks promising).
    public async Task<int> RunNowcastBakeoffAsync(string? cutoffStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var cutoff = DateOnly.TryParse(cutoffStr, out var dc)
            ? dc.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var min3c = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        int[] bakeLeads = { 0, 24 };
        int[] taus = { 0, 3, 6, 9, 12 };
        var hp = PrecipOccurrenceTrainer.Hyperparameters.Default();
        var ml = new MLContext(seed: 42);
        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var emptyUa = System.Array.Empty<(DateTime, double[])>();
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");

        // Two scan-once caches: TRAIN (≤cutoff, all sources incl hist_forecast +
        // offset_day) and EVAL (>cutoff, non-offset_day = hist_forecast + live).
        var root = Path.GetDirectoryName(_cfg.Storage.ModelsPath)!;
        var trFc = Path.Combine(root, "scratch", "nowcast_bake", "tr"); Directory.CreateDirectory(Path.Combine(trFc, "p"));
        var evFc = Path.Combine(root, "scratch", "nowcast_bake", "ev"); Directory.CreateDirectory(Path.Combine(evFc, "p"));
        var rnPath = Path.Combine(root, "scratch", "nowcast_bake", "rn"); Directory.CreateDirectory(Path.Combine(rnPath, "p"));
        var fcGlob = Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"));
        var rnGlob = Esc(Path.Combine(_cfg.Storage.RainfallPath, "location=bonehill_rocks", "**", "*.parquet"));
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true) " +
                $"WHERE ValidTimeUtc <= TIMESTAMP '{cutoff:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(trFc, "p", "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true) " +
                $"WHERE (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day') AND ValidTimeUtc > TIMESTAMP '{cutoff:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(evFc, "p", "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{rnGlob}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(rnPath, "p", "rn.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }
        _log.LogInformation("Nowcast bake-off — train ≤{Cut:yyyy-MM-dd}, eval ({Cut:yyyy-MM-dd}, {End:yyyy-MM-dd}], τ=[{T}], leads=[{L}].",
            cutoff, asOf, string.Join(",", taus), string.Join(",", bakeLeads));

        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var models = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var (slug, friendly) in gauges)
        {
            var byLead = new Dictionary<int, (ITransformer, BlenderSpec)>();
            foreach (var lead in bakeLeads)
            {
                ct.ThrowIfCancellationRequested();
                var spec = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: false);
                var rows = PrecipRichFeatureBuilder.BuildForLead(
                    trFc, rnPath, location.Name, friendly, spec, minValidTime: min3c, ct: ct, maxValidTime: cutoff);
                if (rows.Count < 300) { _log.LogWarning("  {Slug} L{Lead}h: only {N} train rows — skipping.", slug, lead, rows.Count); continue; }
                var ds = BinaryDataset.Split(rows);
                var trained = PrecipOccurrenceTrainer.TrainVector(ds.Train, ds.Val, spec, hp);
                byLead[lead] = (trained.Model, spec);
                _log.LogInformation("  trained 3c {Slug} L{Lead}h: rows={N} (wet {W:P1}, last {E:yyyy-MM-dd}).",
                    slug, lead, rows.Count, rows.Count(r => r.Label) / (double)rows.Count, rows[^1].ValidTimeUtc);
            }
            models[slug] = byLead;
        }

        // Score: per (gauge, tau, lead) Brier + per-gauge climatology baseline.
        var acc = new Dictionary<string, EvalAcc>();
        EvalAcc Acc(string k) { if (!acc.TryGetValue(k, out var a)) { a = new EvalAcc(); acc[k] = a; } return a; }
        var truthByGauge = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rainByGauge = new Dictionary<string, Dictionary<DateTime, double>>();
        foreach (var (slug, friendly) in gauges)
        {
            truthByGauge[slug] = LoadHourlyTruth(rnPath, location.Name, friendly, cutoff, asOf, ct);
            rainByGauge[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(rnPath, location.Name, friendly, null, ct);
        }
        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestForecastRows(evFc, location.Name, cutoff, asOf, asOf, tau, ct);
            foreach (var (slug, friendly) in gauges)
            {
                if (!models.TryGetValue(slug, out var byLead)) continue;
                foreach (var lead in bakeLeads.Where(byLead.ContainsKey))
                {
                    var a = Acc($"{tau}|{lead}|{slug}");
                    var ap = Acc($"{tau}|{lead}|pooled");
                    var (m, sp) = byLead[lead];
                    foreach (var (V, mm) in truthByGauge[slug])
                    {
                        if (!pivot.TryGetValue(V, out var pv) || !pv.Precip.Any(p => p.HasValue)) continue;
                        var p = Predict3c(ml, m, sp, canon, V, tau, pv, rainByGauge[slug], emptyUa);
                        if (p is null) continue;
                        var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                        var e2 = (p.Value - y) * (p.Value - y);
                        a.Sq += e2; a.N++; if (y > 0) a.Wet++;
                        ap.Sq += e2; ap.N++; if (y > 0) ap.Wet++;
                    }
                }
            }
            _log.LogInformation("  τ={Tau}h scored.", tau);
        }

        // Report: pooled Brier by τ, m0 vs m24, with the climatology baseline
        // (constant pooled wet-rate) for context. Δ% = m0 relative to m24.
        Console.WriteLine();
        Console.WriteLine("=== Nowcast bake-off: 3c lead-0 (hist_forecast) vs lead-24, Bonehill pooled, walk-forward OOS ===");
        Console.WriteLine($"  {"τ",4} {"N",6} {"wet%",5} {"m0",9} {"m24",9} {"Δ% (m0 vs m24)",16} {"clim",9}");
        foreach (var tau in taus)
        {
            var a0 = acc.GetValueOrDefault($"{tau}|0|pooled");
            var a24 = acc.GetValueOrDefault($"{tau}|24|pooled");
            var any = a0 is { N: > 0 } ? a0 : a24;
            if (any is null || any.N == 0) { Console.WriteLine($"  {tau + "h",4}  (no rows)"); continue; }
            double? b0 = a0 is { N: > 0 } ? a0.Sq / a0.N : null;
            double? b24 = a24 is { N: > 0 } ? a24.Sq / a24.N : null;
            var wetRate = (double)any.Wet / any.N;
            var clim = wetRate * (1 - wetRate);   // Brier of predicting the constant base rate
            var delta = (b0 is double x && b24 is double yv && yv > 0) ? (x - yv) / yv * 100.0 : double.NaN;
            Console.WriteLine($"  {tau + "h",4} {any.N,6} {wetRate,5:P0} " +
                $"{(b0 is double v0 ? v0.ToString("F4") : "-"),9} " +
                $"{(b24 is double v24 ? v24.ToString("F4") : "-"),9} " +
                $"{(double.IsNaN(delta) ? "-" : delta.ToString("+0.0;-0.0") + "%"),16} " +
                $"{clim,9:F4}");
        }
        Console.WriteLine();
        Console.WriteLine("  (negative Δ% = nowcast better. clim = base-rate Brier reference.)");
        Console.WriteLine();
        return acc.Count == 0 ? 3 : 0;
    }

    // ===================== NOWCAST (lead-0) bake-off — TEMPERATURE 2c =====================
    //
    // Temp twin of the precip nowcast bake-off, SAME methodology: train m0
    // (lead-0, hist_forecast) + m24 (lead-24, offset_day) on ≤cutoff, then score
    // BOTH at target leads τ ∈ {0,3,6,9,12} on the held-out window (cutoff, now],
    // each fed the freshest live cycle ≥τh stale (PredictForecastFilters
    // .LiveCycleAsOf, non-offset_day), vs ERA5 (MAE °C). τ=0's pivot grabs the
    // hist_forecast archive row (≈analysis, RunTime=valid out-ranks live cycles —
    // not available at live predict time); τ≥3 use live cycles only, the live-
    // reproducible comparison. Headline = τ=12. Feature layout is lead-
    // independent so a model trained at one lead scores any τ's pivot.
    private sealed record TempPivot(
        double?[] Temp, double?[] Dew, double?[] Rh, double?[] Cloud,
        double?[] CloudLow, double?[] CloudMid, double?[] CloudHigh,
        double?[] WindSpeed, double?[] WindDir, double?[] WindGust, double?[] Pressure);

    public async Task<int> RunNowcastBakeoffTempAsync(string? cutoffStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var cutoff = DateOnly.TryParse(cutoffStr, out var dc)
            ? dc.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 15, 0, 0, 0, DateTimeKind.Utc);
        var minValid = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        int[] taus = { 0, 3, 6, 9, 12 };
        var hp = TempTrainer.Hyperparameters.Default();
        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        _log.LogInformation("Temp nowcast bake-off (2c) — loc={Loc}, train ≤{Cut:yyyy-MM-dd}, eval ({Cut:yyyy-MM-dd},{End:yyyy-MM-dd}], τ=[{T}]; ERA5 truth.",
            location.Name, cutoff, asOf, string.Join(",", taus));

        // Scan-once caches: all Bonehill forecasts (training BuildForLead + the
        // τ pivot read it; both filter source/lead internally) + Bonehill ERA5
        // (training join + truth). LocationName preserved so the SQL filters match.
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        var root = Path.GetDirectoryName(_cfg.Storage.ModelsPath)!;
        var fcCache = Path.Combine(root, "scratch", "temp_nowcast_bake", "fc"); Directory.CreateDirectory(Path.Combine(fcCache, "p"));
        var eraCache = Path.Combine(root, "scratch", "temp_nowcast_bake", "era"); Directory.CreateDirectory(Path.Combine(eraCache, "p"));
        var fcGlob = Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"));
        var eraGlob = Esc(Path.Combine(_cfg.Storage.Era5Path, "**", "*.parquet"));
        var escLoc = location.Name.Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{fcGlob}', hive_partitioning=false, union_by_name=true)) TO '{Esc(Path.Combine(fcCache, "p", "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{eraGlob}', hive_partitioning=false, union_by_name=true) WHERE LocationName = '{escLoc}') TO '{Esc(Path.Combine(eraCache, "p", "era.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }

        // Train m0 (lead 0) + m24 (lead 24) on ≤cutoff.
        var spec0 = TempRichFeatureBuilder.BuildSpec(_cfg.Blenders, NowcastSource.LeadHours);
        var spec24 = TempRichFeatureBuilder.BuildSpec(_cfg.Blenders, 24);
        var train0 = TempRichFeatureBuilder.BuildForLead(fcCache, eraCache, location.Name, spec0, minValid, ct)
            .Where(r => r.ValidTimeUtc <= cutoff).ToList();
        var train24 = TempRichFeatureBuilder.BuildForLead(fcCache, eraCache, location.Name, spec24, minValid, ct)
            .Where(r => r.ValidTimeUtc <= cutoff).ToList();
        if (train0.Count < 300 || train24.Count < 300)
        {
            _log.LogError("  too few train rows (lead0={N0}, lead24={N24}) ≤cutoff — widen window / check backfill.", train0.Count, train24.Count);
            return 2;
        }
        var ds0 = RegressionDataset.Split(train0);
        var ds24 = RegressionDataset.Split(train24);
        var m0 = TempTrainer.TrainVector(ds0.Train, ds0.Val, spec0, hp);
        var m24 = TempTrainer.TrainVector(ds24.Train, ds24.Val, spec24, hp);
        _log.LogInformation("  trained m0 (lead-0, n={N0}) + m24 (lead-24, n={N24}).", ds0.Train.Count, ds24.Train.Count);

        // ERA5 truth over the held-out window.
        var truth = LoadEra5Temp(eraCache, location.Name, cutoff, asOf, ct);

        static string Pct(double a, double b) => (double.IsNaN(a) || double.IsNaN(b) || b <= 0)
            ? "-" : ((a - b) / b * 100.0).ToString("+0.0;-0.0") + "%";

        Console.WriteLine();
        Console.WriteLine("=== Temp nowcast bake-off (2c): MAE °C vs ERA5, Bonehill, walk-forward OOS (live cycles, τ-fed) ===");
        Console.WriteLine($"  {"τ",4} {"N",6} {"m0 (nowcast)",13} {"m24",10} {"Δ% (m0 vs m24)",16}");
        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestTempRows(fcCache, location.Name, cutoff, asOf, asOf, tau, canon, ct);
            double s0 = 0, s24 = 0; int n0 = 0, n24 = 0;
            foreach (var (v, t) in truth)
            {
                if (!pivot.TryGetValue(v, out var pv)) continue;
                var p0 = PredictTemp(m0, spec0, canon, v, pv);
                var p24 = PredictTemp(m24, spec24, canon, v, pv);
                if (p0 is double x0) { s0 += Math.Abs(x0 - t); n0++; }
                if (p24 is double x24) { s24 += Math.Abs(x24 - t); n24++; }
            }
            var mae0 = n0 > 0 ? s0 / n0 : double.NaN;
            var mae24 = n24 > 0 ? s24 / n24 : double.NaN;
            Console.WriteLine($"  {tau + "h",4} {Math.Min(n0, n24),6} {mae0,13:F3} {mae24,10:F3} {Pct(mae0, mae24),16}");
        }
        Console.WriteLine();
        Console.WriteLine("  (negative Δ% = nowcast better. τ=0 is fed the hist_forecast archive analysis,");
        Console.WriteLine("   which isn't available live; τ≥3 are live-reproducible. Headline = τ=12.)");
        Console.WriteLine();
        return 0;
    }

    // ERA5 hourly 2m-temperature truth (valid→°C) for one location + window.
    private static IReadOnlyList<(DateTime Valid, double TempC)> LoadEra5Temp(
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
    private static IReadOnlyDictionary<DateTime, TempPivot> QueryLatestTempRows(
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

    // ===================== per-lead BLEND crossover (study bundles, OOS) =====================
    //
    // For a fixed bracketing pair (leadA below, leadB above), sweep target-lead τ and score
    // BOTH models PLUS their equal-weight blend on the SAME live input at lead τ, vs EA truth,
    // for 3c AND 3o, pooled over the 4 Bonehill gauges. Study bundles = no-UA, ≤cutoff (live
    // window OOS). Tests "near the bucket boundary a blend(A,B) beats either single, then the
    // upper model takes over". NOTE: input is cycle-selected (lead ≥ τ), so the *actual* lead
    // pools overlap between adjacent hourly τ — effective input resolution is ~6h (NWP cadence),
    // even though τ is swept hourly. The MODEL/blend comparison at each τ is exact.

    private sealed class BlendAcc { public int N, Wet; public double SqA, SqB, SqBl; }

    public async Task<int> RunPolicyBlendCrossoverAsync(string? startDateStr, IReadOnlyList<int>? tausOverride, int leadA, int leadB, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        int[] taus = tausOverride is { Count: > 0 } ? tausOverride.ToArray()
            : new[] { 36, 42, 45, 47, 48, 49, 51, 54, 60 };

        _log.LogInformation("Policy blend-crossover — loc={Loc} window {S:yyyy-MM-dd}..{E:yyyy-MM-dd} pair (m{A},m{B}) + 50/50 blend, τ=[{T}], 3c+3o study (no-UA), live OOS",
            location.Name, windowStart, windowEnd, leadA, leadB, string.Join(",", taus));

        // Scan-once live cache (non-offset_day rows in window) → one parquet.
        var fcPart = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc", "p");
        Directory.CreateDirectory(fcPart);
        var fcPath = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
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

        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var (slug, friendly) in gauges)
        {
            truth[slug] = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            rain[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, friendly, null, ct);
            foreach (var (dir, dst) in new[] {
                (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3c_noua"), m3c),
                (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3o_noua"), m3o) })
            {
                if (!Directory.Exists(dir)) continue;
                var specs = ModelArtifact.LoadBlenderSpecs(dir);
                var byLead = new Dictionary<int, (ITransformer, BlenderSpec)>();
                foreach (var lead in new[] { leadA, leadB })
                    if (specs.TryGetValue(lead, out var sp))
                        byLead[lead] = (ModelArtifact.LoadLeadModel(ml, dir, lead, out _), sp);
                dst[slug] = byLead;
            }
        }
        var anySpec3o = m3o.Values.FirstOrDefault()?.Values.FirstOrDefault().Spec;
        var aux = anySpec3o is null ? new() : PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(fcPath, location.Name, anySpec3o, windowStart, windowEnd, ct);

        var acc = new Dictionary<string, BlendAcc>();
        BlendAcc Acc(string k) { if (!acc.TryGetValue(k, out var a)) { a = new BlendAcc(); acc[k] = a; } return a; }

        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestForecastRows(fcPath, location.Name, windowStart, windowEnd, asOf, tau, ct);
            foreach (var (slug, friendly) in gauges)
            {
                if (!Phase3oStationIndex.TryGetValue(slug, out var stIdx)) stIdx = -1;
                oroBySlug.TryGetValue(slug, out var oro);
                foreach (var (phase, store) in new[] { ("3c", m3c), ("3o", m3o) })
                {
                    if (!store.TryGetValue(slug, out var byLead)) continue;
                    if (!byLead.TryGetValue(leadA, out var msA) || !byLead.TryGetValue(leadB, out var msB)) continue;
                    if (phase == "3o" && (oro is null || stIdx < 0)) continue;
                    var a = Acc($"{phase}|{tau}");
                    foreach (var (V, mm) in truth[slug])
                    {
                        if (!pivot.TryGetValue(V, out var pv) || !pv.Precip.Any(p => p.HasValue)) continue;
                        double? pA = phase == "3c"
                            ? Predict3c(ml, msA.Model, msA.Spec, canon, V, tau, pv, rain[slug], emptyUa)
                            : Predict3o(ml, msA.Model, msA.Spec, canon, V, tau, pv, rain[slug], emptyUa, oro!, aux, stIdx);
                        double? pB = phase == "3c"
                            ? Predict3c(ml, msB.Model, msB.Spec, canon, V, tau, pv, rain[slug], emptyUa)
                            : Predict3o(ml, msB.Model, msB.Spec, canon, V, tau, pv, rain[slug], emptyUa, oro!, aux, stIdx);
                        if (pA is null || pB is null) continue;
                        var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                        var bl = 0.5 * (pA.Value + pB.Value);
                        a.SqA += (pA.Value - y) * (pA.Value - y);
                        a.SqB += (pB.Value - y) * (pB.Value - y);
                        a.SqBl += (bl - y) * (bl - y);
                        a.N++; if (y > 0) a.Wet++;
                    }
                }
            }
            _log.LogInformation("  τ={Tau}h scored.", tau);
        }

        foreach (var phase in new[] { "3c", "3o" })
        {
            Console.WriteLine();
            Console.WriteLine($"=== {phase} blend-crossover: m{leadA} vs m{leadB} vs blend50 by target-lead τ (Bonehill pooled, live OOS, no-UA) ===");
            Console.WriteLine(" Brier lower=better. winner = lowest of the three. τ input is cycle-selected (≈6h effective resolution).");
            Console.WriteLine($"  {"τ",5} {"N",6} {"wet%",5} {$"m{leadA}",9} {$"m{leadB}",9} {"blend50",9} {"winner",8} {"blend Δ vs best-single",22}");
            foreach (var tau in taus)
            {
                if (!acc.TryGetValue($"{phase}|{tau}", out var a) || a.N == 0) continue;
                double bA = a.SqA / a.N, bB = a.SqB / a.N, bBl = a.SqBl / a.N;
                var bestSingle = Math.Min(bA, bB);
                var trip = new[] { (bA, $"m{leadA}"), (bB, $"m{leadB}"), (bBl, "blend50") };
                var win = trip.OrderBy(x => x.Item1).First().Item2;
                var dpct = (bBl - bestSingle) / bestSingle * 100;
                Console.WriteLine($"  {tau + "h",5} {a.N,6} {(double)a.Wet / a.N,5:P0} {bA,9:F4} {bB,9:F4} {bBl,9:F4} {win,8} {dpct,9:+0.0;-0.0;0.0}% vs best");
            }
        }
        Console.WriteLine();
        return acc.Values.Any(a => a.N > 0) ? 0 : 3;
    }

    // ===================== per-lead BAND policy (all candidates + best equal-weight top-2 blend) =====================
    //
    // For each target-lead τ (default 3h-spaced, 12..120), score EVERY candidate model
    // {24,48,72,96,120} fed the SAME live τ input (bidirectional — a longer-trained model on a
    // fresher-than-nominal input is allowed), for 3c AND 3o, pooled over the 4 Bonehill gauges, vs
    // EA truth. Also accumulate every PAIR's equal-weight blend. Then aggregate per-τ into 3h and 6h
    // bands and, per band, report the best single model and the best equal-weight 2-model blend, with
    // a recommended policy (blend only if it clears a margin over the best single). Study bundles =
    // no-UA ≤cutoff so the live window is OOS. A row counts only when ALL candidates produce a
    // prediction, so every series shares identical rows (apples-to-apples). same-window select=score
    // (mild optimism on marginal picks) — the SELECT/SCORE split + quarterly cadence are the guards.

    public async Task<int> RunPolicyBandAsync(string? startDateStr, IReadOnlyList<int>? tausOverride, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var windowStart = DateOnly.TryParse(startDateStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow;
        var windowEnd = asOf;
        int[] cands = { 24, 48, 72, 96, 120 };
        int[] taus = tausOverride is { Count: > 0 } ? tausOverride.ToArray()
            : Enumerable.Range(0, (120 - 12) / 3 + 1).Select(i => 12 + 3 * i).ToArray();
        const double BlendMarginPct = 0.5; // blend recommended only if ≥0.5% better than best single

        _log.LogInformation("Policy band — loc={Loc} window {S:yyyy-MM-dd}..{E:yyyy-MM-dd} all-candidate {C} + best top-2 blend, τ=[{T}], 3c+3o study (no-UA), live OOS",
            location.Name, windowStart, windowEnd, string.Join(",", cands), string.Join(",", taus));

        var fcPart = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc", "p");
        Directory.CreateDirectory(fcPart);
        var fcPath = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        using (var conn = new DuckDBConnection("DataSource=:memory:"))
        {
            conn.Open();
            using var c = conn.CreateCommand();
            c.CommandText =
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
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

        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var (slug, friendly) in gauges)
        {
            truth[slug] = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            rain[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, friendly, null, ct);
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
        var anySpec3o = m3o.Values.FirstOrDefault()?.Values.FirstOrDefault().Spec;
        var aux = anySpec3o is null ? new() : PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(fcPath, location.Name, anySpec3o, windowStart, windowEnd, ct);

        var pairs = new List<(int Lo, int Hi)>();
        for (int i = 0; i < cands.Length; i++) for (int j = i + 1; j < cands.Length; j++) pairs.Add((cands[i], cands[j]));

        var sq = new Dictionary<string, double>();   // phase|tau|series  → Σ squared error
        var nByPt = new Dictionary<string, int>();    // phase|tau         → N
        var wByPt = new Dictionary<string, int>();    // phase|tau         → wet
        void AddSq(string k, double v) => sq[k] = sq.GetValueOrDefault(k) + v;

        foreach (var tau in taus)
        {
            ct.ThrowIfCancellationRequested();
            var pivot = QueryLatestForecastRows(fcPath, location.Name, windowStart, windowEnd, asOf, tau, ct);
            foreach (var (slug, friendly) in gauges)
            {
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
                        nByPt[ptk] = nByPt.GetValueOrDefault(ptk) + 1;
                        if (y > 0) wByPt[ptk] = wByPt.GetValueOrDefault(ptk) + 1;
                        for (int i = 0; i < cands.Length; i++)
                            AddSq($"{ptk}|s{cands[i]}", (preds[i] - y) * (preds[i] - y));
                        foreach (var (lo, hi) in pairs)
                        {
                            var bl = 0.5 * (preds[Array.IndexOf(cands, lo)] + preds[Array.IndexOf(cands, hi)]);
                            AddSq($"{ptk}|b{lo}x{hi}", (bl - y) * (bl - y));
                        }
                    }
                }
            }
            _log.LogInformation("  τ={Tau}h scored.", tau);
        }

        foreach (var bandWidth in new[] { 3, 6 })
        {
            foreach (var phase in new[] { "3c", "3o" })
            {
                Console.WriteLine();
                Console.WriteLine($"=== {phase}: {bandWidth}h-band policy (Bonehill pooled, live OOS, no-UA; all candidates + best equal-weight top-2 blend) ===");
                Console.WriteLine($"  {"band",10} {"N",6} {"wet%",5} {"best single",14} {"best blend",18} {"recommend",16}");
                for (int lo = 12; lo < 120; lo += bandWidth)
                {
                    int hi = lo + bandWidth;
                    var tin = taus.Where(t => t >= lo && t < hi).ToArray();
                    if (tin.Length == 0) continue;
                    int N = tin.Sum(t => nByPt.GetValueOrDefault($"{phase}|{t}"));
                    if (N == 0) continue;
                    int wet = tin.Sum(t => wByPt.GetValueOrDefault($"{phase}|{t}"));
                    double Brier(string series) => tin.Sum(t => sq.GetValueOrDefault($"{phase}|{t}|{series}")) / N;
                    var (bestSLead, bestSBr) = cands.Select(c => (c, Brier($"s{c}"))).OrderBy(x => x.Item2).First();
                    var (bestPair, bestBBr) = pairs.Select(p => (p, Brier($"b{p.Lo}x{p.Hi}"))).OrderBy(x => x.Item2).First();
                    var useBlend = bestBBr < bestSBr * (1 - BlendMarginPct / 100.0);
                    var rec = useBlend ? $"blend {bestPair.Lo}+{bestPair.Hi}" : $"m{bestSLead}";
                    Console.WriteLine($"  {lo + "-" + hi + "h",10} {N,6} {(double)wet / N,5:P0} {$"m{bestSLead} {bestSBr:F4}",14} {$"{bestPair.Lo}+{bestPair.Hi} {bestBBr:F4}",18} {rec,16}");
                }
            }
        }
        Console.WriteLine();
        return nByPt.Count == 0 ? 3 : 0;
    }

    // ===================== per-lead policy EVAL v2: full bidirectional matrix + SELECT/SCORE split + blend =====================
    //
    // Per (phase, τ): score ALL candidate models (both shorter AND longer than τ — full overlap,
    // each composed with its own spec from the shared lead-τ pivot), split rows by date into SELECT
    // (fit blend weights + pick best single) and SCORE (held-out). Report the full per-model SCORE
    // Brier matrix + baseline (production bucket) + best-single + blend, so nothing is scored on its
    // own selection data.

    public async Task<int> RunPolicyEvalSplitAsync(string? startDateStr, string? splitDateStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
        var studyRoot = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "models_study");
        var windowStart = DateOnly.TryParse(startDateStr, out var d0) ? d0.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc) : new DateTime(2026, 3, 19, 0, 0, 0, DateTimeKind.Utc);
        var splitDate = DateOnly.TryParse(splitDateStr, out var d1) ? d1.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc) : new DateTime(2026, 5, 5, 0, 0, 0, DateTimeKind.Utc);
        var asOf = DateTime.UtcNow; var windowEnd = asOf;
        int[] cands = { 24, 48, 72, 96, 120 };   // ALL models at every lead (full bidirectional overlap)
        int[] taus = { 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 108, 120 };
        static int Bucket(int tau) => tau >= 120 ? 120 : tau >= 96 ? 96 : tau >= 72 ? 72 : tau >= 48 ? 48 : 24;

        var fcPart = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc", "p");
        Directory.CreateDirectory(fcPart);
        var fcPath = Path.Combine(Path.GetDirectoryName(_cfg.Storage.ModelsPath)!, "scratch", "policy_eval", "fc");
        static string Esc(string p) => p.Replace('\\', '/').Replace("'", "''");
        if (!File.Exists(Path.Combine(fcPart, "fc.parquet")))
        {
            using var conn = new DuckDBConnection("DataSource=:memory:");
            conn.Open(); using var c = conn.CreateCommand();
            // Live forecasts only (exclude offset_day backfill + hist_forecast
            // archive) — score on what predict sees at runtime. See the fuller
            // note in RunFitLeadPolicyAsync's cache.
            c.CommandText = $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) WHERE (RunTimeSource IS NULL OR RunTimeSource NOT IN ('offset_day', 'hist_forecast')) AND ValidTimeUtc BETWEEN TIMESTAMP '{windowStart:yyyy-MM-dd HH:mm:ss}' AND TIMESTAMP '{windowEnd:yyyy-MM-dd HH:mm:ss}') TO '{Esc(Path.Combine(fcPart, "fc.parquet"))}' (FORMAT PARQUET);";
            c.ExecuteNonQuery();
        }
        _log.LogInformation("Policy eval v2 (full overlap) — SELECT < {Split:yyyy-MM-dd} ≤ SCORE; window {S:yyyy-MM-dd}..{E:yyyy-MM-dd}", splitDate, windowStart, windowEnd);

        var canon = TempFeatureBuilder.CanonicalModelOrder.ToList();
        var ml = new MLContext(seed: 42);
        var emptyUa = System.Array.Empty<(DateTime, double[])>();
        var oroBySlug = OroStaticFeatures.LoadAll(Path.Combine(Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic"));
        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer M, BlenderSpec S)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer M, BlenderSpec S)>>();
        foreach (var (slug, friendly) in gauges)
        {
            truth[slug] = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            rain[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, friendly, null, ct);
            foreach (var (dir, dst) in new[] { (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3c_noua"), m3c), (Path.Combine(studyRoot, "precipitation", slug, "vstudy_phase3o_noua"), m3o) })
            {
                if (!Directory.Exists(dir)) continue;
                var specs = ModelArtifact.LoadBlenderSpecs(dir); var byLead = new Dictionary<int, (ITransformer, BlenderSpec)>();
                foreach (var lead in cands) if (specs.TryGetValue(lead, out var sp)) byLead[lead] = (ModelArtifact.LoadLeadModel(ml, dir, lead, out _), sp);
                dst[slug] = byLead;
            }
        }
        var anySpec3o = m3o.Values.FirstOrDefault()?.Values.FirstOrDefault().S;
        var aux = anySpec3o is null ? new() : PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive(fcPath, location.Name, anySpec3o, windowStart, windowEnd, ct);

        Console.WriteLine();
        Console.WriteLine("=== Policy eval v2: full bidirectional matrix on held-out SCORE (Bonehill pooled, OOS) — Brier, *best, [base] ===");
        foreach (var phase in new[] { "3c", "3o" })
        {
            Console.WriteLine();
            Console.WriteLine($"### {phase}   (SELECT<{splitDate:MM-dd}, SCORE≥; all 5 models at every τ)");
            Console.WriteLine($"  {"τ",4} {"Nsc",6} " + string.Join(" ", cands.Select(m => $"{("m" + m),8}")) + $"  {"best1",6} {"blend",7}  weights");
            var store = phase == "3c" ? m3c : m3o;
            foreach (var tau in taus)
            {
                var pivot = QueryLatestForecastRows(fcPath, location.Name, windowStart, windowEnd, asOf, tau, ct);
                var sel = new List<(double[] p, double y)>(); var sco = new List<(double[] p, double y)>();
                foreach (var (slug, friendly) in gauges)
                {
                    if (!store.TryGetValue(slug, out var byLead)) continue;
                    Phase3oStationIndex.TryGetValue(slug, out var stIdx); oroBySlug.TryGetValue(slug, out var oro);
                    foreach (var (V, mm) in truth[slug])
                    {
                        if (!pivot.TryGetValue(V, out var pv) || !pv.Precip.Any(p => p.HasValue)) continue;
                        var preds = new double[cands.Length]; bool ok = true;
                        for (int i = 0; i < cands.Length; i++)
                        {
                            if (!byLead.TryGetValue(cands[i], out var ms)) { ok = false; break; }
                            double? p = phase == "3c"
                                ? Predict3c(ml, ms.Item1, ms.Item2, canon, V, tau, pv, rain[slug], emptyUa)
                                : (oro is null || stIdx < 0 ? null : Predict3o(ml, ms.Item1, ms.Item2, canon, V, tau, pv, rain[slug], emptyUa, oro, aux, stIdx));
                            if (p is null) { ok = false; break; } preds[i] = p.Value;
                        }
                        if (!ok) continue;
                        var y = mm >= WetThresholdMm ? 1.0 : 0.0;
                        (V < splitDate ? sel : sco).Add((preds, y));
                    }
                }
                if (sel.Count < 50 || sco.Count < 50) { Console.WriteLine($"  {tau + "h",4}  (insufficient: sel={sel.Count} sco={sco.Count})"); continue; }
                double Brier(List<(double[] p, double y)> rows, Func<double[], double> f) => rows.Average(r => { var e = f(r.p) - r.y; return e * e; });
                var scoB = cands.Select((m, i) => Brier(sco, p => p[i])).ToArray();
                int baseIdx = Math.Max(0, Array.IndexOf(cands, Bucket(tau)));
                int bestSel = 0; double bv = double.MaxValue;
                for (int i = 0; i < cands.Length; i++) { var b = Brier(sel, p => p[i]); if (b < bv) { bv = b; bestSel = i; } }
                var w = FitSimplexWeights(sel, cands.Length);
                double blendSc = Brier(sco, p => { double s = 0; for (int i = 0; i < w.Length; i++) s += w[i] * p[i]; return s; });
                int bestScIdx = 0; double bsv = double.MaxValue;
                for (int i = 0; i < cands.Length; i++) if (scoB[i] < bsv) { bsv = scoB[i]; bestScIdx = i; }
                var cells = string.Join(" ", cands.Select((m, i) =>
                    $"{(i == bestScIdx ? "*" : i == baseIdx ? "[" : " ")}{scoB[i],6:F4}{(i == baseIdx ? "]" : " ")}"));
                var wStr = string.Join("/", cands.Select((m, i) => w[i] > 0.02 ? $"{m}:{w[i]:F2}" : null).Where(x => x != null));
                Console.WriteLine($"  {tau + "h",4} {sco.Count,6} {cells}  m{cands[bestSel],-4} {blendSc,7:F4}  {wStr}");
            }
        }
        Console.WriteLine();
        return 0;
    }

    /// <summary>Non-negative weights summing to 1 minimising mean (Σ wᵢ pᵢ − y)² (Brier), via
    /// projected-gradient over the simplex.</summary>
    private static double[] FitSimplexWeights(List<(double[] p, double y)> rows, int k)
    {
        var w = Enumerable.Repeat(1.0 / k, k).ToArray();
        int n = rows.Count; double lr = 0.5;
        for (int it = 0; it < 4000; it++)
        {
            var g = new double[k];
            foreach (var (p, y) in rows) { double r = 0; for (int j = 0; j < k; j++) r += w[j] * p[j]; r -= y; for (int j = 0; j < k; j++) g[j] += 2.0 * r * p[j]; }
            for (int j = 0; j < k; j++) w[j] -= lr * g[j] / n;
            w = ProjectSimplex(w);
        }
        return w;
    }

    private static double[] ProjectSimplex(double[] v)
    {
        int k = v.Length; var u = (double[])v.Clone(); Array.Sort(u); Array.Reverse(u);
        double css = 0, theta = 0;
        for (int j = 0; j < k; j++) { css += u[j]; var t = (css - 1.0) / (j + 1); if (u[j] - t > 0) theta = t; }
        var w = new double[k]; for (int i = 0; i < k; i++) w[i] = Math.Max(v[i] - theta, 0.0); return w;
    }

    private static double[] AssembleUaBlock(double[] perModelCol)
    {
        int mc = PrecipFeatureBuilder.UpperAirModels.Length;
        int pc = PrecipFeatureBuilder.UaPressureCols.Length;
        var outv = new double[pc * mc + 2];
        Array.Copy(perModelCol, outv, Math.Min(perModelCol.Length, pc * mc));
        int t850Off = Array.FindIndex(PrecipFeatureBuilder.UaPressureCols, c => c.Short == "t850");
        int rh850Off = Array.FindIndex(PrecipFeatureBuilder.UaPressureCols, c => c.Short == "rh850");
        double ts = 0, rs = 0; int tn = 0, rn = 0;
        for (int k = 0; k < mc; k++)
        {
            var tv = perModelCol[pc * k + t850Off]; if (!double.IsNaN(tv)) { ts += tv; tn++; }
            var rv = perModelCol[pc * k + rh850Off]; if (!double.IsNaN(rv)) { rs += rv; rn++; }
        }
        outv[pc * mc] = tn == 0 ? double.NaN : ts / tn;
        outv[pc * mc + 1] = rn == 0 ? double.NaN : rs / rn;
        return outv;
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
        string? startDateStr, string? cutoffStr, CancellationToken ct)
    {
        await Task.Yield();
        var location = _cfg.Location;
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

        int[] cands = { 24, 48, 72, 96, 120 };
        var taus = Enumerable.Range(0, (117 - 12) / 3 + 1).Select(i => 12 + 3 * i).ToArray();
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
                $"COPY (SELECT * FROM read_parquet('{Esc(Path.Combine(_cfg.Storage.ForecastsPath, "location=bonehill_rocks", "**", "*.parquet"))}', hive_partitioning=false, union_by_name=true) " +
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
        var gauges = location.Rainfall.Stations.Select(s => (Slug: StationSlug.WithEaPrefix(s.Name), Friendly: s.Name)).ToList();
        var truth = new Dictionary<string, IReadOnlyList<(DateTime, double)>>();
        var rain = new Dictionary<string, Dictionary<DateTime, double>>();
        var m3c = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        var m3o = new Dictionary<string, Dictionary<int, (ITransformer Model, BlenderSpec Spec)>>();
        foreach (var (slug, friendly) in gauges)
        {
            truth[slug] = LoadHourlyTruth(_cfg.Storage.RainfallPath, location.Name, friendly, windowStart, windowEnd, ct);
            rain[slug] = PrecipRichFeatureBuilder.LoadHourlyRain(_cfg.Storage.RainfallPath, location.Name, friendly, null, ct);
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
            foreach (var (slug, friendly) in gauges)
            {
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
        var incumbent = PrecipLeadPolicy.TryLoad(modelsRoot);
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
            for (int lo = 12; lo < 120; lo += 6)
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

                // Candidate set: singles always; pairs when the phase allows.
                var candidates = new List<(string Series, string Kind, List<int> Leads)>();
                foreach (var m in cands) candidates.Add(($"s{m}", "single", new List<int> { m }));
                if (allowBlend)
                    foreach (var (l, h) in pairs) candidates.Add(($"b{l}x{h}", "blend", new List<int> { l, h }));

                // 1. Pick on SELECT.
                var pick = candidates.OrderBy(c => SelB(c.Series)).First();
                // 2. Hysteresis vs incumbent: a different challenger must beat the
                //    incumbent on SCORE by HysteresisPct or the incumbent stays.
                var inc = incumbent?.Lookup(phase, lo);
                string IncSeries(PrecipLeadPolicy.BandEntry e) =>
                    e.Kind == "blend" ? $"b{e.Leads[0]}x{e.Leads[1]}" : $"s{e.Leads[0]}";
                if (inc is not null)
                {
                    var incSco = ScoB(IncSeries(inc));
                    if (!double.IsNaN(incSco)
                        && IncSeries(inc) != pick.Series
                        && !(ScoB(pick.Series) <= incSco * (1 - thresholds.HysteresisPct / 100.0)))
                        pick = candidates.First(c => c.Series == IncSeries(inc));
                }
                // 3. Margin gate vs baseline on SCORE. Incumbents re-qualify at
                //    (margin − hysteresis) so a sub-threshold wobble can't churn them out.
                var pickSco = ScoB(pick.Series);
                var isIncumbentPick = inc is not null && IncSeries(inc) == pick.Series;
                var requiredPct = isIncumbentPick
                    ? Math.Max(0.0, thresholds.MarginPct - thresholds.HysteresisPct)
                    : thresholds.MarginPct;
                var isBaselinePick = pick.Kind == "single" && pick.Leads[0] == baseLead;
                var passes = !isBaselinePick && pickSco <= baseSco * (1 - requiredPct / 100.0);

                var deltaPct = 100.0 * (baseSco - pickSco) / baseSco;
                var decision = passes
                    ? (pick.Kind == "blend" ? $"blend {pick.Leads[0]}+{pick.Leads[1]}" : $"m{pick.Leads[0]}")
                    : $"baseline m{baseLead}";
                Console.WriteLine($"  {lo + "-" + hi + "h",9} {nSco,6} {$"m{baseLead} {baseSco:F4}",13} {$"{pick.Series} sel",16} {pickSco,8:F4} {decision,18}{(passes ? $"  (+{deltaPct:F2}%)" : "")}");

                if (passes)
                    entries.Add(new PrecipLeadPolicy.BandEntry
                    {
                        LeadLo = lo, LeadHi = hi, Kind = pick.Kind, Leads = pick.Leads,
                        BaselineBrier = baseSco, PolicyBrier = pickSco,
                        DeltaPct = deltaPct, ScoreN = nSco,
                    });
            }
            if (entries.Count > 0) policy.Phases[phase] = entries;
        }

        policy.Save(modelsRoot);
        Console.WriteLine();
        Console.WriteLine($"LEAD_POLICY.json written → {PrecipLeadPolicy.PathFor(modelsRoot)} " +
                          $"({policy.Phases.Sum(p => p.Value.Count)} deviation band(s); absent bands = production buckets)");
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
    private static IReadOnlyList<(DateTime Valid, double Mm)> LoadHourlyTruth(
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
}
