using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Evaluate.Precip;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Commands;

/// <summary>
/// Stage-1 bake-off for Phase 3f (rainfall_amount, Membury). Trains three
/// candidate hourly-P(wet) classifiers on identical row keys + chronological
/// 70/15/15 split, dumps per-station test predictions:
///   3a       per-station LightGBM lean    (current 3f stage-1)
///   3c       per-station LightGBM rich
///   3c-oro   pooled (3 Membury stations) LightGBM rich + 9 terrain features
///
/// Output is consumed by scripts/run_membury_two_stage_bakeoff_stage1.py which
/// re-fits stage-2 NGBoost-LogNormal on the existing intensity cache and
/// scores CRPS for each (station, lead, stage-1 variant) cell.
///
/// Important guarantee: all three variants train on the rich-filter row set
/// (the largest of the three filters — lean is a subset). 3a is therefore
/// trained on slightly more rows than its production self, but the same as
/// 3c / 3c-oro here. That's the only way to score them on identical test rows.
/// </summary>
public sealed class Phase3fStage1BakeoffCommand
{
    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly string[] MemburyStations = { "Chards Snowdon Hill", "Goren", "Raymonds Hill" };

    private readonly AppConfig _cfg;
    private readonly ILogger<Phase3fStage1BakeoffCommand> _log;

    public Phase3fStage1BakeoffCommand(AppConfig cfg, ILogger<Phase3fStage1BakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var outRoot = Path.Combine(_cfg.Storage.ReportsPath, $"3f_stage1_bakeoff_{DateTime.UtcNow:yyyy-MM-dd}");
        Directory.CreateDirectory(Path.Combine(outRoot, "3a"));
        Directory.CreateDirectory(Path.Combine(outRoot, "3c"));
        Directory.CreateDirectory(Path.Combine(outRoot, "3c_oro"));
        _log.LogInformation("Output dir: {Path}", outRoot);

        // Resolve Membury station configs + oro records
        var memburyLoc = _cfg.Locations.First(l => l.Name == "membury_devon");
        var stations = new List<(string Name, string Slug, OroStaticFeatures Oro, int Index)>();
        var oroBySlug = OroStaticFeatures.LoadAll(Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic"));
        for (int i = 0; i < MemburyStations.Length; i++)
        {
            var name = MemburyStations[i];
            var match = memburyLoc.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Name}' not in membury_devon rainfall config", name);
                return 2;
            }
            var slug = StationSlug.WithEaPrefix(match.Name);
            if (!oroBySlug.TryGetValue(slug, out var oro))
            {
                _log.LogError("No oro static record for {Slug}", slug);
                return 2;
            }
            stations.Add((match.Name, slug, oro, i));
        }
        _log.LogInformation("Stations: {N} resolved", stations.Count);

        var hpGbt = new PrecipOccurrenceTrainer.Hyperparameters();
        var manifestRows = new List<ManifestRow>();
        var pred3a    = stations.ToDictionary(s => s.Slug, _ => new List<PredictionRow>());
        var pred3c    = stations.ToDictionary(s => s.Slug, _ => new List<PredictionRow>());
        var pred3cOro = stations.ToDictionary(s => s.Slug, _ => new List<PredictionRow>());

        foreach (var lead in Leads)
        {
            ct.ThrowIfCancellationRequested();
            _log.LogInformation("===== Lead {Lead}h =====", lead);

            var leanSpec    = PrecipFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var richSpec    = PrecipRichFeatureBuilder.BuildSpec(_cfg.Blenders, lead);
            var richOroSpec = PrecipRichOroFeatureBuilder.BuildSpec(_cfg.Blenders, lead);

            var perStation = new List<StationDataset>();
            foreach (var st in stations)
            {
                ct.ThrowIfCancellationRequested();
                var lean    = PrecipFeatureBuilder.BuildForLead(_cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    memburyLoc.Name, st.Name, leanSpec, minValidTime: null, ct);
                var rich    = PrecipRichFeatureBuilder.BuildForLead(_cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    memburyLoc.Name, st.Name, richSpec, minValidTime: null, ct);
                var richOro = PrecipRichOroFeatureBuilder.BuildForLead(_cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath,
                    memburyLoc.Name, st.Name, st.Oro, st.Index, richOroSpec, ct);

                _log.LogInformation("  {Slug}: lean={L} rich={R} rich-oro={RO}", st.Slug, lean.Count, rich.Count, richOro.Count);
                if (lean.Count < 100 || rich.Count < 100)
                {
                    _log.LogWarning("  {Slug} lead {Lead}h: too few rows, skipping", st.Slug, lead);
                    continue;
                }

                // Apply chronological 70/15/15 to RICH rows (the larger filter
                // canonically). 3a will inner-join its lean rows to this manifest.
                var dsRich    = BinaryDataset.Split(rich);
                var dsRichOro = BinaryDataset.Split(richOro);

                // Lean rows -> dictionary by ValidTimeUtc for fast join
                var leanByVt = new Dictionary<DateTime, BinaryTrainingRow>(lean.Count);
                foreach (var r in lean)
                    leanByVt[DateTime.SpecifyKind(r.ValidTimeUtc, DateTimeKind.Utc)] = r;

                List<BinaryTrainingRow> JoinLeanTo(IReadOnlyList<BinaryTrainingRow> richRows)
                {
                    var picked = new List<BinaryTrainingRow>(richRows.Count);
                    foreach (var r in richRows)
                        if (leanByVt.TryGetValue(DateTime.SpecifyKind(r.ValidTimeUtc, DateTimeKind.Utc), out var leanRow))
                            picked.Add(leanRow);
                    return picked;
                }

                var leanTrain = JoinLeanTo(dsRich.Train);
                var leanVal   = JoinLeanTo(dsRich.Val);
                var leanTest  = JoinLeanTo(dsRich.Test);
                _log.LogInformation("    {Slug} lean inner-joined to rich split: train {T}/{TR} val {V}/{VR} test {E}/{ER}",
                    st.Slug, leanTrain.Count, dsRich.Train.Count, leanVal.Count, dsRich.Val.Count, leanTest.Count, dsRich.Test.Count);

                perStation.Add(new StationDataset(st.Slug, st.Name, st.Oro, st.Index, dsRich, dsRichOro,
                    new BinaryDataset(leanTrain, leanVal, leanTest)));

                // Manifest: one row per (station, valid_time, lead) for every rich row.
                void AddRows(IReadOnlyList<BinaryTrainingRow> rows, string split)
                {
                    foreach (var r in rows)
                        manifestRows.Add(new ManifestRow
                        {
                            station = st.Slug, valid_time = r.ValidTimeUtc, lead = lead,
                            observed_wet = (byte)(r.Label ? 1 : 0), split = split,
                        });
                }
                AddRows(dsRich.Train, "train"); AddRows(dsRich.Val, "val"); AddRows(dsRich.Test, "test");
            }
            if (perStation.Count < 2) continue;

            // ARM 1: per-station 3a (lean)
            _log.LogInformation("--- Arm: per-station 3a (LightGBM lean) ---");
            foreach (var ps in perStation)
            {
                ct.ThrowIfCancellationRequested();
                if (ps.Lean.Train.Count < 100 || ps.Lean.Test.Count < 20) { _log.LogWarning("    {Slug} lean too few, skip", ps.Slug); continue; }
                var trained = PrecipOccurrenceTrainer.TrainVector(ps.Lean.Train, ps.Lean.Val, leanSpec, hpGbt);
                var probs = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, leanSpec, ps.Lean.Test);
                var truth = ps.Lean.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                _log.LogInformation("    {Slug} 3a lead {Lead}h Brier={B:0.0000} n_test={N}", ps.Slug, lead, PrecipMetrics.Brier(probs, truth), truth.Length);
                for (int i = 0; i < ps.Lean.Test.Count; i++)
                    pred3a[ps.Slug].Add(new PredictionRow {
                        valid_time = ps.Lean.Test[i].ValidTimeUtc, station = ps.Slug, lead = lead,
                        p_wet = probs[i], observed_wet = (byte)(ps.Lean.Test[i].Label ? 1 : 0),
                    });
            }

            // ARM 2: per-station 3c (rich)
            _log.LogInformation("--- Arm: per-station 3c (LightGBM rich) ---");
            foreach (var ps in perStation)
            {
                ct.ThrowIfCancellationRequested();
                var trained = PrecipOccurrenceTrainer.TrainVector(ps.Rich.Train, ps.Rich.Val, richSpec, hpGbt);
                var probs = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, richSpec, ps.Rich.Test);
                var truth = ps.Rich.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                _log.LogInformation("    {Slug} 3c lead {Lead}h Brier={B:0.0000} n_test={N}", ps.Slug, lead, PrecipMetrics.Brier(probs, truth), truth.Length);
                for (int i = 0; i < ps.Rich.Test.Count; i++)
                    pred3c[ps.Slug].Add(new PredictionRow {
                        valid_time = ps.Rich.Test[i].ValidTimeUtc, station = ps.Slug, lead = lead,
                        p_wet = probs[i], observed_wet = (byte)(ps.Rich.Test[i].Label ? 1 : 0),
                    });
            }

            // ARM 3: pooled 3c-oro (rich + 9 terrain), trained across all 3 Membury stations
            _log.LogInformation("--- Arm: pooled 3c-oro (LightGBM rich + 9 terrain), Membury-only ---");
            var pooledTrain = perStation.SelectMany(ps => ps.RichOro.Train).ToList();
            var pooledVal   = perStation.SelectMany(ps => ps.RichOro.Val).ToList();
            _log.LogInformation("    pooled train={N} val={V}", pooledTrain.Count, pooledVal.Count);
            var pooledOro = PrecipOccurrenceTrainer.TrainVector(pooledTrain, pooledVal, richOroSpec, hpGbt);
            foreach (var ps in perStation)
            {
                var probs = PrecipOccurrenceTrainer.PredictVectorProbability(pooledOro.Ml, pooledOro.Model, richOroSpec, ps.RichOro.Test);
                var truth = ps.RichOro.Test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
                _log.LogInformation("    {Slug} 3c-oro lead {Lead}h Brier={B:0.0000}", ps.Slug, lead, PrecipMetrics.Brier(probs, truth));
                for (int i = 0; i < ps.RichOro.Test.Count; i++)
                    pred3cOro[ps.Slug].Add(new PredictionRow {
                        valid_time = ps.RichOro.Test[i].ValidTimeUtc, station = ps.Slug, lead = lead,
                        p_wet = probs[i], observed_wet = (byte)(ps.RichOro.Test[i].Label ? 1 : 0),
                    });
            }
        }

        _log.LogInformation("Writing manifest + per-phase predictions to {Path}", outRoot);
        await ParquetSerializer.SerializeAsync(manifestRows, Path.Combine(outRoot, "manifest.parquet"), cancellationToken: ct);
        foreach (var st in stations)
        {
            await ParquetSerializer.SerializeAsync(pred3a[st.Slug],    Path.Combine(outRoot, "3a",     st.Slug + ".parquet"), cancellationToken: ct);
            await ParquetSerializer.SerializeAsync(pred3c[st.Slug],    Path.Combine(outRoot, "3c",     st.Slug + ".parquet"), cancellationToken: ct);
            await ParquetSerializer.SerializeAsync(pred3cOro[st.Slug], Path.Combine(outRoot, "3c_oro", st.Slug + ".parquet"), cancellationToken: ct);
        }
        _log.LogInformation("Done. Next: run scripts/run_membury_two_stage_bakeoff_stage1.py to fit stage-2 + score CRPS.");
        return 0;
    }

    private sealed record StationDataset(
        string Slug, string Name, OroStaticFeatures Oro, int Index,
        BinaryDataset Rich, BinaryDataset RichOro, BinaryDataset Lean);

    // Inlined from the retired Phase4bComparisonBakeoffCommand 2026-05-25
    // (model-cleanup Phase 1). The Python join_and_report consumer reads
    // these schemas, so the field names and types are load-bearing.
    public sealed class ManifestRow
    {
        public string station { get; set; } = string.Empty;
        public DateTime valid_time { get; set; }
        public int lead { get; set; }
        public byte observed_wet { get; set; }
        public string split { get; set; } = string.Empty;
    }

    public sealed class PredictionRow
    {
        public DateTime valid_time { get; set; }
        public string station { get; set; } = string.Empty;
        public int lead { get; set; }
        public double p_wet { get; set; }
        public byte observed_wet { get; set; }
    }
}
