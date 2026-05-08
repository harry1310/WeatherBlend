using System.Globalization;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.PrecipExact;

namespace WeatherBlend.Commands;

/// <summary>
/// Brier-scored P(wet) bake-off for the exact-runtime precipitation blender.
/// Mirrors the production 3a metric (binary wet/dry classification, threshold
/// 0.1 mm/h) against the same exact-runtime sources the temperature 2d
/// blender uses (GFS + IFS + AIFS required, MO Global optional, plus UKV
/// when <c>--include-ukv</c> is set — 5th always-optional input via the
/// per-V-hour conditional rule from 03Z + 15Z cycles). Single tier P1,
/// single lead 12h or 24h.
///
/// Per-model raw precip values are still fed as features (LightGBM tree
/// splits handle the unit/semantic differences between APCP / cumulative
/// tp / instantaneous rate). The decision boundary is the wet/dry call,
/// not the absolute precip amount, so the unit mismatches matter much
/// less here than in the earlier MAE-regression bake-off.
///
/// Reports per-model + mean-of-models + climatology Brier alongside the
/// blender. The "did the blender earn its keep?" comparison is vs
/// climatology — Brier Skill Score (BSS) > 0 means the blender beats
/// always-predict-the-base-rate.
/// </summary>
public sealed class PrecipExactBakeoffCommand
{
    public const int HoldoutDays = 90;
    public const int ValidationDays = 30;

    private readonly AppConfig _cfg;
    private readonly ILogger<PrecipExactBakeoffCommand> _log;

    public PrecipExactBakeoffCommand(AppConfig cfg, ILogger<PrecipExactBakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public async Task<int> RunAsync(int targetLead, bool includeUkv, string? stationOverride, CancellationToken ct)
    {
        // Default to the first configured rainfall station if none given.
        // Bake-off is exploratory; pinning one station keeps the print clean.
        // Pass --station to compare a different one head-to-head.
        if (_cfg.Location.Rainfall.Stations.Count == 0)
        {
            _log.LogError("No rainfall stations configured — cannot bake off precip-exact.");
            return 2;
        }
        string station;
        if (string.IsNullOrWhiteSpace(stationOverride))
        {
            station = _cfg.Location.Rainfall.Stations[0].Name;
        }
        else
        {
            var match = _cfg.Location.Rainfall.Stations
                .FirstOrDefault(s => s.Name.Equals(stationOverride, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("Station '{Station}' not found in config. Available: {Available}",
                    stationOverride, string.Join(", ", _cfg.Location.Rainfall.Stations.Select(s => s.Name)));
                return 2;
            }
            station = match.Name;
        }

        foreach (var tier in PrecipExactFeatureBuilder.AllTiers)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Run(() => RunTier(tier, targetLead, includeUkv, station, ct), ct);
        }
        return 0;
    }

    private void RunTier(PrecipExactFeatureBuilder.TierSpec tier, int targetLead, bool includeUkv, string stationName, CancellationToken ct)
    {
        _log.LogInformation("=== Precip-Brier exact bake-off: tier {Tier} target {Lead}h UKV={Ukv} station='{Station}' ===",
            tier.Name, targetLead, includeUkv, stationName);
        var spec = PrecipExactFeatureBuilder.BuildSpec(tier, targetLead, includeUkv);
        var rows = PrecipExactFeatureBuilder.Build(
            _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath, _cfg.Location.Name, stationName,
            tier, spec, targetLead, includeUkv, ct: ct);

        if (rows.Count == 0) { _log.LogWarning("  no rows"); return; }

        var sorted = rows.OrderBy(r => r.ValidTimeUtc).ToList();
        var maxValid = sorted[^1].ValidTimeUtc;
        var testStart = maxValid.AddDays(-HoldoutDays);
        var valStart  = testStart.AddDays(-ValidationDays);
        var train = sorted.Where(r => r.ValidTimeUtc < valStart).ToList();
        var val   = sorted.Where(r => r.ValidTimeUtc >= valStart && r.ValidTimeUtc < testStart).ToList();
        var test  = sorted.Where(r => r.ValidTimeUtc >= testStart).ToList();
        if (train.Count == 0 || val.Count == 0 || test.Count == 0)
        { _log.LogWarning("  empty split"); return; }

        _log.LogInformation("  rows: total={N} train={Tr} val={V} test={Te}", rows.Count, train.Count, val.Count, test.Count);

        // Train classifier with production 3a hyperparameters as defaults.
        var hp = new PrecipOccurrenceTrainer.Hyperparameters();
        var trained = PrecipOccurrenceTrainer.TrainVector(train, val, spec, hp);
        var probs = PrecipOccurrenceTrainer.PredictVectorProbability(trained.Ml, trained.Model, spec, test);

        var labels = test.Select(r => r.Label ? 1.0 : 0.0).ToArray();
        var blendBrier = Brier(probs, labels);

        // Climatology baseline = train-set wet rate, applied as a constant
        // probability on every test row. Brier Skill Score floor.
        var clim = train.Count(r => r.Label) / (double)train.Count;
        var climBrier = Brier(Enumerable.Repeat(clim, test.Count).ToArray(), labels);
        var blendBss = (climBrier - blendBrier) / climBrier;

        // Per-model "raw" Brier — convert each model's precip column to a
        // 0/1 wet prediction at the same threshold and score. Crude but
        // matches what 3a's "best single" baseline does.
        var perModelBrier = new Dictionary<string, double>();
        for (int m = 0; m < spec.Models.Count; m++)
        {
            var col = m;
            var preds = test.Select(r =>
            {
                var v = r.Features[col];
                return float.IsNaN(v) ? clim : (v >= PrecipExactFeatureBuilder.WetThresholdMm ? 1.0 : 0.0);
            }).ToArray();
            perModelBrier[spec.Models[m]] = Brier(preds, labels);
        }
        // Mean-of-models 0/1 vote: average the 0/1 wet calls across present-only models per row.
        var meanPreds = test.Select(r =>
        {
            double sum = 0; int n = 0;
            for (int m = 0; m < spec.Models.Count; m++)
            {
                var v = r.Features[m];
                if (float.IsNaN(v)) continue;
                sum += v >= PrecipExactFeatureBuilder.WetThresholdMm ? 1.0 : 0.0;
                n++;
            }
            return n == 0 ? clim : sum / n;
        }).ToArray();
        var meanBrier = Brier(meanPreds, labels);

        var wetN = test.Count(r => r.Label);

        Console.WriteLine();
        Console.WriteLine($"=== Tier {tier.Name} target {targetLead}h ===");
        Console.WriteLine($"  description: {tier.Description}");
        Console.WriteLine($"  rows: train={train.Count} val={val.Count} test={test.Count}");
        Console.WriteLine($"  test period: {testStart:yyyy-MM-dd}..{maxValid:yyyy-MM-dd}");
        Console.WriteLine($"  test wet rate: {(double)wetN / test.Count:P1} ({wetN}/{test.Count})");
        Console.WriteLine($"  train wet rate (climatology): {clim:P1}");
        Console.WriteLine();
        Console.WriteLine($"  Test Brier (lower=better) — wet threshold ≥ 0.1 mm/h:");
        Console.WriteLine($"    → BLENDER (LightGBM)             {blendBrier:F4}   (BSS vs clim: {blendBss:+0.000;-0.000;0.000})");
        Console.WriteLine($"      mean-of-models                  {meanBrier:F4}");
        Console.WriteLine($"      climatology (constant {clim:P0})  {climBrier:F4}");
        foreach (var kv in perModelBrier.OrderBy(kv => kv.Value))
            Console.WriteLine($"      single (0/1 vote): {kv.Key,-22} {kv.Value:F4}");

        var bestSingle = perModelBrier.Values.Where(v => !double.IsNaN(v)).DefaultIfEmpty(double.NaN).Min();
        if (!double.IsNaN(bestSingle))
        {
            var d = blendBrier - bestSingle;
            var pct = d / bestSingle * 100;
            Console.WriteLine();
            Console.WriteLine($"  Δ vs best single (0/1):  {d:+0.0000;-0.0000;0.0000} ({pct:+0.0;-0.0;0.0}%)");
        }
    }

    private static double Brier(IReadOnlyList<double> probs, IReadOnlyList<double> labels)
    {
        if (probs.Count != labels.Count) throw new ArgumentException("length mismatch");
        double sumSq = 0;
        for (int i = 0; i < probs.Count; i++)
        {
            var d = probs[i] - labels[i];
            sumSq += d * d;
        }
        return sumSq / probs.Count;
    }
}
