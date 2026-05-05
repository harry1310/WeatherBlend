using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.PrecipExact;

namespace WeatherBlend.Commands;

/// <summary>
/// First-cut precipitation bake-off using the exact-runtime sources in the
/// 2d temperature blender (minus AIFS — its precip parser has a units bug,
/// see <see cref="PrecipExactFeatureBuilder"/> docstring). Single tier P1
/// (GFS + IFS required, MO Global optional). Single-lead 12h or 24h.
///
/// Caveat: per-model precip semantics differ (GFS = APCP interval, IFS = tp
/// cumulative, MO Global = instantaneous rate). LightGBM handles each as a
/// separate column so this is workable for blending, but the absolute MAE
/// won't be as clean as temperature's. Treat the first cut as
/// directional — does the blender lift over best-single? Drop production
/// 3a (Brier-based wet/dry) as the comparison since the metric differs;
/// this is a side-by-side of "how does an exact-runtime precip blender
/// perform vs its inputs" only.
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

    public async Task<int> RunAsync(int targetLead, CancellationToken ct)
    {
        foreach (var tier in PrecipExactFeatureBuilder.AllTiers)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Run(() => RunTier(tier, targetLead, ct), ct);
        }
        return 0;
    }

    private void RunTier(PrecipExactFeatureBuilder.TierSpec tier, int targetLead, CancellationToken ct)
    {
        _log.LogInformation("=== Precip exact bake-off: tier {Tier} target {Lead}h ===", tier.Name, targetLead);
        var spec = PrecipExactFeatureBuilder.BuildSpec(tier, targetLead);
        var rows = PrecipExactFeatureBuilder.Build(
            _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, _cfg.Location.Name,
            tier, spec, targetLead, ct);

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

        // Use the same TempTrainer (LightGBM regressor). Default HPs to start;
        // we can HP-tune later if there's signal.
        var hp = new TempTrainer.Hyperparameters();
        var trained = TempTrainer.TrainVector(train, val, spec, hp);
        var preds = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, test);
        var blendMae = MeanAbs(test.Select((r, i) => preds[i] - r.Label));

        // Per-model + mean-of-models baselines on canonical-lead values.
        var perModelMae = new Dictionary<string, double>();
        for (int m = 0; m < spec.Models.Count; m++)
        {
            var col = m;
            var diffs = test.Select(r => (pred: r.Features[col], label: r.Label))
                .Where(t => !float.IsNaN(t.pred))
                .Select(t => (double)(t.pred - t.label))
                .ToList();
            perModelMae[spec.Models[m]] = diffs.Count == 0 ? double.NaN : diffs.Select(Math.Abs).Average();
        }
        var meanMae = MeanAbs(test.Select(r =>
        {
            double sum = 0; int n = 0;
            for (int m = 0; m < spec.Models.Count; m++)
            {
                var v = r.Features[m];
                if (!float.IsNaN(v)) { sum += v; n++; }
            }
            return (n == 0 ? double.NaN : sum / n) - r.Label;
        }));

        // Wet/dry context: how often is ERA5 truth ≥0.1 mm/h (the
        // production 3a wet threshold)? Helps interpret MAE — most rows
        // are dry-dry which is easy.
        var truthMean = (double)test.Average(r => r.Label);
        var wetN = test.Count(r => r.Label >= 0.1);
        var wetFrac = (double)wetN / Math.Max(1, test.Count);

        Console.WriteLine();
        Console.WriteLine($"=== Tier {tier.Name} target {targetLead}h ===");
        Console.WriteLine($"  description: {tier.Description}");
        Console.WriteLine($"  rows: train={train.Count} val={val.Count} test={test.Count}");
        Console.WriteLine($"  test period: {testStart:yyyy-MM-dd}..{maxValid:yyyy-MM-dd}");
        Console.WriteLine($"  test wet fraction (≥0.1mm): {wetFrac:P1} ({wetN}/{test.Count})");
        Console.WriteLine($"  test ERA5 mean precip: {truthMean:F3}");
        Console.WriteLine();
        Console.WriteLine($"  Test MAE (mm, lower=better):");
        Console.WriteLine($"    → BLENDER (LightGBM)             {blendMae:F3}");
        Console.WriteLine($"      mean-of-models                  {meanMae:F3}");
        foreach (var kv in perModelMae.OrderBy(kv => kv.Value))
            Console.WriteLine($"      single: {kv.Key,-25} {kv.Value:F3}");
        var bestSingle = perModelMae.Values.Where(v => !double.IsNaN(v)).DefaultIfEmpty(double.NaN).Min();
        if (!double.IsNaN(bestSingle))
        {
            var d = blendMae - bestSingle;
            var pct = (d / bestSingle) * 100.0;
            Console.WriteLine();
            Console.WriteLine($"  Δ vs best single model: {d:+0.000;-0.000;0.000} mm  ({pct:+0.0;-0.0;0.0}%)");
            var dm = blendMae - meanMae;
            var pctm = (dm / meanMae) * 100.0;
            Console.WriteLine($"  Δ vs mean-of-models   : {dm:+0.000;-0.000;0.000} mm  ({pctm:+0.0;-0.0;0.0}%)");
        }
    }

    private static double MeanAbs(IEnumerable<double> diffs)
    {
        double sum = 0; int n = 0;
        foreach (var d in diffs)
        {
            if (double.IsNaN(d)) continue;
            sum += Math.Abs(d); n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }
}
