using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Exact12h;

namespace WeatherBlend.Commands;

/// <summary>
/// Bake-off for the exact-runtime 12h temperature blender (the experiment
/// scoped 2026-05-05). For each tier (T1 strict / T2 GFS+AIFS-required /
/// T3 GFS-only-required):
///
///   1. Build training rows via <see cref="Exact12hFeatureBuilder"/>.
///   2. Time-split: last <see cref="HoldoutDays"/> days are TEST, rest TRAIN.
///      Within TRAIN we further hold the last <see cref="ValidationDays"/>
///      as a validation set so LightGBM has a target for early-stopping.
///   3. Fit LightGBM via <see cref="TempTrainer.TrainVector"/> at the
///      production hyperparameter defaults (no per-tier tuning — first
///      pass is "is the signal there at all").
///   4. Score on TEST:
///        - blender MAE
///        - per-model MAE (single-model baseline for each input)
///        - mean-of-models MAE (naive ensemble baseline)
///        - constant-mean MAE (always predict the train-set ERA5 mean —
///          the floor anything has to beat to be called "skill")
///
/// No artifact persistence — this is exploratory. If a tier looks strong
/// we'll spec out the production wiring (artifact save, predict command,
/// verify wiring) as a separate workstream.
/// </summary>
public sealed class Exact12hBakeoffCommand
{
    /// <summary>Held-out test window (days). 90 = roughly the last
    /// quarter, gives a meaningful test set even for the smallest tier.</summary>
    public const int HoldoutDays = 90;

    /// <summary>Validation window inside TRAIN (days), used by LightGBM for
    /// early stopping. Sits BEFORE the test window so train/val/test are
    /// strictly ordered in time — no leakage.</summary>
    public const int ValidationDays = 30;

    private readonly AppConfig _cfg;
    private readonly ILogger<Exact12hBakeoffCommand> _log;

    public Exact12hBakeoffCommand(AppConfig cfg, ILogger<Exact12hBakeoffCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public async Task<int> RunAsync(
        string? tierName,
        int targetLead,
        IReadOnlyList<int>? inputLeads,
        bool maskNoncanonicalAtTest,
        CancellationToken ct)
    {
        var tiers = string.IsNullOrWhiteSpace(tierName)
            ? Exact12hFeatureBuilder.AllTiers
            : new[] { Exact12hFeatureBuilder.GetTier(tierName) };

        var results = new List<TierResult>();
        foreach (var tier in tiers)
        {
            ct.ThrowIfCancellationRequested();
            results.Add(await Task.Run(() => RunTier(tier, targetLead, inputLeads, maskNoncanonicalAtTest, ct), ct));
        }

        WriteReport(results, targetLead, inputLeads, maskNoncanonicalAtTest);
        return 0;
    }

    private TierResult RunTier(
        Exact12hFeatureBuilder.TierSpec tier,
        int targetLead,
        IReadOnlyList<int>? inputLeads,
        bool maskNoncanonicalAtTest,
        CancellationToken ct)
    {
        inputLeads ??= new[] { targetLead };
        _log.LogInformation("=== Tier {Tier}: {Desc} ===", tier.Name, tier.Description);
        var spec = Exact12hFeatureBuilder.BuildSpec(tier, targetLead, inputLeads);
        _log.LogInformation("  spec: required=[{R}] optional=[{O}] target={Tgt}h leads=[{Leads}] features={F}",
            string.Join(",", spec.RequiredModels),
            string.Join(",", spec.OptionalModels),
            targetLead,
            string.Join(",", inputLeads),
            spec.FeatureCount);

        var rows = Exact12hFeatureBuilder.Build(
            _cfg.Storage.ForecastsPath,
            _cfg.Storage.Era5Path,
            _cfg.Location.Name,
            tier,
            spec,
            targetLead,
            inputLeads,
            ct);

        if (rows.Count == 0)
        {
            _log.LogWarning("  Tier {Tier}: 0 rows — skipping.", tier.Name);
            return new TierResult(tier, spec, rows.Count, 0, 0, 0, double.NaN, new Dictionary<string, double>(), double.NaN, double.NaN, default, default);
        }

        // Time-split: TEST = last HoldoutDays, VAL = HoldoutDays days before that, TRAIN = everything earlier.
        var sorted = rows.OrderBy(r => r.ValidTimeUtc).ToList();
        var maxValid = sorted[^1].ValidTimeUtc;
        var testStart = maxValid.AddDays(-HoldoutDays);
        var valStart  = testStart.AddDays(-ValidationDays);

        var train = sorted.Where(r => r.ValidTimeUtc < valStart).ToList();
        var val   = sorted.Where(r => r.ValidTimeUtc >= valStart && r.ValidTimeUtc < testStart).ToList();
        var test  = sorted.Where(r => r.ValidTimeUtc >= testStart).ToList();

        _log.LogInformation("  rows: total={N} train={Tr} val={V} test={Te}  (test from {TS:yyyy-MM-dd})",
            rows.Count, train.Count, val.Count, test.Count, testStart);

        if (train.Count == 0 || val.Count == 0 || test.Count == 0)
        {
            _log.LogWarning("  Tier {Tier}: empty split (train/val/test = {Tr}/{V}/{Te}) — skipping training.",
                tier.Name, train.Count, val.Count, test.Count);
            return new TierResult(tier, spec, rows.Count, train.Count, val.Count, test.Count,
                double.NaN, new Dictionary<string, double>(), double.NaN, double.NaN,
                sorted[0].ValidTimeUtc, maxValid);
        }

        // Train. Production defaults — no tuning on first pass.
        var hp = new TempTrainer.Hyperparameters();
        var trained = TempTrainer.TrainVector(train, val, spec, hp);

        // Optional: mask non-canonical-lead columns to NaN in TEST rows
        // before scoring. Tests "did multi-lead training help the blender's
        // canonical-lead use, or was it just leaning on lead-6 leakage?".
        // Lead-6 in our setup is the forecast made at ValidTime - 6h, which
        // for a 12h-ahead prediction is in the future from the prediction
        // time — leakage if used at inference. NaN-masking at test time
        // forces the model to rely only on its trained canonical-lead path,
        // exposing whether multi-lead training transferred any honest gain.
        var testForScoring = test;
        if (maskNoncanonicalAtTest && inputLeads.Count > 1)
        {
            var leadsSortedLocal = inputLeads.OrderBy(l => l).ToList();
            var canonicalIdxLocal = leadsSortedLocal.IndexOf(targetLead);
            var nLeadsLocal = leadsSortedLocal.Count;
            testForScoring = test.Select(r =>
            {
                var masked = (float[])r.Features.Clone();
                for (int m = 0; m < spec.Models.Count; m++)
                    for (int l = 0; l < nLeadsLocal; l++)
                        if (l != canonicalIdxLocal)
                            masked[m * nLeadsLocal + l] = float.NaN;
                return new RegressionTrainingRow
                {
                    ValidTimeUtc = r.ValidTimeUtc,
                    Features = masked,
                    Label = r.Label,
                    WindDirMean = r.WindDirMean,
                };
            }).ToList();
            _log.LogInformation("  test-time mask: non-canonical leads (≠{Tgt}h) zeroed to NaN in {N} test rows.",
                targetLead, testForScoring.Count);
        }

        // Score on (possibly masked) TEST.
        var preds = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, testForScoring);
        var blendMae = MeanAbs(testForScoring.Select((r, i) => preds[i] - r.Label));

        // Per-model baseline: each model's CANONICAL-LEAD column vs ERA5
        // on the same TEST rows. In multi-lead mode the column vector
        // contains one slot per (model, lead); here we want the target-lead
        // slot for each model so the baseline means "what would the single
        // model at this lead predict?". Layout is model-major, lead-minor —
        // so model m at canonical lead lives at index `m * nLeads + canonicalIdx`.
        var leadsSorted = inputLeads.OrderBy(l => l).ToList();
        var nLeads = leadsSorted.Count;
        var canonicalIdx = leadsSorted.IndexOf(targetLead);
        // Per-model + mean-of-models baselines use the canonical-lead column
        // from the original (un-masked) test rows — they're "what would
        // single-model lead-12 give you" baselines, independent of whatever
        // multi-lead masking the blender went through. This keeps the
        // comparison interpretable: the BLENDER number reflects the masking
        // choice, but baselines stay constant across modes.
        var perModelMae = new Dictionary<string, double>(StringComparer.Ordinal);
        for (int m = 0; m < spec.Models.Count; m++)
        {
            var col = m * nLeads + canonicalIdx;
            var diffs = test
                .Select(r => (pred: r.Features[col], label: r.Label))
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
                var v = r.Features[m * nLeads + canonicalIdx];
                if (!float.IsNaN(v)) { sum += v; n++; }
            }
            var mean = n == 0 ? double.NaN : sum / n;
            return mean - r.Label;
        }));

        // Constant-mean baseline: always predict the TRAIN ERA5 mean. Floor
        // for "is the model doing anything beyond memorising the climate?".
        var trainMean = (double)train.Average(r => r.Label);
        var constMae = MeanAbs(test.Select(r => trainMean - (double)r.Label));

        return new TierResult(tier, spec, rows.Count, train.Count, val.Count, test.Count,
            blendMae, perModelMae, meanMae, constMae, sorted[0].ValidTimeUtc, maxValid);
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

    private void WriteReport(
        IReadOnlyList<TierResult> results,
        int targetLead,
        IReadOnlyList<int>? inputLeads,
        bool maskNoncanonicalAtTest)
    {
        inputLeads ??= new[] { targetLead };
        var leadsSorted = inputLeads.OrderBy(l => l).ToList();
        var leadDesc = leadsSorted.Count == 1
            ? $"single lead {targetLead}h"
            : (maskNoncanonicalAtTest
                ? $"target {targetLead}h, train on leads [{string.Join(",", leadsSorted)}], TEST ON CANONICAL ONLY (non-{targetLead}h cols masked)"
                : $"target {targetLead}h, input leads [{string.Join(",", leadsSorted)}]");

        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("=========================================================");
        sb.AppendLine($"  EXACT-RUNTIME TEMPERATURE BLENDER — bake-off report");
        sb.AppendLine($"  {leadDesc}");
        sb.AppendLine("=========================================================");
        sb.AppendLine();

        foreach (var r in results)
        {
            sb.AppendLine($"## Tier {r.Tier.Name}");
            sb.AppendLine($"   {r.Tier.Description}");
            sb.AppendLine($"   models in column vector : {string.Join(", ", r.Spec.Models)}");
            sb.AppendLine($"   required                : {string.Join(", ", r.Spec.RequiredModels)}");
            sb.AppendLine($"   optional                : {(r.Spec.OptionalModels.Count == 0 ? "(none)" : string.Join(", ", r.Spec.OptionalModels))}");
            sb.AppendLine($"   rows: total={r.Total}  train={r.Train}  val={r.Val}  test={r.Test}");
            sb.AppendLine($"   date span: {r.FirstValid:yyyy-MM-dd} .. {r.LastValid:yyyy-MM-dd}");
            sb.AppendLine();

            if (double.IsNaN(r.BlendMae))
            {
                sb.AppendLine("   (insufficient data — see warnings above)");
                sb.AppendLine();
                continue;
            }

            // Sort baselines for readability — best (lowest MAE) first.
            var rows = new List<(string label, double mae)>
            {
                ("BLENDER (LightGBM)", r.BlendMae),
                ("mean-of-models", r.MeanMae),
                ("climatology (train mean)", r.ConstantMeanMae),
            };
            foreach (var kv in r.PerModelMae)
                rows.Add(($"single: {kv.Key}", kv.Value));
            var ordered = rows.OrderBy(t => double.IsNaN(t.mae) ? double.MaxValue : t.mae).ToList();

            sb.AppendLine("   Test-set MAE (°C, lower=better):");
            foreach (var (label, mae) in ordered)
            {
                var marker = label.StartsWith("BLENDER", StringComparison.Ordinal) ? "→" : " ";
                sb.AppendLine($"     {marker} {label,-30} {(double.IsNaN(mae) ? "—" : mae.ToString("F3", CultureInfo.InvariantCulture))}");
            }

            // Headline gain vs the best single-model baseline (this is the
            // "did the blend earn its keep?" number — beating mean-of-models
            // alone is a much weaker bar).
            var bestSingle = r.PerModelMae.Values.Where(v => !double.IsNaN(v)).DefaultIfEmpty(double.NaN).Min();
            if (!double.IsNaN(bestSingle) && !double.IsNaN(r.BlendMae))
            {
                var deltaSingle = r.BlendMae - bestSingle;
                var pct = (deltaSingle / bestSingle) * 100.0;
                sb.AppendLine();
                sb.AppendLine($"   Δ vs best single model    : {deltaSingle:+0.000;-0.000;0.000}°C   ({pct:+0.0;-0.0;0.0}%)");
            }
            if (!double.IsNaN(r.MeanMae) && !double.IsNaN(r.BlendMae))
            {
                var deltaMean = r.BlendMae - r.MeanMae;
                var pctMean = (deltaMean / r.MeanMae) * 100.0;
                sb.AppendLine($"   Δ vs mean-of-models       : {deltaMean:+0.000;-0.000;0.000}°C   ({pctMean:+0.0;-0.0;0.0}%)");
            }
            sb.AppendLine();
        }

        // Cross-tier summary table.
        sb.AppendLine("---");
        sb.AppendLine("Cross-tier summary (blender MAE vs best single model on each tier):");
        sb.AppendLine("    tier | rows | train/val/test | blend MAE | best single | mean-of | climatology");
        sb.AppendLine("    -----|------|----------------|-----------|-------------|---------|------------");
        foreach (var r in results)
        {
            var bestSingle = r.PerModelMae.Values.Where(v => !double.IsNaN(v)).DefaultIfEmpty(double.NaN).Min();
            sb.AppendLine($"    {r.Tier.Name,4} | {r.Total,4} | {r.Train,5}/{r.Val,3}/{r.Test,4} | {Fmt(r.BlendMae)} | {Fmt(bestSingle)} | {Fmt(r.MeanMae)} | {Fmt(r.ConstantMeanMae)}");
        }
        sb.AppendLine();

        var report = sb.ToString();
        Console.Write(report);

        // Tag the filename with the lead config so single-lead, multi-lead,
        // and masked-multi-lead runs sit side-by-side without overwriting.
        var maskTag = maskNoncanonicalAtTest && leadsSorted.Count > 1 ? "_maskedtest" : "";
        var leadTag = leadsSorted.Count == 1
            ? $"l{targetLead:00}"
            : "leads" + string.Concat(leadsSorted.Select(l => l.ToString("00"))) + maskTag;
        var outDir = Path.Combine(_cfg.Storage.ReportsPath, "exact_12h_bakeoff");
        Directory.CreateDirectory(outDir);
        var outFile = Path.Combine(outDir, $"bakeoff_{leadTag}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt");
        File.WriteAllText(outFile, report);
        _log.LogInformation("Report written to {Path}", outFile);
    }

    private static string Fmt(double v) =>
        double.IsNaN(v) ? "    —    " : v.ToString("F3", CultureInfo.InvariantCulture).PadLeft(9);

    private sealed record TierResult(
        Exact12hFeatureBuilder.TierSpec Tier,
        BlenderSpec Spec,
        int Total, int Train, int Val, int Test,
        double BlendMae,
        IReadOnlyDictionary<string, double> PerModelMae,
        double MeanMae,
        double ConstantMeanMae,
        DateTime FirstValid,
        DateTime LastValid);
}
