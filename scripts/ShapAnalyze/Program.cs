// SHAP-style feature contribution + per-NWP MAE breakdown for the
// Element blenders (cloud-cover, shortwave-radiation). One-shot
// post-hoc analysis — no new production code, no retraining, no
// manifest changes. Writes a markdown report to data/reports/.
//
// Two diagnostics per (target, lead) cell:
//
//   1. SHAP — ML.NET's FeatureContributionCalculatingEstimator
//      (TreeSHAP-approximate, raw-score space). Identifies dead-weight
//      features, dominant features, and bidirectional correctors. The
//      goal here is to figure out WHY the cloud + radiation blenders
//      are no better than best-single-NWP — typically that's a sign of
//      one input dominating (so the blender just learns "use NWP X")
//      or all inputs being highly correlated (so blending adds noise).
//
//   2. Per-NWP MAE breakdown — for each per-NWP feature column in
//      spec, compute the test-slice MAE of using THAT NWP's raw
//      forecast as the prediction. Compare to the blender's test MAE
//      (BlendTestMae) and to BestSingleTestMae from training metadata.
//      Reveals whether the blender is genuinely close to the best
//      individual NWP or if there's still headroom on offer.
//
// Output: data/reports/element_shap_{date}.md with one section per
// (target, lead) cell.

using System.Globalization;
using System.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using NetEscapades.Configuration.Yaml;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Radiation;

namespace WeatherBlend.Scripts.ElementShap;

internal static class Program
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;
    private const string LocationName = "bonehill_rocks";
    private static readonly string RepoRoot = FindRepoRoot();
    private static readonly string ForecastsPath = Path.Combine(RepoRoot, "data", "forecasts");
    private static readonly string Era5Path      = Path.Combine(RepoRoot, "data", "truth", "era5");
    private static readonly string ReportsPath   = Path.Combine(RepoRoot, "data", "reports");
    private static readonly string ModelsRoot    = Path.Combine(RepoRoot, "data", "models");

    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly DateTime Today = DateTime.UtcNow.Date;

    public static int Main(string[] args)
    {
        Console.WriteLine($"[element-shap] repo root: {RepoRoot}");
        Directory.CreateDirectory(ReportsPath);

        var configPath = Path.Combine(RepoRoot, "src", "WeatherBlend", "config.yaml");
        var configRoot = new ConfigurationBuilder()
            .AddYamlFile(configPath, optional: false, reloadOnChange: false)
            .Build();
        var appCfg = new WeatherBlend.Config.AppConfig();
        configRoot.Bind(appCfg);
        var blendersCfg = appCfg.Blenders;

        var sections = new List<TargetReport>();
        sections.Add(AnalyseTarget(blendersCfg, ElementTargets.CloudCover,
            (cfg, lead) => CloudFeatureBuilder.BuildSpec(cfg, lead),
            (forecastsPath, era5Path, locationName, spec, ct) =>
                CloudFeatureBuilder.BuildForLead(forecastsPath, era5Path, locationName, spec, ct)));
        sections.Add(AnalyseTarget(blendersCfg, ElementTargets.ShortwaveRadiation,
            (cfg, lead) => RadiationFeatureBuilder.BuildSpec(cfg, lead),
            (forecastsPath, era5Path, locationName, spec, ct) =>
                RadiationFeatureBuilder.BuildForLead(forecastsPath, era5Path, locationName, spec, ct)));

        var md = BuildReport(sections);
        var outPath = Path.Combine(ReportsPath, $"element_shap_{Today:yyyy-MM-dd}.md");
        File.WriteAllText(outPath, md);
        Console.WriteLine($"[element-shap] wrote {outPath}");
        return 0;
    }

    // ---- Per-target analysis ------------------------------------------------

    private static TargetReport AnalyseTarget(
        WeatherBlend.Config.BlendersConfig blendersCfg,
        ElementTarget target,
        Func<WeatherBlend.Config.BlendersConfig, int, BlenderSpec> buildSpec,
        Func<string, string, string, BlenderSpec, CancellationToken, List<RegressionTrainingRow>> buildForLead)
    {
        var version = ResolveLatestVersion(target.ModelDirName);
        Console.WriteLine($"\n=== Element target {target.CliName} — version {version} ===");
        var versionDir = Path.Combine(ModelsRoot, target.ModelDirName, version);

        var leadResults = new List<LeadReport>();
        foreach (var lead in Leads)
        {
            Console.WriteLine($"--- {target.CliName} lead {lead}h ---");
            var spec = buildSpec(blendersCfg, lead);
            var rows = buildForLead(ForecastsPath, Era5Path, LocationName, spec, CancellationToken.None);
            var ds = RegressionDataset.Split(rows);
            Console.WriteLine($"  loaded {rows.Count} rows; test slice {ds.Test.Count} rows ({ds.TestStart:yyyy-MM-dd}..{ds.TestEnd:yyyy-MM-dd})");

            var ml = new MLContext(seed: 42);
            var loaded = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);
            var contributions = ComputeRegressionContributions(ml, loaded, spec, ds.Test);
            Console.WriteLine($"  computed contributions for {contributions.Length} test rows × {spec.FeatureCount} features");

            // Per-NWP MAE: each per-model feature lives at spec.FeatureNames[i]
            // for i < spec.Models.Count. Score that column directly against truth.
            var perNwpMae = new List<(string Model, string Feature, double Mae, double Bias)>();
            var truth = ds.Test.Select(r => (double)r.Label).ToArray();
            for (int i = 0; i < spec.Models.Count; i++)
            {
                var modelId = spec.Models[i];
                var featureName = spec.FeatureNames[i];
                var featureIdx = spec.IndexOf(featureName);
                var preds = ds.Test.Select(r => (double)r.Features[featureIdx]).ToArray();
                var (mae, bias, n) = MaeBias(preds, truth);
                perNwpMae.Add((modelId, featureName, mae, bias));
                Console.WriteLine($"    {modelId,-22} ({featureName,-30}) MAE={mae:0.0000} bias={bias:+0.000;-0.000;0.000} n={n}");
            }

            // Blend MAE on test — recompute fresh rather than rely on metadata
            // so it lines up exactly with the SHAP test slice.
            var blendPreds = TempTrainer.PredictVector(ml, loaded, spec, ds.Test);
            var (blendMae, blendBias, _) = MaeBias(blendPreds, truth);
            Console.WriteLine($"    BLEND                                         MAE={blendMae:0.0000} bias={blendBias:+0.000;-0.000;0.000}");

            leadResults.Add(new LeadReport(lead, spec, contributions, perNwpMae,
                blendMae, blendBias, ds.Test.Count, target.Units));
        }

        return new TargetReport(target, version, versionDir, leadResults);
    }

    // ---- SHAP via ML.NET FeatureContribution (TreeSHAP-approx) -------------

    private static float[][] ComputeRegressionContributions(
        MLContext ml, ITransformer loaded, BlenderSpec spec, IReadOnlyList<RegressionTrainingRow> testRows)
    {
        // Element + dry-window blenders save a bare RegressionPredictionTransformer
        // (TempTrainer.TrainVector — features arrive pre-vectorised on the row, no
        // featurizer chain needed). Older RichFeatureBuilder models in the 2c/3c
        // path saved a chain; handle both for compatibility.
        var predictor = ResolveRegressionPredictor(loaded);

        var schemaDef = SchemaDefinition.Create(typeof(RegressionTrainingRow));
        schemaDef[nameof(RegressionTrainingRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, spec.FeatureCount);
        var rawDv = ml.Data.LoadFromEnumerable(testRows, schemaDef);

        var contribEst = ml.Transforms.CalculateFeatureContribution(
            predictor,
            numberOfPositiveContributions: spec.FeatureCount,
            numberOfNegativeContributions: spec.FeatureCount,
            normalize: false);
        var fitted = contribEst.Fit(rawDv);
        var withContrib = fitted.Transform(rawDv);
        return ReadContributionMatrix(withContrib, spec.FeatureCount);
    }

    private static RegressionPredictionTransformer<LightGbmRegressionModelParameters>
        ResolveRegressionPredictor(ITransformer loaded)
    {
        if (loaded is RegressionPredictionTransformer<LightGbmRegressionModelParameters> bare)
            return bare;
        if (loaded is IEnumerable<ITransformer> chain)
        {
            var items = chain.ToArray();
            if (items.Length > 0
                && items[^1] is RegressionPredictionTransformer<LightGbmRegressionModelParameters> tail)
                return tail;
        }
        throw new InvalidOperationException(
            $"Loaded model is not a RegressionPredictionTransformer<LightGbmRegressionModelParameters>: {loaded.GetType().FullName}");
    }

    private static float[][] ReadContributionMatrix(IDataView dv, int featureCount)
    {
        var col = dv.Schema["FeatureContributions"];
        var result = new List<float[]>();
        using var cursor = dv.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        VBuffer<float> buf = default;
        while (cursor.MoveNext())
        {
            getter(ref buf);
            var dense = buf.DenseValues().ToArray();
            var arr = new float[featureCount];
            for (int i = 0; i < Math.Min(dense.Length, featureCount); i++) arr[i] = dense[i];
            result.Add(arr);
        }
        return result.ToArray();
    }

    // ---- MAE/bias helpers --------------------------------------------------

    /// MAE + signed mean error over rows where both pred and truth are finite
    /// (drops rows with NaN in either — happens when an NWP didn't ship a value
    /// for that hour and the row was kept on a different NWP's signal).
    private static (double Mae, double Bias, int N) MaeBias(double[] preds, double[] truth)
    {
        if (preds.Length != truth.Length) throw new InvalidOperationException("length mismatch");
        double sumAbs = 0, sumSigned = 0;
        int n = 0;
        for (int i = 0; i < preds.Length; i++)
        {
            if (double.IsNaN(preds[i]) || double.IsNaN(truth[i])) continue;
            sumAbs += Math.Abs(preds[i] - truth[i]);
            sumSigned += preds[i] - truth[i];
            n++;
        }
        if (n == 0) return (double.NaN, double.NaN, 0);
        return (sumAbs / n, sumSigned / n, n);
    }

    // ---- Version + path discovery ------------------------------------------

    private static string ResolveLatestVersion(string modelDirName)
    {
        var dir = Path.Combine(ModelsRoot, modelDirName);
        if (!Directory.Exists(dir))
            throw new InvalidOperationException($"No model dir at {dir}");
        var versions = Directory.GetDirectories(dir)
            .Select(Path.GetFileName)
            .Where(n => n is not null && !n.Contains("_phase"))
            .OrderByDescending(n => n!, StringComparer.Ordinal)
            .ToList();
        if (versions.Count == 0)
            throw new InvalidOperationException($"No suitable versions under {dir}");
        return versions[0]!;
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir) && !File.Exists(Path.Combine(dir, "WeatherBlend.slnx"))
               && !File.Exists(Path.Combine(dir, "WeatherBlend.sln")))
            dir = Path.GetDirectoryName(dir)!;
        if (string.IsNullOrEmpty(dir)) throw new InvalidOperationException("Could not locate repo root.");
        return dir!;
    }

    // ---- Report aggregation ------------------------------------------------

    private sealed record LeadReport(
        int LeadHours,
        BlenderSpec Spec,
        float[][] Contributions,
        IReadOnlyList<(string Model, string Feature, double Mae, double Bias)> PerNwpMae,
        double BlendMae,
        double BlendBias,
        int TestRows,
        string Units);

    private sealed record TargetReport(
        ElementTarget Target,
        string Version,
        string VersionDir,
        IReadOnlyList<LeadReport> Leads);

    private static string BuildReport(IReadOnlyList<TargetReport> targets)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Element blender SHAP + per-NWP audit");
        sb.AppendLine();
        sb.AppendLine($"- Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm}Z");
        sb.AppendLine("- Method: ML.NET `CalculateFeatureContribution` (TreeSHAP-approx, raw-score space).");
        sb.AppendLine("- Test slice per lead: chronological 15% tail (matches training).");
        sb.AppendLine();
        sb.AppendLine("**How to read.** Per-NWP MAE compares each input NWP's raw forecast against truth on the test slice.");
        sb.AppendLine("The blend's MAE is shown alongside — when blend MAE doesn't beat best-single-NWP MAE by much,");
        sb.AppendLine("the blender is essentially passing through the dominant NWP. SHAP shows where the model spends");
        sb.AppendLine("its weight: a single per-NWP feature dominating with low std means \"blender = picker\";");
        sb.AppendLine("balanced contributions across NWPs means real blending is happening.");
        sb.AppendLine();

        foreach (var t in targets)
        {
            sb.AppendLine(Ci, $"## {t.Target.CliName} (units: {t.Target.Units})");
            sb.AppendLine();
            sb.AppendLine(Ci, $"- Model version: `{t.Version}`");
            sb.AppendLine();

            foreach (var l in t.Leads)
            {
                sb.AppendLine(Ci, $"### Lead {l.LeadHours}h ({l.TestRows} test rows)");
                sb.AppendLine();

                sb.AppendLine("**Per-NWP test MAE vs blend (lower is better)**");
                sb.AppendLine();
                sb.AppendLine("| NWP | Feature | MAE | Bias | Δ vs blend |");
                sb.AppendLine("|---|---|---:|---:|---:|");
                var blendMae = l.BlendMae;
                foreach (var (model, feature, mae, bias) in l.PerNwpMae.OrderBy(x => x.Mae))
                {
                    var delta = mae - blendMae;
                    sb.AppendLine(Ci,
                        $"| {model} | `{feature}` | {mae:0.000} | {bias:+0.000;-0.000;0.000} | {delta:+0.000;-0.000;0.000} |");
                }
                sb.AppendLine(Ci, $"| **BLEND** | — | **{blendMae:0.000}** | {l.BlendBias:+0.000;-0.000;0.000} | — |");
                sb.AppendLine();

                sb.AppendLine("**Top features by mean|SHAP|**");
                sb.AppendLine();
                sb.AppendLine("| Rank | Feature | mean\\|SHAP\\| | mean SHAP | std SHAP | non-zero % |");
                sb.AppendLine("|---|---|---:|---:|---:|---:|");

                var stats = ComputeFeatureStats(l.Spec.FeatureNames.ToArray(), l.Contributions);
                foreach (var s in stats.Take(20))
                {
                    sb.AppendLine(Ci,
                        $"| {s.Rank} | `{s.Name}` | {s.MeanAbs:0.0000} | {s.Mean:+0.0000;-0.0000;0.0000} | {s.Std:0.0000} | {s.NonzeroPct:0.0}% |");
                }

                var dead = stats.Where(s => s.MeanAbs < 0.001 && s.Std < 0.001).ToList();
                if (dead.Count > 0)
                {
                    sb.AppendLine();
                    sb.AppendLine(Ci, $"**{dead.Count} dead-weight features** (mean|SHAP| < 0.001 AND std < 0.001):");
                    foreach (var d in dead.Take(10)) sb.AppendLine(Ci, $"- `{d.Name}`");
                    if (dead.Count > 10) sb.AppendLine(Ci, $"- ...+{dead.Count - 10} more");
                }
                sb.AppendLine();
            }
        }

        return sb.ToString();
    }

    private record FeatureStat(int Rank, string Name, double MeanAbs, double Mean, double Std, double NonzeroPct);

    private static IReadOnlyList<FeatureStat> ComputeFeatureStats(string[] names, float[][] contributions)
    {
        var rows = contributions.Length;
        var feats = names.Length;
        var raw = new List<(string Name, double MeanAbs, double Mean, double Std, double Nonzero)>(feats);
        for (int j = 0; j < feats; j++)
        {
            double sumAbs = 0, sum = 0, nonzero = 0;
            for (int i = 0; i < rows; i++)
            {
                var c = contributions[i][j];
                sumAbs += Math.Abs(c);
                sum += c;
                if (Math.Abs(c) > 1e-9) nonzero++;
            }
            var meanAbs = rows > 0 ? sumAbs / rows : 0;
            var mean = rows > 0 ? sum / rows : 0;
            double sumSq = 0;
            for (int i = 0; i < rows; i++)
            {
                var d = contributions[i][j] - mean;
                sumSq += d * d;
            }
            var std = rows > 1 ? Math.Sqrt(sumSq / (rows - 1)) : 0;
            raw.Add((names[j], meanAbs, mean, std, rows > 0 ? 100.0 * nonzero / rows : 0));
        }
        return raw.OrderByDescending(x => x.MeanAbs)
            .Select((x, i) => new FeatureStat(i + 1, x.Name, x.MeanAbs, x.Mean, x.Std, x.Nonzero))
            .ToList();
    }
}
