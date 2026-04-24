// SHAP-style feature-contribution analysis for the 2c temperature and 3c precip
// rich-feature models. One-shot post-hoc analysis — no new production code, no
// retraining, no manifest changes. Writes markdown reports under data/reports/.
//
// Uses ML.NET's FeatureContributionCalculatingEstimator, which is a TreeSHAP-style
// (Saabas-approximate) per-feature decomposition of the score. For ranking,
// directionality, and dead-weight detection, that's functionally equivalent to
// the shap.TreeExplainer output; it's not the exact Lundberg TreeSHAP, but the
// difference doesn't matter for "which features carry weight, which don't".

using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Calibrators;
using WeatherBlend.Train;

namespace WeatherBlend.Scripts.ShapAnalyze;

internal static class Program
{
    private static readonly CultureInfo Ci = CultureInfo.InvariantCulture;
    private const string LocationName = "bonehill_rocks";
    private static readonly string RepoRoot = FindRepoRoot();
    private static readonly string ForecastsPath = Path.Combine(RepoRoot, "data", "forecasts");
    private static readonly string Era5Path      = Path.Combine(RepoRoot, "data", "truth", "era5");
    private static readonly string RainfallPath  = Path.Combine(RepoRoot, "data", "truth", "rainfall");
    private static readonly string ReportsPath   = Path.Combine(RepoRoot, "data", "reports");
    private static readonly string ModelsRoot    = Path.Combine(RepoRoot, "data", "models");

    // Phase suffix used to pick the rich-feature challenger out of the MANIFEST's
    // Active list. These are the *challengers* (2c / 3c), not the champions (2b /
    // 3a) — MANIFEST.Current still points at the lean baseline. The analyser
    // scans Active and picks the entry whose version name ends in the suffix
    // below, which means future retrains land on the new version automatically.
    private const string DefaultTempPhase   = "phase2c";
    private const string DefaultPrecipPhase = "phase3c";

    private static readonly int[] Leads = { 24, 48, 72 };
    private static readonly DateTime Today = DateTime.UtcNow.Date;

    public static int Main(string[] args)
    {
        var mode = "all";
        var tempPhase = DefaultTempPhase;
        var precipPhase = DefaultPrecipPhase;

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
                case "--temp-phase" when i + 1 < args.Length: tempPhase = args[++i]; break;
                case "--precip-phase" when i + 1 < args.Length: precipPhase = args[++i]; break;
                case "all" or "temp" or "2c" or "precip" or "3c": mode = a.ToLowerInvariant(); break;
                case "-h" or "--help":
                    Console.WriteLine("Usage: shap-analyze [all|temp|precip] [--temp-phase <suffix>] [--precip-phase <suffix>]");
                    Console.WriteLine($"  defaults: mode=all, temp-phase={DefaultTempPhase}, precip-phase={DefaultPrecipPhase}");
                    Console.WriteLine("  phase suffix matches the trailing part of a version name (e.g. phase2c -> v...._phase2c).");
                    return 0;
                default:
                    Console.Error.WriteLine($"[shap-analyze] unknown arg: {a}");
                    return 1;
            }
        }

        Console.WriteLine($"[shap-analyze] repo root: {RepoRoot}");
        Console.WriteLine($"[shap-analyze] reports dir: {ReportsPath}");
        Console.WriteLine($"[shap-analyze] mode={mode} temp-phase={tempPhase} precip-phase={precipPhase}");
        Directory.CreateDirectory(ReportsPath);

        try
        {
            if (mode is "all" or "temp" or "2c")
            {
                RunTemperature(tempPhase);
            }
            if (mode is "all" or "precip" or "3c")
            {
                RunPrecipitation(precipPhase);
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[shap-analyze] FAILED: {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    // Find the single version in MANIFEST.Active whose name ends with the
    // supplied phase suffix. Fails loudly if zero or multiple match — we'd
    // rather abort than silently analyse the wrong artefact.
    private static string ResolveTempVersion(string phaseSuffix)
    {
        var active = ModelArtifact.ResolveActive(ModelsRoot, "temperature");
        return PickBySuffix(active, phaseSuffix, "temperature");
    }

    private static string ResolvePrecipVersion(string station, string phaseSuffix)
    {
        var active = ModelArtifact.ResolveStationActive(ModelsRoot, "precipitation", station);
        return PickBySuffix(active, phaseSuffix, $"precipitation/{station}");
    }

    private static string PickBySuffix(IReadOnlyList<string> active, string phaseSuffix, string context)
    {
        var matches = active.Where(v => v.EndsWith("_" + phaseSuffix, StringComparison.Ordinal)).ToList();
        if (matches.Count == 0)
            throw new InvalidOperationException(
                $"No Active version ending in '_{phaseSuffix}' for {context}. Active: [{string.Join(", ", active)}]");
        if (matches.Count > 1)
            throw new InvalidOperationException(
                $"Ambiguous: multiple Active versions end in '_{phaseSuffix}' for {context}: [{string.Join(", ", matches)}]");
        return matches[0];
    }

    // Stations for the precip run: every station recorded in the precipitation
    // MANIFEST that has a matching phase version in its Active list. Falls back
    // to an empty list if the manifest is absent — caller handles the error.
    private static IReadOnlyList<(string Slug, string Version)> ResolvePrecipTargets(string phaseSuffix)
    {
        var stations = ModelArtifact.ListStations(ModelsRoot, "precipitation");
        var targets = new List<(string, string)>();
        foreach (var slug in stations)
        {
            var active = ModelArtifact.ResolveStationActive(ModelsRoot, "precipitation", slug);
            var match = active.FirstOrDefault(v => v.EndsWith("_" + phaseSuffix, StringComparison.Ordinal));
            if (match is not null) targets.Add((slug, match));
        }
        return targets;
    }

    // ---- Phase 2c -----------------------------------------------------------

    private static void RunTemperature(string phaseSuffix)
    {
        var version = ResolveTempVersion(phaseSuffix);
        Console.WriteLine($"=== Temperature blender (phase suffix '{phaseSuffix}') — version {version} ===");
        var versionDir = Path.Combine(ModelsRoot, "temperature", version);
        var featureNames = RichFeatureBuilder.FeatureNames.ToArray();
        var featureCols = TemperatureTrainer.RichFeatureColumnInternalNames();

        var sections = new List<LeadShapResult>();
        foreach (var lead in Leads)
        {
            Console.WriteLine($"--- lead {lead}h ---");
            var rows = RichFeatureBuilder.BuildForLead(ForecastsPath, Era5Path, LocationName, lead);
            var ds   = TrainingDataset.Split(rows);
            var test = ds.Test.Cast<RichTrainingRow>().ToList();
            Console.WriteLine($"  loaded {rows.Count} rows; test slice {test.Count} rows ({ds.TestStart:yyyy-MM-dd}..{ds.TestEnd:yyyy-MM-dd})");

            var ml = new MLContext(seed: 42);
            var loaded = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);

            var contributions = ComputeRegressionContributions(ml, loaded, test, featureCols);
            Console.WriteLine($"  computed contributions for {contributions.Length} test rows × {featureNames.Length} features");

            sections.Add(new LeadShapResult(lead, featureNames, contributions, test.Count));
        }

        var md = BuildTempReport(sections, versionDir, featureNames, phaseSuffix);
        var tag = ShortTag(phaseSuffix); // phase2c -> 2c
        var outPath = Path.Combine(ReportsPath, $"shap_{tag}_{Today:yyyy-MM-dd}.md");
        File.WriteAllText(outPath, md);
        Console.WriteLine($"[shap-analyze] wrote {outPath}");
    }

    // ---- Phase 3c -----------------------------------------------------------

    private static void RunPrecipitation(string phaseSuffix)
    {
        var targets = ResolvePrecipTargets(phaseSuffix);
        if (targets.Count == 0)
            throw new InvalidOperationException(
                $"No stations in precipitation MANIFEST.Active have a version ending in '_{phaseSuffix}'.");
        Console.WriteLine($"=== Precip blender (phase suffix '{phaseSuffix}') — {targets.Count} station(s) ===");
        var featureNames = PrecipRichFeatureBuilder.OccurrenceFeatureNames.ToArray();
        var featureCols = PrecipOccurrenceTrainer.RichFeatureColumnInternalNames();

        var perStation = new List<StationShapResult>();
        foreach (var (slug, version) in targets)
        {
            Console.WriteLine($"=== station {slug} ({version}) ===");
            var versionDir = Path.Combine(ModelsRoot, "precipitation", slug, version);
            var stationName = SlugToStationName(slug);

            var leadResults = new List<LeadShapResult>();
            foreach (var lead in Leads)
            {
                Console.WriteLine($"--- lead {lead}h ---");
                var rows = PrecipRichFeatureBuilder.BuildForLead(ForecastsPath, RainfallPath, LocationName, stationName, lead);
                var ds   = PrecipDataset.Split(rows.Cast<PrecipTrainingRow>().ToList());
                var test = ds.Test.Cast<RichPrecipTrainingRow>().ToList();
                Console.WriteLine($"  loaded {rows.Count} rows; test slice {test.Count} rows (wet={ds.TestWet})");

                var ml = new MLContext(seed: 42);
                var loaded = ModelArtifact.LoadLeadModel(ml, versionDir, lead, out _);

                var contributions = ComputeBinaryContributions(ml, loaded, test, featureCols);
                Console.WriteLine($"  computed contributions for {contributions.Length} test rows × {featureNames.Length} features");

                leadResults.Add(new LeadShapResult(lead, featureNames, contributions, test.Count));
            }
            perStation.Add(new StationShapResult(slug, stationName, version, leadResults));
        }

        var md = BuildPrecipReport(perStation, featureNames, phaseSuffix);
        var tag = ShortTag(phaseSuffix); // phase3c -> 3c
        var outPath = Path.Combine(ReportsPath, $"shap_{tag}_{Today:yyyy-MM-dd}.md");
        File.WriteAllText(outPath, md);
        Console.WriteLine($"[shap-analyze] wrote {outPath}");
    }

    // ---- Contribution computation (regression) ------------------------------

    // The saved zip is a TransformerChain whose last element is a
    // RegressionPredictionTransformer<LightGbmRegressionModelParameters>. We
    // featurize the test rows, pull out the predictor, wrap it with
    // CalculateFeatureContribution, and transform to read out per-row
    // FeatureContributions vectors.
    private static float[][] ComputeRegressionContributions<T>(
        MLContext ml,
        ITransformer loaded,
        IReadOnlyList<T> testRows,
        string[] featureCols) where T : class
    {
        var (featurizer, predictor) = SplitChainRegression(loaded);
        var rawDv = ml.Data.LoadFromEnumerable(testRows);
        var featDv = featurizer.Transform(rawDv);

        var contribEstimator = ml.Transforms.CalculateFeatureContribution(
            predictor,
            numberOfPositiveContributions: featureCols.Length,
            numberOfNegativeContributions: featureCols.Length,
            normalize: false);
        var fittedContrib = contribEstimator.Fit(featDv);
        var withContrib = fittedContrib.Transform(featDv);

        return ReadContributionMatrix(withContrib, featureCols.Length);
    }

    // ---- Contribution computation (binary) ----------------------------------

    // 3c models land as BinaryPredictionTransformer wrapping a Platt-calibrated
    // LightGBM binary classifier. The calibrated wrapper doesn't implement
    // ICalculateFeatureContribution directly; we peel it off, construct a
    // regression-style prediction transformer around the raw LightGBM binary
    // predictor (which does), and read contributions off that. The resulting
    // contributions are in raw-score space (logits before Platt), which is the
    // right space for ranking/directionality anyway — Platt is monotone so
    // ordering and signs carry through to probability space.
    private static float[][] ComputeBinaryContributions<T>(
        MLContext ml,
        ITransformer loaded,
        IReadOnlyList<T> testRows,
        string[] featureCols) where T : class
    {
        var (featurizer, predTransformer) = SplitChainBinary(loaded);
        var rawDv = ml.Data.LoadFromEnumerable(testRows);
        var featDv = featurizer.Transform(rawDv);

        // Unwrap calibrated → raw LightGBM binary predictor. Direct cast avoids a
        // reflection ambiguity between the base-type and derived-type SubModel
        // declarations (`ValueMapperCalibratedModelParameters<...>` redeclares it).
        var calibratedPredictor = predTransformer.Model;
        if (calibratedPredictor is not CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> cal)
            throw new InvalidOperationException(
                $"Expected CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>, got {calibratedPredictor?.GetType().FullName}");
        var rawSub = cal.SubModel;

        // Wrap the raw binary predictor in a RegressionPredictionTransformer so
        // the FeatureContribution API accepts it. Raw score space is what we
        // want for interpretation — Platt is a monotone rescaling.
        // The 4-arg (host, model, schema, featureCol) constructor is internal in
        // ML.NET 4.0, so we grab it via NonPublic reflection.
        var genericType = typeof(Microsoft.ML.Data.RegressionPredictionTransformer<>).MakeGenericType(rawSub.GetType());
        var featureColName = predTransformer.FeatureColumnName;
        const System.Reflection.BindingFlags ctorFlags =
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
        var ctor = genericType.GetConstructors(ctorFlags)
            .FirstOrDefault(c =>
            {
                var p = c.GetParameters();
                return p.Length == 4
                    && typeof(IHostEnvironment).IsAssignableFrom(p[0].ParameterType)
                    && p[2].ParameterType == typeof(DataViewSchema)
                    && p[3].ParameterType == typeof(string);
            });
        if (ctor is null)
        {
            var allCtors = string.Join("\n", genericType.GetConstructors(ctorFlags)
                .Select(c => $"  {c} (params: {string.Join(", ", c.GetParameters().Select(p => p.ParameterType.Name))})"));
            throw new InvalidOperationException(
                $"No matching RegressionPredictionTransformer ctor. Available:\n{allCtors}");
        }
        // MLContext itself implements IHostEnvironment.
        var host = (IHostEnvironment)ml;
        var wrapped = ctor.Invoke(new object[] { host, rawSub, featDv.Schema, featureColName });

        // Hand this regression-like transformer to the contribution estimator.
        var catalog = ml.Transforms;
        // Need to call the non-generic extension: CalculateFeatureContribution(this, ISingleFeaturePredictionTransformer<ICalculateFeatureContribution>, int, int, bool)
        var miExt = typeof(ExplainabilityCatalog).GetMethods()
            .First(m => m.Name == "CalculateFeatureContribution"
                        && m.GetParameters().Length == 5);
        var contribEstObj = miExt.Invoke(null, new object?[]
        {
            catalog, wrapped, featureCols.Length, featureCols.Length, false
        })!;
        var fitMethod = contribEstObj.GetType().GetMethod("Fit")!;
        var fittedContrib = (ITransformer)fitMethod.Invoke(contribEstObj, new object[] { featDv })!;
        var withContrib = fittedContrib.Transform(featDv);

        return ReadContributionMatrix(withContrib, featureCols.Length);
    }

    private static (ITransformer featurizer,
                    Microsoft.ML.Data.RegressionPredictionTransformer<LightGbmRegressionModelParameters> predictor)
        SplitChainRegression(ITransformer loaded)
    {
        var (feats, last) = SplitChain(loaded);
        if (last is Microsoft.ML.Data.RegressionPredictionTransformer<LightGbmRegressionModelParameters> rpt)
            return (feats, rpt);
        throw new InvalidOperationException(
            $"Expected RegressionPredictionTransformer<LightGbmRegressionModelParameters>, got {last.GetType().FullName}");
    }

    private static (ITransformer featurizer,
                    Microsoft.ML.Data.BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> predictor)
        SplitChainBinary(ITransformer loaded)
    {
        var (feats, last) = SplitChain(loaded);
        // The exact calibrated generic may vary (ValueMapperCalibratedModelParameters), so
        // match on the outer BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
        if (last is Microsoft.ML.Data.BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> bpt)
            return (feats, bpt);
        // Fallback: the concrete type may be a derived calibrated type. Match by assignability.
        var lastType = last.GetType();
        if (lastType.IsGenericType
            && lastType.GetGenericTypeDefinition().Name.StartsWith("BinaryPredictionTransformer"))
        {
            // Accept via reflection; caller only needs Model + FeatureColumnName.
            return (feats, (dynamic)last);
        }
        throw new InvalidOperationException(
            $"Expected BinaryPredictionTransformer<CalibratedModelParametersBase<...>>, got {last.GetType().FullName}");
    }

    private static (ITransformer featurizer, ITransformer last) SplitChain(ITransformer loaded)
    {
        if (loaded is not IEnumerable<ITransformer> chain)
            throw new InvalidOperationException($"Loaded model is not a TransformerChain: {loaded.GetType().FullName}");

        var items = chain.ToArray();
        if (items.Length < 2)
            throw new InvalidOperationException($"Chain has only {items.Length} steps; expected >= 2");

        // Everything except the last is the featurizer pipeline; fold them into a single
        // composite transformer by chaining.
        ITransformer feat = items[0];
        for (int i = 1; i < items.Length - 1; i++)
        {
            // There isn't a public Append(ITransformer, ITransformer) — build via
            // TransformerChain<ITransformer>.
        }
        // For our simple case (concat + predictor), items.Length == 2 and feat = items[0].
        // If longer, fall back to re-wrapping minus the last item.
        if (items.Length == 2) return (items[0], items[1]);
        var head = items.Take(items.Length - 1).ToArray();
        var composed = new Microsoft.ML.Data.TransformerChain<ITransformer>(head);
        return (composed, items[^1]);
    }

    private static float[][] ReadContributionMatrix(IDataView dv, int featureCount)
    {
        // "FeatureContributions" is a VBuffer<float> of length == feature count.
        var col = dv.Schema["FeatureContributions"];
        var result = new List<float[]>();
        using var cursor = dv.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        VBuffer<float> buf = default;
        while (cursor.MoveNext())
        {
            getter(ref buf);
            var arr = new float[featureCount];
            var denseValues = buf.DenseValues().ToArray();
            // VBuffer may be sparse; DenseValues expands to full length.
            if (denseValues.Length != featureCount)
            {
                // Fallback: fill by manual copy.
                for (int i = 0; i < Math.Min(denseValues.Length, featureCount); i++)
                    arr[i] = denseValues[i];
            }
            else
            {
                Array.Copy(denseValues, arr, featureCount);
            }
            result.Add(arr);
        }
        return result.ToArray();
    }

    // ---- Report aggregation --------------------------------------------------

    private sealed record LeadShapResult(
        int LeadHours,
        string[] FeatureNames,
        float[][] Contributions,   // [row][feature]
        int TestRows);

    private sealed record StationShapResult(
        string Slug,
        string Name,
        string Version,
        IReadOnlyList<LeadShapResult> Leads);

    private sealed record PerFeatureStats(
        string Name,
        double MeanAbs,
        double MeanSigned,
        double StdDev,
        int NonZeroRows,
        int TotalRows)
    {
        public double NonZeroFrac => TotalRows == 0 ? 0 : (double)NonZeroRows / TotalRows;
    }

    private static PerFeatureStats[] Aggregate(LeadShapResult r)
    {
        var n = r.FeatureNames.Length;
        var stats = new PerFeatureStats[n];
        var rows = r.Contributions;
        for (int f = 0; f < n; f++)
        {
            double sumAbs = 0, sumSigned = 0;
            int nz = 0;
            for (int i = 0; i < rows.Length; i++)
            {
                var v = rows[i][f];
                sumAbs += Math.Abs(v);
                sumSigned += v;
                if (Math.Abs(v) > 1e-9) nz++;
            }
            double meanAbs = rows.Length == 0 ? 0 : sumAbs / rows.Length;
            double meanSigned = rows.Length == 0 ? 0 : sumSigned / rows.Length;
            double variance = 0;
            for (int i = 0; i < rows.Length; i++)
            {
                var d = rows[i][f] - meanSigned;
                variance += d * d;
            }
            variance = rows.Length == 0 ? 0 : variance / rows.Length;
            stats[f] = new PerFeatureStats(
                r.FeatureNames[f], meanAbs, meanSigned, Math.Sqrt(variance), nz, rows.Length);
        }
        return stats;
    }

    // ---- Markdown rendering --------------------------------------------------

    private static string BuildTempReport(
        IReadOnlyList<LeadShapResult> leads,
        string versionDir,
        string[] featureNames,
        string phaseSuffix)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# SHAP report — {PrettyPhase(phaseSuffix)} temperature blender");
        sb.AppendLine();
        sb.AppendLine($"- Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm}Z");
        sb.AppendLine($"- Model version: `{Path.GetFileName(versionDir)}`");
        sb.AppendLine($"- Feature count: {featureNames.Length} (rich-feature blender)");
        sb.AppendLine($"- Method: ML.NET `CalculateFeatureContribution` (TreeSHAP-approximate), raw-score space");
        sb.AppendLine($"- Test slice per lead: chronological 15% tail");
        sb.AppendLine();
        sb.AppendLine("## How to read this report");
        sb.AppendLine();
        sb.AppendLine("- **mean |SHAP|**: average magnitude of the feature's contribution to the prediction across the test slice. Higher = feature swings predictions more. Primary ranking metric.");
        sb.AppendLine("- **mean SHAP**: directional bias. Positive → feature pushes predicted temp above baseline more often than below (or pushes further up when it does). Close to 0 with large mean|SHAP| = bidirectional corrector.");
        sb.AppendLine("- **std SHAP**: spread of contribution values. Low std + low mean|SHAP| = dead weight. High std = regime-dependent feature.");
        sb.AppendLine("- **non-zero %**: fraction of test rows where the feature affected the prediction at all (non-zero contribution).");
        sb.AppendLine();
        var tierOf = TierForTempFeature;
        foreach (var lead in leads)
        {
            var stats = Aggregate(lead).OrderByDescending(s => s.MeanAbs).ToArray();
            RenderLeadSection(sb, $"Lead {lead.LeadHours}h", lead.TestRows, stats, tierOf);
        }

        // Cross-lead narrative.
        sb.AppendLine("## Cross-lead narrative");
        sb.AppendLine();
        var perLeadTop = leads.Select(l => Aggregate(l).OrderByDescending(s => s.MeanAbs).Take(5).Select(s => s.Name).ToArray()).ToArray();
        var allTop5 = perLeadTop.SelectMany(x => x).Distinct().ToArray();
        sb.AppendLine("Top-5 features by mean|SHAP|, per lead:");
        sb.AppendLine();
        sb.Append("| rank |");
        foreach (var l in leads) sb.Append($" lead {l.LeadHours}h |");
        sb.AppendLine();
        sb.Append("|---|");
        foreach (var _ in leads) sb.Append("---|");
        sb.AppendLine();
        for (int rank = 0; rank < 5; rank++)
        {
            sb.Append($"| {rank + 1} |");
            foreach (var lt in perLeadTop) sb.Append($" `{lt[rank]}` |");
            sb.AppendLine();
        }
        sb.AppendLine();

        // Tier-level roll-up (averaged mean|SHAP| within each tier, per lead).
        sb.AppendLine("### Tier contribution share (sum of mean|SHAP| within tier ÷ total)");
        sb.AppendLine();
        sb.Append("| tier |");
        foreach (var l in leads) sb.Append($" lead {l.LeadHours}h |");
        sb.AppendLine();
        sb.Append("|---|");
        foreach (var _ in leads) sb.Append("---|");
        sb.AppendLine();
        foreach (var tier in TempTierOrder)
        {
            sb.Append($"| {tier} |");
            foreach (var l in leads)
            {
                var stats = Aggregate(l);
                double tierSum = stats.Where(s => tierOf(s.Name) == tier).Sum(s => s.MeanAbs);
                double totalSum = stats.Sum(s => s.MeanAbs);
                double frac = totalSum == 0 ? 0 : tierSum / totalSum;
                sb.Append($" {frac:P1} |");
            }
            sb.AppendLine();
        }
        sb.AppendLine();

        // Dead weight across all leads.
        sb.AppendLine("### Dead-weight candidates (mean|SHAP| ≈ 0 at every lead)");
        sb.AppendLine();
        var dead = new List<string>();
        foreach (var name in featureNames)
        {
            bool allDead = leads.All(l =>
            {
                var s = Aggregate(l).First(x => x.Name == name);
                return s.MeanAbs < 1e-5 || s.NonZeroFrac < 0.01;
            });
            if (allDead) dead.Add(name);
        }
        if (dead.Count == 0)
        {
            sb.AppendLine("_None — every feature contributes non-trivially at at least one lead._");
        }
        else
        {
            foreach (var d in dead) sb.AppendLine($"- `{d}` (tier: {tierOf(d)})");
        }
        sb.AppendLine();

        sb.AppendLine("### Lead-to-lead signal shifts");
        sb.AppendLine();
        sb.AppendLine(DescribeLeadShifts(leads, tierOf));

        return sb.ToString();
    }

    private static string BuildPrecipReport(
        IReadOnlyList<StationShapResult> stations,
        string[] featureNames,
        string phaseSuffix)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# SHAP report — {PrettyPhase(phaseSuffix)} rich precip blender");
        sb.AppendLine();
        sb.AppendLine($"- Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm}Z");
        sb.AppendLine($"- Feature count: {featureNames.Length}");
        sb.AppendLine($"- Method: ML.NET `CalculateFeatureContribution` applied to the raw LightGBM binary sub-model (pre-Platt). Raw-score space = logit; ordering and signs survive the monotone Platt step.");
        sb.AppendLine($"- Test slice per (station, lead): chronological 15% tail of 3c rich-feature dataset.");
        sb.AppendLine();
        sb.AppendLine("## How to read this report");
        sb.AppendLine();
        sb.AppendLine("- Same definitions as the 2c report: mean|SHAP|, mean SHAP, std SHAP, non-zero %.");
        sb.AppendLine("- Values are in logit space, not probability. Sign is still interpretable: positive = pushes P(wet) up.");
        sb.AppendLine();
        var tierOf = TierForPrecipFeature;

        foreach (var station in stations)
        {
            sb.AppendLine($"## Station: {station.Name} (`{station.Slug}`)");
            sb.AppendLine();
            sb.AppendLine($"- Model version: `{station.Version}`");
            sb.AppendLine();
            foreach (var lead in station.Leads)
            {
                var stats = Aggregate(lead).OrderByDescending(s => s.MeanAbs).ToArray();
                RenderLeadSection(sb, $"{station.Name} — lead {lead.LeadHours}h", lead.TestRows, stats, tierOf);
            }
        }

        // Cross-lead narrative per station.
        sb.AppendLine("## Cross-station / cross-lead narrative");
        sb.AppendLine();

        sb.AppendLine("### Top-5 features per (station, lead)");
        sb.AppendLine();
        sb.Append("| rank |");
        foreach (var s in stations)
            foreach (var l in s.Leads) sb.Append($" {s.Slug.Replace("ea_", "")} {l.LeadHours}h |");
        sb.AppendLine();
        sb.Append("|---|");
        foreach (var s in stations) foreach (var _ in s.Leads) sb.Append("---|");
        sb.AppendLine();
        for (int rank = 0; rank < 5; rank++)
        {
            sb.Append($"| {rank + 1} |");
            foreach (var s in stations)
                foreach (var l in s.Leads)
                {
                    var st = Aggregate(l).OrderByDescending(x => x.MeanAbs).ElementAt(rank);
                    sb.Append($" `{st.Name}` |");
                }
            sb.AppendLine();
        }
        sb.AppendLine();

        sb.AppendLine("### Tier contribution share");
        sb.AppendLine();
        sb.Append("| tier |");
        foreach (var s in stations)
            foreach (var l in s.Leads) sb.Append($" {s.Slug.Replace("ea_", "")} {l.LeadHours}h |");
        sb.AppendLine();
        sb.Append("|---|");
        foreach (var s in stations) foreach (var _ in s.Leads) sb.Append("---|");
        sb.AppendLine();
        foreach (var tier in PrecipTierOrder)
        {
            sb.Append($"| {tier} |");
            foreach (var s in stations)
                foreach (var l in s.Leads)
                {
                    var stats = Aggregate(l);
                    double tierSum = stats.Where(x => tierOf(x.Name) == tier).Sum(x => x.MeanAbs);
                    double totalSum = stats.Sum(x => x.MeanAbs);
                    double frac = totalSum == 0 ? 0 : tierSum / totalSum;
                    sb.Append($" {frac:P1} |");
                }
            sb.AppendLine();
        }
        sb.AppendLine();

        sb.AppendLine("### Dead-weight candidates (mean|SHAP| ≈ 0 everywhere)");
        sb.AppendLine();
        var dead = new List<string>();
        foreach (var name in featureNames)
        {
            bool allDead = stations.All(s => s.Leads.All(l =>
            {
                var st = Aggregate(l).First(x => x.Name == name);
                return st.MeanAbs < 1e-5 || st.NonZeroFrac < 0.01;
            }));
            if (allDead) dead.Add(name);
        }
        if (dead.Count == 0)
        {
            sb.AppendLine("_None — every feature contributes non-trivially in at least one (station, lead)._");
        }
        else
        {
            foreach (var d in dead) sb.AppendLine($"- `{d}` (tier: {tierOf(d)})");
        }
        sb.AppendLine();

        // Bellever vs Princetown comparison.
        var bell = stations.FirstOrDefault(s => s.Slug == "ea_bellever_dartmoor");
        var prnc = stations.FirstOrDefault(s => s.Slug == "ea_princetown");
        if (bell is not null && prnc is not null)
        {
            sb.AppendLine("### Bellever vs Princetown");
            sb.AppendLine();
            sb.AppendLine(DescribeBellevverPrincetown(bell, prnc, tierOf));
        }
        return sb.ToString();
    }

    private static void RenderLeadSection(
        StringBuilder sb, string heading, int testRows, PerFeatureStats[] stats,
        Func<string, string> tierOf)
    {
        sb.AppendLine($"### {heading}");
        sb.AppendLine();
        sb.AppendLine($"- Test rows: {testRows}");
        sb.AppendLine();
        sb.AppendLine("**Top 20 by mean|SHAP|**");
        sb.AppendLine();
        sb.AppendLine("| rank | feature | tier | mean\\|SHAP\\| | mean SHAP | std SHAP | non-zero % |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---:|");
        for (int i = 0; i < Math.Min(20, stats.Length); i++)
        {
            var s = stats[i];
            sb.AppendFormat(Ci,
                "| {0} | `{1}` | {2} | {3:0.0000} | {4:+0.0000;-0.0000;0.0000} | {5:0.0000} | {6:P1} |{7}",
                i + 1, s.Name, tierOf(s.Name), s.MeanAbs, s.MeanSigned, s.StdDev, s.NonZeroFrac, Environment.NewLine);
        }
        sb.AppendLine();

        // Flagged features: high std + near-zero mean (swingy, bidirectional),
        // very signed-biased (persistent pushes), very-low-usage.
        var swingy = stats
            .Where(s => s.MeanAbs > 0.01 && Math.Abs(s.MeanSigned) / Math.Max(s.MeanAbs, 1e-9) < 0.05)
            .OrderByDescending(s => s.StdDev).Take(5).ToArray();
        var biased = stats
            .Where(s => s.MeanAbs > 0.01 && Math.Abs(s.MeanSigned) / Math.Max(s.MeanAbs, 1e-9) > 0.6)
            .OrderByDescending(s => Math.Abs(s.MeanSigned)).Take(5).ToArray();
        var stale = stats
            .Where(s => s.MeanAbs < 1e-4 || s.NonZeroFrac < 0.02)
            .OrderBy(s => s.MeanAbs).ThenBy(s => s.NonZeroFrac).Take(5).ToArray();

        if (swingy.Length > 0)
        {
            sb.AppendLine("**Most bidirectional (high-variance, near-zero mean — pulls both directions depending on context):**");
            foreach (var s in swingy) sb.AppendLine($"- `{s.Name}` (mean|SHAP|={s.MeanAbs:0.000}, std={s.StdDev:0.000}, signed bias {Ratio(s):P1})");
            sb.AppendLine();
        }
        if (biased.Length > 0)
        {
            sb.AppendLine("**Most directionally biased (sign dominates — persistent nudge, not a corrector):**");
            foreach (var s in biased) sb.AppendLine($"- `{s.Name}` (mean SHAP={s.MeanSigned:+0.000;-0.000;0.000}, bias fraction {Ratio(s):P1})");
            sb.AppendLine();
        }
        if (stale.Length > 0)
        {
            sb.AppendLine("**Near-dead at this lead (low mean|SHAP| or rarely fires):**");
            foreach (var s in stale) sb.AppendLine($"- `{s.Name}` (mean|SHAP|={s.MeanAbs:0.000000}, non-zero {s.NonZeroFrac:P1})");
            sb.AppendLine();
        }
    }

    private static double Ratio(PerFeatureStats s) => Math.Abs(s.MeanSigned) / Math.Max(s.MeanAbs, 1e-9);

    // ---- Feature-tier classification ----------------------------------------

    private static readonly string[] TempTierOrder =
    {
        "temperature (per-model)", "temperature aggregates", "calendar",
        "dew point", "humidity", "cloud", "wind", "pressure", "other",
    };

    private static string TierForTempFeature(string name)
    {
        if (name is "temp_gfs" or "temp_ecmwf" or "temp_icon" or "temp_mf" or "temp_ukmo" or "temp_gem")
            return "temperature (per-model)";
        if (name is "temp_mean" or "temp_std" or "temp_range")
            return "temperature aggregates";
        if (name is "hour_sin" or "hour_cos" or "doy_sin" or "doy_cos")
            return "calendar";
        if (name.StartsWith("dew_")) return "dew point";
        if (name.StartsWith("rh_")) return "humidity";
        if (name.StartsWith("cloud")) return "cloud";
        if (name.StartsWith("wind_")) return "wind";
        if (name.StartsWith("pressure")) return "pressure";
        return "other";
    }

    private static readonly string[] PrecipTierOrder =
    {
        "precip (per-model)", "wet-prob (per-model)", "precip aggregates",
        "humidity / moisture", "cloud", "cape", "wind", "pressure",
        "ea persistence", "calendar",
    };

    private static string TierForPrecipFeature(string name)
    {
        if (name is "precip_gfs" or "precip_ecmwf" or "precip_icon" or "precip_mf" or "precip_ukmo" or "precip_gem")
            return "precip (per-model)";
        if (name is "prob_gfs" or "prob_ecmwf" or "prob_icon" or "prob_mf" or "prob_ukmo" or "prob_gem")
            return "wet-prob (per-model)";
        if (name is "precip_mean" or "precip_std" or "precip_max" or "precip_agreement_wet_01")
            return "precip aggregates";
        if (name.StartsWith("dew_depression") || name.StartsWith("dew_") || name.StartsWith("rh_"))
            return "humidity / moisture";
        if (name.StartsWith("cloud")) return "cloud";
        if (name.StartsWith("cape")) return "cape";
        if (name.StartsWith("wind_")) return "wind";
        if (name.StartsWith("pressure")) return "pressure";
        if (name.StartsWith("ea_")) return "ea persistence";
        if (name is "hour_sin" or "hour_cos" or "doy_sin" or "doy_cos")
            return "calendar";
        return "other";
    }

    // ---- Narrative helpers ---------------------------------------------------

    private static string DescribeLeadShifts(IReadOnlyList<LeadShapResult> leads, Func<string, string> tierOf)
    {
        var sb = new StringBuilder();
        var tiers = TempTierOrder;
        var tierFracs = leads.Select(l =>
        {
            var stats = Aggregate(l);
            double total = stats.Sum(s => s.MeanAbs);
            return tiers.ToDictionary(t => t, t => total == 0 ? 0.0 : stats.Where(s => tierOf(s.Name) == t).Sum(s => s.MeanAbs) / total);
        }).ToArray();
        for (int t = 0; t < tiers.Length; t++)
        {
            var name = tiers[t];
            var fracs = tierFracs.Select(d => d[name]).ToArray();
            var shift = fracs[^1] - fracs[0];
            if (Math.Abs(shift) < 0.02) continue;
            sb.AppendLine($"- **{name}**: share shifts from {fracs[0]:P1} at {leads[0].LeadHours}h to {fracs[^1]:P1} at {leads[^1].LeadHours}h ({(shift > 0 ? "gains" : "loses")} {Math.Abs(shift):P1}).");
        }
        if (sb.Length == 0)
            sb.AppendLine("_No tier shifts > 2% between shortest and longest lead — feature usage is stable across horizons._");
        return sb.ToString();
    }

    private static string DescribeBellevverPrincetown(
        StationShapResult bell, StationShapResult prnc, Func<string, string> tierOf)
    {
        var sb = new StringBuilder();
        foreach (var lead in Leads)
        {
            var bL = bell.Leads.FirstOrDefault(l => l.LeadHours == lead);
            var pL = prnc.Leads.FirstOrDefault(l => l.LeadHours == lead);
            if (bL is null || pL is null) continue;
            var bStats = Aggregate(bL);
            var pStats = Aggregate(pL);
            var bTop = bStats.OrderByDescending(s => s.MeanAbs).First();
            var pTop = pStats.OrderByDescending(s => s.MeanAbs).First();
            sb.AppendLine($"**Lead {lead}h** — Bellever top: `{bTop.Name}` (mean|SHAP|={bTop.MeanAbs:0.000}); Princetown top: `{pTop.Name}` (mean|SHAP|={pTop.MeanAbs:0.000}).");
            // Biggest relative differences.
            var deltaByFeature = bStats.Join(pStats, b => b.Name, p => p.Name,
                (b, p) => (Name: b.Name, BMeanAbs: b.MeanAbs, PMeanAbs: p.MeanAbs, Diff: b.MeanAbs - p.MeanAbs))
                .OrderByDescending(x => Math.Abs(x.Diff))
                .Take(3).ToArray();
            foreach (var d in deltaByFeature)
                sb.AppendLine($"  - `{d.Name}`: Bellever {d.BMeanAbs:0.000} vs Princetown {d.PMeanAbs:0.000} (Δ {d.Diff:+0.000;-0.000;0.000})");
        }
        return sb.ToString();
    }

    // ---- Utilities -----------------------------------------------------------

    private static string SlugToStationName(string slug) => slug switch
    {
        "ea_bellever_dartmoor"     => "Bellever Dartmoor",
        "ea_princetown"            => "Princetown",
        "ea_dartmoor_nr_hexworthy" => "Dartmoor nr Hexworthy",
        // Fall back to a human-readable form derived from the slug so newly
        // added stations don't crash the analyser.
        _ => CultureInfo.InvariantCulture.TextInfo.ToTitleCase(
                (slug.StartsWith("ea_") ? slug[3..] : slug).Replace('_', ' ')),
    };

    // Turn "phase2c" into "2c" for file naming; pass through anything that
    // doesn't start with "phase".
    private static string ShortTag(string phaseSuffix) =>
        phaseSuffix.StartsWith("phase", StringComparison.Ordinal) ? phaseSuffix[5..] : phaseSuffix;

    // "phase2c" -> "Phase 2c" for the report title.
    private static string PrettyPhase(string phaseSuffix) =>
        phaseSuffix.StartsWith("phase", StringComparison.Ordinal)
            ? "Phase " + phaseSuffix[5..]
            : phaseSuffix;

    private static string FindRepoRoot()
    {
        // Walk up from AppContext.BaseDirectory looking for the `data/` dir + `src/` dir.
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "data")) &&
                Directory.Exists(Path.Combine(dir, "src", "WeatherBlend")))
            {
                return dir;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        // Fallback: current working directory.
        return Directory.GetCurrentDirectory();
    }
}
