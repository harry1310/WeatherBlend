using FluentAssertions;
using Xunit;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Mlp;

namespace WeatherBlend.Tests;

/// <summary>
/// Phase 3e MLP smoke. Trains on a synthetic dataset where the label is a
/// known function of the features (so the MLP must beat climatology by a
/// wide margin if the train loop is wired up at all), then round-trips the
/// bundle through save+load and confirms predictions match within float
/// noise.
///
/// These are NOT correctness tests for the modelling itself — that's what
/// the bake-off vs 3a/3c on real EA data answers. These are wiring tests:
/// "does TrainVector actually fit", "does SaveLeadModel + LoadLeadModel
/// preserve predictions", "do the tensor shapes line up". Anything beyond
/// that goes in the bake-off, not here.
/// </summary>
public class MlpTrainerTests
{
    /// <summary>Synthetic 8-feature dataset: y = (x0 + x1 - x2 > 0).
    /// The MLP must learn this relationship to beat climatology;
    /// asserting Brier < 0.15 confirms the train loop is wired up.
    /// (Climatology Brier on a balanced binary is 0.25.)</summary>
    private static (List<BinaryTrainingRow> Train, List<BinaryTrainingRow> Val, List<BinaryTrainingRow> Test, BlenderSpec Spec)
        MakeSynthetic(int seed, int n_train = 800, int n_val = 200, int n_test = 200, int n_features = 8)
    {
        var rng = new Random(seed);
        BinaryTrainingRow MakeRow()
        {
            var f = new float[n_features];
            for (int k = 0; k < n_features; k++) f[k] = (float)(rng.NextDouble() * 2 - 1);
            var label = (f[0] + f[1] - f[2]) > 0;
            return new BinaryTrainingRow { Features = f, Label = label };
        }
        var train = Enumerable.Range(0, n_train).Select(_ => MakeRow()).ToList();
        var val   = Enumerable.Range(0, n_val).Select(_ => MakeRow()).ToList();
        var test  = Enumerable.Range(0, n_test).Select(_ => MakeRow()).ToList();
        var featureNames = Enumerable.Range(0, n_features).Select(i => $"x{i}").ToArray();
        var spec = new BlenderSpec
        {
            Target = "precipitation",
            FeatureSet = "mlp-smoke",
            LeadHours = 24,
            Models = Array.Empty<string>(),
            RequiredModels = Array.Empty<string>(),
            OptionalModels = Array.Empty<string>(),
            FeatureNames = featureNames,
            DataSource = "synthetic",
            Tier = "smoke",
            UkvStrategy = null,
        };
        return (train, val, test, spec);
    }

    private static double Brier(double[] probs, IReadOnlyList<BinaryTrainingRow> rows)
    {
        if (probs.Length != rows.Count) throw new InvalidOperationException("length mismatch");
        double s = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            var y = rows[i].Label ? 1.0 : 0.0;
            var d = probs[i] - y;
            s += d * d;
        }
        return s / probs.Length;
    }

    [Fact]
    public void TrainVector_learns_synthetic_signal_well_below_climatology()
    {
        var (train, val, test, spec) = MakeSynthetic(seed: 7);

        // Aggressive early-stop + small MaxEpochs because the synthetic problem
        // converges fast and we don't want the test taking forever.
        var hp = new MlpTrainer.Hyperparameters(
            HiddenSizes: new[] { 32, 16 },
            Dropout: 0.0,
            LearningRate: 5e-3,
            BatchSize: 64,
            MaxEpochs: 100,
            EarlyStoppingPatience: 20,
            Seed: 7);

        var trained = MlpTrainer.TrainVector(train, val, spec, hp);
        var probs = MlpTrainer.PredictVectorProbability(trained, test);

        probs.Should().HaveCount(test.Count);
        probs.Should().OnlyContain(p => p >= 0.0 && p <= 1.0);

        var brier = Brier(probs, test);
        // Climatology on a balanced binary is 0.25; the MLP should be well
        // under that on a learnable y = (x0 + x1 - x2 > 0) signal. A loose
        // 0.15 ceiling avoids flakiness without rubber-stamping a broken fit.
        brier.Should().BeLessThan(0.15, "MLP should learn the synthetic signal");
        trained.BestValBrier.Should().BeLessThan(0.20);
    }

    /// <summary>Dead-column safety net regression test. Reproduces the
    /// 3e-Bellever 2026-05-13 failure mode: an extra feature whose
    /// training values are 100% NaN (so ComputeStandardiser writes
    /// mean=0, scale=1) collapses into the post-standardisation mean
    /// at train time. At predict time the SAME column carries a real
    /// value (mimicking offset_day vs reported cloud cover). Without
    /// the fix, that live value standardises to (60-0)/1 = 60 and
    /// saturates the MLP; with the fix it forces to 0 (= what training
    /// saw) and the prediction matches what we'd get with the column
    /// NaN'd at predict too.
    /// </summary>
    [Fact]
    public void PredictVectorProbability_zeroes_dead_columns_to_match_training_distribution()
    {
        var (train, val, test, spec) = MakeSynthetic(seed: 13, n_features: 8);

        // Force feature index 7 to NaN across train+val+test — i.e. the
        // model has never seen a real value for it. Train sees NaN →
        // imputed to 0 by MakeStandardisedTensor anyway, so this column
        // contributes nothing during fitting; ComputeStandardiser writes
        // mean=0, scale=1 for it (the dead-column sentinel).
        foreach (var row in train.Concat(val).Concat(test))
            row.Features[7] = float.NaN;

        var hp = new MlpTrainer.Hyperparameters(
            HiddenSizes: new[] { 16, 8 },
            Dropout: 0.0, LearningRate: 5e-3, BatchSize: 64,
            MaxEpochs: 50, EarlyStoppingPatience: 50, Seed: 13);
        var trained = MlpTrainer.TrainVector(train, val, spec, hp);

        // Confirm ComputeStandardiser detected the dead column.
        trained.ScalerMean[7].Should().Be(0f);
        trained.ScalerScale[7].Should().Be(1f);

        // Predict twice: once with the dead column held at NaN (train-like)
        // and once with the dead column populated with a large OOD value
        // (= the offset_day vs reported mismatch). The fix forces both to
        // the same post-standardisation 0, so predictions must match.
        var withNan = test.Select(r => new BinaryTrainingRow {
            Features = r.Features.ToArray(), Label = r.Label
        }).ToList();
        var withOod = test.Select(r => {
            var f = r.Features.ToArray();
            f[7] = 60f;   // big OOD value, c.f. live cloud_low_mean
            return new BinaryTrainingRow { Features = f, Label = r.Label };
        }).ToList();

        var probsNan = MlpTrainer.PredictVectorProbability(trained, withNan);
        var probsOod = MlpTrainer.PredictVectorProbability(trained, withOod);

        for (int i = 0; i < probsNan.Length; i++)
            probsOod[i].Should().BeApproximately(probsNan[i], 1e-6,
                $"row {i}: dead-column OOD value must standardise to the same 0 as train-time NaN");
    }

    [Fact]
    public void SaveLeadModel_then_LoadLeadModel_round_trips_predictions()
    {
        var (train, val, test, spec) = MakeSynthetic(seed: 11);
        var hp = new MlpTrainer.Hyperparameters(
            HiddenSizes: new[] { 16, 8 },
            Dropout: 0.0,
            LearningRate: 5e-3,
            BatchSize: 64,
            MaxEpochs: 30,
            EarlyStoppingPatience: 30,
            Seed: 11);

        var trained = MlpTrainer.TrainVector(train, val, spec, hp);
        var beforeProbs = MlpTrainer.PredictVectorProbability(trained, test);

        // Round-trip via the bundle dir.
        var bundleDir = Path.Combine(Path.GetTempPath(), $"mlp-smoke-{Guid.NewGuid():N}");
        try
        {
            var perLead = MlpArtifact.SaveLeadModel(bundleDir, leadHours: 24, trained, spec);
            var preprocess = new MlpArtifact.Preprocess(
                PerLead: new Dictionary<string, MlpArtifact.PerLeadPreprocess>(StringComparer.Ordinal)
                {
                    ["24"] = perLead,
                });
            MlpArtifact.WritePreprocess(bundleDir, preprocess);

            var (loadedModule, loadedCfg) = MlpArtifact.LoadLeadModel(bundleDir, leadHours: 24);
            loadedCfg.FeatureNames.Should().BeEquivalentTo(spec.FeatureNames);
            loadedCfg.ScalerMean.Should().BeEquivalentTo(trained.ScalerMean);
            loadedCfg.ScalerScale.Should().BeEquivalentTo(trained.ScalerScale);

            // Run predict against the LOADED module via a fresh TrainedMlp wrapper
            // — predict cares about Module + ScalerMean + ScalerScale, nothing else.
            var loadedTrained = new MlpTrainer.TrainedMlp(
                Module: loadedModule,
                ScalerMean: loadedCfg.ScalerMean.ToArray(),
                ScalerScale: loadedCfg.ScalerScale.ToArray(),
                Hyperparameters: hp,
                FeatureNames: loadedCfg.FeatureNames,
                EpochsRun: loadedCfg.EpochsRun,
                BestValBrier: loadedCfg.BestValBrier);
            var afterProbs = MlpTrainer.PredictVectorProbability(loadedTrained, test);

            afterProbs.Should().HaveCount(beforeProbs.Length);
            for (int i = 0; i < beforeProbs.Length; i++)
            {
                // Expect bit-exact (no dropout active in eval mode + same
                // weights). A loose 1e-5 tolerance covers any incidental
                // float-vs-double precision drift in the save/load path.
                afterProbs[i].Should().BeApproximately(beforeProbs[i], 1e-5,
                    $"row {i}: load should reproduce save's prediction");
            }
        }
        finally
        {
            if (Directory.Exists(bundleDir)) Directory.Delete(bundleDir, recursive: true);
        }
    }
}
