using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using WeatherBlend.Train.Common;
using WeatherBlend.Models;

namespace WeatherBlend.Train.Mlp;

/// <summary>
/// Plain MLP binary classifier for P(wet hour) — Phase 3e. .NET-native via
/// TorchSharp (CPU-only libtorch). Same per-(station, lead) shape as
/// 3a/3c so it slots into the existing precip predict + verify pipeline
/// unchanged. Comparison target: 3c (rich, 59 features) — a head-to-head
/// "does NN beat GBT on tabular at 30k rows" check.
///
/// Architecture (defaults; tweakable via Hyperparameters):
///   Linear(d_in, 128) → ReLU → Dropout(0.2)
///   → Linear(128, 64) → ReLU → Dropout(0.2)
///   → Linear(64,  32) → ReLU → Dropout(0.2)
///   → Linear(32,  1)  → (sigmoid applied OUTSIDE the module via BCEWithLogitsLoss
///                        for numerical stability; predict-side applies sigmoid).
///
/// Why BCEWithLogitsLoss not BCELoss: combines sigmoid + BCE in one pass for
/// numerical stability (avoids log(0) at saturated outputs). Predict side
/// applies torch.sigmoid to the raw logits to emit P(wet) in [0, 1].
///
/// Why standardise inputs: gradient descent converges much faster on z-scored
/// features. Scaler mean+scale persisted in the artefact so predict applies
/// the EXACT same standardisation the model was trained on.
///
/// Determinism: TorchSharp seeds via torch.manual_seed and we keep CPU-only;
/// per-run reproducibility is best-effort. Different OS / TorchSharp version
/// may produce slightly different floats. Per-cell test Brier is the
/// reproducibility check.
/// </summary>
public sealed class MlpTrainer
{
    public sealed record Hyperparameters(
        int[]? HiddenSizes = null,
        double Dropout = 0.2,
        double LearningRate = 1e-3,
        int BatchSize = 256,
        int MaxEpochs = 200,
        int EarlyStoppingPatience = 20,
        int Seed = 42)
    {
        public int[] HiddenSizesEffective => HiddenSizes ?? new[] { 128, 64, 32 };
    }

    public sealed record TrainedMlp(
        Module<Tensor, Tensor> Module,
        // Per-feature standardisation params, length = spec.FeatureCount.
        // Predict applies (x - mean) / scale before forward pass.
        float[] ScalerMean,
        float[] ScalerScale,
        Hyperparameters Hyperparameters,
        IReadOnlyList<string> FeatureNames,
        int EpochsRun,
        double BestValBrier);

    /// <summary>
    /// Vector-native fit on <see cref="BinaryTrainingRow"/>. Standardises
    /// features on TRAIN ONLY (no val leakage), runs Adam mini-batch gradient
    /// descent against BCEWithLogitsLoss, early-stops on val Brier, restores
    /// best-val weights before returning.
    /// </summary>
    public static TrainedMlp TrainVector(
        IReadOnlyList<BinaryTrainingRow> train,
        IReadOnlyList<BinaryTrainingRow> val,
        BlenderSpec spec,
        Hyperparameters hp)
    {
        if (train.Count == 0)
            throw new ArgumentException("No training rows", nameof(train));
        if (train[0].Features.Length != spec.FeatureCount)
            throw new InvalidOperationException(
                $"Training row Features length {train[0].Features.Length} != spec.FeatureCount {spec.FeatureCount} for {spec}");

        // Reproducibility: TorchSharp inherits the global seed for module
        // init + dropout + shuffles. CPU-only so no CUDA RNG to worry about.
        torch.manual_seed(hp.Seed);

        var d_in = spec.FeatureCount;
        var (mean, scale) = ComputeStandardiser(train, d_in);

        // Materialise train + val tensors once (small enough at 30k × 59 to
        // fit comfortably in memory: ~7 MiB float32). Avoids re-tensorising
        // each epoch.
        var x_train = MakeStandardisedTensor(train, mean, scale, d_in);
        var y_train = MakeLabelTensor(train);
        var x_val   = MakeStandardisedTensor(val, mean, scale, d_in);
        var y_val   = MakeLabelTensor(val);

        var model = BuildMlp(d_in, hp);
        var optimiser = torch.optim.Adam(model.parameters(), lr: hp.LearningRate);
        var loss_fn = BCEWithLogitsLoss();

        // Best-val tracking for early stopping. Brier (= MSE between
        // sigmoid(logit) and 0/1 label) is the production metric for 3a/3c so
        // we use it directly here too — keeps the bake-off honest.
        double bestValBrier = double.PositiveInfinity;
        int epochsSinceBest = 0;
        int bestEpoch = 0;
        Dictionary<string, Tensor>? bestWeights = null;

        var rng = new Random(hp.Seed);
        var n_train = train.Count;
        var indices = new int[n_train];
        for (int i = 0; i < n_train; i++) indices[i] = i;

        int epochsRun = 0;
        for (int epoch = 1; epoch <= hp.MaxEpochs; epoch++)
        {
            epochsRun = epoch;
            model.train();

            // Shuffle training indices each epoch — Fisher-Yates.
            for (int i = n_train - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            for (int batchStart = 0; batchStart < n_train; batchStart += hp.BatchSize)
            {
                var batchEnd = Math.Min(batchStart + hp.BatchSize, n_train);
                var batchIdx = indices[batchStart..batchEnd];
                using var idxTensor = torch.tensor(batchIdx, dtype: ScalarType.Int64);
                using var xb = x_train.index_select(0, idxTensor);
                using var yb = y_train.index_select(0, idxTensor);

                optimiser.zero_grad();
                using var logits = model.forward(xb);
                using var loss = loss_fn.forward(logits.squeeze(-1), yb);
                loss.backward();
                optimiser.step();
            }

            // Eval — Brier on val.
            model.eval();
            double valBrier;
            using (no_grad())
            {
                using var valLogits = model.forward(x_val).squeeze(-1);
                using var valProbs = torch.sigmoid(valLogits);
                using var diff = valProbs - y_val;
                using var sqDiff = diff * diff;
                using var mean_t = sqDiff.mean();
                valBrier = mean_t.item<float>();
            }

            if (valBrier < bestValBrier)
            {
                bestValBrier = valBrier;
                bestEpoch = epoch;
                epochsSinceBest = 0;
                bestWeights = SnapshotState(model);
            }
            else
            {
                epochsSinceBest++;
                if (epochsSinceBest >= hp.EarlyStoppingPatience)
                    break;
            }
        }

        if (bestWeights is not null)
            RestoreState(model, bestWeights);

        // Tensors created for the train loop are tied to the module/optimiser
        // — disposing here would invalidate the model state. Let GC + module
        // dispose in the caller's using block reclaim them.
        return new TrainedMlp(
            Module: model,
            ScalerMean: mean,
            ScalerScale: scale,
            Hyperparameters: hp,
            FeatureNames: spec.FeatureNames.ToArray(),
            EpochsRun: epochsRun,
            BestValBrier: bestValBrier);
    }

    /// <summary>
    /// Vector-native predict. Standardises rows with the trained scaler then
    /// runs forward pass; emits P(wet) in [0, 1] (raw sigmoid on the logits).
    /// </summary>
    public static double[] PredictVectorProbability(TrainedMlp trained, IReadOnlyList<BinaryTrainingRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var d_in = trained.ScalerMean.Length;
        if (rows[0].Features.Length != d_in)
            throw new InvalidOperationException(
                $"Predict row Features length {rows[0].Features.Length} != trained d_in {d_in}");

        trained.Module.eval();
        using var x = MakeStandardisedTensor(rows, trained.ScalerMean, trained.ScalerScale, d_in);
        using (no_grad())
        {
            using var logits = trained.Module.forward(x).squeeze(-1);
            using var probs = torch.sigmoid(logits);
            // Tensor → double[]. probs is shape [N], copy to managed array.
            var probsHost = probs.to(ScalarType.Float64).cpu().data<double>().ToArray();
            return probsHost;
        }
    }

    /// <summary>Build the Sequential MLP per Hyperparameters.</summary>
    private static Module<Tensor, Tensor> BuildMlp(int d_in, Hyperparameters hp)
    {
        var hidden = hp.HiddenSizesEffective;
        var layers = new List<Module<Tensor, Tensor>>();
        var prev = d_in;
        foreach (var h in hidden)
        {
            layers.Add(Linear(prev, h));
            layers.Add(ReLU());
            layers.Add(Dropout(hp.Dropout));
            prev = h;
        }
        layers.Add(Linear(prev, 1));
        // No final sigmoid — we use BCEWithLogitsLoss for numerical
        // stability and apply sigmoid at predict time.
        return Sequential(layers.ToArray());
    }

    /// <summary>Per-feature mean + scale on TRAIN ONLY, NaN-aware.
    ///
    /// Critical for the rich (3c-shape) feature vector: persistence
    /// features (eaRainPrev24h/72h, wet/dry hours) are NaN at the start
    /// of the time series; per-model NWP fields are NaN when an optional
    /// model is absent (e.g. AIFS pre-2026-04-27, JMA pre-2026-04-28).
    /// LightGBM handles NaN natively, so 3a/3c never had to care; MLPs
    /// propagate NaN → BCE of NaN → NaN gradient → all weights NaN, then
    /// every val Brier is NaN/∞, early stop never sees improvement, and
    /// the model collapses to the random init (verified failure mode on
    /// Hexworthy 2026-05-11). Skip NaN values when computing per-column
    /// mean + variance; columns with no observed values fall back to
    /// (mean=0, scale=1) so post-standardisation they're still 0.
    ///
    /// Scale clamped at 1e-6 to avoid divide-by-zero on constant
    /// features (e.g. an integer-coded feature that's the same across
    /// the whole train slice).</summary>
    private static (float[] Mean, float[] Scale) ComputeStandardiser(
        IReadOnlyList<BinaryTrainingRow> train, int d_in)
    {
        var n = train.Count;
        var sum = new double[d_in];
        var count = new int[d_in];

        for (int i = 0; i < n; i++)
        {
            var f = train[i].Features;
            for (int k = 0; k < d_in; k++)
            {
                if (float.IsNaN(f[k])) continue;
                sum[k] += f[k];
                count[k]++;
            }
        }
        var mean = new float[d_in];
        for (int k = 0; k < d_in; k++)
            mean[k] = count[k] > 0 ? (float)(sum[k] / count[k]) : 0f;

        var sumSq = new double[d_in];
        for (int i = 0; i < n; i++)
        {
            var f = train[i].Features;
            for (int k = 0; k < d_in; k++)
            {
                if (float.IsNaN(f[k])) continue;
                var d = f[k] - mean[k];
                sumSq[k] += d * d;
            }
        }
        var scale = new float[d_in];
        for (int k = 0; k < d_in; k++)
        {
            if (count[k] > 1)
                scale[k] = Math.Max((float)Math.Sqrt(sumSq[k] / (count[k] - 1)), 1e-6f);
            else
                scale[k] = 1f;
        }

        return (mean, scale);
    }

    /// <summary>Standardise rows + pack into a [N, d_in] float32 tensor.
    /// NaN inputs are imputed to 0 POST-standardisation (= the
    /// per-column mean) so they contribute a neutral signal rather
    /// than poisoning the forward pass.</summary>
    private static Tensor MakeStandardisedTensor(
        IReadOnlyList<BinaryTrainingRow> rows, float[] mean, float[] scale, int d_in)
    {
        var n = rows.Count;
        var flat = new float[n * d_in];
        for (int i = 0; i < n; i++)
        {
            var f = rows[i].Features;
            var off = i * d_in;
            for (int k = 0; k < d_in; k++)
            {
                if (float.IsNaN(f[k]))
                    flat[off + k] = 0f;          // = post-standardisation mean
                else
                    flat[off + k] = (f[k] - mean[k]) / scale[k];
            }
        }
        return torch.tensor(flat, new long[] { n, d_in }, ScalarType.Float32);
    }

    private static Tensor MakeLabelTensor(IReadOnlyList<BinaryTrainingRow> rows)
    {
        var n = rows.Count;
        var y = new float[n];
        for (int i = 0; i < n; i++) y[i] = rows[i].Label ? 1f : 0f;
        return torch.tensor(y, new long[] { n }, ScalarType.Float32);
    }

    /// <summary>Detached clone of the module's state_dict so we can restore
    /// best-val weights after early stopping.</summary>
    private static Dictionary<string, Tensor> SnapshotState(Module<Tensor, Tensor> module)
    {
        var snap = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, t) in module.state_dict())
            snap[name] = t.detach().clone();
        return snap;
    }

    private static void RestoreState(Module<Tensor, Tensor> module, Dictionary<string, Tensor> snap)
    {
        // Wrap in no_grad so autograd doesn't trip on copy_ into a leaf
        // parameter that has requires_grad=True. The state_dict() returns
        // the actual parameter tensors, not detached views, so the in-place
        // copy is fine semantically — we just need to tell autograd we know
        // what we're doing.
        using (no_grad())
        {
            var live = module.state_dict();
            foreach (var (name, t) in snap)
            {
                if (live.TryGetValue(name, out var liveTensor))
                    liveTensor.copy_(t);
            }
        }
    }
}
