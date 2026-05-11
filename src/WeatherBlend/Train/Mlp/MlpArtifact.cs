using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Train.Mlp;

/// <summary>
/// Phase 3e MLP bundle save / load. Bundle layout per (station, version):
///
/// <code>
/// data/models/precipitation/{station}/v..._phase3e/
///   mlp_lead_24h.pt          TorchSharp state_dict for the lead-24 MLP
///   mlp_lead_48h.pt          (one per trained lead)
///   mlp_lead_72h.pt
///   mlp_lead_96h.pt
///   mlp_lead_120h.pt
///   preprocess.json          per-lead scaler mean+scale + hyperparams
///   training_metadata.json   standard PerLead stats — site / verify
///   feature_schema.json      standard per-lead spec — Spec page
///   test_predictions.parquet per-row test (val_time, station, lead, p_wet, observed)
/// </code>
///
/// Mirrors 4a's per-lead-files pattern (state_lead_NNh.rds + arrays_lead
/// _NNh.npz + preprocess.json) so the bundle directory is "obviously a 3e
/// bundle" at a glance and predict-side dispatch can scan for the
/// `mlp_lead_*.pt` glob to confirm the layout.
///
/// Why TorchSharp's native .pt format (not PyTorch-compatible): we never
/// round-trip to Python so the format only needs to be self-consistent.
/// `Module.save(path)` writes the state_dict as a TorchSharp binary;
/// `Module.load(path)` reads it back into an existing module of matching
/// shape (which we rebuild from preprocess.json's hyperparams at load
/// time).
/// </summary>
public static class MlpArtifact
{
    public const string PreprocessFileName = "preprocess.json";
    public const string MlpFilePattern = "mlp_lead_{0}h.pt";

    /// <summary>Per-lead block of the preprocess.json file.</summary>
    public sealed record PerLeadPreprocess(
        int LeadHours,
        IReadOnlyList<string> FeatureNames,
        IReadOnlyList<float> ScalerMean,
        IReadOnlyList<float> ScalerScale,
        IReadOnlyList<int> HiddenSizes,
        double Dropout,
        double LearningRate,
        int BatchSize,
        int MaxEpochs,
        int EarlyStoppingPatience,
        int Seed,
        int EpochsRun,
        double BestValBrier);

    /// <summary>Top-level preprocess.json wrapper. <c>perLead</c> keyed by
    /// lead-hours-as-string (matches preprocess.json convention in 4a).</summary>
    public sealed record Preprocess(
        IReadOnlyDictionary<string, PerLeadPreprocess> PerLead,
        // Free-form bookkeeping. Useful when reading the bundle by eye to know
        // which fitter version produced it, in case we ever move the trainer
        // signature in a non-back-compat way.
        string TrainerVersion = "1.0");

    /// <summary>Save one trained lead's MLP into the bundle dir. Idempotent —
    /// overwrite-on-write so re-runs leave the bundle in a clean state.
    /// Caller is responsible for assembling preprocess.json from the per-lead
    /// blocks at the end (see <see cref="WritePreprocess"/>).</summary>
    public static PerLeadPreprocess SaveLeadModel(
        string bundleDir, int leadHours, MlpTrainer.TrainedMlp trained, BlenderSpec spec)
    {
        Directory.CreateDirectory(bundleDir);
        var path = Path.Combine(bundleDir, string.Format(MlpFilePattern, leadHours));
        trained.Module.save(path);

        return new PerLeadPreprocess(
            LeadHours: leadHours,
            FeatureNames: spec.FeatureNames.ToArray(),
            ScalerMean: trained.ScalerMean,
            ScalerScale: trained.ScalerScale,
            HiddenSizes: trained.Hyperparameters.HiddenSizesEffective,
            Dropout: trained.Hyperparameters.Dropout,
            LearningRate: trained.Hyperparameters.LearningRate,
            BatchSize: trained.Hyperparameters.BatchSize,
            MaxEpochs: trained.Hyperparameters.MaxEpochs,
            EarlyStoppingPatience: trained.Hyperparameters.EarlyStoppingPatience,
            Seed: trained.Hyperparameters.Seed,
            EpochsRun: trained.EpochsRun,
            BestValBrier: trained.BestValBrier);
    }

    public static void WritePreprocess(string bundleDir, Preprocess preprocess)
    {
        Directory.CreateDirectory(bundleDir);
        var path = Path.Combine(bundleDir, PreprocessFileName);
        var opts = new JsonSerializerOptions
        {
            WriteIndented = true,
            // Keep enums + records human-readable.
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(preprocess, opts));
    }

    public static Preprocess ReadPreprocess(string bundleDir)
    {
        var path = Path.Combine(bundleDir, PreprocessFileName);
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<Preprocess>(json)
            ?? throw new InvalidOperationException(
                $"preprocess.json at {path} deserialised to null");
    }

    /// <summary>Rebuild the MLP for one lead and load its saved state_dict.
    /// Module shape is reconstructed from <paramref name="leadCfg"/> so
    /// load() finds matching parameter shapes; if the saved file was written
    /// with different HiddenSizes / d_in this throws at load time.</summary>
    public static (Module<Tensor, Tensor> Module, PerLeadPreprocess Cfg) LoadLeadModel(
        string bundleDir, int leadHours)
    {
        var preprocess = ReadPreprocess(bundleDir);
        if (!preprocess.PerLead.TryGetValue(leadHours.ToString(), out var leadCfg))
            throw new InvalidOperationException(
                $"preprocess.json has no entry for lead {leadHours}h in {bundleDir}");
        var d_in = leadCfg.FeatureNames.Count;
        var module = BuildMlpFromCfg(d_in, leadCfg);
        var path = Path.Combine(bundleDir, string.Format(MlpFilePattern, leadHours));
        module.load(path);
        module.eval();
        return (module, leadCfg);
    }

    /// <summary>Reconstruct the same Sequential MLP shape that the trainer
    /// built. Kept here (not on MlpTrainer) because Load needs it without
    /// reaching into the trainer's private helper.</summary>
    private static Module<Tensor, Tensor> BuildMlpFromCfg(int d_in, PerLeadPreprocess cfg)
    {
        var layers = new List<Module<Tensor, Tensor>>();
        var prev = d_in;
        foreach (var h in cfg.HiddenSizes)
        {
            layers.Add(Linear(prev, h));
            layers.Add(ReLU());
            layers.Add(Dropout(cfg.Dropout));
            prev = h;
        }
        layers.Add(Linear(prev, 1));
        return Sequential(layers.ToArray());
    }
}
