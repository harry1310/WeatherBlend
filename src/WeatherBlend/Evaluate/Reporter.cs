using System.Globalization;
using System.Text;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate;

/// <summary>
/// Assembles the phase 2a verification report as markdown. No charts — tables only;
/// an MAE-vs-lead plot is pointless here because phase 2a is lead-bucket-free.
/// </summary>
public static class Reporter
{
    public sealed record ModelPrediction(string Name, double[] Predicted);

    public sealed class ReportInput
    {
        public required DateTime GeneratedAtUtc { get; init; }
        public required string ModelVersion { get; init; }
        public required string Phase { get; init; }
        public required ModelArtifact.TrainingMetadata Metadata { get; init; }
        public required IReadOnlyList<TrainingRow> TestRows { get; init; }
        public required ModelPrediction BlendTest { get; init; }
        public required IReadOnlyList<ModelPrediction> BaselinesTest { get; init; }
        public required string BestSingleName { get; init; }
        public required IReadOnlyList<(string Name, double Gain)> FeatureImportance { get; init; }
        public ModelPrediction? BlendMetar { get; init; }
        public IReadOnlyList<double>? ActualMetar { get; init; }
        public ModelPrediction? BestSingleMetar { get; init; }
        public int MetarTestRowsAvailable { get; init; }
        public string? LeadTimeBackfillMemoPath { get; init; }
    }

    public static string BuildMarkdown(ReportInput r)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;
        var actual = r.TestRows.Select(x => (double)x.Era5Temp).ToArray();

        sb.AppendLine($"# Phase {r.Phase} verification report");
        sb.AppendLine();
        sb.AppendLine($"- Generated (UTC): {r.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Model version: `{r.ModelVersion}`");
        sb.AppendLine($"- Train range: {r.Metadata.DataRangeTrain}  ({r.Metadata.TrainRows} rows)");
        sb.AppendLine($"- Val range:   {r.Metadata.DataRangeVal}  ({r.Metadata.ValRows} rows)");
        sb.AppendLine($"- Test range:  {r.Metadata.DataRangeTest}  ({r.Metadata.TestRows} rows)");
        sb.AppendLine($"- Best single model (selected on validation MAE): `{r.BestSingleName}`");
        sb.AppendLine();

        // Headline: MAE / RMSE / bias table
        sb.AppendLine("## Headline — error vs ERA5 on the test set");
        sb.AppendLine();
        sb.AppendLine("| Predictor | MAE (°C) | RMSE (°C) | Bias (°C) | N |");
        sb.AppendLine("|---|---:|---:|---:|---:|");
        AppendErrorRow(sb, r.BlendTest.Name, r.BlendTest.Predicted, actual, ci, bold: true);
        foreach (var b in r.BaselinesTest)
            AppendErrorRow(sb, b.Name, b.Predicted, actual, ci,
                bold: b.Name.Equals(r.BestSingleName, StringComparison.OrdinalIgnoreCase));
        sb.AppendLine();

        // Stratified MAE: by month, hour, wind quadrant, spread quintile.
        AppendStratification(sb, r, actual, ci);

        // Feature importance.
        sb.AppendLine("## Feature importance (gain)");
        sb.AppendLine();
        sb.AppendLine("| Feature | Gain |");
        sb.AppendLine("|---|---:|");
        foreach (var fi in r.FeatureImportance)
            sb.AppendLine($"| `{fi.Name}` | {fi.Gain.ToString("0.0000", ci)} |");
        sb.AppendLine();

        // Secondary truth check against METAR.
        sb.AppendLine("## Secondary truth check — vs Exeter METAR (EGTE)");
        sb.AppendLine();
        if (r.BlendMetar is null || r.ActualMetar is null || r.BestSingleMetar is null || r.ActualMetar.Count == 0)
        {
            sb.AppendLine("- No overlapping METAR rows in the test window (or METAR data missing).");
            sb.AppendLine("- This is a known limitation of phase 1's data coverage, not a training failure.");
        }
        else
        {
            var blendMae = Metrics.Compute(r.BlendMetar.Predicted, r.ActualMetar.ToList()).Mae;
            var bestMae = Metrics.Compute(r.BestSingleMetar.Predicted, r.ActualMetar.ToList()).Mae;
            sb.AppendLine($"- METAR rows available: {r.MetarTestRowsAvailable}");
            sb.AppendLine();
            sb.AppendLine("| Predictor | MAE vs METAR (°C) |");
            sb.AppendLine("|---|---:|");
            sb.AppendLine($"| **{r.BlendMetar.Name}** | {blendMae.ToString("0.000", ci)} |");
            sb.AppendLine($"| {r.BestSingleMetar.Name} | {bestMae.ToString("0.000", ci)} |");
            sb.AppendLine();
            var blendTestMae = Metrics.Compute(r.BlendTest.Predicted, actual).Mae;
            var bestTestMae = Metrics.Compute(
                r.BaselinesTest.First(b => b.Name.Equals(r.BestSingleName, StringComparison.OrdinalIgnoreCase))
                 .Predicted, actual).Mae;
            var era5WinsForBlend = blendTestMae < bestTestMae;
            var metarWinsForBlend = blendMae < bestMae;
            if (era5WinsForBlend && !metarWinsForBlend)
            {
                sb.AppendLine("**Flag:** blend beats best-single vs ERA5 but loses vs METAR. " +
                              "Possible ERA5-specific overfitting — worth investigating.");
                sb.AppendLine();
            }
        }

        // Caveats.
        sb.AppendLine("## Known caveats");
        sb.AppendLine();
        sb.AppendLine("- **No real lead time in backfill.** Open-Meteo's historical-forecast API returns " +
                      "\"best-available per valid-time\" with no cycle metadata; phase 1 synthesises " +
                      "`RunTime = midnight` so `LeadHours` is effectively hour-of-day. Phase 2a trains one " +
                      "blender across all valid-times (Option A). Per-lead-time training needs real issue-time " +
                      "archives — scoped for phase 3.");
        if (!string.IsNullOrWhiteSpace(r.LeadTimeBackfillMemoPath))
            sb.AppendLine($"  See: `{r.LeadTimeBackfillMemoPath}`.");
        sb.AppendLine("- **ERA5 as training truth** is grid-averaged (~30km) — systematically smoother than point obs.");
        sb.AppendLine("- **Lowland METAR as secondary truth** for a 393m tor — absolute numbers shift vs ERA5, " +
                      "but the *ranking* of predictors should be consistent.");
        sb.AppendLine("- **Deviations from the original brief (Microsoft.ML constraints):** " +
                      "L2 training objective instead of `regression_l1` (MAE only used for early stopping); " +
                      "no monotone constraints on per-model inputs. See `training_metadata.json`.");
        sb.AppendLine();

        return sb.ToString();
    }

    private static void AppendErrorRow(
        StringBuilder sb,
        string name,
        IReadOnlyList<double> predicted,
        IReadOnlyList<double> actual,
        CultureInfo ci,
        bool bold)
    {
        var s = Metrics.Compute(predicted, actual);
        var disp = bold ? $"**{name}**" : name;
        sb.AppendLine(
            $"| {disp} | {s.Mae.ToString("0.000", ci)} | {s.Rmse.ToString("0.000", ci)} | " +
            $"{s.Bias.ToString("+0.000;-0.000;0.000", ci)} | {s.N} |");
    }

    private static void AppendStratification(
        StringBuilder sb,
        ReportInput r,
        double[] actual,
        CultureInfo ci)
    {
        var blend = r.BlendTest.Predicted;
        var best = r.BaselinesTest
            .First(b => b.Name.Equals(r.BestSingleName, StringComparison.OrdinalIgnoreCase)).Predicted;
        var meanOfModels = r.BaselinesTest
            .First(b => b.Name.Equals("mean_of_models", StringComparison.OrdinalIgnoreCase)).Predicted;

        // By month.
        var monthKeys = r.TestRows.Select(x => x.ValidTimeUtc.Month).ToArray();
        sb.AppendLine("## Stratified MAE — by month");
        sb.AppendLine();
        WriteStrat(sb, "Month", r.BlendTest.Name, r.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, monthKeys, ci);

        // By hour-of-day.
        var hourKeys = r.TestRows.Select(x => x.ValidTimeUtc.Hour).ToArray();
        sb.AppendLine("## Stratified MAE — by hour of day");
        sb.AppendLine();
        WriteStrat(sb, "Hour", r.BlendTest.Name, r.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, hourKeys, ci);

        // Wind quadrant.
        var quadKeys = r.TestRows.Select(x => Metrics.WindQuadrant(x.WindDirMean)).ToArray();
        sb.AppendLine("## Stratified MAE — by mean wind direction (quadrant)");
        sb.AppendLine();
        WriteStrat(sb, "Quadrant", r.BlendTest.Name, r.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, quadKeys, ci);

        // Spread quintile — does the blend help most when models disagree?
        var spreads = r.TestRows.Select(x => (double)x.TempStd).ToArray();
        var quint = Metrics.Quintiles(spreads);
        sb.AppendLine("## Stratified MAE — by inter-model spread quintile (0=tightest, 4=widest)");
        sb.AppendLine();
        WriteStrat(sb, "Spread Q", r.BlendTest.Name, r.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, quint, ci);
    }

    private static void WriteStrat<TKey>(
        StringBuilder sb, string keyHeader,
        string blendName, string bestName, string meanName,
        double[] blend, double[] best, double[] meanModels, double[] actual,
        TKey[] keys, CultureInfo ci)
        where TKey : notnull, IComparable<TKey>
    {
        var blendMae = Metrics.StratifiedMae(blend, actual, keys);
        var bestMae  = Metrics.StratifiedMae(best, actual, keys);
        var meanMae  = Metrics.StratifiedMae(meanModels, actual, keys);

        sb.AppendLine($"| {keyHeader} | {blendName} MAE | {bestName} MAE | {meanName} MAE | N |");
        sb.AppendLine("|---|---:|---:|---:|---:|");
        foreach (var k in blendMae.Keys)
        {
            var b = blendMae[k];
            bestMae.TryGetValue(k, out var bs);
            meanMae.TryGetValue(k, out var mm);
            sb.AppendLine(
                $"| {k} | {b.Mae.ToString("0.000", ci)} | " +
                $"{(bs.N > 0 ? bs.Mae.ToString("0.000", ci) : "—")} | " +
                $"{(mm.N > 0 ? mm.Mae.ToString("0.000", ci) : "—")} | {b.N} |");
        }
        sb.AppendLine();
    }
}
