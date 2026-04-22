using System.Globalization;
using System.Text;
using WeatherBlend.Train;

namespace WeatherBlend.Evaluate;

/// <summary>
/// Assembles the per-lead phase 2b verification report as markdown. Tables only, no charts.
/// </summary>
public static class Reporter
{
    public sealed record ModelPrediction(string Name, double[] Predicted);

    /// <summary>Per-lead bundle for the phase 2b report. One of these per lead ∈ {24,48,72}.</summary>
    public sealed class LeadBundle
    {
        public required int LeadHours { get; init; }
        public required ModelArtifact.PerLeadStats Stats { get; init; }
        public required IReadOnlyList<TrainingRow> TestRows { get; init; }
        public required ModelPrediction BlendTest { get; init; }
        public required IReadOnlyList<ModelPrediction> BaselinesTest { get; init; }
        public required string BestSingleName { get; init; }
        public required IReadOnlyList<(string Name, double Gain)> FeatureImportance { get; init; }
        public required int PersistenceDropped { get; init; }

        // Optional METAR cross-check — same valid-time matching as the ERA5 path.
        public ModelPrediction? BlendMetar { get; init; }
        public ModelPrediction? BestSingleMetar { get; init; }
        public IReadOnlyList<double>? ActualMetar { get; init; }
        public int MetarRowsAvailable { get; init; }
    }

    public sealed class Phase2bReportInput
    {
        public required DateTime GeneratedAtUtc { get; init; }
        public required string ModelVersion { get; init; }
        public required ModelArtifact.TrainingMetadata Metadata { get; init; }
        public required IReadOnlyList<LeadBundle> Leads { get; init; }
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

    public static string BuildPhase2bMarkdown(Phase2bReportInput r)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;

        sb.AppendLine("# Phase 2b verification report");
        sb.AppendLine();
        sb.AppendLine($"- Generated (UTC): {r.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"- Model version: `{r.ModelVersion}`");
        sb.AppendLine($"- Data source: `{r.Metadata.DataSource}` (RunTimeSource = `offset_day`, lead-stratified)");
        sb.AppendLine();

        // Headline: per-lead MAE for blend + key baselines, stacked.
        sb.AppendLine("## Headline — MAE vs ERA5 on the test set, per lead");
        sb.AppendLine();
        sb.AppendLine("| Lead | Blend | Best single (val-selected) | Mean of models | Persistence ERA5(t−lead) | Climatology(month,hour) | N |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|");
        foreach (var lb in r.Leads)
        {
            var actual = lb.TestRows.Select(x => (double)x.Era5Temp).ToArray();
            var blendMae = Metrics.Compute(lb.BlendTest.Predicted, actual).Mae;
            var bestMae  = Metrics.Compute(Pick(lb, lb.BestSingleName), actual).Mae;
            var meanMae  = Metrics.Compute(Pick(lb, "mean_of_models"), actual).Mae;
            var persMae  = SafeMaeWithNaNs(Pick(lb, $"persistence_-{lb.LeadHours}h"), actual);
            var climMae  = Metrics.Compute(Pick(lb, "climatology(month,hour)"), actual).Mae;

            sb.AppendLine(
                $"| **{lb.LeadHours}h** | " +
                $"**{blendMae.ToString("0.000", ci)}** | " +
                $"{bestMae.ToString("0.000", ci)} (`{lb.BestSingleName}`) | " +
                $"{meanMae.ToString("0.000", ci)} | " +
                $"{(persMae is double p ? p.ToString("0.000", ci) : "—")} | " +
                $"{climMae.ToString("0.000", ci)} | {actual.Length} |");
        }
        sb.AppendLine();

        // Per-lead sections.
        foreach (var lb in r.Leads)
        {
            sb.AppendLine($"## Lead {lb.LeadHours}h");
            sb.AppendLine();
            sb.AppendLine($"- Train range: {lb.Stats.DataRangeTrain}  ({lb.Stats.TrainRows} rows)");
            sb.AppendLine($"- Val range:   {lb.Stats.DataRangeVal}  ({lb.Stats.ValRows} rows)");
            sb.AppendLine($"- Test range:  {lb.Stats.DataRangeTest}  ({lb.Stats.TestRows} rows, {lb.Stats.TestCalendarMonths} calendar months)");
            sb.AppendLine($"- Best single (val-selected): `{lb.BestSingleName}` (val MAE {lb.Stats.BestSingleValMae.ToString("0.000", ci)}°C)");
            if (lb.PersistenceDropped > 0)
                sb.AppendLine($"- Persistence rows dropped (no ERA5 at t−{lb.LeadHours}h): {lb.PersistenceDropped}");
            sb.AppendLine();

            var actual = lb.TestRows.Select(x => (double)x.Era5Temp).ToArray();

            sb.AppendLine("### MAE / RMSE / bias vs ERA5");
            sb.AppendLine();
            sb.AppendLine("| Predictor | MAE (°C) | RMSE (°C) | Bias (°C) | N |");
            sb.AppendLine("|---|---:|---:|---:|---:|");
            AppendErrorRow(sb, lb.BlendTest.Name, lb.BlendTest.Predicted, actual, ci, bold: true);
            foreach (var b in lb.BaselinesTest)
                AppendErrorRow(sb, b.Name, b.Predicted, actual, ci,
                    bold: b.Name.Equals(lb.BestSingleName, StringComparison.OrdinalIgnoreCase));
            sb.AppendLine();

            AppendPerLeadStratification(sb, lb, actual, ci);

            // Feature importance — same order as FeatureBuilder.FeatureNames.
            sb.AppendLine("### Feature importance (gain)");
            sb.AppendLine();
            sb.AppendLine("| Feature | Gain |");
            sb.AppendLine("|---|---:|");
            foreach (var fi in lb.FeatureImportance)
                sb.AppendLine($"| `{fi.Name}` | {fi.Gain.ToString("0.0000", ci)} |");
            sb.AppendLine();

            // METAR check for this lead.
            sb.AppendLine("### Secondary truth check — vs Exeter METAR (EGTE)");
            sb.AppendLine();
            if (lb.BlendMetar is null || lb.ActualMetar is null || lb.BestSingleMetar is null || lb.ActualMetar.Count == 0)
            {
                sb.AppendLine("- No overlapping METAR rows in this lead's test window (or METAR data missing).");
            }
            else
            {
                var blendMetarMae = Metrics.Compute(lb.BlendMetar.Predicted, lb.ActualMetar.ToList()).Mae;
                var bestMetarMae  = Metrics.Compute(lb.BestSingleMetar.Predicted, lb.ActualMetar.ToList()).Mae;
                sb.AppendLine($"- METAR rows available: {lb.MetarRowsAvailable}");
                sb.AppendLine();
                sb.AppendLine("| Predictor | MAE vs METAR (°C) |");
                sb.AppendLine("|---|---:|");
                sb.AppendLine($"| **{lb.BlendMetar.Name}** | {blendMetarMae.ToString("0.000", ci)} |");
                sb.AppendLine($"| {lb.BestSingleMetar.Name} | {bestMetarMae.ToString("0.000", ci)} |");
                sb.AppendLine();

                var blendTestMae = Metrics.Compute(lb.BlendTest.Predicted, actual).Mae;
                var bestTestMae  = Metrics.Compute(Pick(lb, lb.BestSingleName), actual).Mae;
                var era5WinsForBlend  = blendTestMae < bestTestMae;
                var metarWinsForBlend = blendMetarMae < bestMetarMae;
                if (era5WinsForBlend && !metarWinsForBlend)
                {
                    sb.AppendLine("**Flag:** blend beats best-single vs ERA5 but loses vs METAR. " +
                                  "Possible ERA5-specific overfitting for this lead.");
                    sb.AppendLine();
                }
            }
        }

        // Caveats.
        sb.AppendLine("## Known caveats");
        sb.AppendLine();
        sb.AppendLine("- **Previous Runs API depth** — data begins around Jan 2024 for most models, " +
                      "mid-2024 once UKMO's partial coverage is intersected with the all-6-present filter.");
        sb.AppendLine("- **ERA5 as training truth** is grid-averaged (~30km) — systematically smoother than point obs, " +
                      "especially on a 393m tor surrounded by lower terrain.");
        sb.AppendLine("- **Lowland METAR (EGTE) as secondary truth** for Bonehill Rocks — absolute MAE differs from ERA5; " +
                      "the *ranking* of predictors is the interpretable signal.");
        sb.AppendLine("- **Météo-France wind direction is 100% null** on the Previous Runs API — ensemble wind-direction " +
                      "quadrant stratification averages over 5 models, not 6.");
        sb.AppendLine("- **Deviations from brief (ML.NET constraints):** L2 training objective instead of `regression_l1` " +
                      "(MAE used only for early stopping); no monotone constraints on per-model inputs; " +
                      "artefacts saved as `.zip` (ITransformer pipeline) not `.lgb`. See `training_metadata.json`.");
        sb.AppendLine();

        return sb.ToString();
    }

    private static double[] Pick(LeadBundle lb, string name)
    {
        if (name.Equals(lb.BlendTest.Name, StringComparison.OrdinalIgnoreCase))
            return lb.BlendTest.Predicted;
        foreach (var b in lb.BaselinesTest)
            if (b.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                return b.Predicted;
        throw new KeyNotFoundException($"No baseline named '{name}' for lead {lb.LeadHours}h");
    }

    // Persistence predictions have NaN placeholders where the ERA5 lookup fails.
    // Metrics.Compute aborts on NaN, so compute MAE only over non-NaN pairs.
    private static double? SafeMaeWithNaNs(double[] predicted, double[] actual)
    {
        double sum = 0; int n = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            if (double.IsNaN(predicted[i]) || double.IsNaN(actual[i])) continue;
            sum += Math.Abs(predicted[i] - actual[i]);
            n++;
        }
        return n == 0 ? null : sum / n;
    }

    private static void AppendPerLeadStratification(
        StringBuilder sb, LeadBundle lb, double[] actual, CultureInfo ci)
    {
        var blend = lb.BlendTest.Predicted;
        var best  = Pick(lb, lb.BestSingleName);
        var meanOfModels = Pick(lb, "mean_of_models");

        var monthKeys = lb.TestRows.Select(x => x.ValidTimeUtc.Month).ToArray();
        sb.AppendLine("### Stratified MAE — by month");
        sb.AppendLine();
        WriteStrat(sb, "Month", lb.BlendTest.Name, lb.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, monthKeys, ci);

        // 4-bin hour-of-day: 00-05, 06-11, 12-17, 18-23.
        var hourBinKeys = lb.TestRows.Select(x => x.ValidTimeUtc.Hour / 6).ToArray(); // 0..3
        sb.AppendLine("### Stratified MAE — by hour-of-day bin (0=00-05, 1=06-11, 2=12-17, 3=18-23)");
        sb.AppendLine();
        WriteStrat(sb, "Hour bin", lb.BlendTest.Name, lb.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, hourBinKeys, ci);

        var quadKeys = lb.TestRows.Select(x => Metrics.WindQuadrant(x.WindDirMean)).ToArray();
        sb.AppendLine("### Stratified MAE — by mean wind direction (quadrant)");
        sb.AppendLine();
        WriteStrat(sb, "Quadrant", lb.BlendTest.Name, lb.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, quadKeys, ci);

        var spreads = lb.TestRows.Select(x => (double)x.TempStd).ToArray();
        var quint = Metrics.Quintiles(spreads);
        sb.AppendLine("### Stratified MAE — by inter-model spread quintile (0=tightest, 4=widest)");
        sb.AppendLine();
        WriteStrat(sb, "Spread Q", lb.BlendTest.Name, lb.BestSingleName, "mean_of_models",
                   blend, best, meanOfModels, actual, quint, ci);
    }
}
