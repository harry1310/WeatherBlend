using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Gust;
using WeatherBlend.Train.Element.Wind;

namespace WeatherBlend.Commands;

/// <summary>
/// EXPERIMENT (uncommitted bake-off): does the UkmoCleanWindowStart (2024-09-01)
/// floor help or hurt the ERA5-truth `wind` and `wind_gust` element blenders?
///
/// Proper A/B: build the full-span rows (floor overridden to 2022-01-01), hold out
/// a FIXED recent OOS window (UKMO present there), and train two arms on a common
/// cutoff — FLOORED (rows ≥ 2024-09-01) vs NO-FLOOR (rows ≥ 2022-01-01) — then score
/// BOTH on the SAME held-out window vs ERA5 truth. Same features, same spec, same
/// LightGBM hp (wind = no-bagging, gust = default), same test rows → the MAE delta is
/// purely the floor. Per lead.
/// </summary>
public sealed class WindFloorBakeoffCommand
{
    private readonly ILogger<WindFloorBakeoffCommand> _log;
    private readonly AppConfig _cfg;

    public WindFloorBakeoffCommand(ILogger<WindFloorBakeoffCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    private sealed record ArmResult(int Lead, int NTest, int NTrainFloored, int NTrainNoFloor, double MaeFloored, double MaeNoFloor);

    public async Task<int> RunAsync(string? testStartStr, string? targetFilter, CancellationToken ct)
    {
        await Task.Yield();
        var loc = _cfg.Location.Name;
        var fc = _cfg.Storage.ForecastsPath;
        var era5 = _cfg.Storage.Era5Path;
        var testStart = DateOnly.TryParse(testStartStr, out var d)
            ? d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc)
            : new DateTime(2026, 3, 1, 0, 0, 0, DateTimeKind.Utc);
        var noFloor = "2022-01-01";
        var flooredFrom = DateTime.Parse(TrainingWindow.UkmoCleanWindowStart + " 00:00:00",
            System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.AssumeUniversal | System.Globalization.DateTimeStyles.AdjustToUniversal);
        int[] leads = { 24, 48, 72 };

        var windHp = TempTrainer.Hyperparameters.Default() with
        { SubsampleFraction = 1.0, SubsampleFrequency = 0, FeatureFraction = 1.0 };
        var defaultHp = TempTrainer.Hyperparameters.Default();

        _log.LogInformation("Wind floor bake-off — loc={Loc} testStart={T:yyyy-MM-dd} (FLOORED ≥{F:yyyy-MM-dd} vs NO-FLOOR off; required-not-null clips to ~2024-01/02)",
            loc, testStart, flooredFrom);

        var allTargets = new (string Name, Func<int, BlenderSpec> BuildSpec,
            Func<BlenderSpec, List<RegressionTrainingRow>> BuildRows, TempTrainer.Hyperparameters Hp)[]
        {
            ("wind",
                lead => WindFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
                spec => WindFeatureBuilder.BuildForLead(fc, era5, loc, spec, ct, floorOverride: noFloor),
                windHp),
            ("wind_gust",
                lead => WindGustFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
                spec => WindGustFeatureBuilder.BuildForLead(fc, era5, loc, spec, ct, floorOverride: noFloor),
                defaultHp),
            ("cloud_cover",
                lead => CloudFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
                spec => CloudFeatureBuilder.BuildForLead(fc, era5, loc, spec, ct, floorOverride: noFloor),
                defaultHp),
            // cloud with UKMO demoted required→optional: with UKMO no longer required,
            // the no-floor rows extend back to ~2024-02 (ecmwf binds) instead of being
            // clipped to UKMO's 2024-08 start. Tests whether unlocking that earlier data
            // (UKMO-NaN there) helps cloud. floored arm ≈ production cloud minus the
            // UKMO-required constraint.
            ("cloud_ukmo_opt",
                lead => DemoteUkmo(CloudFeatureBuilder.BuildSpec(_cfg.Blenders, lead)),
                spec => CloudFeatureBuilder.BuildForLead(fc, era5, loc, spec, ct, floorOverride: noFloor),
                defaultHp),
        };
        var filter = string.IsNullOrWhiteSpace(targetFilter) ? null
            : targetFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var targets = filter is null ? allTargets : allTargets.Where(t => filter.Contains(t.Name)).ToArray();

        var results = new Dictionary<string, List<ArmResult>>();

        foreach (var tgt in targets)
        {
            var rows = new List<ArmResult>();
            foreach (var lead in leads)
            {
                ct.ThrowIfCancellationRequested();
                var spec = tgt.BuildSpec(lead);
                var all = tgt.BuildRows(spec);   // full span, ordered by ValidTimeUtc
                var test = all.Where(r => r.ValidTimeUtc >= testStart).ToList();
                var trainNoFloor = all.Where(r => r.ValidTimeUtc < testStart).ToList();
                var trainFloored = trainNoFloor.Where(r => r.ValidTimeUtc >= flooredFrom).ToList();
                _log.LogInformation("  {T} lead {L}h — all={A} test={Te} trainFloored={Tf} trainNoFloor={Tn}",
                    tgt.Name, lead, all.Count, test.Count, trainFloored.Count, trainNoFloor.Count);
                if (test.Count < 100 || trainFloored.Count < 500 || trainNoFloor.Count < 500)
                {
                    _log.LogWarning("  {T} lead {L}h — too few rows; skipping.", tgt.Name, lead);
                    continue;
                }

                double maeF = TrainAndScore(trainFloored, test, spec, tgt.Hp);
                double maeN = TrainAndScore(trainNoFloor, test, spec, tgt.Hp);
                rows.Add(new ArmResult(lead, test.Count, trainFloored.Count, trainNoFloor.Count, maeF, maeN));
                _log.LogInformation("  {T} lead {L}h — MAE floored={F:0.0000} no-floor={N:0.0000}", tgt.Name, lead, maeF, maeN);
            }
            results[tgt.Name] = rows;
        }

        foreach (var (name, rows) in results)
        {
            Console.WriteLine();
            Console.WriteLine($"=== {name}: floor bake-off (ERA5 truth, common OOS test ≥ {testStart:yyyy-MM-dd}, Bonehill) ===");
            Console.WriteLine(" MAE, lower=better (m/s for wind/gust, % for cloud). NO-FLOOR adds the ~2024-01/02→2024-08 block (UKMO all-NaN); row counts below confirm the real span.");
            Console.WriteLine($"  {"lead",5} {"N_test",7} {"trN_floor",10} {"trN_nofloor",12} {"MAE floored",12} {"MAE no-floor",13} {"Δ no-floor",11}");
            foreach (var r in rows)
            {
                var dpct = (r.MaeNoFloor - r.MaeFloored) / r.MaeFloored * 100;
                var verdict = dpct < 0 ? "no-floor better" : "floored better";
                Console.WriteLine($"  {r.Lead + "h",5} {r.NTest,7} {r.NTrainFloored,10} {r.NTrainNoFloor,12} {r.MaeFloored,12:F4} {r.MaeNoFloor,13:F4} {dpct,9:+0.0;-0.0;0.0}%  {verdict}");
            }
        }
        Console.WriteLine();
        return results.Values.Any(v => v.Count > 0) ? 0 : 3;
    }

    private static double TrainAndScore(
        List<RegressionTrainingRow> pool, List<RegressionTrainingRow> test,
        BlenderSpec spec, TempTrainer.Hyperparameters hp)
    {
        // Chronological val = last 15% of the (already time-ordered) pool.
        int valStart = (int)(pool.Count * 0.85);
        var train = pool.Take(valStart).ToList();
        var val = pool.Skip(valStart).ToList();
        var trained = TempTrainer.TrainVector(train, val, spec, hp);
        var pred = TempTrainer.PredictVector(trained.Ml, trained.Model, spec, test);
        var actual = test.Select(r => (double)r.Label).ToArray();
        double sum = 0; int nn = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (double.IsNaN(pred[i]) || double.IsNaN(actual[i])) continue;
            sum += Math.Abs(pred[i] - actual[i]); nn++;
        }
        return nn == 0 ? double.NaN : sum / nn;
    }

    /// <summary>Return a copy of the spec with UKMO moved from required → optional
    /// (Models/FeatureNames unchanged — UKMO stays a feature, just no longer
    /// row-gating). Lets the no-floor arm keep pre-UKMO (≥~2024-02) rows.</summary>
    private static BlenderSpec DemoteUkmo(BlenderSpec s)
    {
        const string ukmo = "ukmo_seamless";
        if (!s.RequiredModels.Contains(ukmo)) return s;
        return new BlenderSpec
        {
            Target = s.Target,
            FeatureSet = s.FeatureSet,
            LeadHours = s.LeadHours,
            RequiredModels = s.RequiredModels.Where(m => m != ukmo).ToList(),
            OptionalModels = s.OptionalModels.Append(ukmo).ToList(),
            Models = s.Models,
            FeatureNames = s.FeatureNames,
            DataSource = s.DataSource,
            Tier = s.Tier,
            UkvStrategy = s.UkvStrategy,
        };
    }
}
