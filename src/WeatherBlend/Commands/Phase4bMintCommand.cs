using System.Text.Json;
using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 4b one-time bundle minter. 4b isn't a trained model — it's
/// the arithmetic mean of phase 4a + phase 3o P(wet), identified as
/// the best production stack in the 2026-05-12 bake-off. The bundle
/// exists so render / verify / Models page treat 4b as a first-class
/// phase: there's a versioned dir per station with metadata,
/// climatology, feature schema, training_summary, and a
/// test_predictions.parquet derived from the inner-join of 4a + 3o's
/// own test_predictions parquets.
///
/// Runs once per Sunday auto-retrain (after 4a + 3o are minted) plus
/// any time the user wants to refresh the bundle manually. Updates
/// MANIFEST.Active via PromoteStationVersion so the new 4b version
/// replaces the previous one (lead-overlap-aware: same phase, same lead
/// set → replace, no accumulation). 4b is a challenger — phases.yaml,
/// not this promote, decides the champion (3a).
///
/// What the bundle contains:
///   training_metadata.json   Phase=4b, LocationName, PerLead Brier
///                            on the joined test slice, source bundle
///                            versions in DeviationsFromBrief.
///   training_summary.json    Minimal stub for RetrainGuard.
///   feature_schema.json      One BlenderSpec per lead listing p_4a +
///                            p_3o as inputs (so the Spec page renders
///                            a meaningful card).
///   test_predictions.parquet Inner-join of 4a + 3o test_predictions
///                            with p_wet = mean(p_4a, p_3o).
///   climatology.json         Copied from 3o's bundle — same station,
///                            same wet indicator, identical baseline.
/// </summary>
public sealed class Phase4bMintCommand
{
    private readonly ILogger<Phase4bMintCommand> _log;
    private readonly AppConfig _cfg;
    private const string Phase = "4b";

    public Phase4bMintCommand(ILogger<Phase4bMintCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    /// <summary>
    /// Per-station outcome. <see cref="Skipped"/> is NOT a failure: a
    /// station with no 4a or 3o bundle simply isn't provisioned for 4b
    /// (e.g. Membury — 4a was never trained there). Skips must not fail
    /// the mint step, otherwise the Sunday retrain red-Xes over a model
    /// that was never meant to exist at that location.
    /// </summary>
    private enum MintOutcome { Minted, Skipped, Failed }

    public async Task<int> RunAsync(CancellationToken ct)
    {
        var stations = ResolveStations();
        if (stations.Count == 0)
        {
            _log.LogError("No precipitation stations configured — cannot mint 4b.");
            return 2;
        }

        var modelsRoot = _cfg.Storage.ModelsPath;
        var version = DateTime.UtcNow.ToString("'v'yyyy-MM-dd_HHmmss'_phase4b'");
        var trainedAt = DateTime.UtcNow;

        _log.LogInformation(
            "Mint Phase 4b: version={Version}, stations=[{Stations}]",
            version, string.Join(", ", stations));

        int nOk = 0, nSkip = 0, nErr = 0;
        foreach (var station in stations)
        {
            ct.ThrowIfCancellationRequested();
            var (outcome, note) = await MintForStationAsync(modelsRoot, station, version, trainedAt, ct);
            var tag = outcome switch
            {
                MintOutcome.Minted  => "OK  ",
                MintOutcome.Skipped => "SKIP",
                _                   => "ERR ",
            };
            _log.LogInformation("{Tag} {Station}: {Note}", tag, station, note);
            switch (outcome)
            {
                case MintOutcome.Minted:  nOk++;   break;
                case MintOutcome.Skipped: nSkip++; break;
                default:                  nErr++;  break;
            }
        }
        _log.LogInformation("Done. Minted: {OK}, skipped: {Skip}, failed: {Err}.", nOk, nSkip, nErr);
        // Skips don't fail the step — only genuine errors (4a+3o present
        // but the mint itself broke) escalate to exit 3.
        return nErr == 0 ? 0 : 3;
    }

    private IReadOnlyList<string> ResolveStations()
    {
        var slugs = new List<string>();
        foreach (var loc in _cfg.Locations)
        {
            foreach (var s in loc.Rainfall.Stations)
                slugs.Add(StationSlug.WithEaPrefix(s.Name));
        }
        return slugs;
    }

    private async Task<(MintOutcome outcome, string note)> MintForStationAsync(
        string modelsRoot, string station, string version, DateTime trainedAt, CancellationToken ct)
    {
        var stationDir = Path.Combine(modelsRoot, "precipitation", station);
        if (!Directory.Exists(stationDir))
            return (MintOutcome.Skipped, $"no precipitation models dir at {stationDir} — not provisioned for 4b");

        // 4a / 3o supply the test_predictions to join. Location pin +
        // climatology come from 3o — 4b is composed FROM 4a + 3o, and
        // 3o's bundle already ships both training_metadata.json (with
        // LocationName) and climatology.json at the same station, so
        // there's no reason to drag a separate 3a dependency through.
        // (Pre-2026-05-26 the mint pulled both fields from 3a because
        // 4b's stage-2 source was then 3e, which didn't carry
        // climatology. The 3e→3o composition swap removed that
        // constraint; this commit closes it out.)
        var bundle4a = FindLatestBundle(stationDir, "_phase4a", "test_predictions.parquet");
        var bundle3o = FindLatestBundle(stationDir, "_phase3o", "test_predictions.parquet");
        // No 4a or no 3o ⇒ this station isn't a 4b station (e.g. Membury,
        // where 4a has never been trained). Skip, don't fail.
        if (bundle4a is null) return (MintOutcome.Skipped, $"no 4a bundle — station not provisioned for 4b ({stationDir})");
        if (bundle3o is null) return (MintOutcome.Skipped, $"no 3o bundle — station not provisioned for 4b ({stationDir})");

        // Pull the active location from the 3o bundle's metadata — 4b
        // inherits 3o's location pinning. If a station ever ended up
        // with 3o + 4a trained on different locations, RetrainGuard
        // would have refused those promotions long before we got here.
        var meta3o = ModelArtifact.LoadTrainingMetadata(bundle3o);
        var location = string.IsNullOrEmpty(meta3o.LocationName)
            ? _cfg.Location.Name
            : meta3o.LocationName;

        // Read both test_predictions parquets via DuckDB, inner-join on
        // (valid_time, lead) — the canonical schema is
        // {valid_time, station, lead, p_wet, observed_wet}.
        var joined = ReadAndJoinTestPredictions(
            Path.Combine(bundle4a, "test_predictions.parquet"),
            Path.Combine(bundle3o, "test_predictions.parquet"));
        if (joined.Count == 0)
            return (MintOutcome.Failed, "inner-join of 4a + 3o test_predictions produced 0 rows");

        // Write bundle dir + artefacts.
        var outDir = Path.Combine(stationDir, version);
        Directory.CreateDirectory(outDir);

        // 1. test_predictions.parquet — same schema as every other binary phase.
        var testRows = joined
            .Select(j => new TestPredictionRow
            {
                valid_time   = j.ValidTime,
                station      = j.Station,
                lead         = j.Lead,
                p_wet        = j.PWet,
                observed_wet = (byte)(j.Observed ? 1 : 0),
            })
            .ToList();
        await ParquetSerializer.SerializeAsync(testRows,
            Path.Combine(outDir, "test_predictions.parquet"), cancellationToken: ct);

        // 2. Per-lead Brier — same convention every binary phase uses
        // (BlendTestMae carries Brier; BlendTestRmse / Bias not
        // meaningful for an unfit mean, set to 0/mean(p-y)).
        var perLead = new Dictionary<string, ModelArtifact.PerLeadStats>();
        foreach (var leadGroup in joined.GroupBy(j => j.Lead).OrderBy(g => g.Key))
        {
            var rows = leadGroup.ToList();
            var y = rows.Select(r => r.Observed ? 1.0 : 0.0).ToArray();
            var p = rows.Select(r => r.PWet).ToArray();
            var p4a = rows.Select(r => r.PWet4a).ToArray();
            var p3o = rows.Select(r => r.PWet3o).ToArray();
            var brierBlend = Brier(p, y);
            var brier4a    = Brier(p4a, y);
            var brier3o    = Brier(p3o, y);
            var bestBrier  = Math.Min(brier4a, brier3o);
            var bestName   = brier3o <= brier4a ? "p_3o" : "p_4a";
            perLead[leadGroup.Key.ToString()] = new ModelArtifact.PerLeadStats
            {
                LeadHours          = leadGroup.Key,
                DataRangeTrain     = "",
                DataRangeVal       = "",
                DataRangeTest      = "",
                TrainRows          = 0,
                ValRows            = 0,
                TestRows           = rows.Count,
                TestCalendarMonths = 0,
                BestSingle         = bestName,
                BestSingleValMae   = null,
                BestSingleTestMae  = bestBrier,
                BlendTestMae       = brierBlend,
                BlendTestRmse      = 0.0,
                BlendTestBias      = p.Zip(y, (pp, yy) => pp - yy).Average(),
                CalibratedBlendTestMae = 0.0,
            };
        }

        // 3. training_metadata.json — first-class phase, real PerLead numbers.
        var metadata = new ModelArtifact.TrainingMetadata
        {
            Version       = version,
            Target        = "precipitation",
            Phase         = Phase,
            LocationName  = location,
            DataSource    = $"derived: mean(p_4a, p_3o) — 4a={Path.GetFileName(bundle4a)}, 3o={Path.GetFileName(bundle3o)}",
            TrainedAtUtc  = trainedAt,
            Hyperparameters = new Dictionary<string, object>
            {
                ["method"]            = "arithmetic_mean",
                ["components"]        = new[] { "p_4a", "p_3o" },
                ["source_bundle_4a"]  = Path.GetFileName(bundle4a)!,
                ["source_bundle_3o"]  = Path.GetFileName(bundle3o)!,
            },
            TestMae = perLead.ToDictionary(
                kv => $"lead_{kv.Key}h_brier",
                kv => kv.Value.BlendTestMae),
            DeviationsFromBrief = new List<string>
            {
                "Phase 4b is NOT a trained model — it is the arithmetic mean of phase 4a's BART P(wet) and phase 3o's MLP P(wet). The bundle is minted by Phase4bMintCommand (typically after Sunday auto-retrain) so the ModelMetadata + Manifest plumbing treats it as a first-class phase (renders on rain forecasts, scored by verify, shown on the Models page). The 'training' step is the join + average.",
                "test_predictions.parquet is the inner-join of 4a and 3o's test_predictions parquets with p_wet = (p_4a + p_3o) / 2. PerLead.BlendTestMae reports Brier on that joined slice — matches the 2026-05-12 LightGBM-meta-learner bake-off finding (mean Brier 0.0830 vs best single 3o at 0.0844).",
                "No feature schema, no LightGBM/MLP/BART training step. Phase4bPredictCommand performs the same arithmetic on the live cycle's 4a + 3o prediction parquets — predict-and-render runs it inline between predict-all and the R2 push.",
                "PerLeadStats fields repurposed: BlendTestMae=Brier on joined test slice, BlendTestBias=mean(p − y), BlendTestRmse=0.0 (not meaningful), BestSingleTestMae=Brier of the better of 4a/3o alone on the same slice.",
                "Climatology copied from the station's 3o bundle — 4b targets the same wet/dry indicator at the same station, so the baseline is identical.",
            },
            PerLead = perLead,
        };
        await File.WriteAllTextAsync(Path.Combine(outDir, ModelArtifact.TrainingMetadataFileName),
            JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }), ct);

        // 4. feature_schema.json — one BlenderSpec per lead, two
        // synthetic "models" (p_4a, p_3o). Lets the Spec page render
        // 4b's card without bespoke handling.
        var leadInts = perLead.Keys.Select(int.Parse).OrderBy(x => x).ToArray();
        var schemaPerLead = leadInts.ToDictionary(
            L => L,
            L => new BlenderSpec
            {
                Target         = "precipitation",
                FeatureSet     = "phase4b-2way-mean",
                LeadHours      = L,
                RequiredModels = Array.Empty<string>(),
                OptionalModels = Array.Empty<string>(),
                Models         = new[] { "phase4a", "phase3o" },
                FeatureNames   = new[] { "p_4a", "p_3o" },
                DataSource     = BlenderDataSource.OpenMeteoPreviousRuns,  // closest match — 4b doesn't have its own
                Tier           = "4b-synth",
                UkvStrategy    = WeatherBlend.Train.Exact12h.Exact12hFeatureBuilder.UkvPickStrategy.Strict,
            });
        ModelArtifact.SaveBlenderSpecs(outDir, schemaPerLead);

        // 5. climatology.json — copy from 3o's bundle. Same station,
        // same wet indicator, identical baseline rate.
        var climSrc = Path.Combine(bundle3o, ModelArtifact.ClimatologyFileName);
        if (File.Exists(climSrc))
            File.Copy(climSrc, Path.Combine(outDir, ModelArtifact.ClimatologyFileName), overwrite: true);

        // 6. training_summary.json — minimal stub for RetrainGuard.
        // No real features → empty PerFeature dict. Row count = test
        // slice size. Label rate from the joined slice's observed_wet.
        var summary = new TrainingSummary
        {
            SchemaVersion     = "1",
            Composite         = $"precipitation/{station}",
            Phase             = Phase,
            Version           = version,
            LocationName      = location,
            ComputedAtUtc     = trainedAt,
            RowsTrain         = 0,
            RowsVal           = 0,
            RowsTest          = joined.Count,
            FeaturesEffective = 2,
            PerFeature        = new Dictionary<string, FeatureStats>(),
            LabelRates        = new Dictionary<string, double>
            {
                [station] = joined.Count(j => j.Observed) / (double)joined.Count,
            },
        };
        ModelArtifact.SaveTrainingSummary(outDir, summary);

        // 7. Promote into MANIFEST.Active — replaces any prior _phase4b
        // entry for this station (ComposeActive is lead-overlap-aware).
        ModelArtifact.PromoteStationVersion(modelsRoot, "precipitation", station, version, newPhase: Phase);

        return (MintOutcome.Minted, $"minted {version} ({joined.Count:N0} test rows, mean Brier {perLead.Values.Average(s => s.BlendTestMae):F4}); promoted to MANIFEST.Active");
    }

    /// <summary>
    /// Latest version dir under <paramref name="stationDir"/> that carries
    /// <paramref name="requiredFile"/>. <paramref name="phaseSuffix"/> null
    /// matches the suffix-less 3a champion dirs; a non-null suffix matches
    /// exactly (e.g. <c>_phase4a</c>). The required-file gate differs per
    /// caller: 4a/3o are resolved on <c>test_predictions.parquet</c> (the
    /// data 4b joins), 3a on <c>training_metadata.json</c> (4b only needs
    /// 3a for the location pin + a best-effort climatology copy).
    /// </summary>
    private static string? FindLatestBundle(string stationDir, string? phaseSuffix, string requiredFile)
    {
        if (!Directory.Exists(stationDir)) return null;
        var candidates = new List<string>();
        foreach (var d in Directory.GetDirectories(stationDir))
        {
            var name = Path.GetFileName(d);
            if (phaseSuffix is null)
            {
                if (name.Contains("phase")) continue;     // 3a champion has no phase suffix
            }
            else
            {
                if (!name.EndsWith(phaseSuffix, StringComparison.Ordinal)) continue;
            }
            if (File.Exists(Path.Combine(d, requiredFile))) candidates.Add(d);
        }
        return candidates.OrderByDescending(d => Path.GetFileName(d), StringComparer.Ordinal).FirstOrDefault();
    }

    private record JoinedRow(DateTime ValidTime, string Station, int Lead,
                             double PWet4a, double PWet3o, double PWet, bool Observed);

    private static List<JoinedRow> ReadAndJoinTestPredictions(string parquet4a, string parquet3o)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var p4a = parquet4a.Replace("\\", "/");
        var p3o = parquet3o.Replace("\\", "/");
        var sql = $@"
SELECT a.valid_time, a.station, a.lead,
       a.p_wet AS p_4a, e.p_wet AS p_3o, a.observed_wet
FROM read_parquet('{p4a}') a
INNER JOIN read_parquet('{p3o}') e
  ON a.valid_time = e.valid_time
 AND a.station    = e.station
 AND a.lead       = e.lead
ORDER BY a.valid_time, a.lead";
        var rows = new List<JoinedRow>();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            var vt   = r.GetDateTime(0);
            var st   = r.GetString(1);
            var lead = (int)r.GetInt64(2);
            var p4   = r.GetDouble(3);
            var p3   = r.GetDouble(4);
            var obs  = !r.IsDBNull(5) && r.GetInt32(5) > 0;
            rows.Add(new JoinedRow(vt, st, lead, p4, p3, (p4 + p3) / 2.0, obs));
        }
        return rows;
    }

    private static double Brier(double[] p, double[] y)
    {
        double sum = 0.0;
        for (int i = 0; i < p.Length; i++)
        {
            var d = p[i] - y[i];
            sum += d * d;
        }
        return sum / p.Length;
    }
}
