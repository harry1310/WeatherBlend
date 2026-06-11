using Microsoft.Extensions.Logging;
using Parquet.Serialization;
using WeatherBlend.Config;
using WeatherBlend.Models;
using WeatherBlend.Train;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Commands;

/// <summary>
/// Dump the C#-built rich-oro feature rows for one (station, lead) to a
/// parquet + FeatureNames sidecar — the reference fixture for
/// WeatherProbabilistic's <c>test_rich_oro_python_vs_csharp.py</c>
/// bit-parity test.
///
/// Why this is a COMMITTED command (2026-06-11): the rich-oro spec exists
/// twice — here (feeds 3o) and in WP's <c>_shared.py</c> (feeds 4a) — and
/// the parity test is the only enforcement that the two stay bit-identical.
/// The original 2026-05-29 dump came from bake-off scratch code that never
/// landed, so when the data tree moved under it (rolling previous-runs
/// refresh + the GEM-freeze recovery) the fixture rotted with no way to
/// regenerate it, and the contract went unenforced. Regenerate with:
///
///   dotnet run -- dump-oro-features
///   (defaults: Bellever Dartmoor, lead 24, min-valid 2024-01-01 — the
///    canonical fixture cell)
///
/// then re-run the WP test in the SAME data-tree state (no noon refresh
/// between dump and test, or the comparison is across two tree states —
/// exactly the rot this command exists to recover from).
///
/// NO upper-air: the Python twin mirrors the 68-feature (59 rich + 9
/// terrain) pre-UA spec that 4a consumes; 3o's production 110-feature
/// UA spec is C#-only and out of the parity contract's scope.
/// </summary>
public sealed class DumpOroFeaturesCommand
{
    private readonly ILogger<DumpOroFeaturesCommand> _log;
    private readonly AppConfig _cfg;

    public DumpOroFeaturesCommand(ILogger<DumpOroFeaturesCommand> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    /// <summary>Parquet row shape the WP test reads: ValidTimeUtc + the
    /// packed feature vector (Label included for completeness).</summary>
    public sealed class OroDumpRow
    {
        public DateTime ValidTimeUtc { get; set; }
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }

    public async Task<int> RunAsync(
        string stationFriendly, int lead, string minValidTimeStr, int stationIndex, CancellationToken ct)
    {
        var minValid = DateTime.SpecifyKind(DateTime.Parse(minValidTimeStr), DateTimeKind.Utc);

        // Resolve the station's home location from config (the dump cell is
        // Bonehill's Bellever by convention, but any configured rainfall
        // station works).
        var location = _cfg.Locations.FirstOrDefault(l =>
            l.Rainfall.Stations.Any(s => s.Name.Equals(stationFriendly, StringComparison.OrdinalIgnoreCase)));
        if (location is null)
        {
            _log.LogError("No configured location has rainfall station '{Station}'.", stationFriendly);
            return 2;
        }
        var slug = StationSlug.WithEaPrefix(stationFriendly);

        // stationIndex MUST match PrecipTrainCommand.BonehillStationOrder3o's
        // position for the station (Bellever=0, Bovey=1, Hexworthy=2,
        // Princetown=3) — it is packed into oro_station_id and the Python
        // twin passes the same index. Default 0 = Bellever.
        var oroRoot = Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "static", "orographic");
        var oroBySlug = OroStaticFeatures.LoadAll(oroRoot);
        if (!oroBySlug.TryGetValue(slug, out var oro))
        {
            _log.LogError("No orographic record for {Slug} under {Root}.", slug, oroRoot);
            return 2;
        }

        // 68-feature no-UA rich-oro spec + its embedded rich spec — the same
        // construction PrecipRichOroFeatureBuilder.BuildForLead performs.
        var richOroSpec = PrecipRichOroFeatureBuilder.BuildSpec(_cfg.Blenders, lead, withUpperAir: false);
        var richSpec = new Train.Common.BlenderSpec
        {
            Target = richOroSpec.Target,
            FeatureSet = PrecipRichFeatureBuilder.SpecFeatureSet,
            LeadHours = richOroSpec.LeadHours,
            RequiredModels = richOroSpec.RequiredModels,
            OptionalModels = richOroSpec.OptionalModels,
            Models = richOroSpec.Models,
            FeatureNames = richOroSpec.FeatureNames
                .Take(richOroSpec.FeatureNames.Count - PrecipRichOroFeatureBuilder.TerrainFeatureCount).ToList(),
            DataSource = richOroSpec.DataSource,
            Tier = PrecipRichFeatureBuilder.SpecFeatureSet,
            UkvStrategy = richOroSpec.UkvStrategy,
        };

        // Unlike 3o training (min_valid exempt), the parity fixture is built
        // WITH the 2024-01-01 floor — the Python twin passes the same value.
        _log.LogInformation("dump-oro-features — {Station} (idx {Idx}) lead {Lead}h, minValid {Min:yyyy-MM-dd}, {N} features",
            stationFriendly, stationIndex, lead, minValid, richOroSpec.FeatureCount);
        var richRows = PrecipRichFeatureBuilder.BuildForLead(
            _cfg.Storage.ForecastsPath, _cfg.Storage.RainfallPath, location.Name,
            stationFriendly, richSpec, minValidTime: minValid, ct: ct);
        if (richRows.Count == 0)
        {
            _log.LogError("Rich builder returned 0 rows — check the local data tree.");
            return 3;
        }
        var aux = PrecipRichOroFeatureBuilder.LoadAuxNwpMeans(
            _cfg.Storage.ForecastsPath, location.Name, richSpec, ct);

        var richDim = richSpec.FeatureCount;
        var rows = new List<OroDumpRow>(richRows.Count);
        foreach (var rr in richRows)
        {
            ct.ThrowIfCancellationRequested();
            if (!aux.TryGetValue(rr.ValidTimeUtc, out var a)) continue;
            var terrain = PrecipRichOroFeatureBuilder.ComposeTerrainBlock(oro, a, stationIndex);
            var features = new float[richOroSpec.FeatureCount];
            Array.Copy(rr.Features, features, richDim);
            for (int i = 0; i < terrain.Length; i++) features[richDim + i] = terrain[i];
            rows.Add(new OroDumpRow { ValidTimeUtc = rr.ValidTimeUtc, Features = features, Label = rr.Label ? 1f : 0f });
        }

        var outDir = Path.Combine(
            Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!, "scratch", "oro_dump");
        Directory.CreateDirectory(outDir);
        var stem = $"{slug}_lead{lead}h_rich-oro";
        var parquetPath = Path.Combine(outDir, $"{stem}.parquet");
        using (var fs = File.Create(parquetPath))
            await ParquetSerializer.SerializeAsync(rows, fs, cancellationToken: ct);

        var sidecar = System.Text.Json.JsonSerializer.Serialize(new
        {
            FeatureNames = richOroSpec.FeatureNames,
            Station = slug,
            StationIndex = stationIndex,
            LeadHours = lead,
            MinValidTime = minValid.ToString("yyyy-MM-dd"),
            DumpedAtUtc = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"),
            Note = "Reference fixture for WP test_rich_oro_python_vs_csharp; regenerate via dotnet run -- dump-oro-features",
        }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(Path.Combine(outDir, $"{stem}.schema.json"), sidecar, ct);

        _log.LogInformation("Dumped {N} rows × {F} features → {Path}", rows.Count, richOroSpec.FeatureCount, parquetPath);
        return 0;
    }
}
