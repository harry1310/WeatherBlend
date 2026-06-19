using Microsoft.Extensions.Logging;
using WeatherBlend.Collect;
using WeatherBlend.Config;
using WeatherBlend.Storage;

namespace WeatherBlend.Commands;

/// <summary>
/// Historical Davis WeatherLink backfill. For each configured location with a
/// <c>weatherLinkStationId</c> (or the one named via <c>--location</c>), pages
/// the WeatherLink /historic archive in 24-HOUR windows from <c>--start</c> to
/// <c>--end</c> (inclusive), aggregates each window's 15-minute records to
/// hourly truth, and writes per-day parquet under
/// <c>data/truth/weatherlink/location=&lt;name&gt;/date=&lt;yyyy-MM-dd&gt;/observations.parquet</c>.
///
/// WeatherLink caps each /historic request to 24h, so the window size is fixed.
/// A small polite delay sits between requests to be gentle on the API.
/// </summary>
public sealed class WeatherLinkBackfillCommand
{
    // Be polite to the API between 24h windows. WeatherLink's published rate
    // limit is generous, but a backfill is the one place we hammer it.
    private static readonly TimeSpan RequestDelay = TimeSpan.FromSeconds(1.5);

    private readonly AppConfig _cfg;
    private readonly WeatherLinkClient _client;
    private readonly ILogger<WeatherLinkBackfillCommand> _log;

    public WeatherLinkBackfillCommand(
        AppConfig cfg,
        WeatherLinkClient client,
        ILogger<WeatherLinkBackfillCommand> log)
    {
        _cfg = cfg;
        _client = client;
        _log = log;
    }

    public async Task<int> RunAsync(DateOnly start, DateOnly end, string? location, CancellationToken ct)
    {
        if (end < start)
        {
            _log.LogError("weatherlink-backfill: --end {End} is before --start {Start}", end, start);
            return 2;
        }

        var wlCfg = _cfg.WeatherLink;
        var creds = wlCfg.Enabled
            ? WeatherLinkSecrets.TryLoad(wlCfg.ApiKeyEnvVar, wlCfg.ApiSecretEnvVar, wlCfg.KeyFile)
            : null;
        if (creds is null)
        {
            _log.LogError(
                "weatherlink-backfill: no credentials ({KeyEnv}/{SecretEnv} or {File}){Disabled}. "
                + "Set the env vars or create the key file (key=… / secret=… lines).",
                wlCfg.ApiKeyEnvVar, wlCfg.ApiSecretEnvVar, wlCfg.KeyFile,
                wlCfg.Enabled ? "" : " — WeatherLink also disabled in config");
            return 2;
        }

        // Resolve locations: every one with a station id, or the single named one.
        var all = _cfg.Locations.Count > 0 ? _cfg.Locations : new List<LocationConfig> { _cfg.Location };
        IEnumerable<LocationConfig> candidates = all;
        if (!string.IsNullOrWhiteSpace(location))
        {
            var match = all.FirstOrDefault(l => string.Equals(l.Name, location, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                _log.LogError("--location '{L}' not in configured locations. Available: [{Avail}]",
                    location, string.Join(", ", all.Select(l => l.Name)));
                return 2;
            }
            candidates = new[] { match };
        }

        var targets = candidates
            .Where(l => !string.IsNullOrWhiteSpace(l.WeatherLinkStationId))
            .ToList();
        if (targets.Count == 0)
        {
            _log.LogWarning("weatherlink-backfill: no configured location has a weatherLinkStationId — nothing to do.");
            return 0;
        }

        _log.LogInformation("weatherlink-backfill {Start:yyyy-MM-dd}..{End:yyyy-MM-dd} — {N} station(s): [{Names}]",
            start, end, targets.Count,
            string.Join(", ", targets.Select(l => $"{l.Name}={l.WeatherLinkStationId}")));

        var errors = 0;
        foreach (var loc in targets)
        {
            _log.LogInformation("weatherlink-backfill: location={Name} station={Station}",
                loc.Name, loc.WeatherLinkStationId);

            // Page one 24h window per UTC day [00:00, next 00:00).
            var cursor = start;
            while (cursor <= end)
            {
                ct.ThrowIfCancellationRequested();
                var windowStart = new DateTimeOffset(cursor.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc));
                var windowEnd = windowStart.AddDays(1);
                try
                {
                    var rows = await _client.FetchHistoricAsync(
                        loc.Name, loc.WeatherLinkStationId!,
                        windowStart, windowEnd, creds.Value.Key, creds.Value.Secret, ct);
                    await ParquetWriter.WriteWeatherLinkAsync(_cfg.Storage.WeatherLinkPath, rows, ct);
                    _log.LogInformation("  weatherlink/{Name} {Date:yyyy-MM-dd}: wrote {Rows} hourly rows",
                        loc.Name, cursor, rows.Count);
                }
                catch (Exception ex)
                {
                    errors++;
                    _log.LogError(ex, "  weatherlink/{Name} {Date:yyyy-MM-dd} FAILED", loc.Name, cursor);
                }
                await Task.Delay(RequestDelay, ct);
                cursor = cursor.AddDays(1);
            }
        }

        return errors == 0 ? 0 : 1;
    }
}
