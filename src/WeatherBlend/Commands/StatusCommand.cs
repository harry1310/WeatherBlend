using DuckDB.NET.Data;
using Microsoft.Extensions.Logging;
using WeatherBlend.Config;

namespace WeatherBlend.Commands;

/// <summary>
/// Quick "what do I have?" summary: row counts per model, date coverage,
/// observation count. Handy to run after a backfill or after the cron has
/// been ticking over for a while.
/// </summary>
public sealed class StatusCommand
{
    private readonly AppConfig _cfg;
    private readonly ILogger<StatusCommand> _log;

    public StatusCommand(AppConfig cfg, ILogger<StatusCommand> log)
    {
        _cfg = cfg;
        _log = log;
    }

    public Task<int> RunAsync(CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();

        Console.WriteLine();
        Console.WriteLine($"Location: {_cfg.Location.DisplayName}");
        Console.WriteLine(new string('-', 60));

        var fcGlob = Path.Combine(_cfg.Storage.ForecastsPath, "**", "*.parquet").Replace('\\', '/');
        if (Directory.Exists(_cfg.Storage.ForecastsPath))
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = $@"
                SELECT Model AS model,
                       COUNT(*) AS rows,
                       MIN(ValidTimeUtc) AS earliest,
                       MAX(ValidTimeUtc) AS latest
                FROM read_parquet('{fcGlob}', hive_partitioning = false, union_by_name = true)
                GROUP BY Model
                ORDER BY Model";
            using var r = cmd.ExecuteReader();
            Console.WriteLine("Forecasts:");
            Console.WriteLine($"  {"model",-25} {"rows",10} {"earliest",-22} {"latest",-22}");
            while (r.Read())
            {
                Console.WriteLine($"  {r.GetString(0),-25} {r.GetInt64(1),10} {r.GetValue(2),-22} {r.GetValue(3),-22}");
            }
        }
        else
        {
            Console.WriteLine("Forecasts: (no data yet)");
        }

        Console.WriteLine();
        var obsGlob = Path.Combine(_cfg.Storage.ObservationsPath, "**", "*.parquet").Replace('\\', '/');
        if (Directory.Exists(_cfg.Storage.ObservationsPath))
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = $@"
                SELECT Station AS station,
                       COUNT(*) AS rows,
                       MIN(ObservedTimeUtc) AS earliest,
                       MAX(ObservedTimeUtc) AS latest,
                       SUM(CASE WHEN HasPrecipitation THEN 1 ELSE 0 END) AS precip_obs
                FROM read_parquet('{obsGlob}', hive_partitioning = false, union_by_name = true)
                GROUP BY Station
                ORDER BY Station";
            using var r = cmd.ExecuteReader();
            Console.WriteLine("Observations:");
            Console.WriteLine($"  {"station",-10} {"rows",8} {"earliest",-22} {"latest",-22} {"precip",8}");
            while (r.Read())
            {
                Console.WriteLine($"  {r.GetString(0),-10} {r.GetInt64(1),8} {r.GetValue(2),-22} {r.GetValue(3),-22} {r.GetInt64(4),8}");
            }
        }
        else
        {
            Console.WriteLine("Observations: (no data yet)");
        }

        Console.WriteLine();
        return Task.FromResult(0);
    }
}
