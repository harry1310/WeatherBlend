using DuckDB.NET.Data;

namespace WeatherBlend.Commands;

// Throwaway: dumps a few columns of a forecast parquet for ad-hoc inspection.
public sealed class InspectCommand
{
    public Task<int> RunAsync(string path, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var glob = path.Replace('\\', '/');

        Console.WriteLine($"\nFile: {path}\n");

        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"SELECT COUNT(*) AS rows,
                                      COUNT(Precipitation) AS precip_non_null,
                                      SUM(CASE WHEN Precipitation > 0 THEN 1 ELSE 0 END) AS precip_hours,
                                      ROUND(SUM(Precipitation)::DOUBLE, 2) AS total_mm,
                                      ROUND(MAX(Precipitation)::DOUBLE, 2) AS max_mm
                               FROM read_parquet('{glob}', hive_partitioning = false)";
            using var r = c.ExecuteReader();
            r.Read();
            Console.WriteLine($"rows={r["rows"]}  precip_non_null={r["precip_non_null"]}  hours_with_precip={r["precip_hours"]}  total_mm={r["total_mm"]}  max_mm={r["max_mm"]}\n");
        }

        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"SELECT ValidTimeUtc,
                                      LeadHours,
                                      ROUND(Temperature2m::DOUBLE, 1) AS t2m,
                                      ROUND(Precipitation::DOUBLE, 2) AS precip_mm,
                                      ROUND(PrecipitationProbability::DOUBLE, 0) AS pop_pct,
                                      ROUND(WindSpeed10m::DOUBLE, 1) AS wind_ms,
                                      ROUND(CloudCover::DOUBLE, 0) AS cloud_pct
                               FROM read_parquet('{glob}', hive_partitioning = false)
                               ORDER BY ValidTimeUtc
                               LIMIT 18";
            using var r = c.ExecuteReader();
            Console.WriteLine($"{"valid_utc",-22} {"lead",4} {"t2m",6} {"precip",8} {"pop%",6} {"wind",6} {"cloud",6}");
            while (r.Read())
            {
                Console.WriteLine($"{r.GetValue(0),-22} {r.GetValue(1),4} {r.GetValue(2),6} {r.GetValue(3),8} {r.GetValue(4),6} {r.GetValue(5),6} {r.GetValue(6),6}");
            }
        }

        Console.WriteLine();

        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"SELECT ValidTimeUtc,
                                      LeadHours,
                                      ROUND(Precipitation::DOUBLE, 2) AS precip_mm,
                                      ROUND(PrecipitationProbability::DOUBLE, 0) AS pop_pct
                               FROM read_parquet('{glob}', hive_partitioning = false)
                               WHERE Precipitation > 0
                               ORDER BY ValidTimeUtc";
            using var r = c.ExecuteReader();
            Console.WriteLine("Hours with precipitation > 0:");
            Console.WriteLine($"  {"valid_utc",-22} {"lead",4} {"precip_mm",10} {"pop%",6}");
            int n = 0;
            while (r.Read())
            {
                Console.WriteLine($"  {r.GetValue(0),-22} {r.GetValue(1),4} {r.GetValue(2),10} {r.GetValue(3),6}");
                n++;
            }
            if (n == 0) Console.WriteLine("  (none)");
        }

        return Task.FromResult(0);
    }
}
