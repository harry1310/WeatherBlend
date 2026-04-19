using DuckDB.NET.Data;

namespace WeatherBlend.Commands;

// Cross-model agreement view: pivots forecasts by model for one run cycle.
public sealed class CompareCommand
{
    public Task<int> RunAsync(string glob, CancellationToken ct)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        var g = glob.Replace('\\', '/');

        Console.WriteLine($"\nGlob: {glob}\n");

        // Per-model summary across the whole forecast horizon.
        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"
                SELECT Model,
                       COUNT(*) AS rows,
                       ROUND(MIN(Temperature2m)::DOUBLE, 1) AS t_min,
                       ROUND(AVG(Temperature2m)::DOUBLE, 1) AS t_avg,
                       ROUND(MAX(Temperature2m)::DOUBLE, 1) AS t_max,
                       SUM(CASE WHEN Precipitation > 0 THEN 1 ELSE 0 END) AS wet_hrs,
                       ROUND(SUM(Precipitation)::DOUBLE, 2) AS total_mm,
                       ROUND(MAX(Precipitation)::DOUBLE, 2) AS max_mm
                FROM read_parquet('{g}', hive_partitioning = false, union_by_name = true)
                WHERE LeadHours >= 0
                GROUP BY Model
                ORDER BY Model";
            using var r = c.ExecuteReader();
            Console.WriteLine("Per-model summary (lead >= 0):");
            Console.WriteLine($"  {"model",-22} {"rows",5} {"t_min",6} {"t_avg",6} {"t_max",6} {"wet_h",6} {"tot_mm",8} {"max_mm",8}");
            while (r.Read())
            {
                Console.WriteLine($"  {r.GetValue(0),-22} {r.GetValue(1),5} {r.GetValue(2),6} {r.GetValue(3),6} {r.GetValue(4),6} {r.GetValue(5),6} {r.GetValue(6),8} {r.GetValue(7),8}");
            }
        }

        // Side-by-side temperature at every 6h lead.
        Console.WriteLine();
        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"
                PIVOT (
                    SELECT LeadHours, Model, ROUND(Temperature2m::DOUBLE, 1) AS t
                    FROM read_parquet('{g}', hive_partitioning = false, union_by_name = true)
                    WHERE LeadHours >= 0 AND LeadHours % 6 = 0 AND LeadHours <= 72
                ) ON Model USING FIRST(t) GROUP BY LeadHours ORDER BY LeadHours";
            using var r = c.ExecuteReader();
            Console.WriteLine("Temperature (°C) by lead, every 6h to 72h:");
            var headers = new List<string> { r.GetName(0) };
            for (int i = 1; i < r.FieldCount; i++) headers.Add(r.GetName(i));
            Console.WriteLine("  " + string.Join("  ", headers.Select((h, i) => i == 0 ? h.PadLeft(4) : h.PadLeft(10))));
            while (r.Read())
            {
                var cells = new List<string> { r.GetValue(0).ToString()!.PadLeft(4) };
                for (int i = 1; i < r.FieldCount; i++) cells.Add((r.IsDBNull(i) ? "-" : r.GetValue(i).ToString())!.PadLeft(10));
                Console.WriteLine("  " + string.Join("  ", cells));
            }
        }

        // Side-by-side precipitation totals per 24h window per model.
        Console.WriteLine();
        using (var c = conn.CreateCommand())
        {
            c.CommandText = $@"
                PIVOT (
                    SELECT CAST(LeadHours / 24 AS INTEGER) AS day_ahead, Model,
                           ROUND(SUM(Precipitation)::DOUBLE, 2) AS mm
                    FROM read_parquet('{g}', hive_partitioning = false, union_by_name = true)
                    WHERE LeadHours >= 0 AND LeadHours < 168
                    GROUP BY day_ahead, Model
                ) ON Model USING FIRST(mm) GROUP BY day_ahead ORDER BY day_ahead";
            using var r = c.ExecuteReader();
            Console.WriteLine("Precipitation total (mm) per 24h window per model:");
            var headers = new List<string> { r.GetName(0) };
            for (int i = 1; i < r.FieldCount; i++) headers.Add(r.GetName(i));
            Console.WriteLine("  " + string.Join("  ", headers.Select((h, i) => i == 0 ? h.PadLeft(4) : h.PadLeft(10))));
            while (r.Read())
            {
                var cells = new List<string> { r.GetValue(0).ToString()!.PadLeft(4) };
                for (int i = 1; i < r.FieldCount; i++) cells.Add((r.IsDBNull(i) ? "-" : r.GetValue(i).ToString())!.PadLeft(10));
                Console.WriteLine("  " + string.Join("  ", cells));
            }
        }

        Console.WriteLine();
        return Task.FromResult(0);
    }
}
