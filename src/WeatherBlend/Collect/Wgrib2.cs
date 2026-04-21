using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Thin subprocess wrapper around NOAA's wgrib2. We use it for two things:
///   1. Listing messages in a GRIB2 file (wgrib2 &lt;file&gt; -s).
///   2. Extracting a single point value at a given lat/lon from every message in
///      a file (wgrib2 &lt;file&gt; -lon &lt;lon&gt; &lt;lat&gt;).
///
/// We shell out rather than binding a library because there is no mature GRIB2
/// decoder in .NET — see docs/LEAD_TIME_BACKFILL.md for the full analysis.
/// </summary>
public sealed class Wgrib2
{
    private readonly string _exe;
    private readonly ILogger<Wgrib2> _log;

    public Wgrib2(string exePath, ILogger<Wgrib2> log)
    {
        _exe = exePath;
        _log = log;
    }

    /// <summary>
    /// Returns (index, offset, trailer) triples for every message in the file.
    /// The trailer is the `:`-separated identifier (variable:level:forecast).
    /// </summary>
    public async Task<List<(int Index, long Offset, string Trailer)>> ListAsync(
        string gribPath, CancellationToken ct)
    {
        var lines = await RunAsync(new[] { gribPath, "-s" }, ct);
        var result = new List<(int, long, string)>(lines.Count);
        foreach (var line in lines)
        {
            // Format: "<n>:<offset>:d=<date>:<trailer>"
            var parts = line.Split(':', 4);
            if (parts.Length < 4) continue;
            if (!int.TryParse(parts[0], out var n)) continue;
            if (!long.TryParse(parts[1], out var off)) continue;
            result.Add((n, off, parts[3]));
        }
        return result;
    }

    /// <summary>
    /// Runs `wgrib2 &lt;grib&gt; -lon &lt;lon&gt; &lt;lat&gt;`. Returns one (messageIndex, value)
    /// per message in the file — wgrib2 finds the nearest grid point and prints
    /// `<i>:<offset>:lon=X,lat=Y,val=Z`.
    /// </summary>
    public async Task<List<(int Index, double Value)>> ExtractPointAsync(
        string gribPath, double latitude, double longitude, CancellationToken ct)
    {
        var args = new[]
        {
            gribPath,
            "-lon",
            longitude.ToString("F5", CultureInfo.InvariantCulture),
            latitude.ToString("F5", CultureInfo.InvariantCulture),
        };
        var lines = await RunAsync(args, ct);
        var result = new List<(int, double)>(lines.Count);
        foreach (var line in lines)
        {
            // "1:0:lon=356.250000,lat=50.500000,val=278.127"
            var parts = line.Split(':');
            if (parts.Length < 3) continue;
            if (!int.TryParse(parts[0], out var n)) continue;

            var valIdx = line.LastIndexOf("val=", StringComparison.Ordinal);
            if (valIdx < 0) continue;
            var valStr = line[(valIdx + 4)..];
            if (double.TryParse(valStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                result.Add((n, v));
        }
        return result;
    }

    private async Task<List<string>> RunAsync(string[] args, CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _exe,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);

        using var proc = new Process { StartInfo = psi };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        proc.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) stderr.AppendLine(e.Data); };

        proc.Start();
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync(ct);

        if (proc.ExitCode != 0)
        {
            _log.LogWarning("wgrib2 args=[{Args}] exit={Exit} stderr={Err}",
                string.Join(' ', args), proc.ExitCode, stderr.ToString().Trim());
            throw new InvalidOperationException(
                $"wgrib2 exited {proc.ExitCode}: {stderr.ToString().Trim()}");
        }

        return stdout.ToString()
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();
    }
}
