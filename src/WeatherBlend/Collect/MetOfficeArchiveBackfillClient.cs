using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Thin subprocess wrapper around the Python backfill script
/// <c>scripts/met_office_archive_backfill.py</c>. Mirrors the
/// <see cref="Wgrib2"/> pattern — we shell out because the file format
/// (NetCDF4/HDF5 with 47 surface variables on a global 0.1° grid) has no
/// mature .NET reader. xarray + h5netcdf in a Python venv is the path of
/// least resistance and matches what every other tool in the meteo space
/// does for this dataset.
///
/// The Python script does the actual work: anonymous S3 GET against
/// <c>met-office-atmospheric-model-data</c>, single-cell extraction at
/// Bonehill, write to WeatherBlend's hive forecast tree as
/// <c>model=met_office_global</c>. The wrapper streams stdout/stderr to
/// the logger so progress is visible during the long backfill (~15s/cycle
/// × 250+ cycles = 60+ minutes typical).
///
/// Configuration (env vars):
///   WEATHERBLEND_PYTHON  — interpreter to use. Defaults to a sensible path
///                          on this dev box; override per-environment.
///   WEATHERBLEND_REPO    — repo root used to locate the script. Defaults to
///                          AppContext.BaseDirectory walked up to the repo root.
/// </summary>
public sealed class MetOfficeArchiveBackfillClient
{
    private readonly ILogger<MetOfficeArchiveBackfillClient> _log;

    public MetOfficeArchiveBackfillClient(ILogger<MetOfficeArchiveBackfillClient> log)
    {
        _log = log;
    }

    public async Task<int> RunAsync(
        DateOnly start, DateOnly end,
        IReadOnlyList<int> cycles, IReadOnlyList<int> leads,
        int parallelism, CancellationToken ct)
    {
        var python = Environment.GetEnvironmentVariable("WEATHERBLEND_PYTHON")
                     ?? @"C:\Projects\Weather\WeatherProbabilistic\.venv\Scripts\python.exe";
        var scriptPath = ResolveScriptPath();

        if (!File.Exists(python))
            throw new FileNotFoundException(
                $"Python interpreter not found at {python}. Set WEATHERBLEND_PYTHON env var.", python);
        if (!File.Exists(scriptPath))
            throw new FileNotFoundException(
                $"Met Office backfill script not found at {scriptPath}.", scriptPath);

        var args = new[]
        {
            "-u", scriptPath,
            "--start", start.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            "--end",   end.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            "--cycles", string.Join(",", cycles),
            "--leads",  string.Join(",", leads),
            "--parallelism", parallelism.ToString(CultureInfo.InvariantCulture),
        };

        _log.LogInformation(
            "met-office-archive-backfill: {Python} {Args}",
            python, string.Join(' ', args));

        var psi = new ProcessStartInfo
        {
            FileName = python,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        // PYTHONIOENCODING=utf-8 keeps the Python side happy on Windows where
        // cp1252 trips up on the script's logging output.
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        foreach (var a in args) psi.ArgumentList.Add(a);

        using var proc = new Process { StartInfo = psi };
        proc.OutputDataReceived += (_, e) => { if (e.Data is not null) _log.LogInformation("py> {Line}", e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) _log.LogWarning("py! {Line}", e.Data); };
        proc.Start();
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync(ct);

        if (proc.ExitCode != 0)
            _log.LogError("met-office-archive-backfill exited {Exit}", proc.ExitCode);
        else
            _log.LogInformation("met-office-archive-backfill done (exit 0)");

        return proc.ExitCode;
    }

    private static string ResolveScriptPath()
    {
        var explicitRoot = Environment.GetEnvironmentVariable("WEATHERBLEND_REPO");
        if (!string.IsNullOrEmpty(explicitRoot))
            return Path.Combine(explicitRoot, "scripts", "met_office_archive_backfill.py");

        // Walk up from AppContext.BaseDirectory (typically
        // <repo>/src/WeatherBlend/bin/Debug/net10.0) until we find a "scripts"
        // sibling. Same pattern GfsBackfillCommand uses to find its scratch dir.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "scripts", "met_office_archive_backfill.py");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return Path.Combine("scripts", "met_office_archive_backfill.py");
    }
}
