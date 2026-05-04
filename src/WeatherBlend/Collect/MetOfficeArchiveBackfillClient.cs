using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Logging;

namespace WeatherBlend.Collect;

/// <summary>
/// Thin subprocess wrapper around the Met Office Python backfill scripts
/// (<c>scripts/met_office_archive_backfill.py</c> for the global 10km dataset
/// and <c>scripts/met_office_ukv_archive_backfill.py</c> for the UKV 2km
/// regional dataset). Mirrors the <see cref="Wgrib2"/> pattern — we shell out
/// because the file format (NetCDF4/HDF5 on either a global 0.1° grid or a
/// 2km Lambert Azimuthal projection) has no mature .NET reader. xarray +
/// h5netcdf in a Python venv is the path of least resistance and matches
/// what every other tool in the meteo space does for this dataset.
///
/// The Python script does the actual work: anonymous S3 GET against
/// <c>met-office-atmospheric-model-data</c>, single-cell extraction at
/// Bonehill, write to WeatherBlend's hive forecast tree under the script's
/// own model id. The wrapper streams stdout/stderr to the logger so progress
/// is visible during the long backfill (~15s/cycle × 250+ cycles = 60+
/// minutes typical for global; UKV is similar).
///
/// Both scripts accept the same CLI surface (--start/--end/--cycles/--leads/
/// --parallelism/--min-cycle-age-hours) so this single wrapper serves both —
/// the caller picks the script name per dataset.
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
        string scriptName,
        DateOnly start, DateOnly end,
        IReadOnlyList<int> cycles, IReadOnlyList<int> leads,
        int parallelism, int minCycleAgeHours, CancellationToken ct)
    {
        var python = ResolvePython();
        var scriptPath = ResolveScriptPath(scriptName);

        if (!File.Exists(scriptPath))
            throw new FileNotFoundException(
                $"Met Office backfill script not found at {scriptPath}.", scriptPath);

        // Build args list — only include --cycles / --leads if the caller
        // supplied a non-empty set. Otherwise the script picks up its own
        // source-native defaults (passing an empty string would override
        // those defaults with literal "" and the script would fail to parse).
        var argList = new List<string>
        {
            "-u", scriptPath,
            "--start", start.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            "--end",   end.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            "--parallelism", parallelism.ToString(CultureInfo.InvariantCulture),
            "--min-cycle-age-hours", minCycleAgeHours.ToString(CultureInfo.InvariantCulture),
        };
        if (cycles.Count > 0) { argList.Add("--cycles"); argList.Add(string.Join(",", cycles)); }
        if (leads.Count  > 0) { argList.Add("--leads");  argList.Add(string.Join(",", leads)); }
        var args = argList.ToArray();

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

    /// <summary>Resolve a usable Python interpreter. Order:
    ///   1. WEATHERBLEND_PYTHON env var if set and pointing at a real file.
    ///   2. Local dev convenience — the WeatherProbabilistic venv on this box.
    ///   3. `python3` then `python` from PATH (CI / Linux runners after
    ///      actions/setup-python@v5 — both names are usually exposed).</summary>
    private static string ResolvePython()
    {
        var fromEnv = Environment.GetEnvironmentVariable("WEATHERBLEND_PYTHON");
        if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
            return fromEnv;

        const string winDevVenv = @"C:\Projects\Weather\WeatherProbabilistic\.venv\Scripts\python.exe";
        if (File.Exists(winDevVenv))
            return winDevVenv;

        foreach (var name in new[] { "python3", "python" })
        {
            var resolved = TryResolveOnPath(name);
            if (resolved is not null) return resolved;
        }

        throw new FileNotFoundException(
            "No Python interpreter found. Set WEATHERBLEND_PYTHON, or install python3/python on PATH " +
            "(GH Actions: use actions/setup-python@v5).");
    }

    private static string? TryResolveOnPath(string exeName)
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(path)) return null;
        var sep = Path.PathSeparator;
        var exts = OperatingSystem.IsWindows()
            ? new[] { ".exe", ".bat", ".cmd", "" }
            : new[] { "" };
        foreach (var dir in path.Split(sep))
        {
            if (string.IsNullOrEmpty(dir)) continue;
            foreach (var ext in exts)
            {
                var candidate = Path.Combine(dir, exeName + ext);
                if (File.Exists(candidate)) return candidate;
            }
        }
        return null;
    }

    private static string ResolveScriptPath(string scriptName)
    {
        var explicitRoot = Environment.GetEnvironmentVariable("WEATHERBLEND_REPO");
        if (!string.IsNullOrEmpty(explicitRoot))
            return Path.Combine(explicitRoot, "scripts", scriptName);

        // Walk up from AppContext.BaseDirectory (typically
        // <repo>/src/WeatherBlend/bin/Debug/net10.0) until we find a "scripts"
        // sibling. Same pattern GfsBackfillCommand uses to find its scratch dir.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "scripts", scriptName);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return Path.Combine("scripts", scriptName);
    }
}
