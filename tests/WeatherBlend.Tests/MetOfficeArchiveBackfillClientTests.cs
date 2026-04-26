using System;
using System.IO;
using System.Reflection;
using FluentAssertions;
using WeatherBlend.Collect;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the cross-platform Python resolver in MetOfficeArchiveBackfillClient.
/// Originally the wrapper hardcoded a Windows .venv path with a single
/// WEATHERBLEND_PYTHON env override; the GH-Actions integration of MO Global
/// into `collect` requires a Linux fallback (python3/python on PATH after
/// actions/setup-python@v5). These tests pin that resolver so the GH-runner
/// path doesn't silently regress.
/// </summary>
public class MetOfficeArchiveBackfillClientTests
{
    private static MethodInfo GetResolveMethod()
    {
        var t = typeof(MetOfficeArchiveBackfillClient);
        var m = t.GetMethod("ResolvePython", BindingFlags.NonPublic | BindingFlags.Static);
        if (m is null)
            throw new InvalidOperationException("ResolvePython not found — has it been renamed/removed?");
        return m;
    }

    private static string? Invoke(Func<string?> swap)
    {
        var prev = Environment.GetEnvironmentVariable("WEATHERBLEND_PYTHON");
        var prevPath = Environment.GetEnvironmentVariable("PATH");
        try
        {
            return swap();
        }
        finally
        {
            Environment.SetEnvironmentVariable("WEATHERBLEND_PYTHON", prev);
            Environment.SetEnvironmentVariable("PATH", prevPath);
        }
    }

    [Fact]
    public void ResolvePython_uses_env_var_when_set_and_path_exists(
        )
    {
        var fakeExe = Path.Combine(Path.GetTempPath(), $"fake_python_{Guid.NewGuid():N}.exe");
        File.WriteAllText(fakeExe, "");
        try
        {
            var result = Invoke(() =>
            {
                Environment.SetEnvironmentVariable("WEATHERBLEND_PYTHON", fakeExe);
                return (string?)GetResolveMethod().Invoke(null, null);
            });
            result.Should().Be(fakeExe);
        }
        finally
        {
            if (File.Exists(fakeExe)) File.Delete(fakeExe);
        }
    }

    [Fact]
    public void ResolvePython_ignores_env_var_when_path_does_not_exist()
    {
        // Falls through to next strategy (Windows venv → PATH probe). On a
        // typical dev/CI machine python3 or python is on PATH, so this should
        // resolve to something existing rather than throw.
        var nonexistent = Path.Combine(Path.GetTempPath(), $"definitely-not-here_{Guid.NewGuid():N}");
        var result = Invoke(() =>
        {
            Environment.SetEnvironmentVariable("WEATHERBLEND_PYTHON", nonexistent);
            try
            {
                return (string?)GetResolveMethod().Invoke(null, null);
            }
            catch (TargetInvocationException tie) when (tie.InnerException is FileNotFoundException)
            {
                // Acceptable on a totally bare machine without Python anywhere.
                return null;
            }
        });
        if (result is not null)
            File.Exists(result).Should().BeTrue("resolver returned a path; it must exist");
    }

    [Fact]
    public void ResolvePython_throws_clear_error_when_nothing_resolves()
    {
        // Force every fallback to fail: bad env var, scrub PATH so PATH probe
        // can't find python. The resolver should throw FileNotFoundException
        // with a hint about WEATHERBLEND_PYTHON / setup-python.
        // Skip on a dev machine that has the Windows venv at the hardcoded
        // path — we can't easily mock that out without changing the resolver.
        const string winDevVenv = @"C:\Projects\Weather\WeatherProbabilistic\.venv\Scripts\python.exe";
        if (File.Exists(winDevVenv))
        {
            // Hardcoded dev fallback wins; can't drive this test path.
            return;
        }
        var act = () => Invoke(() =>
        {
            Environment.SetEnvironmentVariable("WEATHERBLEND_PYTHON",
                Path.Combine(Path.GetTempPath(), "nonexistent_python"));
            Environment.SetEnvironmentVariable("PATH", "");
            try
            {
                return (string?)GetResolveMethod().Invoke(null, null);
            }
            catch (TargetInvocationException tie)
            {
                throw tie.InnerException ?? tie;
            }
        });
        act.Should().Throw<FileNotFoundException>()
            .WithMessage("*Python*");
    }
}
