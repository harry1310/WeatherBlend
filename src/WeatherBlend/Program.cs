using System.CommandLine;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.Extensions.Logging;
using Serilog;
using WeatherBlend.Collect;
using WeatherBlend.Commands;
using WeatherBlend.Config;

namespace WeatherBlend;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        // Config + DI host
        var host = Host.CreateDefaultBuilder()
            .ConfigureAppConfiguration((ctx, builder) =>
            {
                var configPath = Environment.GetEnvironmentVariable("WEATHERBLEND_CONFIG")
                                 ?? Path.Combine(AppContext.BaseDirectory, "config.yaml");
                builder.AddYamlFile(configPath, optional: false, reloadOnChange: false);
            })
            .UseSerilog((ctx, lc) => lc
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File("logs/weatherblend-.log", rollingInterval: RollingInterval.Day))
            .ConfigureServices((ctx, services) =>
            {
                var cfg = new AppConfig();
                ctx.Configuration.Bind(cfg);
                services.AddSingleton(cfg);

                services.AddHttpClient<OpenMeteoClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(cfg.Http.TimeoutSeconds);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler(opts =>
                {
                    // Default 10s attempt / 30s total is too tight for meteofrance_seamless
                    // historical chunks — slow archive responses were timing out on backfill.
                    // Widened with extra headroom; SamplingDuration must be ≥ 2×AttemptTimeout.
                    opts.AttemptTimeout.Timeout = TimeSpan.FromSeconds(60);
                    opts.TotalRequestTimeout.Timeout = TimeSpan.FromSeconds(180);
                    opts.CircuitBreaker.SamplingDuration = TimeSpan.FromSeconds(240);
                });

                services.AddHttpClient<MetarClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(cfg.Http.TimeoutSeconds);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler();

                services.AddHttpClient<Era5Client>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(cfg.Http.TimeoutSeconds);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler();

                services.AddHttpClient<EaHydrologyClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(cfg.Http.TimeoutSeconds);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler();

                // OGIMET: longer timeout, no aggressive retry — the rate limit means
                // a hammered retry loop is the fastest way to get the IP blocked.
                services.AddHttpClient<OgimetClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(120);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                });

                // GFS S3 archive — large Range-downloads per cycle, needs long timeouts
                // and resilience against transient S3 errors.
                services.AddHttpClient<GfsClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(120);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler(opts =>
                {
                    opts.AttemptTimeout.Timeout = TimeSpan.FromSeconds(60);
                    opts.TotalRequestTimeout.Timeout = TimeSpan.FromSeconds(180);
                    opts.CircuitBreaker.SamplingDuration = TimeSpan.FromSeconds(240);
                });

                services.AddSingleton<Wgrib2>(sp =>
                {
                    var exe = Environment.GetEnvironmentVariable("WEATHERBLEND_WGRIB2")
                              ?? @"C:\Tools\wgrib2\wgrib2.exe";
                    return new Wgrib2(exe, sp.GetRequiredService<ILogger<Wgrib2>>());
                });

                services.AddTransient<CollectCommand>();
                services.AddTransient<BackfillCommand>();
                services.AddTransient<GfsBackfillCommand>();
                services.AddTransient<StatusCommand>();
                services.AddTransient<TrainCommand>();
                services.AddTransient<EvaluateCommand>();
                services.AddTransient<InspectCommand>();
                services.AddTransient<CompareCommand>();
                services.AddTransient<PredictCommand>();
                services.AddTransient<PrecipPredictCommand>();
                services.AddTransient<VerifyCommand>();
                services.AddTransient<PrecipVerifyCommand>();
                services.AddTransient<RenderSiteCommand>();
                services.AddTransient<DryWindowDiagnosticCommand>();
                services.AddTransient<DryWindowTrainCommand>();
                services.AddTransient<DryWindowReportCommand>();
            })
            .Build();

        // CLI wiring
        var root = new RootCommand("WeatherBlend - multi-model weather forecast blending PoC");

        var collect = new Command("collect", "Pull one cycle of forecasts + latest observations");
        collect.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<CollectCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(collect);

        var startOpt = new Option<DateOnly>("--start", "Start date (yyyy-MM-dd), UTC")
            { IsRequired = true };
        var endOpt = new Option<DateOnly>(
            name: "--end",
            description: "End date (yyyy-MM-dd), UTC (default: yesterday)",
            getDefaultValue: () => DateOnly.FromDateTime(DateTime.UtcNow.AddDays(-1)));
        var sourceOpt = new Option<string>(
            name: "--source",
            description: "forecasts | previous-runs | era5 | metar | rainfall | all",
            getDefaultValue: () => "all");
        var backfill = new Command("backfill", "Fetch historical data (forecasts, ERA5, OGIMET METAR, EA rainfall)")
            { sourceOpt, startOpt, endOpt };
        backfill.SetHandler(async (source, start, end) =>
        {
            var cmd = host.Services.GetRequiredService<BackfillCommand>();
            await cmd.RunAsync(source, start, end, CancellationToken.None);
        }, sourceOpt, startOpt, endOpt);
        root.AddCommand(backfill);

        var gfsStartOpt = new Option<DateOnly>("--start", "Start cycle date (yyyy-MM-dd), UTC")
            { IsRequired = true };
        var gfsEndOpt = new Option<DateOnly>(
            name: "--end",
            description: "End cycle date (yyyy-MM-dd), UTC (default: yesterday)",
            getDefaultValue: () => DateOnly.FromDateTime(DateTime.UtcNow.AddDays(-1)));
        var gfsCyclesOpt = new Option<string>(
            name: "--cycles",
            description: "Comma-separated cycle hours, e.g. 0,6,12,18 (default: all four)",
            getDefaultValue: () => "0,6,12,18");
        var gfsBackfill = new Command(
            "gfs-backfill",
            "Phase 3: fetch GFS cycles from NOAA S3 archive with exact run-times/lead-hours")
            { gfsStartOpt, gfsEndOpt, gfsCyclesOpt };
        gfsBackfill.SetHandler(async (start, end, cyclesStr) =>
        {
            var cycles = cyclesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray();
            var cmd = host.Services.GetRequiredService<GfsBackfillCommand>();
            await cmd.RunAsync(start, end, cycles, CancellationToken.None);
        }, gfsStartOpt, gfsEndOpt, gfsCyclesOpt);
        root.AddCommand(gfsBackfill);

        var status = new Command("status", "Show what data is on disk");
        status.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<StatusCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(status);

        var targetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precipitation | dry-window",
            getDefaultValue: () => "temperature");
        var leadOpt = new Option<string>(
            name: "--lead",
            description: "Lead hours: 24 | 48 | 72 | all",
            getDefaultValue: () => "all");
        var stationOpt = new Option<string?>(
            name: "--station",
            description: "Rainfall station name (precipitation / dry-window targets). Defaults to the first station (precipitation) or all phase-3b stations (dry-window).",
            getDefaultValue: () => null);
        var windowOpt = new Option<string?>(
            name: "--window",
            description: "Dry-window target only: window length in hours (3, 4, 6, or all). Default: all.",
            getDefaultValue: () => null);
        var train = new Command("train", "Train the blender (phase 2b temperature / phase 3a precipitation / phase 3b dry-window)")
        {
            targetOpt, leadOpt, stationOpt, windowOpt,
        };
        train.SetHandler(async (target, lead, station, window) =>
        {
            var cmd = host.Services.GetRequiredService<TrainCommand>();
            await cmd.RunAsync(target, lead, station, window, CancellationToken.None);
        }, targetOpt, leadOpt, stationOpt, windowOpt);
        root.AddCommand(train);

        var evalTargetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature",
            getDefaultValue: () => "temperature");
        var modelVersionOpt = new Option<string>(
            name: "--model-version",
            description: "Version directory name (e.g. 'v2026-04-20_140000') or 'current'",
            getDefaultValue: () => "current");
        var evaluate = new Command("evaluate", "Run verification report against the held-out test set")
        {
            evalTargetOpt, modelVersionOpt,
        };
        evaluate.SetHandler(async (target, version) =>
        {
            var cmd = host.Services.GetRequiredService<EvaluateCommand>();
            await cmd.RunAsync(target, version, CancellationToken.None);
        }, evalTargetOpt, modelVersionOpt);
        root.AddCommand(evaluate);

        var pathOpt = new Option<string>("--path", "Path to a parquet file") { IsRequired = true };
        var inspect = new Command("inspect", "Dump a parquet file") { pathOpt };
        inspect.SetHandler(async (path) =>
        {
            var cmd = host.Services.GetRequiredService<InspectCommand>();
            await cmd.RunAsync(path, CancellationToken.None);
        }, pathOpt);
        root.AddCommand(inspect);

        var predictTargetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precipitation",
            getDefaultValue: () => "temperature");
        var predictVersionOpt = new Option<string>(
            name: "--model-version",
            description: "Version directory name or 'current'",
            getDefaultValue: () => "current");
        var predictForDateOpt = new Option<DateOnly?>(
            name: "--for-date",
            description: "Retroactive fill: pretend anchor is this date at 08:00 UTC (yyyy-MM-dd). Omit for live run.");
        var predictTruthStationOpt = new Option<string>(
            name: "--truth-station",
            description: "Precipitation target only: truth station slug (e.g. ea_bellever_dartmoor), config station name, or 'all'",
            getDefaultValue: () => "all");
        var predict = new Command(
            "predict",
            "Produce blended forecasts for the next 24/48/72h from the current blender")
        {
            predictTargetOpt, predictVersionOpt, predictForDateOpt, predictTruthStationOpt,
        };
        predict.SetHandler(async (target, version, forDate, truthStation) =>
        {
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipPredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(truthStation, version, forDate, CancellationToken.None);
            }
            else
            {
                var cmd = host.Services.GetRequiredService<PredictCommand>();
                await cmd.RunAsync(target, version, forDate, CancellationToken.None);
            }
        }, predictTargetOpt, predictVersionOpt, predictForDateOpt, predictTruthStationOpt);
        root.AddCommand(predict);

        var verifyTargetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precipitation",
            getDefaultValue: () => "temperature");
        var verifyAsOfOpt = new Option<DateOnly?>(
            name: "--as-of",
            description: "Anchor date for the rolling window (yyyy-MM-dd). Default: now.");
        var verifyWindowOpt = new Option<int?>(
            name: "--window-days",
            description: "Rolling window size in days. Default: 14 for temperature, 30 for precipitation.");
        var verifyLatencyOpt = new Option<int>(
            name: "--latency-days",
            description: "Truth-release latency — exclude this many most-recent days",
            getDefaultValue: () => 5);
        var verifyDriftOpt = new Option<double>(
            name: "--drift",
            description: "Drift threshold multiplier (rolling metric vs training test metric)",
            getDefaultValue: () => 1.5);
        var verifyTruthStationOpt = new Option<string>(
            name: "--truth-station",
            description: "Precipitation target only: truth station slug, config name, or 'all'",
            getDefaultValue: () => "all");
        var verify = new Command(
            "verify",
            "Rolling verification vs ERA5 (temperature) or EA rainfall (precipitation), stratified by (version, lead). Flags drift.")
        {
            verifyTargetOpt, verifyAsOfOpt, verifyWindowOpt, verifyLatencyOpt, verifyDriftOpt, verifyTruthStationOpt,
        };
        verify.SetHandler(async (target, asOf, windowDays, latencyDays, drift, truthStation) =>
        {
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipVerifyCommand>();
                Environment.ExitCode = await cmd.RunAsync(
                    truthStation, asOf, windowDays ?? 30, latencyDays, drift, CancellationToken.None);
            }
            else
            {
                var cmd = host.Services.GetRequiredService<VerifyCommand>();
                Environment.ExitCode = await cmd.RunAsync(
                    target, asOf, windowDays ?? 14, latencyDays, drift, CancellationToken.None);
            }
        }, verifyTargetOpt, verifyAsOfOpt, verifyWindowOpt, verifyLatencyOpt, verifyDriftOpt, verifyTruthStationOpt);
        root.AddCommand(verify);

        var siteOutputOpt = new Option<string>(
            name: "--output",
            description: "Directory to write the static site into",
            getDefaultValue: () => Path.Combine("data", "site"));
        var siteWindowOpt = new Option<int>(
            name: "--window-days",
            description: "How many days of predictions to include (table + charts)",
            getDefaultValue: () => 30);
        var siteRollingOpt = new Option<int>(
            name: "--rolling-window-days",
            description: "Rolling-MAE window size for verification charts",
            getDefaultValue: () => 14);
        var renderSite = new Command(
            "render-site",
            "Render a self-contained static site (home/predictions/verify/about)")
        {
            siteOutputOpt, siteWindowOpt, siteRollingOpt,
        };
        renderSite.SetHandler(async (output, window, rolling) =>
        {
            var cmd = host.Services.GetRequiredService<RenderSiteCommand>();
            Environment.ExitCode = await cmd.RunAsync(output, window, rolling, CancellationToken.None);
        }, siteOutputOpt, siteWindowOpt, siteRollingOpt);
        root.AddCommand(renderSite);

        var dryWindowDiag = new Command(
            "dry-window-diagnostic",
            "Phase 3b pre-training label diagnostic (per-station, per-window rates + sanity checks)");
        dryWindowDiag.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<DryWindowDiagnosticCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(dryWindowDiag);

        var dryWindowReport = new Command(
            "dry-window-report",
            "Phase 3b post-training evaluation: reload current artefacts, score on test partition, write markdown report");
        dryWindowReport.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<DryWindowReportCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(dryWindowReport);

        var globOpt = new Option<string>("--glob", "Parquet glob across models for one run") { IsRequired = true };
        var compare = new Command("compare", "Cross-model agreement summary for a run") { globOpt };
        compare.SetHandler(async (glob) =>
        {
            var cmd = host.Services.GetRequiredService<CompareCommand>();
            await cmd.RunAsync(glob, CancellationToken.None);
        }, globOpt);
        root.AddCommand(compare);

        try
        {
            return await root.InvokeAsync(args);
        }
        finally
        {
            Log.CloseAndFlush();
        }
    }
}
