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
using WeatherBlend.Storage;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using WeatherBlend.Train.Element.Wind;

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

                services.AddHttpClient<MetOfficeSpotClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(cfg.Http.TimeoutSeconds);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                })
                .AddStandardResilienceHandler();

                services.AddHttpClient<MetOfficeObservationsClient>(c =>
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

                // Storage repositories — thin layer over ParquetReader that
                // de-duplicates SQL/row-mapping previously copy-pasted across
                // commands. Singleton: stateless, just config + logger.
                services.AddSingleton<TruthRepository>();
                services.AddSingleton<ModelMetadataRepository>();
                services.AddSingleton<PredictionsRepository>();

                services.AddTransient<CollectCommand>();
                services.AddTransient<MetOfficeBootstrapCommand>();
                services.AddTransient<BackfillCommand>();
                services.AddTransient<GfsBackfillCommand>();
                services.AddTransient<StatusCommand>();
                services.AddTransient<TempTrainCommand>();
                services.AddTransient<InspectCommand>();
                services.AddTransient<CompareCommand>();
                services.AddTransient<TempPredictCommand>();
                services.AddTransient<PrecipPredictCommand>();
                services.AddTransient<PrecipReplayCommand>();
                // PrecipCalibrateCommand (Phase 3a_isotonic PAV calibration) +
                // DryWindowCalibrateCommand (Phase 3d-calibrated) removed
                // 2026-04-29 — bake-off found PAV didn't move test Brier on
                // either target. Deletion includes IsotonicCalibrator.cs and
                // its tests. Old artefacts on R2 are inert.
                // PrecipAblateCommand removed in Phase 6 of unify-model-membership refactor.
                services.AddTransient<TempVerifyCommand>();
                services.AddTransient<PrecipVerifyCommand>();
                services.AddTransient<RenderSiteCommand>();
                services.AddTransient<DryWindowDiagnosticCommand>();
                services.AddTransient<DryWindowConformalFitCommand>();
                services.AddTransient<PrecipConformalFitCommand>();
                services.AddTransient<DryWindowTrainCommand>();
                services.AddTransient<DryWindowPredictCommand>();
                services.AddTransient<DryWindowVerifyCommand>();
                services.AddTransient<StartHourVerifyCommand>();
                services.AddTransient<ElementTrainCommand>();
                services.AddTransient<ElementPredictCommand>();
                services.AddTransient<ElementVerifyCommand>();
                // ElementBakeoffCommand removed in Phase 5 of unify-model-membership refactor.
                services.AddTransient<FeelsLikePredictCommand>();
                services.AddTransient<StartHourPredictCommand>();
                // Met Office Global / UKV (raw AWS S3 backfill via Python) removed
                // 2026-04-29 — bake-off rejected as blender inputs (negative result)
                // and their parquet writer was the sole source of TIMESTAMPTZ schema
                // that mis-aligned forecast↔truth JOINs during BST. DataHub Spot +
                // Land Observations C# clients (MetOfficeSpotClient,
                // MetOfficeObservationsClient) are unrelated and stay.
                services.AddTransient<IElementBlender, WindBlender>();
                services.AddTransient<IElementBlender, HumidityBlender>();
                services.AddTransient<IElementBlender, RadiationBlender>();
                services.AddTransient<IElementBlender, CloudBlender>();
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

        var metOfficeBootstrap = new Command(
            "met-office-bootstrap",
            "One-off: pull the current Met Office Spot forecast + last 48h of Land Observations. Run once at the start of the capture window; normal 'collect' handles every cycle after that.");
        metOfficeBootstrap.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<MetOfficeBootstrapCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(metOfficeBootstrap);

        var startOpt = new Option<DateOnly>("--start", "Start date (yyyy-MM-dd), UTC")
            { IsRequired = true };
        var endOpt = new Option<DateOnly>(
            name: "--end",
            description: "End date (yyyy-MM-dd), UTC (default: yesterday)",
            getDefaultValue: () => DateOnly.FromDateTime(DateTime.UtcNow.AddDays(-1)));
        var sourceOpt = new Option<string>(
            name: "--source",
            description: "previous-runs | era5 | metar | rainfall | all",
            getDefaultValue: () => "all");
        var modelOpt = new Option<string?>(
            name: "--model",
            description: "Optional: backfill only this Open-Meteo model id (previous-runs only). Defaults to all configured models.",
            getDefaultValue: () => null);
        var backfill = new Command("backfill", "Fetch historical data (previous-runs forecasts, ERA5, OGIMET METAR, EA rainfall)")
            { sourceOpt, startOpt, endOpt, modelOpt };
        backfill.SetHandler(async (source, start, end, model) =>
        {
            var cmd = host.Services.GetRequiredService<BackfillCommand>();
            await cmd.RunAsync(source, start, end, model, CancellationToken.None);
        }, sourceOpt, startOpt, endOpt, modelOpt);
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

        // The `met-office-archive-backfill` CLI command was removed 2026-04-29 along
        // with the Python S3 collector. See Program.cs DI block for the why.

        var status = new Command("status", "Show what data is on disk");
        status.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<StatusCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(status);

        var targetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precipitation | dry-window | wind | humidity | shortwave-radiation | cloud-cover",
            getDefaultValue: () => "temperature");
        var leadOpt = new Option<string>(
            name: "--lead",
            description: "Lead hours: 24 | 48 | 72 | 120 | all (120 supported for temperature + precipitation only)",
            getDefaultValue: () => "all");
        var stationOpt = new Option<string?>(
            name: "--station",
            description: "Rainfall station name (precipitation / dry-window targets). Defaults to the first station (precipitation) or all phase-3b stations (dry-window).",
            getDefaultValue: () => null);
        var windowOpt = new Option<string?>(
            name: "--window",
            description: "Dry-window target only: window length in hours (3, 4, 6, or all). Default: all.",
            getDefaultValue: () => null);
        var featureSetOpt = new Option<string>(
            name: "--feature-set",
            description: "Temperature target only: 'lean' (Phase 2b — 13 features) or 'rich' (Phase 2c — 88 features incl. per-model dew/RH/cloud/wind/pressure secondaries).",
            getDefaultValue: () => "lean");
        var train = new Command("train", "Train the blender (phase 2b temperature / phase 3a precipitation / phase 3b dry-window)")
        {
            targetOpt, leadOpt, stationOpt, windowOpt, featureSetOpt,
        };
        train.SetHandler(async (ctx) =>
        {
            var target = ctx.ParseResult.GetValueForOption(targetOpt)!;
            var lead = ctx.ParseResult.GetValueForOption(leadOpt)!;
            var station = ctx.ParseResult.GetValueForOption(stationOpt);
            var window = ctx.ParseResult.GetValueForOption(windowOpt);
            var featureSet = ctx.ParseResult.GetValueForOption(featureSetOpt)!;
            var cmd = host.Services.GetRequiredService<TempTrainCommand>();
            ctx.ExitCode = await cmd.RunAsync(target, lead, station, window, featureSet, ctx.GetCancellationToken());
        });
        root.AddCommand(train);

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
            description: "Target variable: temperature | precipitation | dry-window | wind | humidity | shortwave-radiation | cloud-cover | feels-like | start-hour",
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
            description: "Precipitation / dry-window only: truth station slug (e.g. ea_bellever_dartmoor), config station name, or 'all'",
            getDefaultValue: () => "all");
        var predictWindowOpt = new Option<string>(
            name: "--window",
            description: "Dry-window only: window length in hours (3 | 4 | 6 | all)",
            getDefaultValue: () => "all");
        var predict = new Command(
            "predict",
            "Produce blended forecasts (24/48/72h, plus 120h for temperature + precipitation) from the current blender")
        {
            predictTargetOpt, predictVersionOpt, predictForDateOpt, predictTruthStationOpt, predictWindowOpt,
        };
        predict.SetHandler(async (target, version, forDate, truthStation, window) =>
        {
            var elementTarget = ElementTargets.TryFromCli(target);
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipPredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(truthStation, version, forDate, CancellationToken.None);
            }
            else if (string.Equals(target, "dry-window", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<DryWindowPredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(truthStation, window, version, forDate, CancellationToken.None);
            }
            else if (string.Equals(target, "feels-like", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<FeelsLikePredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(forDate, CancellationToken.None);
            }
            else if (string.Equals(target, "start-hour", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<StartHourPredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(forDate, CancellationToken.None);
            }
            else if (elementTarget is not null)
            {
                var cmd = host.Services.GetRequiredService<ElementPredictCommand>();
                Environment.ExitCode = await cmd.RunAsync(elementTarget, version, forDate, CancellationToken.None);
            }
            else
            {
                var cmd = host.Services.GetRequiredService<TempPredictCommand>();
                await cmd.RunAsync(target, version, forDate, CancellationToken.None);
            }
        }, predictTargetOpt, predictVersionOpt, predictForDateOpt, predictTruthStationOpt, predictWindowOpt);
        root.AddCommand(predict);

        // ---- precip-replay (one-off research command — see PrecipReplayCommand docs) ----
        var replayStationOpt = new Option<string>(
            name: "--truth-station",
            description: "Truth station slug, e.g. ea_bellever_dartmoor",
            getDefaultValue: () => "ea_bellever_dartmoor");
        var replayVersionOpt = new Option<string>(
            name: "--model-version",
            description: "Version dir or 'current'",
            getDefaultValue: () => "current");
        var replayLeadsOpt = new Option<string>(
            name: "--leads",
            description: "Comma-separated lead hours (default: 24)",
            getDefaultValue: () => "24");
        var precipReplay = new Command(
            "precip-replay",
            "Replay a Phase 3a blender against every historical row, dump per-row P(wet)")
        {
            replayStationOpt, replayVersionOpt, replayLeadsOpt,
        };
        precipReplay.SetHandler(async (slug, version, leadsStr) =>
        {
            var leads = leadsStr.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => int.Parse(s.Trim())).ToArray();
            var cmd = host.Services.GetRequiredService<PrecipReplayCommand>();
            Environment.ExitCode = await cmd.RunAsync(slug, version, leads, CancellationToken.None);
        }, replayStationOpt, replayVersionOpt, replayLeadsOpt);
        root.AddCommand(precipReplay);

        var verifyTargetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precipitation | dry-window | wind | humidity | shortwave-radiation | cloud-cover | start-hour",
            getDefaultValue: () => "temperature");
        var verifyAsOfOpt = new Option<DateOnly?>(
            name: "--as-of",
            description: "Anchor date for the rolling window (yyyy-MM-dd). Default: now.");
        var verifyWindowOpt = new Option<int?>(
            name: "--window-days",
            description: "Rolling window size in days. Default: 14 for temperature, 30 for precipitation/dry-window.");
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
            description: "Precipitation / dry-window target only: truth station slug, config name, or 'all'",
            getDefaultValue: () => "all");
        var verifyDryWindowOpt = new Option<string>(
            name: "--window",
            description: "Dry-window target only: window length in hours (3 | 4 | 6 | all)",
            getDefaultValue: () => "all");
        var verify = new Command(
            "verify",
            "Rolling verification vs ERA5 (temperature) or EA rainfall (precipitation/dry-window), stratified by (version, lead). Flags drift.")
        {
            verifyTargetOpt, verifyAsOfOpt, verifyWindowOpt, verifyLatencyOpt, verifyDriftOpt, verifyTruthStationOpt, verifyDryWindowOpt,
        };
        verify.SetHandler(async (ctx) =>
        {
            var target = ctx.ParseResult.GetValueForOption(verifyTargetOpt)!;
            var asOf = ctx.ParseResult.GetValueForOption(verifyAsOfOpt);
            var windowDays = ctx.ParseResult.GetValueForOption(verifyWindowOpt);
            var latencyDays = ctx.ParseResult.GetValueForOption(verifyLatencyOpt);
            var drift = ctx.ParseResult.GetValueForOption(verifyDriftOpt);
            var truthStation = ctx.ParseResult.GetValueForOption(verifyTruthStationOpt)!;
            var dryWindow = ctx.ParseResult.GetValueForOption(verifyDryWindowOpt)!;

            var elementTarget = ElementTargets.TryFromCli(target);
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    truthStation, asOf, windowDays ?? 30, latencyDays, drift, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "dry-window", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<DryWindowVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    truthStation, dryWindow, asOf, windowDays ?? 30, latencyDays, drift, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "start-hour", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<StartHourVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    asOf, windowDays ?? 30, latencyDays, ctx.GetCancellationToken());
            }
            else if (elementTarget is not null)
            {
                var cmd = host.Services.GetRequiredService<ElementVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    elementTarget, asOf, windowDays ?? 14, latencyDays, drift, ctx.GetCancellationToken());
            }
            else
            {
                var cmd = host.Services.GetRequiredService<TempVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    target, asOf, windowDays ?? 14, latencyDays, drift, ctx.GetCancellationToken());
            }
        });
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

        // The `precip-calibrate` (Phase 3a_isotonic) and `dry-window-calibrate`
        // (Phase 3d-calibrated) CLI commands were removed 2026-04-29 alongside
        // their backing classes — the bake-off concluded PAV calibration didn't
        // move test Brier on either target.
        // The `precip-ablate` command was removed in Phase 6 of the unify-model-membership refactor.
        // Its conclusions are baked into the production rich precip blender.

        var dryWindowDiag = new Command(
            "dry-window-diagnostic",
            "Phase 3b pre-training label diagnostic (per-station, per-window rates + sanity checks)");
        dryWindowDiag.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<DryWindowDiagnosticCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(dryWindowDiag);

        // The `bakeoff` command was removed in Phase 5 of the unify-model-membership refactor.
        // It diagnosed UKMO inclusion vs exclusion on a shared UKMO-present test set; the
        // resulting per-element decisions are now baked into config.yaml's blenders section.

        // The `start-hour-bakeoff` command was removed 2026-05-04 alongside
        // StartHourCurveDerivation. The bake-off was Phase-1 scaffolding for
        // a never-run "current vs option C vs option B" comparison; option C
        // (3g-style MC over 3a hourly q) shipped directly as production in
        // StartHourPredictCommand v2 so "current" no longer exists and the
        // framework would have nothing to compare against. Rebuild a fresh
        // harness if/when option B (Bayesian copula) gets built.

        // ---- dry-window-conformal-fit (one-shot back-fit of conformal calibrators) ----
        var conformalAlphaOpt = new Option<double>(
            name: "--alpha",
            description: "Target miscoverage rate (0.10 = 90% coverage; default 0.10).",
            getDefaultValue: () => 0.10);
        var dryWindowConformalFit = new Command(
            "dry-window-conformal-fit",
            "Back-fit conformal calibrators on the val slice for every Active dry-window version (3b + 3g)")
        {
            conformalAlphaOpt,
        };
        dryWindowConformalFit.SetHandler(async (alpha) =>
        {
            var cmd = host.Services.GetRequiredService<DryWindowConformalFitCommand>();
            Environment.ExitCode = await cmd.RunAsync(alpha, CancellationToken.None);
        }, conformalAlphaOpt);
        root.AddCommand(dryWindowConformalFit);

        // ---- precip-conformal-fit (sibling of dry-window-conformal-fit) ----
        var precipConformalAlphaOpt = new Option<double>(
            name: "--alpha",
            description: "Target miscoverage rate (0.10 = 90% coverage; default 0.10).",
            getDefaultValue: () => 0.10);
        var precipConformalFit = new Command(
            "precip-conformal-fit",
            "Back-fit conformal calibrators on the val slice for every Active precipitation version (3a + 3c)")
        {
            precipConformalAlphaOpt,
        };
        precipConformalFit.SetHandler(async (alpha) =>
        {
            var cmd = host.Services.GetRequiredService<PrecipConformalFitCommand>();
            Environment.ExitCode = await cmd.RunAsync(alpha, CancellationToken.None);
        }, precipConformalAlphaOpt);
        root.AddCommand(precipConformalFit);

        // dry-window-ablate (Phase 3d 3b-vs-3d-shape diagnostic) was retired
        // 2026-05-04 alongside the 3d-shape training path. 3b vs 3g comparison
        // is covered by the standard verify command's per-phase grouping.

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
