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
using WeatherBlend.Train.Element.Gust;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using WeatherBlend.Train.Element.Wind;
using WeatherBlend.Train.Exact12h;
using WeatherBlend.Train.PrecipExact;

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

                // GEFS S3 archive — same Range-download + idx pattern as GFS.
                // Reads the geavg ensemble-mean member from pgrb2ap5/0.5°.
                services.AddHttpClient<GefsClient>(c =>
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

                // ECMWF S3 archive — same Range-download shape as GFS, same
                // resilience profile. Reuses Wgrib2 singleton below.
                services.AddHttpClient<EcmwfClient>(c =>
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
                services.AddTransient<GefsBackfillCommand>();
                services.AddTransient<EcmwfBackfillCommand>();
                services.AddTransient<StatusCommand>();
                services.AddTransient<PrecipTrainCommand>();
                services.AddTransient<TrainCommand>();
                services.AddTransient<InspectCommand>();
                services.AddTransient<CompareCommand>();
                services.AddTransient<TempPredictCommand>();
                services.AddTransient<PrecipPredictCommand>();
                services.AddTransient<PrecipReplayCommand>();
                services.AddTransient<Phase4bPredictCommand>();
                services.AddTransient<Phase4bMintCommand>();
                services.AddTransient<WindBlendPredictCommand>();
                // PrecipCalibrateCommand (Phase 3a_isotonic PAV calibration) +
                // DryWindowCalibrateCommand (Phase 3d-calibrated) removed
                // 2026-04-29 — bake-off found PAV didn't move test Brier on
                // either target. Deletion includes IsotonicCalibrator.cs and
                // its tests. Old artefacts on R2 are inert.
                // PrecipAblateCommand removed in Phase 6 of unify-model-membership refactor.
                services.AddTransient<TempVerifyCommand>();
                services.AddTransient<PrecipVerifyCommand>();
                services.AddTransient<RainfallAmountVerifyCommand>();
                services.AddTransient<RenderSiteCommand>();
                services.AddTransient<DryWindowDiagnosticCommand>();
                services.AddTransient<DryWindowConformalFitCommand>();
                services.AddTransient<PrecipConformalFitCommand>();
                services.AddTransient<DryWindowTrainCommand>();
                services.AddTransient<DryWindowPredictCommand>();
                services.AddTransient<DryWindowVerifyCommand>();
                services.AddTransient<ElementTrainCommand>();
                services.AddTransient<ElementPredictCommand>();
                services.AddTransient<ElementVerifyCommand>();
                // ElementBakeoffCommand removed in Phase 5 of unify-model-membership refactor.
                services.AddTransient<FeelsLikePredictCommand>();
                // Met Office Global / UKV — raw AWS S3 backfill via Python.
                // Reinstated 2026-05-05 with the parquet writer fixed to emit
                // tz-NAIVE timestamps (the BST-skewed JOIN bug that drove the
                // 2026-04-29 deletion). Bake-off still rejects them as blender
                // inputs (uncorrected elevation bias at Bonehill); they're
                // captured here for raw model comparisons + future work.
                services.AddTransient<MetOfficeArchiveBackfillClient>();
                services.AddTransient<MetOfficeArchiveBackfillCommand>();
                // Exact-runtime 12h temperature bake-off (Option-1 design,
                // 2026-05-05). Three tiers, no UKV, no artifact persistence —
                // exploratory only.
                services.AddTransient<Exact12hBakeoffCommand>();
                services.AddTransient<PrecipExactBakeoffCommand>();
                services.AddTransient<PrecipIfsCycleBakeoffCommand>();
                services.AddTransient<Phase3cDataWindowBakeoffCommand>();
                // Live S3 collect — refreshes the five exact-runtime sources
                // the 2d temperature blender consumes (GFS / IFS / AIFS /
                // Met Office Global / Met Office UKV). Each is a thin
                // lookback wrapper around its existing backfill command.
                // Wired into a 6-hourly cron workflow s3-collect.yml that
                // pairs with the live data landing pattern (NOAA ~3-4h,
                // ECMWF ~7-8h, MO ~3-6h).
                services.AddTransient<GfsArchiveCollector>();
                // GEFS wired into S3CollectCommand 2026-05-09 after the 2d/3d
                // bake-offs validated GEFS as a feature column (3d wins or
                // ties in all 12 (station, lead, tier) cells; 2d gains at
                // every lead except 72).
                services.AddTransient<GefsArchiveCollector>();
                services.AddTransient<EcmwfArchiveCollector>();
                services.AddTransient<MetOfficeGlobalArchiveCollector>();
                services.AddTransient<MetOfficeUkvArchiveCollector>();
                services.AddTransient<S3CollectCommand>();
                services.AddTransient<IElementBlender, WindBlender>();
                services.AddTransient<IElementBlender, HumidityBlender>();
                services.AddTransient<IElementBlender, RadiationBlender>();
                services.AddTransient<IElementBlender, CloudBlender>();
                services.AddTransient<IElementBlender, WindGustBlender>();
                services.AddTransient<IElementBlender, WindSpeedLgbBlender>();
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
            description: "previous-runs | era5 | metar | rainfall | hist-forecast | all",
            getDefaultValue: () => "all");
        var modelOpt = new Option<string?>(
            name: "--model",
            description: "Optional: backfill only this Open-Meteo model id (previous-runs / hist-forecast only). Defaults to all configured models.",
            getDefaultValue: () => null);
        var locationOpt = new Option<string?>(
            name: "--location",
            description: "Optional: backfill only this location name (matches AppConfig.Locations[].Name). Applies to previous-runs + era5 + rainfall (METAR dedupes per-ICAO). Defaults to all configured locations.",
            getDefaultValue: () => null);
        var backfill = new Command("backfill", "Fetch historical data (previous-runs forecasts, ERA5, OGIMET METAR, EA rainfall)")
            { sourceOpt, startOpt, endOpt, modelOpt, locationOpt };
        backfill.SetHandler(async (source, start, end, model, location) =>
        {
            var cmd = host.Services.GetRequiredService<BackfillCommand>();
            // Propagate the command's exit code (non-zero when any chunk errored).
            // Without this the process exited 0 even when every API call failed,
            // so a fully-failed backfill showed green in CI (the 2026-05-30
            // pressure-level 400 incident). Other command handlers already do this.
            Environment.ExitCode = await cmd.RunAsync(source, start, end, model, location, CancellationToken.None);
        }, sourceOpt, startOpt, endOpt, modelOpt, locationOpt);
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

        // GEFS backfill — NOAA's Global Ensemble Forecast System ensemble
        // mean (geavg member) from s3://noaa-gefs-pds/. Reuses the same
        // --start/--end/--cycles option shape as the GFS backfill above
        // since the bucket layout is identical apart from the geavg
        // member prefix and the pgrb2ap5/0.5° subfolder.
        var gefsStartOpt = new Option<DateOnly>(
            name: "--start",
            description: "Start cycle date (yyyy-MM-dd), UTC",
            getDefaultValue: () => DateOnly.FromDateTime(DateTime.UtcNow.AddDays(-30)));
        var gefsEndOpt = new Option<DateOnly>(
            name: "--end",
            description: "End cycle date (yyyy-MM-dd), UTC (default: yesterday)",
            getDefaultValue: () => DateOnly.FromDateTime(DateTime.UtcNow.AddDays(-1)));
        var gefsCyclesOpt = new Option<string>(
            name: "--cycles",
            description: "Comma-separated cycle hours, e.g. 0,6,12,18 (default: all four)",
            getDefaultValue: () => "0,6,12,18");
        var gefsBackfill = new Command(
            "gefs-backfill",
            "Fetch GEFS ensemble-mean cycles from NOAA S3 archive with exact run-times/lead-hours")
            { gefsStartOpt, gefsEndOpt, gefsCyclesOpt };
        gefsBackfill.SetHandler(async (start, end, cyclesStr) =>
        {
            var cycles = cyclesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray();
            var cmd = host.Services.GetRequiredService<GefsBackfillCommand>();
            await cmd.RunAsync(start, end, cycles, CancellationToken.None);
        }, gefsStartOpt, gefsEndOpt, gefsCyclesOpt);
        root.AddCommand(gefsBackfill);

        // ECMWF backfill — IFS oper or AIFS oper. Reuses --start/--end/--cycles
        // from the GFS shape; adds --stream {ifs|aifs}.
        var ecmStreamOpt = new Option<string>(
            name: "--stream",
            description: "ECMWF stream: ifs (deterministic) | aifs (AI model)",
            getDefaultValue: () => "ifs");
        var ecmwfBackfill = new Command(
            "ecmwf-backfill",
            "Fetch ECMWF IFS/AIFS oper cycles from AWS Open Data archive with exact run-times/lead-hours")
            { ecmStreamOpt, gfsStartOpt, gfsEndOpt, gfsCyclesOpt };
        ecmwfBackfill.SetHandler(async (stream, start, end, cyclesStr) =>
        {
            var cycles = cyclesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray();
            var cmd = host.Services.GetRequiredService<EcmwfBackfillCommand>();
            // Historical backfill → AWS deep archive (the live ECMWF server
            // only keeps a rolling ~4-day window).
            await cmd.RunAsync(stream, start, end, cycles, EcmwfClient.ArchiveBaseUrl, CancellationToken.None);
        }, ecmStreamOpt, gfsStartOpt, gfsEndOpt, gfsCyclesOpt);
        root.AddCommand(ecmwfBackfill);

        // Met Office archive backfill — Global 10km or UKV 2km. Defaults to
        // the source-native sensible cycles + leads (the script's own arg
        // defaults), so the only required input from the workflow is the
        // date range + source. Per-source defaults inside the Python:
        //   global: cycles=0,12 (the long-range cycles to 168h)
        //   ukv:    cycles=3,15 (the long-range cycles to 120h)
        // Both default to leads=1,3,6,12,24,36,48,72,96,120 to match the
        // GFS/ECMWF archive backfills.
        var moSourceOpt = new Option<string>(
            name: "--source",
            description: "Met Office dataset: global (10km det) | ukv (2km det)",
            getDefaultValue: () => "global");
        var moCyclesOpt = new Option<string>(
            name: "--cycles",
            description: "Override cycle hours (blank = source-native defaults: global=0,12 / ukv=3,15)",
            getDefaultValue: () => "");
        var moLeadsOpt = new Option<string>(
            name: "--leads",
            description: "Override lead hours (blank = source-native default 1,3,6,12,24,36,48,72,96,120)",
            getDefaultValue: () => "");
        var moParallelismOpt = new Option<int>(
            name: "--parallelism",
            description: "Concurrent NetCDF downloads per cycle",
            getDefaultValue: () => 8);
        var metOfficeBackfill = new Command(
            "metoffice-backfill",
            "Fetch Met Office Global / UKV cycles from AWS Open Data NetCDF archive (Python subprocess)")
            { moSourceOpt, gfsStartOpt, gfsEndOpt, moCyclesOpt, moLeadsOpt, moParallelismOpt };
        metOfficeBackfill.SetHandler(async (source, start, end, cyclesStr, leadsStr, parallelism) =>
        {
            // Empty string → let the Python script use its own defaults (we pass
            // an empty list, the wrapper sees zero items and forwards the script
            // default). Non-empty → override per-dispatch.
            var cycles = string.IsNullOrWhiteSpace(cyclesStr)
                ? Array.Empty<int>()
                : cyclesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                    .ToArray();
            var leads = string.IsNullOrWhiteSpace(leadsStr)
                ? Array.Empty<int>()
                : leadsStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                    .ToArray();
            var cmd = host.Services.GetRequiredService<MetOfficeArchiveBackfillCommand>();
            await cmd.RunAsync(source, start, end, cycles, leads, parallelism, CancellationToken.None);
        }, moSourceOpt, gfsStartOpt, gfsEndOpt, moCyclesOpt, moLeadsOpt, moParallelismOpt);
        root.AddCommand(metOfficeBackfill);

        // Exact-runtime 12h temperature bake-off — runs all three tiers
        // (T1 strict / T2 GFS+AIFS-required / T3 GFS-only-required) and
        // prints a comparison report. --tier T1|T2|T3 to run just one.
        var exact12hTierOpt = new Option<string?>(
            name: "--tier",
            description: "Run a specific tier only (T1 / T2 / T3). Defaults to all three.",
            getDefaultValue: () => null);
        // --lead picks the SINGLE-LEAD target. Defaults to 12 (the original
        // experiment). Pass --lead 24 to run apples-to-apples vs production 2b.
        var exact12hLeadOpt = new Option<int>(
            name: "--lead",
            description: "Target lead in hours. Single-lead mode unless --leads is also passed. Default 12.",
            getDefaultValue: () => Exact12hFeatureBuilder.DefaultTargetLead);
        // --leads expands the column vector to multiple leads per model
        // (the 2026-05-05 multi-lead experiment). Must contain --lead value.
        // Default empty = single-lead mode using --lead.
        var exact12hLeadsOpt = new Option<string>(
            name: "--leads",
            description: "Comma-separated input leads for multi-lead mode (e.g. 6,12,18). Must contain --lead. Empty = single-lead.",
            getDefaultValue: () => "");
        // Honest test of multi-lead training: TRAIN with the multi-lead
        // vector, MASK non-canonical leads to NaN at test time. Tests
        // whether multi-lead training boosted the canonical-lead path
        // or was just leaning on lead-6 (made AFTER prediction time, so
        // leakage in any deployment scenario).
        var exact12hMaskOpt = new Option<bool>(
            name: "--mask-noncanonical-at-test",
            description: "Multi-lead only: zero non-canonical-lead columns to NaN in test rows before scoring (fair test of multi-lead training transfer).",
            getDefaultValue: () => false);
        // Cross-lead evaluation — train at --lead, then score the trained
        // model on test sets built from data at OTHER leads. Tells us how
        // the model generalises off its training lead. Single-lead mode only
        // (multi-lead column shape can't be remapped). Comma-separated leads
        // on the 6h grid: 6,12,24,36,48,72,96,120.
        var exact12hEvalAtLeadsOpt = new Option<string>(
            name: "--evaluate-at-leads",
            description: "After training at --lead, also score on test rows built at these alternative leads (comma list, 6h-grid). Single-lead only.",
            getDefaultValue: () => "");
        // HP search: grid-search LightGBM hyperparameters (lr × leaves × min-leaf × feature-frac)
        // and report the winning config + MAE. The "2d candidate" path per the
        // 2026-05-05 productionisation discussion. Skips the regular bake-off
        // output when set; ignores --evaluate-at-leads / --mask.
        var exact12hHpSearchOpt = new Option<bool>(
            name: "--hp-search",
            description: "Grid-search LightGBM hyperparameters and report the winning config (instead of the regular bake-off).",
            getDefaultValue: () => false);
        // --include-ukv (2026-05-05) — adds an extra "temp_ukv" column whose
        // value per ValidTime hour is pulled from the appropriate UKV (cycle,
        // lead) tuple chosen so the average UKV-effective-lead matches the
        // bake-off targetLead (lead-12 → leads {9,15}; lead-24 → leads {21,27}).
        // Tests whether UKV helps the exact-runtime blender despite its odd
        // 03Z/15Z cycle phasing.
        var exact12hUkvOpt = new Option<bool>(
            name: "--include-ukv",
            description: "Add UKV as a 5th feature column (per-V-hour conditional pull, lead-aware: lead-12 reads UKV at leads {9,15}; lead-24 reads at {21,27}). Off by default.",
            getDefaultValue: () => false);
        var exact12hBakeoff = new Command(
            "exact-12h-bakeoff",
            "Train + score the exact-runtime temperature blender across the three coverage tiers")
            { exact12hTierOpt, exact12hLeadOpt, exact12hLeadsOpt, exact12hMaskOpt, exact12hEvalAtLeadsOpt, exact12hHpSearchOpt, exact12hUkvOpt };
        exact12hBakeoff.SetHandler(async (ctx) =>
        {
            var tier = ctx.ParseResult.GetValueForOption(exact12hTierOpt);
            var lead = ctx.ParseResult.GetValueForOption(exact12hLeadOpt);
            var leadsStr = ctx.ParseResult.GetValueForOption(exact12hLeadsOpt) ?? "";
            var mask = ctx.ParseResult.GetValueForOption(exact12hMaskOpt);
            var evalLeadsStr = ctx.ParseResult.GetValueForOption(exact12hEvalAtLeadsOpt) ?? "";
            var hpSearch = ctx.ParseResult.GetValueForOption(exact12hHpSearchOpt);
            var includeUkv = ctx.ParseResult.GetValueForOption(exact12hUkvOpt);
            var inputLeads = string.IsNullOrWhiteSpace(leadsStr)
                ? null
                : (IReadOnlyList<int>)leadsStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                    .ToArray();
            var evalLeads = string.IsNullOrWhiteSpace(evalLeadsStr)
                ? null
                : (IReadOnlyList<int>)evalLeadsStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                    .ToArray();
            var cmd = host.Services.GetRequiredService<Exact12hBakeoffCommand>();
            ctx.ExitCode = await cmd.RunAsync(tier, lead, inputLeads, mask, evalLeads, hpSearch, includeUkv, CancellationToken.None);
        });
        root.AddCommand(exact12hBakeoff);

        // precip-exact-bakeoff — exact-runtime P(wet) blender (mirror of
        // exact-12h-bakeoff for temperature). 4 models required/optional
        // (GFS+IFS+AIFS required, MO Global optional); --include-ukv adds
        // UKV as a 5th always-optional column via the same per-V-hour
        // conditional pull pattern as the temperature builder. Single tier
        // P1, --lead 12 or 24.
        var precipLeadOpt = new Option<int>(
            name: "--lead",
            description: "Target lead in hours. Default 12.",
            getDefaultValue: () => PrecipExactFeatureBuilder.DefaultTargetLead);
        var precipUkvOpt = new Option<bool>(
            name: "--include-ukv",
            description: "Include Met Office UKV as a 5th always-optional precip input via per-V-hour (cycle, lead) conditional pull from 03Z+15Z cycles.",
            getDefaultValue: () => false);
        var precipStationOpt = new Option<string?>(
            name: "--station",
            description: "EA rainfall station name (config-friendly, e.g. 'Bellever Dartmoor'). Defaults to first configured station.",
            getDefaultValue: () => null);
        var precipBakeoff = new Command(
            "precip-exact-bakeoff",
            "Train + score the exact-runtime precipitation blender at a single lead, against EA gauge truth")
            { precipLeadOpt, precipUkvOpt, precipStationOpt };
        precipBakeoff.SetHandler(async (lead, includeUkv, station) =>
        {
            var cmd = host.Services.GetRequiredService<PrecipExactBakeoffCommand>();
            await cmd.RunAsync(lead, includeUkv, station, CancellationToken.None);
        }, precipLeadOpt, precipUkvOpt, precipStationOpt);
        root.AddCommand(precipBakeoff);

        // precip-ifs-cycle-bakeoff — 4-way IFS-cycle comparison for 3d
        // (Both / Oper / Scda / None) on a constant row/eval set. Trains
        // in memory, persists nothing; writes data/reports/ifs_cycle_bakeoff_3d.md.
        var ifsCycleLeadOpt = new Option<int>(
            name: "--lead",
            description: "Single target lead in hours. 0 (default) = run all of 12/24/48/72.",
            getDefaultValue: () => 0);
        var ifsCycleStationOpt = new Option<string?>(
            name: "--station",
            description: "EA rainfall station name. Defaults to every configured station.",
            getDefaultValue: () => null);
        var ifsCycleAsOfOpt = new Option<string>(
            name: "--as-of",
            description: "Cap training/eval data at ValidTime ≤ this date (yyyy-MM-dd).",
            getDefaultValue: () => "2026-05-12");
        var ifsCycleBakeoff = new Command(
            "precip-ifs-cycle-bakeoff",
            "4-way IFS-cycle bake-off for 3d: IFS from both/oper/scda/no cycles, constant eval set")
            { ifsCycleLeadOpt, ifsCycleStationOpt, ifsCycleAsOfOpt };
        ifsCycleBakeoff.SetHandler(async (lead, station, asOf) =>
        {
            var cmd = host.Services.GetRequiredService<PrecipIfsCycleBakeoffCommand>();
            await cmd.RunAsync(lead, station, asOf, CancellationToken.None);
        }, ifsCycleLeadOpt, ifsCycleStationOpt, ifsCycleAsOfOpt);
        root.AddCommand(ifsCycleBakeoff);

        // precip-data-window-bakeoff — A vs B diagnostic. Trains per-station
        // 3c on full data vs 2024+ only, scores on identical test rows.
        // Answers: did the JMA 2022-2023 backfill help or hurt 3c?
        var dataWindowBakeoff = new Command(
            "precip-data-window-bakeoff",
            "A vs B diagnostic: per-station 3c trained on full data vs 2024+ only, same test set");
        dataWindowBakeoff.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<Phase3cDataWindowBakeoffCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(dataWindowBakeoff);

        // s3-collect — live refresh of the five exact-runtime forecast sources
        // the 2d temperature blender consumes. Defaults to all sources; pass
        // --sources gfs,ifs to subset for testing.
        var s3CollectSourcesOpt = new Option<string>(
            name: "--sources",
            description: "Comma-list of sources to collect (default: all). Valid: gfs, ifs, aifs, mo-global, ukv.",
            getDefaultValue: () => "");
        var s3Collect = new Command(
            "s3-collect",
            "Live refresh of exact-runtime forecast sources (GFS / IFS / AIFS / Met Office Global / Met Office UKV) for the 2d blender")
            { s3CollectSourcesOpt };
        s3Collect.SetHandler(async (sourcesStr) =>
        {
            var sources = string.IsNullOrWhiteSpace(sourcesStr)
                ? null
                : (IReadOnlyList<string>)sourcesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToArray();
            var cmd = host.Services.GetRequiredService<S3CollectCommand>();
            await cmd.RunAsync(sources, CancellationToken.None);
        }, s3CollectSourcesOpt);
        root.AddCommand(s3Collect);

        var status = new Command("status", "Show what data is on disk");
        status.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<StatusCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(status);

        // Phase C commit 3c: single source of truth for "which locations does
        // this codebase target?". Workflow bash loops consume the stdout line-
        // by-line — predict-all and verify both honour _cfg.Locations now, so
        // adding a third location is a config.yaml edit, not a YAML edit.
        var listLocations = new Command("list-locations",
            "Print configured location slugs, one per line (single source of truth for CI workflow loops).");
        listLocations.SetHandler(ctx =>
        {
            var cfg = host.Services.GetRequiredService<AppConfig>();
            foreach (var loc in cfg.Locations)
                Console.WriteLine(loc.Name);
            ctx.ExitCode = 0;
            return Task.CompletedTask;
        });
        root.AddCommand(listLocations);

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
        var tierOpt = new Option<string?>(
            name: "--tier",
            description: "Exact-runtime only: model-set tier name. Temperature: T1/T2/T3 (default T2). Precipitation: P1/P2 (default P1; P2 drops IFS).",
            getDefaultValue: () => null);
        var includeUkvOpt = new Option<bool?>(
            name: "--include-ukv",
            description: "Exact-runtime only: include UKV column in feature vector (default true). Pass --include-ukv false to exclude.",
            getDefaultValue: () => null);
        var exactLeadsOpt = new Option<string?>(
            name: "--leads",
            description: "Exact-runtime only: comma-separated lead hours (default 12,24). E.g. --leads 48,72 for the long-range bake-off vs 2b/3a.",
            getDefaultValue: () => null);
        var cyclesOpt = new Option<string?>(
            name: "--cycles",
            description: "Exact-runtime only: IFS-specific cycle filter — comma-separated RunTime cycle hours that the ecmwf_ifs_oper feature column may draw from (default null = all IFS cycles on disk). Other models pass through unfiltered. Use --cycles 0,12 for an oper-only IFS bake-off vs the default which includes the 06/18 scda cycles.",
            getDefaultValue: () => null);
        var trainLocationOpt = new Option<string?>(
            name: "--location",
            description: "Override the primary location for precipitation training (3a/3c). Must match a name in config.yaml's `locations:` list (e.g. 'membury_devon'). Default = primary location (Bonehill). Other targets ignore this flag for now.",
            getDefaultValue: () => null);
        var train = new Command("train", "Train the blender (phase 2b temperature / phase 3a precipitation / phase 3b dry-window)")
        {
            targetOpt, leadOpt, stationOpt, windowOpt, featureSetOpt, tierOpt, includeUkvOpt, exactLeadsOpt, cyclesOpt, trainLocationOpt,
        };
        train.SetHandler(async (ctx) =>
        {
            var target = ctx.ParseResult.GetValueForOption(targetOpt)!;
            var lead = ctx.ParseResult.GetValueForOption(leadOpt)!;
            var station = ctx.ParseResult.GetValueForOption(stationOpt);
            var window = ctx.ParseResult.GetValueForOption(windowOpt);
            var featureSet = ctx.ParseResult.GetValueForOption(featureSetOpt)!;
            var tier = ctx.ParseResult.GetValueForOption(tierOpt);
            var includeUkv = ctx.ParseResult.GetValueForOption(includeUkvOpt);
            var exactLeadsStr = ctx.ParseResult.GetValueForOption(exactLeadsOpt);
            int[]? exactLeads = string.IsNullOrWhiteSpace(exactLeadsStr) ? null
                : exactLeadsStr.Split(',').Select(s => int.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            var cyclesStr = ctx.ParseResult.GetValueForOption(cyclesOpt);
            int[]? cycles = string.IsNullOrWhiteSpace(cyclesStr) ? null
                : cyclesStr.Split(',').Select(s => int.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            var locationOverride = ctx.ParseResult.GetValueForOption(trainLocationOpt);
            var cmd = host.Services.GetRequiredService<TrainCommand>();
            ctx.ExitCode = await cmd.RunAsync(target, lead, station, window, featureSet, tier, includeUkv, exactLeads, cycles, locationOverride, ctx.GetCancellationToken());
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
            description: "Target variable: temperature | precipitation | dry-window | wind | humidity | shortwave-radiation | cloud-cover | feels-like",
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
        var predictLocationOpt = new Option<string?>(
            name: "--location",
            description: "Override the primary location for any predict target (e.g. 'membury_devon') — precipitation, temperature, dry-window, feels-like, start-hour, and the element blenders all honour it. For precip, filters stationsToRun to that location's rainfall config + reads forecasts from that location's tree. For temperature (post-2026-05-14 station-keyed layout), filters to that station entry's bundles + that location's forecasts. Without --location, temperature iterates EVERY station entry in the manifest; every other target defaults to the primary location only.",
            getDefaultValue: () => null);
        var predict = new Command(
            "predict",
            "Produce blended forecasts (24/48/72h, plus 120h for temperature + precipitation) from the current blender")
        {
            predictTargetOpt, predictVersionOpt, predictForDateOpt, predictTruthStationOpt, predictWindowOpt, predictLocationOpt,
        };
        predict.SetHandler(async (ctx) =>
        {
            var target = ctx.ParseResult.GetValueForOption(predictTargetOpt)!;
            var version = ctx.ParseResult.GetValueForOption(predictVersionOpt)!;
            var forDate = ctx.ParseResult.GetValueForOption(predictForDateOpt);
            var truthStation = ctx.ParseResult.GetValueForOption(predictTruthStationOpt)!;
            var window = ctx.ParseResult.GetValueForOption(predictWindowOpt)!;
            var locationOverride = ctx.ParseResult.GetValueForOption(predictLocationOpt);
            var elementTarget = ElementTargets.TryFromCli(target);
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipPredictCommand>();
                ctx.ExitCode = await cmd.RunAsync(truthStation, version, forDate, locationOverride, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "dry-window", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<DryWindowPredictCommand>();
                ctx.ExitCode = await cmd.RunAsync(truthStation, window, version, forDate, locationOverride, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "feels-like", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<FeelsLikePredictCommand>();
                ctx.ExitCode = await cmd.RunAsync(forDate, locationOverride, ctx.GetCancellationToken());
            }
            else if (elementTarget is not null)
            {
                var cmd = host.Services.GetRequiredService<ElementPredictCommand>();
                ctx.ExitCode = await cmd.RunAsync(elementTarget, version, forDate, locationOverride, ctx.GetCancellationToken());
            }
            else
            {
                var cmd = host.Services.GetRequiredService<TempPredictCommand>();
                ctx.ExitCode = await cmd.RunAsync(target, version, forDate, locationOverride, ctx.GetCancellationToken());
            }
        });
        root.AddCommand(predict);

        // ---- phase4b-predict — synthesise the 2-way mean of 4a + 3o ----
        // No options today; reads stations from config.yaml. Optional
        // --for-date supports backfill scenarios (same shape as predict).
        var p4bForDateOpt = new Option<DateOnly?>(
            name: "--for-date",
            description: "Anchor date YYYY-MM-DD (UTC). Omit = today's date (the live cycle).");
        var phase4bPredict = new Command(
            "phase4b-predict",
            "Synthesise Phase 4b predictions (mean of 4a + 3o) for the current cycle.")
        {
            p4bForDateOpt,
        };
        phase4bPredict.SetHandler(async (ctx) =>
        {
            var forDate = ctx.ParseResult.GetValueForOption(p4bForDateOpt);
            var cmd = host.Services.GetRequiredService<Phase4bPredictCommand>();
            ctx.ExitCode = await cmd.RunAsync(forDate, ctx.GetCancellationToken());
        });
        root.AddCommand(phase4bPredict);

        // ---- wind-blend-predict — sigmoid blend of wind_speed_lgb + wind_mvn ----
        // Phase 3 Slice C: composes WB-side wind_speed_lgb (Dunkeswell LGB)
        // with WP-side wind_mvn (PyTorch MLP) speed magnitude into a final
        // wind speed. No bundle/MANIFEST yet — fixed model_version tag
        // `v_wind_blend_live` for v1; bundle integration is a follow-up.
        var windBlendForDateOpt = new Option<DateOnly?>(
            name: "--for-date",
            description: "Anchor date YYYY-MM-DD (UTC). Omit = today (the live cycle).");
        var windBlendPredict = new Command(
            "wind-blend-predict",
            "Synthesise wind_blend predictions (sigmoid composition of wind_speed_lgb + wind_mvn) for the current cycle.")
        {
            windBlendForDateOpt,
        };
        windBlendPredict.SetHandler(async (ctx) =>
        {
            var forDate = ctx.ParseResult.GetValueForOption(windBlendForDateOpt);
            var cmd = host.Services.GetRequiredService<WindBlendPredictCommand>();
            ctx.ExitCode = await cmd.RunAsync(forDate, ctx.GetCancellationToken());
        });
        root.AddCommand(windBlendPredict);

        // ---- phase4b-mint — refresh the 4b bundle after a 4a/3o retrain ----
        var phase4bMint = new Command(
            "phase4b-mint",
            "Mint a fresh Phase 4b bundle from the latest 4a + 3o test_predictions (run after Sunday auto-retrain).");
        phase4bMint.SetHandler(async (ctx) =>
        {
            var cmd = host.Services.GetRequiredService<Phase4bMintCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(phase4bMint);

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
            description: "Target variable: temperature | precipitation | rainfall-amount | dry-window | wind | humidity | shortwave-radiation | cloud-cover",
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
        var verifyLocationOpt = new Option<string?>(
            name: "--location",
            description: "Scope verify to one location's predictions (e.g. 'membury_devon'). Default = primary location (Bonehill / Locations[0]). Honoured by temperature, element, and start-hour verify (Phase C commit 3b). Precip + dry-window are location-agnostic via station partitioning and ignore the flag.",
            getDefaultValue: () => null);
        var verify = new Command(
            "verify",
            "Rolling verification vs ERA5 (temperature) or EA rainfall (precipitation/dry-window), stratified by (version, lead). Flags drift.")
        {
            verifyTargetOpt, verifyAsOfOpt, verifyWindowOpt, verifyLatencyOpt, verifyDriftOpt, verifyTruthStationOpt, verifyDryWindowOpt, verifyLocationOpt,
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
            var locationOverride = ctx.ParseResult.GetValueForOption(verifyLocationOpt);

            var elementTarget = ElementTargets.TryFromCli(target);
            if (string.Equals(target, "precipitation", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<PrecipVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    truthStation, asOf, windowDays ?? 30, latencyDays, drift, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "rainfall-amount", StringComparison.OrdinalIgnoreCase)
                     || string.Equals(target, "rainfall_amount", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<RainfallAmountVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    truthStation, asOf, windowDays ?? 30, latencyDays, drift, ctx.GetCancellationToken());
            }
            else if (string.Equals(target, "dry-window", StringComparison.OrdinalIgnoreCase))
            {
                var cmd = host.Services.GetRequiredService<DryWindowVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    truthStation, dryWindow, asOf, windowDays ?? 30, latencyDays, drift, ctx.GetCancellationToken());
            }
            else if (elementTarget is not null)
            {
                var cmd = host.Services.GetRequiredService<ElementVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    elementTarget, asOf, windowDays ?? 14, latencyDays, drift, locationOverride, ctx.GetCancellationToken());
            }
            else
            {
                var cmd = host.Services.GetRequiredService<TempVerifyCommand>();
                ctx.ExitCode = await cmd.RunAsync(
                    target, asOf, windowDays ?? 14, latencyDays, drift, locationOverride, ctx.GetCancellationToken());
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

        // ---- dry-window-conformal-fit (one-shot back-fit of conformal calibrators) ----
        var conformalAlphaOpt = new Option<double>(
            name: "--alpha",
            description: "Target miscoverage rate (0.10 = 90% coverage; default 0.10).",
            getDefaultValue: () => 0.10);
        var dryWindowConformalFit = new Command(
            "dry-window-conformal-fit",
            "Back-fit conformal calibrators on the val slice for every Active dry-window version (3b)")
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
