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
                .AddStandardResilienceHandler();

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

                // OGIMET: longer timeout, no aggressive retry — the rate limit means
                // a hammered retry loop is the fastest way to get the IP blocked.
                services.AddHttpClient<OgimetClient>(c =>
                {
                    c.Timeout = TimeSpan.FromSeconds(120);
                    c.DefaultRequestHeaders.UserAgent.ParseAdd(cfg.Http.UserAgent);
                });

                services.AddTransient<CollectCommand>();
                services.AddTransient<BackfillCommand>();
                services.AddTransient<StatusCommand>();
                services.AddTransient<TrainCommand>();
                services.AddTransient<EvaluateCommand>();
                services.AddTransient<InspectCommand>();
                services.AddTransient<CompareCommand>();
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
            description: "forecasts | era5 | metar | all",
            getDefaultValue: () => "all");
        var backfill = new Command("backfill", "Fetch historical data (forecasts, ERA5, OGIMET METAR)")
            { sourceOpt, startOpt, endOpt };
        backfill.SetHandler(async (source, start, end) =>
        {
            var cmd = host.Services.GetRequiredService<BackfillCommand>();
            await cmd.RunAsync(source, start, end, CancellationToken.None);
        }, sourceOpt, startOpt, endOpt);
        root.AddCommand(backfill);

        var status = new Command("status", "Show what data is on disk");
        status.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<StatusCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(status);

        var targetOpt = new Option<string>(
            name: "--target",
            description: "Target variable: temperature | precip_occurrence | precip_amount",
            getDefaultValue: () => "temperature");
        var train = new Command("train", "Train the blender (stubbed in phase 1)") { targetOpt };
        train.SetHandler(async (target) =>
        {
            var cmd = host.Services.GetRequiredService<TrainCommand>();
            await cmd.RunAsync(target, CancellationToken.None);
        }, targetOpt);
        root.AddCommand(train);

        var evaluate = new Command("evaluate", "Run verification report (stubbed in phase 1)");
        evaluate.SetHandler(async ctx =>
        {
            var cmd = host.Services.GetRequiredService<EvaluateCommand>();
            ctx.ExitCode = await cmd.RunAsync(ctx.GetCancellationToken());
        });
        root.AddCommand(evaluate);

        var pathOpt = new Option<string>("--path", "Path to a parquet file") { IsRequired = true };
        var inspect = new Command("inspect", "Dump a parquet file") { pathOpt };
        inspect.SetHandler(async (path) =>
        {
            var cmd = host.Services.GetRequiredService<InspectCommand>();
            await cmd.RunAsync(path, CancellationToken.None);
        }, pathOpt);
        root.AddCommand(inspect);

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
