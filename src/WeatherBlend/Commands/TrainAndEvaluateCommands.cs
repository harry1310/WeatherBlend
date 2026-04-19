using Microsoft.Extensions.Logging;

namespace WeatherBlend.Commands;

/// <summary>
/// Phase 2: LightGBM blender, one model per lead-time group, temperature target first.
/// Feature set: every model's forecast of every variable at that lead time, plus
/// inter-model mean/std/range (disagreement = uncertainty), plus cyclical calendar
/// features (hour-of-day, day-of-year), plus recent observation persistence features.
/// Time-based train/test split (no random shuffling - that would leak).
///
/// Stubbed for now; wire up once the collector has accumulated real data.
/// </summary>
public sealed class TrainCommand
{
    private readonly ILogger<TrainCommand> _log;
    public TrainCommand(ILogger<TrainCommand> log) { _log = log; }

    public Task<int> RunAsync(string target, CancellationToken ct)
    {
        _log.LogInformation("TODO: train blender for target '{Target}'", target);
        _log.LogInformation("Planned approach:");
        _log.LogInformation("  1. Load forecasts+obs via DuckDB, pivot model->columns");
        _log.LogInformation("  2. Feature engineering (spread, calendar, persistence)");
        _log.LogInformation("  3. Time-based split, LightGBM per lead-time bucket");
        _log.LogInformation("  4. Persist model + feature schema to data/models/");
        return Task.FromResult(0);
    }
}

public sealed class EvaluateCommand
{
    private readonly ILogger<EvaluateCommand> _log;
    public EvaluateCommand(ILogger<EvaluateCommand> log) { _log = log; }

    public Task<int> RunAsync(CancellationToken ct)
    {
        _log.LogInformation("TODO: evaluation report");
        _log.LogInformation("Planned metrics:");
        _log.LogInformation("  - MAE / RMSE per lead time per model and for blend");
        _log.LogInformation("  - Baselines: persistence, climatology, mean-of-models, best single");
        _log.LogInformation("  - Precip: Brier @ 0.1/1/5mm, reliability diagram, CRPS");
        _log.LogInformation("  - Output: markdown report in data/reports/");
        return Task.FromResult(0);
    }
}
