using Microsoft.Extensions.Logging;
using WeatherBlend.Config;

namespace WeatherBlend.Commands;

/// <summary>
/// Shared scaffold for the per-target train commands — <see cref="TempTrainCommand"/>
/// (Phase 2b/2c/2d) and <see cref="PrecipTrainCommand"/> (Phase 3a/3c/3d/3e).
///
/// Holds only the logger + config every phase trainer needs. The per-phase
/// fit/guard/promote bodies stay in the subclasses: the temperature loop is a
/// regression against ERA5 (MAE) and the precipitation loop is a binary
/// classifier against EA gauge truth (Brier), so a single shared per-lead loop
/// would obscure more than it dedups. The genuinely shared scaffold already
/// lives in the static helpers (<c>RetrainGuard.BuildCheckAndSave</c>,
/// <c>ModelArtifact.PromoteStationVersion</c>, <c>ModelArtifact.SaveTrainingMetadata</c>).
/// </summary>
public abstract class TrainCommandBase
{
    protected readonly ILogger _log;
    protected readonly AppConfig _cfg;

    protected TrainCommandBase(ILogger log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }
}
