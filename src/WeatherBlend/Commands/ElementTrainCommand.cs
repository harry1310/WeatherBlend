using Microsoft.Extensions.Logging;
using WeatherBlend.Train.Common;
using WeatherBlend.Train.Element;

namespace WeatherBlend.Commands;

/// <summary>
/// Train one of the per-variable element blenders (wind, humidity,
/// shortwave-radiation, cloud-cover). Picks the right <see cref="IElementBlender"/>
/// off the registered list by CLI target name and delegates the actual training to
/// it; the per-element blender owns its row class, feature builder, and any quirks
/// (e.g. wind excludes MétéoFrance).
/// </summary>
public sealed class ElementTrainCommand
{
    private readonly ILogger<ElementTrainCommand> _log;
    private readonly IReadOnlyList<IElementBlender> _blenders;

    private static readonly int[] DefaultLeads = Leads.Short;

    public ElementTrainCommand(ILogger<ElementTrainCommand> log, IEnumerable<IElementBlender> blenders)
    {
        _log = log;
        _blenders = blenders.ToList();
    }

    public Task<int> RunAsync(ElementTarget target, int[] leads, CancellationToken ct)
    {
        var blender = _blenders.FirstOrDefault(b => b.Target == target);
        if (blender is null)
        {
            _log.LogError(
                "No registered blender for target '{Target}'. Registered: [{Have}]",
                target.CliName, string.Join(", ", _blenders.Select(b => b.Target.CliName)));
            return Task.FromResult(2);
        }
        return blender.TrainAsync(leads, ct);
    }

    public static int[] DefaultLeadsArray => DefaultLeads;
}
