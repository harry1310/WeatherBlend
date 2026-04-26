namespace WeatherBlend.Train.Element;

/// <summary>
/// Registry of the four per-variable blenders. Adding a fifth element is a
/// one-line addition here plus a new sibling folder under <c>Train/Element/</c>
/// implementing <see cref="IElementBlender"/>.
///
/// PhaseTag is what training_metadata.json carries and what predict-side dispatch
/// keys on. Per-target tags (`lean-wind`, `lean-humidity`, ...) keep the predict
/// dispatcher unambiguous; rich variants would land as `rich-wind` etc. when added.
/// </summary>
public static class ElementTargets
{
    public static readonly ElementTarget Wind = new(
        CliName: "wind",
        ModelDirName: "wind",
        Display: "10 m wind speed",
        Units: "m/s",
        PhaseTag: "lean-wind");

    public static readonly ElementTarget Humidity = new(
        CliName: "humidity",
        ModelDirName: "humidity",
        Display: "2 m relative humidity",
        Units: "%",
        PhaseTag: "lean-humidity");

    public static readonly ElementTarget ShortwaveRadiation = new(
        CliName: "shortwave-radiation",
        ModelDirName: "shortwave_radiation",
        Display: "shortwave radiation",
        Units: "W/m²",
        PhaseTag: "lean-shortwave-radiation");

    public static readonly ElementTarget CloudCover = new(
        CliName: "cloud-cover",
        ModelDirName: "cloud_cover",
        Display: "total cloud cover",
        Units: "%",
        PhaseTag: "lean-cloud-cover");

    public static readonly IReadOnlyList<ElementTarget> All =
        new[] { Wind, Humidity, ShortwaveRadiation, CloudCover };

    public static ElementTarget? TryFromCli(string cliName)
        => All.FirstOrDefault(t => string.Equals(t.CliName, cliName, StringComparison.OrdinalIgnoreCase));

    public static ElementTarget FromCli(string cliName)
        => TryFromCli(cliName)
           ?? throw new ArgumentException(
               $"Unknown element target '{cliName}'. Known: {string.Join(", ", All.Select(t => t.CliName))}");

    public static ElementTarget? TryFromPhaseTag(string phaseTag)
        => All.FirstOrDefault(t => string.Equals(t.PhaseTag, phaseTag, StringComparison.OrdinalIgnoreCase));
}
