namespace WeatherBlend.Train.Element;

/// <summary>
/// Per-variable blender identity. One per weather "element" — wind, humidity, radiation,
/// cloud cover. Pure data: name + display + storage layout. The actual feature shape
/// and SQL live in the per-element <c>TempFeatureBuilder</c> + <c>Blender</c> classes,
/// because each element pulls a different set of forecast columns and would not benefit
/// from a single generic builder (audit 2026-04-25 confirmed wind has direction, humidity
/// has dewpoint, radiation has direct/diffuse, cloud has only total available).
///
/// CLI uses dashes (`shortwave-radiation`); model dir uses underscores
/// (`shortwave_radiation`) — matches the existing dry-window convention.
/// </summary>
public sealed record ElementTarget(
    string CliName,
    string ModelDirName,
    string Display,
    string Units,
    string PhaseTag);
