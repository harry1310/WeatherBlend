namespace WeatherBlend.Train.Element;

/// <summary>
/// Registry of the four per-variable blenders. Adding a fifth element is a
/// one-line addition here plus a new sibling folder under <c>Train/Element/</c>
/// implementing <see cref="IElementBlender"/>.
///
/// PhaseTag is what training_metadata.json carries and what
/// <see cref="ModelArtifact.ResolveStationChampionVersion"/> matches against
/// phases.yaml's champion id. The two must agree per-target.
/// </summary>
public static class ElementTargets
{
    public static readonly ElementTarget Wind = new(
        CliName: "wind",
        ModelDirName: "wind",
        Display: "10 m wind speed",
        Units: "m/s",
        PhaseTag: "wind");

    public static readonly ElementTarget Humidity = new(
        CliName: "humidity",
        ModelDirName: "humidity",
        Display: "2 m relative humidity",
        Units: "%",
        PhaseTag: "humidity");

    public static readonly ElementTarget ShortwaveRadiation = new(
        CliName: "shortwave-radiation",
        ModelDirName: "shortwave_radiation",
        Display: "shortwave radiation",
        Units: "W/m²",
        PhaseTag: "shortwave_radiation");

    public static readonly ElementTarget CloudCover = new(
        CliName: "cloud-cover",
        ModelDirName: "cloud_cover",
        Display: "total cloud cover",
        Units: "%",
        PhaseTag: "cloud_cover");

    // Wind gust — separate target from wind (speed/direction). 4 production
    // NWPs with gust forecasts on Open-Meteo (GFS / ICON / GEM / UKMO);
    // ERA5 WindGusts10m as training truth. PhaseTag suffix matches the plan
    // doc's id (WIND_BLENDER_PLAN.md) — leaves room for a future
    // wind_gust_mvn or similar challenger without re-keying the manifest.
    public static readonly ElementTarget WindGust = new(
        CliName: "wind-gust",
        ModelDirName: "wind_gust",
        Display: "10 m wind gust",
        Units: "m/s",
        PhaseTag: "wind_gust_lgb");

    // NOTE: wind_speed_lgb (the Dunkeswell-SYNOP-truth sibling phase under
    // ModelDirName='wind') is NOT an ElementTarget any more — it moved to
    // Python 2026-06-10 (Option B cutover: quantile-LGB + cross-conformal
    // CQR in WeatherProbabilistic's train/predict_wind_speed_pi.py, which
    // emits the same ElementPredictionRow parquet plus a CQR band sidecar).
    // Its versions still appear in the wind MANIFEST Active list (promoted
    // by the Python trainer as challenger); ElementPredictCommand skips
    // them via HandledElementPhases, and WindBlendPredictCommand consumes
    // the Python-written predictions by model_version glob.
    public static readonly IReadOnlyList<ElementTarget> All =
        new[] { Wind, Humidity, ShortwaveRadiation, CloudCover, WindGust };

    public static ElementTarget? TryFromCli(string cliName)
        => All.FirstOrDefault(t => string.Equals(t.CliName, cliName, StringComparison.OrdinalIgnoreCase));

    public static ElementTarget FromCli(string cliName)
        => TryFromCli(cliName)
           ?? throw new ArgumentException(
               $"Unknown element target '{cliName}'. Known: {string.Join(", ", All.Select(t => t.CliName))}");
}
