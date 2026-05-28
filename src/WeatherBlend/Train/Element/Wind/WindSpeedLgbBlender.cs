using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train.Element.Common;
using WeatherBlend.Train.Oro;

namespace WeatherBlend.Train.Element.Wind;

/// <summary>
/// Wind-speed LightGBM blender trained on **Dunkeswell SYNOP** truth
/// (MIDAS Open 01383) rather than ERA5. Phase 3 of WIND_BLENDER_PLAN
/// (locked 2026-05-27). Sibling phase under <see cref="ElementTargets.Wind"/>'s
/// ModelDirName — bundle lives under <c>data/models/wind/{location}/v{ts}_wind_speed_lgb/</c>,
/// alongside (not replacing) the existing ERA5-truth <see cref="WindBlender"/>.
///
/// Wires <see cref="WindSpeedLgbFeatureBuilder"/> into the generic
/// <see cref="ElementTrainerHarness"/>. Loads the per-location
/// <see cref="OroStaticFeatures"/> JSON at the start of each TrainAsync
/// call so the 5 ORO_LEAN features can be derived per row.
///
/// The minted bundle feeds the Phase 3 <c>wind_blend</c> mint command
/// (sigmoid composition of <c>wind_speed_lgb</c> + <c>wind_mvn</c>'s
/// speed magnitude) — not yet shipped in Slice 3.A.
/// </summary>
public sealed class WindSpeedLgbBlender : IElementBlender
{
    private readonly ILogger<WindSpeedLgbBlender> _log;
    private readonly AppConfig _cfg;

    public WindSpeedLgbBlender(ILogger<WindSpeedLgbBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.WindSpeedLgb;

    public Task<int> TrainAsync(int[] leads, LocationConfig location, CancellationToken ct)
    {
        // Both the orographic JSON and the MIDAS archive resolve
        // relative to the parent of ForecastsPath — that's the
        // canonical "data tree root" in production
        // (ForecastsPath="data/forecasts" → parent="data"; static lives
        // at data/static/orographic, MIDAS at data/truth/midas/raw).
        // Using a config-derived absolute path here (rather than
        // CWD-relative resolution) keeps every TrainAsync invocation
        // self-contained — no Environment.CurrentDirectory mutation
        // needed by callers, which means smoke tests can run in
        // parallel without colliding on the process-global CWD.
        var dataRoot = Path.GetDirectoryName(_cfg.Storage.ForecastsPath)!;
        var staticRoot = Path.Combine(dataRoot, "static", "orographic");
        var oroAll = OroStaticFeatures.LoadAll(staticRoot);
        if (!oroAll.TryGetValue(location.Name, out var oro))
            throw new InvalidOperationException(
                $"Missing orographic JSON for location '{location.Name}' under {staticRoot} — " +
                $"wind_speed_lgb's ORO_LEAN features need it. Generate via " +
                $"scripts/OrographicFeatures/build_static_orographic.py.");

        // MIDAS Open archive lives in the same data-tree root. Path is
        // mirrored in retrain-python.yml (Dunkeswell pull) and the
        // Phase 2 wind_mvn loader.
        var midasRoot = Path.Combine(dataRoot, "truth", "midas", "raw");

        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            // Same anti-bagging tweak the existing WindBlender uses —
            // 2026-04-26 bake-off showed bagging hurts wind at every lead.
            // Wind speed truth swap doesn't change the result; bagging
            // still rejected here.
            Hyperparameters: TempTrainer.Hyperparameters.Default() with
            {
                SubsampleFraction = 1.0, SubsampleFrequency = 0, FeatureFraction = 1.0,
            },
            ModelsRoot: _cfg.Storage.ModelsPath,
            BuildSpec: lead => WindSpeedLgbFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => WindSpeedLgbFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, midasRoot, location.Name, oro, spec, c),
            DeviationsFromBrief: new[]
            {
                "Truth = Dunkeswell SYNOP (MIDAS Open station 01383), NOT ERA5. " +
                "Production-scope bake-off 2026-05-27 picked this over ERA5 because " +
                "Dunkeswell is real obs (not reanalysis interpolation) and the " +
                "Bonehill→Dunkeswell wind correlation is high enough (~0.85+) for " +
                "Dunkeswell to act as a denser, more honest training signal.",
                "29 features (lean+ORO+spread): 6 wsp + 6 wdir (RAW degrees) + " +
                "5 ORO_LEAN + 12 SPREAD (mean+std × {wsp, gust, t, td, p, cc}). " +
                "Bake-off ladder rejected rich variants — 'rich + oro' collapses " +
                "to NWP-mean MAE at this data scale.",
                "AIFS excluded entirely (zero 2024 rows in Open-Meteo Previous Runs " +
                "for the Dunkeswell training window). MétéoFrance excluded " +
                "(no MF wind on Open-Meteo Previous Runs).",
                "Bagging disabled (same as the existing WindBlender; bake-off 2026-04-26 " +
                "showed bagging hurts wind at every lead, target-agnostic).",
            },
            LocationName: location.Name);

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }

}
