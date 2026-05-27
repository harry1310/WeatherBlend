using Microsoft.Extensions.Logging;
using WeatherBlend.Config;
using WeatherBlend.Train;
using WeatherBlend.Train.Element.Common;

namespace WeatherBlend.Train.Element.Gust;

/// <summary>
/// Wind-gust blender — wires <see cref="WindGustFeatureBuilder"/> into the generic
/// <see cref="ElementTrainerHarness"/>. Discovered by DI and dispatched to from
/// <c>ElementTrainCommand</c> when <c>--target wind-gust</c>.
///
/// Hyperparameters: <see cref="TempTrainer.Hyperparameters.Default"/>. The
/// production-scope bake-off 2026-05-27 (gust_4nwp_variants_bakeoff.py) used
/// LightGBM defaults with no bagging tweak — match that here. Wind-blender's
/// bagging opt-out was based on a wind-specific bake-off and doesn't apply.
/// </summary>
public sealed class WindGustBlender : IElementBlender
{
    private readonly ILogger<WindGustBlender> _log;
    private readonly AppConfig _cfg;

    public WindGustBlender(ILogger<WindGustBlender> log, AppConfig cfg)
    {
        _log = log;
        _cfg = cfg;
    }

    public ElementTarget Target => ElementTargets.WindGust;

    public Task<int> TrainAsync(int[] leads, LocationConfig location, CancellationToken ct)
    {
        var inputs = new ElementTrainerHarness.ElementTrainerInputs(
            Target: Target,
            Hyperparameters: TempTrainer.Hyperparameters.Default(),
            ModelsRoot: _cfg.Storage.ModelsPath,
            BuildSpec: lead => WindGustFeatureBuilder.BuildSpec(_cfg.Blenders, lead),
            LoadRowsForSpec: (spec, c) => WindGustFeatureBuilder.BuildForLead(
                _cfg.Storage.ForecastsPath, _cfg.Storage.Era5Path, location.Name, spec, c),
            DeviationsFromBrief: new[]
            {
                "ERA5 WindGusts10m as truth. Dunkeswell SYNOP gust is too sparse " +
                "(241 rows >9 m/s in 2022, zero coverage 2023-24); MO DataHub geohash " +
                "gust live since 2026-04-24 but ~30 days history — revisit ~6-12 months.",

                "4 production NWPs with gust on Open-Meteo: GFS / ICON / GEM required, " +
                "UKMO optional (post-2024-09 clean window). ECMWF / MF / JMA / AIFS " +
                "publish no gust on Open-Meteo Previous Runs (audit 2026-05-27).",

                "10-feature minimal variant (per-NWP gust + per-NWP gust/wsp ratio " +
                "clipped [0.5, 4.0] + ratio mean/std). Bake-off 2026-05-27: MAE 1.0418 " +
                "(-18.4% vs NWP-mean 1.2764) on 2024 Bonehill, 1,251 test rows. Tied " +
                "with the 24-feat rich-no-oro variant; orographic features HURT at " +
                "4-NWP scope (see project_gust_production_scope_2026-05-27.md).",

                "Objective is L2; MAE used only as early-stopping metric " +
                "(Microsoft.ML.LightGbm 4.0 limit). No monotone constraints " +
                "(same library limit).",
            },
            LocationName: location.Name);

        return ElementTrainerHarness.RunAsync(_log, inputs, leads, ct);
    }
}
