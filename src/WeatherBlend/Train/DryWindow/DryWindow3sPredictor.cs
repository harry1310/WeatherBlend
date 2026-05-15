namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3s — parameter-free iid Monte-Carlo dry-window predictor over
/// Phase 3e's hourly P(wet). The 3g algorithm exactly (independent-hour
/// Bernoulli sampling of the daytime q-vector, structural longest-dry-run
/// extraction), with one difference: the marginal source is the 3e MLP
/// occurrence blender instead of the 3a LightGBM blender.
///
/// Why 3s exists — the 2026-05-15 start-hour bake-off
/// (scripts/DryWindowStartHour/phase3g_start_hour_bakeoff.py):
///   * On the dry-window OCCURRENCE question (P(∃ N-hour dry block)) 3e
///     marginals ≈ 3a marginals — 3s is NOT an occurrence improvement
///     over 3g, it ties it.
///   * On the START question (where in the day the block begins) 3e's
///     sharper hourly P(wet) localises the block markedly better:
///     6h start-hour Brier −4.5% / Top-1 +2.7pt vs 3g, improvement at
///     every window {3,4,6}. The copula samplers (3j/3n) did NOT help
///     the start question — plain iid over 3e is the winner.
/// So 3s ships as a dry-window challenger whose value is the start-hour
/// curve; its occurrence card reads ≈ 3g by design.
///
/// There is deliberately no MC code in this class. Every Monte-Carlo
/// primitive — <see cref="DryWindow3gPredictor.SampleStartHourCurve"/>,
/// <see cref="DryWindow3gPredictor.ProbDryWindow(double[],System.Collections.Generic.IReadOnlyList{int},System.Random,int)"/>,
/// <see cref="DryWindow3gPredictor.SampleStats"/>,
/// <see cref="DryWindow3gPredictor.ExtractDaytimeQ"/>,
/// <see cref="DryWindow3gPredictor.LoadLivePredictionsHourly"/> — is
/// source-agnostic (it operates on a bare hourly q-vector), so 3s reuses
/// it verbatim. This class is just the phase identity + the metadata key
/// naming the 3e champion the predictor binds to.
/// </summary>
public static class DryWindow3sPredictor
{
    /// <summary>Phase tag persisted in <c>training_metadata.Phase</c> and
    /// used as the bundle-dir suffix (<c>v..._phase3s</c>).</summary>
    public const string Phase3s = "3s";

    /// <summary>
    /// Hyperparameter key under which the 3s bundle records the Phase 3e
    /// champion version it binds to — the sibling of 3g's
    /// <c>precip_3a_version</c>. Predict reads this to locate the live 3e
    /// hourly-P(wet) parquet to Monte-Carlo over.
    /// </summary>
    public const string Precip3eVersionKey = "precip_3e_version";

    /// <summary>MC sample count — shared with 3g so 3s's occurrence prob and
    /// its start-hour curve carry the same sampling noise as the rest of the
    /// dry-window family.</summary>
    public const int DefaultMcSamples = DryWindow3gPredictor.DefaultMcSamples;
}
