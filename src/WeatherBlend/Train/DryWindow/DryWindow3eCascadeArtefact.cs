using Microsoft.ML;
using WeatherBlend.Train;
using WeatherBlend.Train.Common;
using CommonRow = WeatherBlend.Train.Common.DryWindowTrainingRow;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Per-lead artefact layout + scoring helpers for the Phase 3e cascade.
///
/// On disk, each (station, output-window) version dir under
/// <c>data/models/dry_window/{station}/window_{N}h/v{ts}_phase3e/</c>
/// contains:
///   <c>lead_{L}h.zip</c>           — M_base (P(3h dry block)). At window_3h
///                                    this IS the prediction; at window_4h
///                                    it's the base of the cascade.
///   <c>lead_{L}h_extend.zip</c>    — M_extend4 (P(extends to 4h | has 3h)).
///                                    Present only at window_4h.
///   <c>calibrator_{L}h.json</c>    — PAV the SHIPPING probability against
///                                    the matching truth. At window_3h that's
///                                    M_base vs 3h truth. At window_4h that's
///                                    the PRODUCT (M_base × M_extend4) vs 4h
///                                    truth. PAV-on-product matches what the
///                                    user reads on the page.
///
/// The window_3h artefact is identical in shape to a 3b artefact — same
/// LightGBM zip, same calibrator filename — so the existing
/// <see cref="DryWindowTrainer.PredictVectorProbability"/> +
/// <see cref="ModelArtifact.LoadLeadModel"/> path scores it without
/// modification. Only the window_4h cascade needs the helpers below.
/// </summary>
public static class DryWindow3eCascadeArtefact
{
    /// <summary>Filename for the M_extend4 LightGBM zip at a given lead.</summary>
    public static string ExtendModelFileName(int leadHours) => $"lead_{leadHours}h_extend.zip";

    /// <summary>Suffix appended to the version dir name so 3e is visually
    /// distinguishable from 3b in MANIFEST.Active and on the Models page.
    /// Mirrors the 3d-shape convention.</summary>
    public const string VersionSuffix = "phase3e";

    /// <summary>
    /// Score the 4h cascade for a vector of feature rows. Returns the raw
    /// P(4h dry) = M_base · M_extend4 pre-calibration. Caller PAVs (or not)
    /// downstream — separate so a trainer fitting the calibrator can call
    /// this on the validation split first.
    /// </summary>
    public static double[] PredictRawProductForExtend(
        MLContext ml,
        ITransformer baseModel,
        ITransformer extendModel,
        BlenderSpec spec,
        IReadOnlyList<CommonRow> rows)
    {
        if (rows.Count == 0) return Array.Empty<double>();
        var pBase   = DryWindowTrainer.PredictVectorProbability(ml, baseModel,   spec, rows);
        var pExtend = DryWindowTrainer.PredictVectorProbability(ml, extendModel, spec, rows);
        return MultiplyForCascade(pBase, pExtend);
    }

    /// <summary>
    /// Pure cascade arithmetic, separated so unit tests can pin the
    /// "P(4h) = clamp(P(3h)) · clamp(P(extend))" rule and the implied
    /// monotonicity invariant <c>output[i] ≤ pBase[i]</c> without spinning
    /// up an ML.NET pipeline. Both inputs must be the same length;
    /// out-of-range factors are clamped to [0, 1] before multiplying.
    /// </summary>
    public static double[] MultiplyForCascade(IReadOnlyList<double> pBase, IReadOnlyList<double> pExtend)
    {
        if (pBase.Count != pExtend.Count)
            throw new ArgumentException(
                $"pBase ({pBase.Count}) and pExtend ({pExtend.Count}) must be the same length.");
        var product = new double[pBase.Count];
        for (int i = 0; i < pBase.Count; i++)
        {
            // P(4h) = P(3h) · P(extends | has 3h). Both factors are in [0,1] so
            // the product is too — monotonic-by-construction: P(4h) ≤ P(3h).
            product[i] = Math.Clamp(pBase[i], 0.0, 1.0) * Math.Clamp(pExtend[i], 0.0, 1.0);
        }
        return product;
    }
}
