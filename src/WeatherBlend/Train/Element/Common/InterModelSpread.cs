namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Mean / population std / range across the per-model values for one element.
/// Computed over non-NaN entries only — for elements where one model is always
/// missing (e.g. wind: MF) the spread is over the present 5, not 6.
///
/// Double-based since the 2026-06-10 adoption pass: every ComposeRow call site
/// accumulates these stats in DOUBLE over double inputs and only casts to
/// float when packing the feature vector, so the helper does the same —
/// switching the accumulator to float here would change the packed bits and
/// break the feature-builder unit tests that pin exact packing. Variance is
/// population (N, not N-1 — matches numpy default; fine for a feature) and is
/// clamped at 0 against floating-point error before the sqrt.
///
/// All-NaN / empty input yields (NaN, NaN, NaN) — LightGBM handles the NaN
/// slots natively, and predict-side card writers map NaN→null themselves.
///
/// Deliberate non-adopters (different maths — do NOT funnel them through
/// here without a bake-off):
///   * WindGustFeatureBuilder.ComposeRow's ratio_mean/ratio_std — float
///     accumulation over float ratios (x*x is a FLOAT multiply there), no
///     min/max. Routing it through this double-based helper would change
///     the trained feature bits.
///   * PrecipFeatureBuilder's mean/std/max/agreement block — max (not
///     range) plus the wet-agreement fraction; a different stat set.
///   * TempRich / DryWindow multi-variable aggregate blocks — mean/std
///     pairs without range, interleaved across several variables.
/// </summary>
public readonly record struct InterModelSpread(double Mean, double Std, double Range)
{
    public static InterModelSpread From(IReadOnlyList<double> values)
    {
        double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
        int n = 0;
        for (int i = 0; i < values.Count; i++)
        {
            var x = values[i];
            if (double.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            n++;
        }
        if (n == 0) return new InterModelSpread(double.NaN, double.NaN, double.NaN);
        var mean = sum / n;
        var variance = Math.Max(0.0, (sumSq / n) - (mean * mean));
        return new InterModelSpread(
            Mean: mean,
            Std: Math.Sqrt(variance),
            Range: max - min);
    }
}
