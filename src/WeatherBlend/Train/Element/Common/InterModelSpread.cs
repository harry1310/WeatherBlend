namespace WeatherBlend.Train.Element.Common;

/// <summary>
/// Mean / population std / range across the per-model values for one element.
/// Computed over non-NaN entries only — for elements where one model is always
/// missing (e.g. wind: MF) the spread is over the present 5, not 6.
/// </summary>
public readonly record struct InterModelSpread(float Mean, float Std, float Range)
{
    public static InterModelSpread From(ReadOnlySpan<float> values)
    {
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        int n = 0;
        foreach (var x in values)
        {
            if (float.IsNaN(x)) continue;
            sum += x;
            sumSq += x * x;
            if (x < min) min = x;
            if (x > max) max = x;
            n++;
        }
        if (n == 0) return new InterModelSpread(float.NaN, float.NaN, float.NaN);
        var mean = sum / n;
        var variance = Math.Max(0.0, (sumSq / n) - (mean * mean));
        return new InterModelSpread(
            Mean: (float)mean,
            Std:  (float)Math.Sqrt(variance),
            Range: (float)(max - min));
    }
}
