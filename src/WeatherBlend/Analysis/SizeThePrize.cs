namespace WeatherBlend.Analysis;

/// <summary>
/// Reusable "size the prize" machinery for the residual-explained-variance and
/// multi-seed metric-delta audits that recur every time we ask "does feature/signal
/// X help blender Y?". Extracted 2026-06-21 after the same OLS / R² / Corr / multi-seed
/// code was hand-copied into four one-off harnesses in one session (LongwaveBakeoff,
/// PwatCin/CapeBakeoff, CloudBase/Visibility, GeoNeighbourProbe).
///
/// The PATTERN these all share:
///   1. Train the live baseline blender, take its TEST residuals (truth − prediction).
///   2. Regress the residuals on a candidate design matrix → OOS R². Near 0 ⇒ no
///      exploitable signal (the baseline already captures it); clearly positive ⇒ worth
///      wiring the feature in.  (<see cref="ResidualR2"/> / <see cref="ResidualR2OutOfSample"/>)
///   3. Confirm with a multi-seed metric (Brier for precip, MAE for temp): train
///      baseline vs augmented arm across N seeds, report mean Δ% ± std + win count, so a
///      small gain can be told apart from seed noise.  (<see cref="MultiSeedDelta"/>)
///
/// Only the math lives here — sourcing candidate features and building the baseline stay
/// in each (throwaway) harness. Pure functions, no dependencies, no production code path
/// touches this; it's analysis tooling kept tracked + tested so the next audit is a
/// 20-line script instead of a 250-line copy.
/// </summary>
public static class SizeThePrize
{
    /// <summary>Ordinary least squares via the normal equations with Gaussian
    /// elimination (partial pivoting). Returns the coefficient vector β minimising
    /// ‖Xβ − y‖². <paramref name="x"/> rows are design vectors (include your own
    /// intercept column of 1s if you want one); <paramref name="y"/> the target.
    /// A (near-)singular column is left at β=0 rather than throwing.</summary>
    public static double[] Ols(IReadOnlyList<double[]> x, IReadOnlyList<double> y)
    {
        int m = x.Count, p = m > 0 ? x[0].Length : 0;
        var a = new double[p, p + 1];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++) { double s = 0; for (int r = 0; r < m; r++) s += x[r][i] * x[r][j]; a[i, j] = s; }
            double sy = 0; for (int r = 0; r < m; r++) sy += x[r][i] * y[r]; a[i, p] = sy;
        }
        for (int c = 0; c < p; c++)
        {
            int piv = c; for (int r = c + 1; r < p; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[piv, c])) piv = r;
            for (int j = 0; j <= p; j++) (a[c, j], a[piv, j]) = (a[piv, j], a[c, j]);
            double d = a[c, c]; if (Math.Abs(d) < 1e-12) continue;
            for (int j = c; j <= p; j++) a[c, j] /= d;
            for (int r = 0; r < p; r++) { if (r == c) continue; double f = a[r, c]; for (int j = c; j <= p; j++) a[r, j] -= f * a[c, j]; }
        }
        var beta = new double[p]; for (int i = 0; i < p; i++) beta[i] = a[i, p];
        return beta;
    }

    /// <summary>Coefficient of determination (1 − SSR/SST) of <paramref name="y"/>
    /// against the predictions X·<paramref name="beta"/>. &gt;0 ⇒ the design explains
    /// some variance; ≤0 ⇒ worse than the mean. 0 when y is constant.</summary>
    public static double R2(IReadOnlyList<double[]> x, IReadOnlyList<double> y, double[] beta)
    {
        int m = x.Count; if (m == 0) return 0;
        double mean = 0; for (int r = 0; r < m; r++) mean += y[r]; mean /= m;
        double ssr = 0, sst = 0;
        for (int r = 0; r < m; r++)
        {
            double yhat = 0; for (int j = 0; j < beta.Length; j++) yhat += x[r][j] * beta[j];
            ssr += (y[r] - yhat) * (y[r] - yhat);
            sst += (y[r] - mean) * (y[r] - mean);
        }
        return sst > 0 ? 1.0 - ssr / sst : 0.0;
    }

    /// <summary>IN-SAMPLE residual size-the-prize: fit OLS on (<paramref name="x"/>,
    /// <paramref name="residuals"/>) and score R² on the same set — "how much of THESE
    /// residuals does the candidate design explain". Slightly optimistic (no holdout);
    /// use <see cref="ResidualR2OutOfSample"/> when you have a train/test split of the
    /// residuals. Returns 0 if fewer than <paramref name="minRows"/> rows.</summary>
    public static double ResidualR2(IReadOnlyList<double[]> x, IReadOnlyList<double> residuals, int minRows = 10)
        => x.Count < minRows ? 0.0 : R2(x, residuals, Ols(x, residuals));

    /// <summary>OUT-OF-SAMPLE residual size-the-prize: fit β on the TRAIN residuals,
    /// score R² on the held-out TEST residuals. The honest "does this candidate explain
    /// errors it hasn't seen". Returns 0 if either side is below <paramref name="minRows"/>.</summary>
    public static double ResidualR2OutOfSample(
        IReadOnlyList<double[]> xTrain, IReadOnlyList<double> rTrain,
        IReadOnlyList<double[]> xTest, IReadOnlyList<double> rTest, int minRows = 10)
        => xTrain.Count < minRows || xTest.Count < minRows ? 0.0 : R2(xTest, rTest, Ols(xTrain, rTrain));

    /// <summary>Pearson correlation of design column <paramref name="col"/> with
    /// <paramref name="y"/> — the quick "which candidate feature actually moves with the
    /// residual" diagnostic. 0 if either side is constant.</summary>
    public static double Corr(IReadOnlyList<double[]> x, int col, IReadOnlyList<double> y)
    {
        int m = x.Count; if (m == 0) return 0;
        double mx = 0, my = 0; for (int r = 0; r < m; r++) { mx += x[r][col]; my += y[r]; } mx /= m; my /= m;
        double sxy = 0, sxx = 0, syy = 0;
        for (int r = 0; r < m; r++) { double dx = x[r][col] - mx, dy = y[r] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
        return (sxx > 0 && syy > 0) ? sxy / Math.Sqrt(sxx * syy) : 0.0;
    }

    /// <summary>Multi-seed metric delta: for each seed, <paramref name="baselineMetric"/>
    /// and <paramref name="armMetric"/> each train their model with that seed and return
    /// a test metric where LOWER IS BETTER (Brier, MAE). Reports the mean % change (arm
    /// vs baseline), the sample std across seeds (so a small Δ% can be told from noise),
    /// and how many seeds the arm beat baseline. Train each model fresh per seed.</summary>
    public static SeedDelta MultiSeedDelta(
        IReadOnlyList<int> seeds, Func<int, double> baselineMetric, Func<int, double> armMetric)
    {
        var baseVals = new List<double>(seeds.Count);
        var deltas = new List<double>(seeds.Count);
        int wins = 0;
        foreach (var seed in seeds)
        {
            double b = baselineMetric(seed), a = armMetric(seed);
            baseVals.Add(b);
            double d = b != 0 ? 100.0 * (a - b) / b : 0.0;
            deltas.Add(d);
            if (a < b) wins++;
        }
        return new SeedDelta(Mean(baseVals), Mean(deltas), Std(deltas), wins, seeds.Count);
    }

    private static double Mean(IReadOnlyList<double> xs)
    {
        if (xs.Count == 0) return 0; double s = 0; foreach (var x in xs) s += x; return s / xs.Count;
    }

    private static double Std(IReadOnlyList<double> xs)
    {
        if (xs.Count < 2) return 0;
        double m = Mean(xs), s = 0; foreach (var x in xs) s += (x - m) * (x - m);
        return Math.Sqrt(s / (xs.Count - 1));   // sample std across seeds
    }
}

/// <summary>Result of a <see cref="SizeThePrize.MultiSeedDelta"/> run.</summary>
/// <param name="BaseMean">Mean baseline metric across seeds (the reference level).</param>
/// <param name="MeanDeltaPct">Mean % change of the arm vs baseline (negative = arm wins).</param>
/// <param name="StdPct">Sample std of the per-seed Δ% — if it exceeds |MeanDeltaPct| the result is inside seed noise.</param>
/// <param name="Wins">How many seeds the arm beat baseline.</param>
/// <param name="Seeds">Number of seeds run.</param>
public readonly record struct SeedDelta(double BaseMean, double MeanDeltaPct, double StdPct, int Wins, int Seeds)
{
    /// <summary>True when |meanΔ%| is within one std — i.e. not distinguishable from noise.</summary>
    public bool WithinNoise => Math.Abs(MeanDeltaPct) <= StdPct;

    /// <summary>Compact report cell, e.g. "−0.7±1.0% [4/5]".</summary>
    public override string ToString() => string.Create(System.Globalization.CultureInfo.InvariantCulture,
        $"{MeanDeltaPct:+0.0;-0.0}±{StdPct:0.0}% [{Wins}/{Seeds}]");
}
