namespace WeatherBlend.Evaluate.StartHour;

/// <summary>
/// Scoring helpers for the dry-window start-hour curve. Inputs are the
/// per-row <c>ConditionalProb</c> values from the start-hour predict tree
/// (one π per candidate start hour) plus the observed truth set (which
/// start hours actually had a fully dry N-hour window per EA rainfall).
///
/// All functions are pure — no I/O, no logging, no DateTime calls — so the
/// verify command can call them in a tight loop and tests can exercise
/// every metric in milliseconds.
///
/// Three numbers per (station, window, lead, target_date):
///   - <see cref="Top1Hit"/>: 1 if argmax_s π_s ∈ truthStarts else 0.
///   - <see cref="Brier"/>: Σ (π_s − τ_s)² where τ uniform over truthStarts.
///     Lower = better. Uniform-π baseline depends on |truth|/|starts| so
///     the verify command also tracks the uniform baseline alongside.
///   - <see cref="LogLoss"/>: -Σ τ_s log(π_s + ε). Lower = better; bigger
///     spread between curve and uniform = more skill.
///
/// The curve is "informative" only when 1 ≤ |truthStarts| &lt; |starts| —
/// fully-dry days (every start valid) and no-block days (no start valid)
/// carry no shape signal and are excluded from aggregation by
/// <see cref="IsInformative"/>.
/// </summary>
public static class StartHourMetrics
{
    public const double LogLossEpsilon = 1e-6;

    /// <summary>True when the day exercises the curve's shape — not all
    /// starts valid, not zero starts valid.</summary>
    public static bool IsInformative(int truthStartCount, int totalStartCount)
        => truthStartCount > 0 && truthStartCount < totalStartCount;

    /// <summary>Top-1 hit: did argmax_s π_s land on a truth-valid start?</summary>
    public static bool Top1Hit(IReadOnlyList<(int StartHour, double Pi)> curve,
                              IReadOnlySet<int> truthStarts)
    {
        if (curve.Count == 0) return false;
        var argmaxStart = curve[0].StartHour;
        var argmaxPi = curve[0].Pi;
        for (int i = 1; i < curve.Count; i++)
        {
            if (curve[i].Pi > argmaxPi)
            {
                argmaxPi = curve[i].Pi;
                argmaxStart = curve[i].StartHour;
            }
        }
        return truthStarts.Contains(argmaxStart);
    }

    /// <summary>Brier score against the uniform-over-truth target
    /// distribution τ_s = 1/|truth| if s ∈ truth else 0.</summary>
    public static double Brier(IReadOnlyList<(int StartHour, double Pi)> curve,
                              IReadOnlySet<int> truthStarts)
    {
        if (curve.Count == 0 || truthStarts.Count == 0) return 0.0;
        var tau = 1.0 / truthStarts.Count;
        double sum = 0.0;
        foreach (var (s, pi) in curve)
        {
            var t = truthStarts.Contains(s) ? tau : 0.0;
            var d = pi - t;
            sum += d * d;
        }
        return sum;
    }

    /// <summary>Log-loss against τ. Use <see cref="LogLossUniform"/> for the
    /// no-skill baseline so the verify command can compute skill score as
    /// <c>1 − LogLoss / LogLossUniform</c> per row + aggregate after.</summary>
    public static double LogLoss(IReadOnlyList<(int StartHour, double Pi)> curve,
                                IReadOnlySet<int> truthStarts)
    {
        if (curve.Count == 0 || truthStarts.Count == 0) return 0.0;
        var tau = 1.0 / truthStarts.Count;
        double sum = 0.0;
        foreach (var (s, pi) in curve)
        {
            if (!truthStarts.Contains(s)) continue;
            var clamped = Math.Max(pi, LogLossEpsilon);
            sum += -tau * Math.Log(clamped);
        }
        return sum;
    }

    /// <summary>Log-loss for the uniform-π reference (every start = 1/N).
    /// Same τ; same epsilon. Used as the denominator of the skill score.</summary>
    public static double LogLossUniform(int totalStartCount, int truthStartCount)
    {
        if (totalStartCount <= 0 || truthStartCount <= 0) return 0.0;
        var uniformPi = 1.0 / totalStartCount;
        return -Math.Log(Math.Max(uniformPi, LogLossEpsilon));
    }
}
