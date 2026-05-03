using System.Globalization;
using DuckDB.NET.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// Phase 3g — direct dry-window probability via Monte Carlo over Phase 3a's
/// per-hour P(wet) outputs under independence. No LightGBM, no learned
/// parameters: the prediction is literally the fraction of MC samples
/// (Bernoulli per hour using 3a's q values) where the longest dry run within
/// the daytime window is at least N hours long.
///
/// Why this works (per the 2026-05-03 bake-off): 3a is well-calibrated at the
/// hourly level (no PAV needed), and although the independence assumption is
/// wrong (real rain clusters), the structural rule "longer windows are rarer"
/// averages over the dependency structure across many sample days. Net Brier
/// improvement vs Phase 3b: −6.4% on a 27-cell historical test slice.
///
/// Cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) is GUARANTEED by
/// computing all requested window probabilities in a single MC pass per row
/// (one Bernoulli sequence per sample, three indicators read off it). This is
/// the structural fix the user requested for the 3b "P(4h) > P(3h)" bug.
///
/// Predict-time inputs are read from the live Phase 3a prediction parquet
/// (<c>data/predictions/precipitation/{station}/model_version={3a-version}/
/// date={anchor}/predictions.parquet</c>) — the same artefact the rain-skill
/// page already serves. Train-time the same shape comes from the replay
/// parquet under <c>precipitation_replay/</c>, produced by
/// <see cref="WeatherBlend.Commands.PrecipReplayCommand"/>.
/// </summary>
public static class DryWindow3gPredictor
{
    public const string Phase3g = "3g";

    /// <summary>Default MC sample count. 10,000 keeps Brier estimation noise
    /// well below the per-cell ±5% scale while staying cheap.</summary>
    public const int DefaultMcSamples = 10000;

    /// <summary>
    /// Single MC pass yielding P(longest dry run ≥ L) for every requested
    /// window length <paramref name="windowHours"/>. All windows share the
    /// same Bernoulli draws per sample, so monotonicity P(L=3) ≥ P(L=4) ≥ ...
    /// is preserved exactly (not just in expectation). Returns a dictionary
    /// keyed by window length; missing windows in the input array are not
    /// in the output.
    /// </summary>
    public static Dictionary<int, double> ProbDryWindow(
        double[] qHourly,
        IReadOnlyList<int> windowHours,
        Random rng,
        int nSamples = DefaultMcSamples)
    {
        var result = new Dictionary<int, double>(windowHours.Count);
        if (qHourly.Length == 0 || windowHours.Count == 0)
        {
            foreach (var w in windowHours) result[w] = 0.0;
            return result;
        }

        var sortedWindows = windowHours.Distinct().OrderBy(w => w).ToArray();
        var hits = new int[sortedWindows.Length];

        for (int s = 0; s < nSamples; s++)
        {
            int run = 0, longest = 0;
            for (int h = 0; h < qHourly.Length; h++)
            {
                bool dry = rng.NextDouble() >= qHourly[h];
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            for (int i = 0; i < sortedWindows.Length; i++)
                if (longest >= sortedWindows[i]) hits[i]++;
        }

        for (int i = 0; i < sortedWindows.Length; i++)
            result[sortedWindows[i]] = (double)hits[i] / nSamples;
        return result;
    }

    /// <summary>
    /// Convenience for the single-window case used by the predict command.
    /// </summary>
    public static double ProbDryWindow(
        double[] qHourly, int windowLength, Random rng, int nSamples = DefaultMcSamples)
    {
        if (qHourly.Length < windowLength) return 0.0;
        int hits = 0;
        for (int s = 0; s < nSamples; s++)
        {
            int run = 0, longest = 0;
            for (int h = 0; h < qHourly.Length; h++)
            {
                bool dry = rng.NextDouble() >= qHourly[h];
                run = dry ? run + 1 : 0;
                if (run > longest) longest = run;
            }
            if (longest >= windowLength) hits++;
        }
        return (double)hits / nSamples;
    }

    /// <summary>
    /// Read (ValidTimeUtc → ProbWet) for a specific lead bucket out of the
    /// Phase 3a replay parquet. Used at training time only — the historical
    /// per-hour q for every (station, lead) combination is needed to score
    /// the chronological test split.
    /// </summary>
    public static Dictionary<DateTime, double> LoadReplayHourly(
        string predictionsRoot, string stationSlug, string precip3aVersion, int leadHours)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation_replay", stationSlug, precip3aVersion, $"lead_{leadHours}h.parquet")
            .Replace('\\', '/');
        return ReadValidTimeProbWet($"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path}')");
    }

    /// <summary>
    /// Read (ValidTimeUtc → ProbWet) from the live 3a prediction parquet for
    /// one anchor cycle. Predict-time path: 3a writes one parquet per
    /// (station, model_version, anchor_date) covering ~120h of hourly
    /// forecasts; we read the rows whose ValidTimeUtc falls within the
    /// requested target_date's daytime window (caller filters).
    /// </summary>
    public static Dictionary<DateTime, double> LoadLivePredictionsHourly(
        string predictionsRoot, string stationSlug, string precip3aVersion, DateTime anchorDate)
    {
        var path = Path.Combine(
            predictionsRoot, "precipitation", stationSlug,
            $"model_version={precip3aVersion}",
            $"date={anchorDate:yyyy-MM-dd}",
            "predictions.parquet").Replace('\\', '/');
        return ReadValidTimeProbWet($"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path}')");
    }

    /// <summary>
    /// Pull the daytime hourly q vector for a single target_date from the
    /// supplied (ValidTimeUtc → ProbWet) dictionary. Returns null if any
    /// daytime hour is missing — under independence we can't honestly
    /// produce a probability with a gap (and the 3b training pipeline drops
    /// such days from its rows anyway, so this stays consistent).
    /// </summary>
    public static double[]? ExtractDaytimeQ(
        IReadOnlyDictionary<DateTime, double> hourly,
        DateTime targetDateUtc, int startUtcHour, int endUtcHourExclusive)
    {
        var n = endUtcHourExclusive - startUtcHour;
        if (n <= 0) return null;
        var q = new double[n];
        var midnight = new DateTime(
            targetDateUtc.Year, targetDateUtc.Month, targetDateUtc.Day, 0, 0, 0, DateTimeKind.Utc);
        for (int h = startUtcHour; h < endUtcHourExclusive; h++)
        {
            var t = midnight.AddHours(h);
            if (!hourly.TryGetValue(t, out var p)) return null;
            q[h - startUtcHour] = Math.Clamp(p, 0.0, 1.0);
        }
        return q;
    }

    private static Dictionary<DateTime, double> ReadValidTimeProbWet(string sql)
    {
        using var conn = new DuckDBConnection("DataSource=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        var dict = new Dictionary<DateTime, double>();
        using var rdr = cmd.ExecuteReader();
        while (rdr.Read())
        {
            var t = DateTime.SpecifyKind(rdr.GetDateTime(0), DateTimeKind.Utc);
            dict[t] = rdr.GetDouble(1);
        }
        return dict;
    }
}
