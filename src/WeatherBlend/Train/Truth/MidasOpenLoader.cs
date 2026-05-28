using System.Globalization;

namespace WeatherBlend.Train.Truth;

/// <summary>
/// Reader for Met Office MIDAS Open BADC-CSV files. The wind blender's
/// Dunkeswell SYNOP truth (station 01383) is distributed as a yearly
/// BADC-CSV per station — header preamble, then a line literal
/// <c>data</c>, then the column header, then rows, then <c>end data</c>.
///
/// This loader extracts only the four columns the wind blender needs:
/// <c>ob_time</c>, <c>met_domain_name</c> (filtered to <c>SYNOP</c>),
/// <c>wind_speed</c> (knots, converted to m/s), and <c>wind_direction</c>
/// (degrees true). Every other MIDAS column is dropped on read.
///
/// Production layout: <c>data/truth/midas/raw/midas-open*_{stationId}_*_{year}.csv</c>.
/// Files are static archive downloads, not collector-managed, so this
/// loader is on the train-time path only — there's no Collect/-side
/// equivalent (which is why this lives under Train/Truth/ rather than
/// Collect/).
///
/// Python parity: matches <c>scripts/train_wind_mvn.py</c>'s
/// <c>parse_badc</c> + <c>load_dunkeswell</c> in WeatherProbabilistic.
/// </summary>
public static class MidasOpenLoader
{
    /// <summary>kt → m/s. Same constant as train_wind_mvn.py.</summary>
    public const double KtToMs = 0.514444;

    public sealed record MidasWindObs(DateTime ValidTimeUtc, double WindSpeedMs, double WindDirectionDeg);

    /// <summary>
    /// Load Dunkeswell SYNOP wind observations from every matching BADC-CSV
    /// file under <paramref name="midasRoot"/> whose name matches
    /// <c>midas-open*_{stationId}_*_{year}.csv</c> for any year in
    /// <paramref name="years"/>. Rows with met_domain_name ≠ "SYNOP" or
    /// missing wind_speed / wind_direction are dropped.
    ///
    /// Returns a dictionary keyed by ValidTimeUtc for cheap inner-join
    /// against the pivoted forecast tree in the feature builder. Caller
    /// is expected to inner-join (rows without a forecast match get
    /// dropped naturally).
    /// </summary>
    public static IReadOnlyDictionary<DateTime, MidasWindObs> LoadStationWindByYear(
        string midasRoot,
        string stationId,
        IEnumerable<int> years)
    {
        var byTime = new Dictionary<DateTime, MidasWindObs>(capacity: 26000);  // ~3 yrs hourly
        if (!Directory.Exists(midasRoot)) return byTime;

        foreach (var y in years)
        {
            // Mirrors the rclone include pattern in retrain-python.yml:
            //   --include 'midas-open*_01383_*.csv'
            var pattern = $"midas-open*_{stationId}_*_{y}.csv";
            foreach (var path in Directory.EnumerateFiles(midasRoot, pattern))
            {
                foreach (var obs in ParseFile(path))
                {
                    // Duplicate timestamps shouldn't happen within a year, but if
                    // a file ships two SYNOP rows for the same ob_time we keep
                    // the first — second-write would be a silent data error.
                    if (!byTime.ContainsKey(obs.ValidTimeUtc))
                        byTime[obs.ValidTimeUtc] = obs;
                }
            }
        }
        return byTime;
    }

    /// <summary>
    /// Parse one BADC-CSV file. Yields one <see cref="MidasWindObs"/> per
    /// SYNOP row with non-empty wind_speed AND wind_direction. Rows whose
    /// met_domain_name isn't SYNOP, or that have "NA" / unparseable values
    /// in either field, are skipped silently (matches the Python loader's
    /// <c>dropna(subset=["wsp_ms", "wdir_deg"])</c>).
    /// </summary>
    public static IEnumerable<MidasWindObs> ParseFile(string path)
    {
        var lines = File.ReadAllLines(path);
        int dataIdx = -1, endIdx = -1;
        for (int i = 0; i < lines.Length; i++)
        {
            if (dataIdx < 0 && lines[i] == "data") dataIdx = i;
            else if (lines[i] == "end data") { endIdx = i; break; }
        }
        if (dataIdx < 0 || endIdx <= dataIdx + 1)
            yield break;  // malformed / empty: no rows to yield

        var header = lines[dataIdx + 1].Split(',');
        var idxObTime = Array.IndexOf(header, "ob_time");
        var idxDomain = Array.IndexOf(header, "met_domain_name");
        var idxWsp    = Array.IndexOf(header, "wind_speed");
        var idxWdir   = Array.IndexOf(header, "wind_direction");
        if (idxObTime < 0 || idxDomain < 0 || idxWsp < 0 || idxWdir < 0)
            yield break;  // required columns absent — caller's data is corrupt

        var inv = CultureInfo.InvariantCulture;
        for (int r = dataIdx + 2; r < endIdx; r++)
        {
            var cells = lines[r].Split(',');
            if (cells.Length <= idxWdir) continue;
            if (cells[idxDomain] != "SYNOP") continue;
            var wspStr  = cells[idxWsp];
            var wdirStr = cells[idxWdir];
            if (wspStr == "NA" || wdirStr == "NA" || string.IsNullOrEmpty(wspStr) || string.IsNullOrEmpty(wdirStr))
                continue;
            if (!double.TryParse(wspStr, NumberStyles.Float, inv, out var wspKt)) continue;
            if (!double.TryParse(wdirStr, NumberStyles.Float, inv, out var wdirDeg)) continue;
            if (!DateTime.TryParseExact(cells[idxObTime], "yyyy-MM-dd HH:mm:ss",
                                         inv, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                                         out var obTime))
                continue;
            yield return new MidasWindObs(
                ValidTimeUtc: DateTime.SpecifyKind(obTime, DateTimeKind.Utc),
                WindSpeedMs: wspKt * KtToMs,
                WindDirectionDeg: wdirDeg);
        }
    }
}
