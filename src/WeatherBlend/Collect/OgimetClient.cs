using System.Globalization;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using WeatherBlend.Models;

namespace WeatherBlend.Collect;

/// <summary>
/// OGIMET historical METAR archive: www.ogimet.com/cgi-bin/getmetar
/// Returns raw METAR text — no pre-parsed JSON. We parse the bits we need
/// (temp, dewpoint, wind, pressure, visibility, present-weather codes) with
/// regexes that cover the routine portion of the METAR.
///
/// IMPORTANT: OGIMET is community-funded and rate-limits aggressively.
/// Caller MUST sleep ~5s between requests; ignoring this gets the IP blocked.
/// </summary>
public sealed class OgimetClient
{
    private const string BaseUrl = "https://www.ogimet.com/cgi-bin/getmetar";

    private readonly HttpClient _http;
    private readonly ILogger<OgimetClient> _log;

    public OgimetClient(HttpClient http, ILogger<OgimetClient> log)
    {
        _http = http;
        _log = log;
    }

    /// <summary>
    /// Fetch all METARs for one station between begin/end (UTC).
    /// OGIMET returns up to ~24h per request reliably; caller chunks by day.
    /// </summary>
    public async Task<List<ObservationRow>> FetchAsync(
        string locationName,
        string station,
        DateTime beginUtc,
        DateTime endUtc,
        CancellationToken ct)
    {
        var url = $"{BaseUrl}?icao={Uri.EscapeDataString(station)}" +
                  $"&begin={beginUtc:yyyyMMddHHmm}" +
                  $"&end={endUtc:yyyyMMddHHmm}";
        _log.LogInformation("GET {Url}", url);

        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadAsStringAsync(ct);

        var rows = new List<ObservationRow>();
        // OGIMET response is CSV-prefixed text, one METAR per line:
        //   "EGTE,2026,04,15,06,20,METAR EGTE 150620Z 19010KT 9999 SCT009 11/11 Q1013="
        // Fields: icao, yyyy, mm, dd, hh, min, "METAR <text>"
        // NIL reports ("METAR EGTE 150020Z NIL=") are skipped — no payload to parse.
        foreach (var rawLine in body.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;

            var parsed = TryParseCsvLine(line, locationName, station);
            if (parsed is not null) rows.Add(parsed);
        }

        _log.LogInformation("Parsed {Count} METARs from OGIMET for {Station} {Begin:yyyy-MM-dd}..{End:yyyy-MM-dd}",
            rows.Count, station, beginUtc, endUtc);
        return rows;
    }

    // ---- Minimal METAR parsing -------------------------------------------------

    private static readonly Regex WindRx = new(
        @"\b(?<dir>\d{3}|VRB)(?<spd>\d{2,3})(?:G(?<gust>\d{2,3}))?(?<unit>KT|MPS|KMH)\b",
        RegexOptions.Compiled);

    // Temp/dewpoint group: e.g. "09/05" or "M02/M05" (M = minus).
    private static readonly Regex TempRx = new(
        @"(?<![A-Z0-9/])(?<t>M?\d{2})/(?<d>M?\d{2})(?![A-Z0-9/])",
        RegexOptions.Compiled);

    // Pressure: Q1018 (hPa) or A2992 (inHg, hundredths).
    private static readonly Regex PressureRx = new(@"\b(?<unit>Q|A)(?<val>\d{4})\b", RegexOptions.Compiled);

    // Visibility (metres): four digits standalone, e.g. " 9999 " or " 5000 ".
    private static readonly Regex VisRx = new(@"(?<=\s)(?<v>\d{4})(?=\s)", RegexOptions.Compiled);

    private static readonly Regex PresentWxRx = new(
        @"\b[+\-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|PL|GR|GS|IC|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)\b",
        RegexOptions.Compiled);

    private static readonly Regex PrecipCodeRx =
        new(@"(RA|DZ|SN|SG|PL|GR|GS|IC|UP|SH|TS)", RegexOptions.Compiled);

    internal static ObservationRow? TryParseCsvLine(string line, string locationName, string station)
    {
        // Split off the 6 date fields + METAR text.
        var parts = line.Split(',', 7);
        if (parts.Length < 7) return null;
        if (!parts[0].Equals(station, StringComparison.OrdinalIgnoreCase)) return null;

        if (!int.TryParse(parts[1], out var yyyy) ||
            !int.TryParse(parts[2], out var mm) ||
            !int.TryParse(parts[3], out var dd) ||
            !int.TryParse(parts[4], out var hh) ||
            !int.TryParse(parts[5], out var mi)) return null;

        var observed = new DateTime(yyyy, mm, dd, hh, mi, 0, DateTimeKind.Utc);
        var raw = parts[6].TrimEnd('=', ' ', '\r');

        // Skip NIL reports (station offline / no observation) — no payload to parse.
        if (raw.Contains(" NIL")) return null;

        // Strip "METAR EGTE DDHHMMZ" prefix; everything after is the body.
        var body = raw;
        var afterIcao = body.IndexOf(station, StringComparison.OrdinalIgnoreCase);
        if (afterIcao >= 0) body = body[(afterIcao + station.Length)..];
        // Drop the DDHHMMZ time group if present.
        var timeRx = new Regex(@"^\s*\d{6}Z\s*", RegexOptions.Compiled);
        body = timeRx.Replace(body, " ");

        double? windKt = null, dirDeg = null, gustKt = null;
        string windUnit = "KT";
        var w = WindRx.Match(body);
        if (w.Success)
        {
            windUnit = w.Groups["unit"].Value;
            windKt = double.Parse(w.Groups["spd"].Value, CultureInfo.InvariantCulture);
            if (w.Groups["dir"].Value != "VRB")
                dirDeg = double.Parse(w.Groups["dir"].Value, CultureInfo.InvariantCulture);
            if (w.Groups["gust"].Success)
                gustKt = double.Parse(w.Groups["gust"].Value, CultureInfo.InvariantCulture);
        }

        double? temp = null, dew = null;
        var tm = TempRx.Match(body);
        if (tm.Success)
        {
            temp = ParseSignedTemp(tm.Groups["t"].Value);
            dew = ParseSignedTemp(tm.Groups["d"].Value);
        }

        double? pressure = null;
        var pm = PressureRx.Match(body);
        if (pm.Success)
        {
            var v = double.Parse(pm.Groups["val"].Value, CultureInfo.InvariantCulture);
            pressure = pm.Groups["unit"].Value == "Q" ? v : v / 100.0 * 33.8639;
        }

        double? vis = null;
        if (body.Contains(" CAVOK"))
            vis = 10000;
        else
        {
            var vm = VisRx.Match(body);
            if (vm.Success)
            {
                var v = int.Parse(vm.Groups["v"].Value);
                if (v <= 9999) vis = v;
            }
        }

        var codes = PresentWxRx.Matches(body).Select(m => m.Value).Distinct().ToArray();
        var hasPrecip = codes.Any(c => PrecipCodeRx.IsMatch(c));

        return new ObservationRow
        {
            LocationName = locationName,
            Station = station,
            ObservedTimeUtc = observed,
            RawMetar = raw,
            Temperature2m = temp,
            DewPoint2m = dew,
            WindSpeed10m = ToMs(windKt, windUnit),
            WindDirection10m = dirDeg,
            WindGusts10m = ToMs(gustKt, windUnit),
            SurfacePressure = pressure,
            Visibility = vis,
            PresentWeather = codes,
            HasPrecipitation = hasPrecip
        };
    }

    private static double? ParseSignedTemp(string s)
    {
        var negative = s.StartsWith('M');
        var digits = negative ? s[1..] : s;
        if (!int.TryParse(digits, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v)) return null;
        return negative ? -v : v;
    }

    private static double? ToMs(double? v, string unit) => v switch
    {
        null => null,
        _ => unit switch
        {
            "KT" => v * 0.514444,
            "MPS" => v,
            "KMH" => v / 3.6,
            _ => v * 0.514444
        }
    };

}
