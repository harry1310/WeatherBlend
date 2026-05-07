using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Home day-window: 0 = today's UTC date (offset 0), 1 = tomorrow, … up
    /// to <see cref="MaxHomeDayOffset"/>. The home page now renders one file
    /// per day with a sub-nav at the top to switch between them — the old
    /// stacked-day layout meant scrolling past today before reaching tomorrow.
    /// 5 forward days matches the temp + rain blender's 120h max lead; the
    /// dry-window forecasts page covers per-day outlook beyond that scope.
    /// </summary>
    public const int MaxHomeDayOffset = 5;

    /// <summary>
    /// One forward-day's home page. Tiles are the champion-blender forecast
    /// at the shortest lead, filtered to "outdoor" hours
    /// <c>[<see cref="HomeFirstVisibleHourUtc"/>, <see cref="HomeLastVisibleHourUtcExclusive"/>)</c>
    /// — 21:00-03:59 UTC are dropped because they're irrelevant for climbing
    /// or walking trip planning. Each tile carries a UTCI ⓘ pop-out with the
    /// element-blender values that fed it.
    /// </summary>
    public static string RenderIndex(SiteInputs input, int dayOffset)
    {
        if (dayOffset < 0 || dayOffset > MaxHomeDayOffset)
            throw new ArgumentOutOfRangeException(nameof(dayOffset),
                $"Home dayOffset must be in [0, {MaxHomeDayOffset}].");

        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(HomeFirstVisibleHourUtc);
        var dayWindowEnd   = dayUtc.AddHours(HomeLastVisibleHourUtcExclusive);

        // Champion-only filter, per lead. ChampionByLead pins specific
        // (lead → version) overrides (e.g. 2d champions lead 12h while 2b
        // stays Current at 24+); any lead missing from the dict falls back
        // to CurrentVersion. Empty CurrentVersion = no manifest, fall back
        // to "any" so a freshly-deployed environment still renders cards.
        string ChampionForLead(int lead)
        {
            if (input.ChampionByLead.TryGetValue(lead, out var perLead) && !string.IsNullOrEmpty(perLead))
                return perLead;
            return input.CurrentVersion;
        }
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => p.ModelVersion == ChampionForLead(p.LeadHours));

        // For each future valid_time, take the smallest lead (most recent
        // cycle); within ties, freshest PredictionMadeAt wins. Restrict to
        // this day's outdoor window and drop any past hour (today's tab only).
        var dayPreds = cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc
                        && p.ValidTimeUtc >= dayWindowStart
                        && p.ValidTimeUtc <  dayWindowEnd)
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g
                .OrderBy(p => p.LeadHours)
                .ThenByDescending(p => p.PredictionMadeAtUtc)
                .First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        // P(wet) lookup keyed by valid_time across all leads, smallest-lead-wins
        // mirroring the temperature pick. Bellever as the headline gauge.
        // Per-lead champion override for precip (mirrors temperature
        // ChampionByLead): a (Station, Lead) pin in
        // input.PrecipChampionByStationLead beats the per-station Current.
        const string PwetStation = "ea_bellever_dartmoor";
        input.PrecipCurrentByStation.TryGetValue(PwetStation, out var pwetChampion);
        string PwetChampionForLead(int lead)
        {
            if (input.PrecipChampionByStationLead.TryGetValue((PwetStation, lead), out var perLead)
                && !string.IsNullOrEmpty(perLead))
                return perLead;
            return pwetChampion ?? "";
        }
        var pwetByValid = input.PrecipPredictions
            .Where(r => r.Station == PwetStation
                        && !string.IsNullOrEmpty(PwetChampionForLead(r.LeadHours))
                        && r.Version == PwetChampionForLead(r.LeadHours))
            .GroupBy(r => r.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(r => r.LeadHours)
                      .ThenByDescending(r => r.PredictedAtUtc)
                      .First());

        var feelsLikeByValid = input.FeelsLikePredictions
            .GroupBy(u => u.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(u => u.LeadHours)
                      .ThenByDescending(u => u.PredictedAtUtc)
                      .First());

        // Day summary line above the tile grid: temp range + mean P(wet) +
        // driest hour. Falls back to a temp-only line when no P(wet) is
        // available for Bellever this day.
        string daySummary, tilesHtml;
        if (dayPreds.Count == 0)
        {
            daySummary = "<p class=\"skill-line\"><em>No forward predictions in this day's outdoor window.</em></p>";
            tilesHtml = "";
        }
        else
        {
            var minT = dayPreds.Min(p => p.BlendTemperature);
            var maxT = dayPreds.Max(p => p.BlendTemperature);
            var dayPwets = dayPreds
                .Select(p => TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw) ? pw : null)
                .Where(pw => pw is not null)
                .Select(pw => pw!)
                .ToList();
            if (dayPwets.Count > 0)
            {
                var meanP = dayPwets.Average(p => p.ProbWet);
                var driest = dayPwets.OrderBy(p => p.ProbWet).First();
                daySummary = string.Create(Ci, $"""
                    <p class="skill-line">
                      <strong>{minT:0.0}°C → {maxT:0.0}°C</strong> ·
                      mean P(wet) <strong style="color: {PrecipProbColor(meanP)}">{(meanP * 100):0}%</strong> ·
                      driest hour {driest.ValidTimeUtc:HH'Z'} <strong style="color: {PrecipProbColor(driest.ProbWet)}">{(driest.ProbWet * 100):0}%</strong>
                    </p>
                    """);
            }
            else
            {
                daySummary = string.Create(Ci, $"""<p class="skill-line"><strong>{minT:0.0}°C → {maxT:0.0}°C</strong> · no P(wet) for Bellever this day</p>""");
            }
            var tiles = new StringBuilder();
            int popoverId = 0;
            foreach (var p in dayPreds)
                tiles.Append(RenderHourTile(p, feelsLikeByValid, pwetByValid, input.LowCloudByValid, popoverId++));
            tilesHtml = string.Create(Ci, $"<div class=\"forecast-grid\">{tiles}</div>");
        }

        var body = new StringBuilder();
        body.Append("<section>");
        body.Append(RenderHomeDaySubNav(input, dayOffset));
        body.Append(Ci, $"""
              <hgroup>
                <h2>Forward forecast — {dayUtc:dddd dd MMMM}</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m.</p>
              </hgroup>
            """);
        body.Append(daySummary);
        body.Append(tilesHtml);
        body.Append("</section>");

        // Page filename is set by RenderSiteCommand (index.html for today,
        // index-N.html for day-N). The pageId we pass to WrapPage is purely
        // for the top-nav highlight, so it must always be "index" — using
        // "index-N" silently broke the Home highlight on every forward-day
        // tab because NavActive("index-1", "index") returns false.
        return WrapPage(input, "Home", "index", body.ToString());
    }

    /// <summary>UTC hour at which the home tile grid starts. Tiles before
    /// this time are filtered out per user feedback (overnight hours aren't
    /// useful for planning a climbing trip). Tightened from 04 → 05 on
    /// 2026-05-07 (user request) — a 14h window 05–19Z is enough headroom
    /// for "out before sunrise" through "back before dusk" on Dartmoor in
    /// summer; the dropped 04Z + 20Z tiles were rarely-used edge hours.</summary>
    public const int HomeFirstVisibleHourUtc = 5;

    /// <summary>UTC hour at which the home tile grid ends, exclusive. Tiles
    /// at or after this time are filtered out. Tightened from 21 → 20 on
    /// 2026-05-07 alongside the start-hour change so the visible window is
    /// 05:00Z..19:00Z inclusive.</summary>
    public const int HomeLastVisibleHourUtcExclusive = 20;

    /// <summary>
    /// Day sub-nav for the home page — pill links labelled "Mon 4/5"
    /// style (day-of-week short + day/month numeric, e.g. "Tue 5/5"). Today
    /// is offset 0 (file index.html); each forward day gets index-{n}.html.
    /// Days with zero tiles after the outdoor-window + future-only filter
    /// are skipped so the sub-nav can't link to a blank page (added
    /// 2026-05-07). If the active offset itself has no tiles it's still
    /// rendered as the highlight so the user knows where they are.
    /// </summary>
    private static string RenderHomeDaySubNav(SiteInputs input, int activeOffset)
    {
        var today = input.GeneratedAtUtc.Date;
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        for (int n = 0; n <= MaxHomeDayOffset; n++)
        {
            // Skip empty days, but always keep the active one so the
            // current page has something highlighted in the bar.
            if (n != activeOffset && CountHomeDayTiles(input, n) == 0) continue;

            var date = today.AddDays(n);
            var file = n == 0 ? "index.html" : $"index-{n}.html";
            var label = n == 0 ? "Today" : $"{date:ddd} {date.Day}/{date.Month}";
            var cls = n == activeOffset ? " class=\"active\"" : "";
            s.Append(Ci, $"<li><a href=\"{file}\"{cls}>{Escape(label)}</a></li>");
        }
        s.Append("</ul></nav>");
        return s.ToString();
    }

    /// <summary>
    /// Count of tiles that the home page would render for the given
    /// day offset. Mirrors the filter chain at the top of
    /// <see cref="RenderIndex"/> — champion-only per lead, future-of-now,
    /// inside the outdoor visible-hour window, smallest-lead wins per
    /// valid_time. Used by the day sub-nav to suppress days that would
    /// link to an empty page.
    /// </summary>
    private static int CountHomeDayTiles(SiteInputs input, int dayOffset)
    {
        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(HomeFirstVisibleHourUtc);
        var dayWindowEnd = dayUtc.AddHours(HomeLastVisibleHourUtcExclusive);

        string ChampionForLead(int lead)
        {
            if (input.ChampionByLead.TryGetValue(lead, out var perLead) && !string.IsNullOrEmpty(perLead))
                return perLead;
            return input.CurrentVersion;
        }
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => p.ModelVersion == ChampionForLead(p.LeadHours));

        return cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc
                        && p.ValidTimeUtc >= dayWindowStart
                        && p.ValidTimeUtc < dayWindowEnd)
            .Select(p => p.ValidTimeUtc)
            .Distinct()
            .Count();
    }

    /// <summary>
    /// One forward-hour tile — temperature headline, optional Feels-like /
    /// UTCI / P(wet) chips, and a Pico-styled <c>&lt;details&gt;</c> pop-out
    /// listing the four element-blender values that fed UTCI for this hour.
    /// </summary>
    /// <summary>Visibility-signal firing threshold — number of vis-publishing
    /// NWPs (out of 6) that must forecast sub-1km vis for the badge to fire
    /// on visibility alone.</summary>
    private const int LowCloudVisFireThreshold = 3;

    /// <summary>Cloud-base-signal firing threshold — number of NWPs (out of
    /// up to 11) that must forecast T-Td &lt; 1.5°C for the badge to fire
    /// on cloud-base alone. Higher absolute count than vis because the
    /// denominator is much larger.</summary>
    private const int LowCloudBaseFireThreshold = 6;

    private static string RenderHourTile(
        Models.TempPredictionRow p,
        IReadOnlyDictionary<DateTime, FeelsLikeForecastPoint> feelsLikeByValid,
        IReadOnlyDictionary<DateTime, PrecipForecastPoint> pwetByValid,
        IReadOnlyDictionary<DateTime, LowCloudSignal> lowCloudByValid,
        int popoverId)
    {
        // Low-cloud / mist warning — fires when EITHER signal hits its
        // threshold. Sits in the tile header as a Pico <details>/<summary>
        // pop-out so the trigger details work on touch devices (the earlier
        // title="..." tooltip was unreachable from mobile).
        string lowCloudBadge = "";
        if (lowCloudByValid.TryGetValue(p.ValidTimeUtc, out var lc))
        {
            var visFired = lc.VisFiredCount >= LowCloudVisFireThreshold;
            var cbFired  = lc.CloudBaseFiredCount >= LowCloudBaseFireThreshold;
            if (visFired || cbFired)
            {
                var rows = new StringBuilder();
                if (visFired)
                    rows.Append(Ci, $"<li>{lc.VisFiredCount}/{lc.VisTotalCount} NWPs: mist (vis &lt; 1 km)</li>");
                if (cbFired)
                    rows.Append(Ci, $"<li>{lc.CloudBaseFiredCount}/{lc.CloudBaseTotalCount} NWPs: cloud base below tor (T−Td &lt; 1.5°C)</li>");
                lowCloudBadge = $"""
                    <details class="low-cloud-pop">
                      <summary class="low-cloud-badge">☁ low cloud</summary>
                      <ul>{rows}</ul>
                    </details>
                    """;
            }
        }

        // Feels-like and P(wet) tolerate ±1h drift — predict cycles can land
        // an hour off, and we'd rather show a chip than a gap.
        string feelsCell = "";
        if (TryNearest(feelsLikeByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var fl))
        {
            var apparentColor = TemperatureColor(fl!.ApparentC);
            var utciColor = TemperatureColor(fl.UtciC);

            // UTCI pop-out: only emit the toggle when at least one element
            // value is present (older parquets pre-dating the element-input
            // persistence have all five null and the panel would be empty).
            string toggle = "";
            // Temperature deliberately excluded from the gate too — only show
            // the toggle when at least one of the four "why is UTCI this value"
            // element fields is present.
            if (fl.RelativeHumidityPct is not null
                || fl.WindSpeed10mMs is not null || fl.ShortwaveDownWm2 is not null
                || fl.CloudCoverPct is not null)
            {
                // Temperature row deliberately omitted — it's already the
                // headline value on the tile, repeating it in the pop-out is
                // noise. The other four are the "why is UTCI this value?"
                // story the reader can't get elsewhere.
                var rows = new StringBuilder();
                if (fl.RelativeHumidityPct is double rh)
                    rows.Append(Ci, $"<tr><td>Humidity</td><td class=\"num\">{rh:0} %</td></tr>");
                if (fl.WindSpeed10mMs is double ws)
                    rows.Append(Ci, $"<tr><td>Wind 10 m</td><td class=\"num\">{ws:0.0} m/s</td></tr>");
                if (fl.ShortwaveDownWm2 is double sw)
                    rows.Append(Ci, $"<tr><td>Shortwave down</td><td class=\"num\">{sw:0} W/m²</td></tr>");
                if (fl.CloudCoverPct is double cc)
                    rows.Append(Ci, $"<tr><td>Cloud cover</td><td class=\"num\">{cc:0} %</td></tr>");
                toggle = $"""
                    <details class="utci-pop">
                      <summary title="Element values that fed UTCI for this hour">ⓘ</summary>
                      <table class="utci-pop-table">{rows}</table>
                    </details>
                    """;
            }
            feelsCell = string.Create(Ci, $"""
                <div class="feels">
                  <div>Feels like <strong style="color: {apparentColor}">{fl.ApparentC:0.0}°C</strong></div>
                  <div>UTCI <strong style="color: {utciColor}">{fl.UtciC:0.0}°C</strong> <small>{Escape(PrettyUtciBand(fl.Band))}</small> {toggle}</div>
                </div>
                """);
        }

        string pwetCell = "";
        if (TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw))
        {
            // ☔ when the displayed P(wet) rounds to ≥ 25%
            var rain = pw!.ProbWet >= 0.245 ? " <span class=\"rain\">&#x2614;</span>" : "";
            var pwColor = PrecipProbColor(pw.ProbWet);
            pwetCell = string.Create(Ci, $"<div class=\"pwet\">P(wet) <strong style=\"color: {pwColor}\">{(pw.ProbWet * 100):0}%</strong>{rain}</div>");
        }

        var tempColor = TemperatureColor(p.BlendTemperature);
        return string.Create(Ci, $"""
            <article class="forecast-card">
              <header><h4>{p.ValidTimeUtc:HH:mm}Z</h4>{lowCloudBadge}</header>
              <div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature:0.0}°C</div>
              {feelsCell}
              {pwetCell}
              <footer><small>+{p.LeadHours}h · made {p.PredictionMadeAtUtc:MM-dd HH:mm}Z</small></footer>
            </article>
            """);
    }

    /// <summary>
    /// Exact-key lookup, falling back to the closest entry within
    /// <paramref name="tolerance"/>. Returns false (and a default value) when
    /// no entry is within tolerance — caller treats that as "no chip".
    /// </summary>
    private static bool TryNearest<TValue>(
        IReadOnlyDictionary<DateTime, TValue> map,
        DateTime target,
        TimeSpan tolerance,
        out TValue? value)
    {
        if (map.TryGetValue(target, out value)) return true;
        TValue? best = default;
        var bestDelta = TimeSpan.MaxValue;
        foreach (var kv in map)
        {
            var delta = kv.Key - target;
            if (delta < TimeSpan.Zero) delta = -delta;
            if (delta <= tolerance && delta < bestDelta)
            {
                best = kv.Value;
                bestDelta = delta;
            }
        }
        if (bestDelta == TimeSpan.MaxValue)
        {
            value = default;
            return false;
        }
        value = best;
        return true;
    }
}
