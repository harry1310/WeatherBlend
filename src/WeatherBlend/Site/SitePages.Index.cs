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

        var (firstHour, lastHourExcl) = OverviewWindow(input);
        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(firstHour);
        var dayWindowEnd   = dayUtc.AddHours(lastHourExcl);

        // Champion-PHASE filter, per lead. ChampionByLead pins specific
        // (lead → version) overrides (e.g. 2d champions lead 12h while 2b
        // stays Current at 24+); any lead missing from the dict falls back
        // to CurrentVersion. Empty CurrentVersion = no manifest, fall back
        // to "any" so a freshly-deployed environment still renders cards.
        //
        // Match by PHASE rather than strict version (2026-05-26): a retrain
        // mints a new champion version whose predictions only target the
        // anchor day forward, leaving today's window orphaned for any phase
        // without a sub-24h lead (i.e. anything other than 2d / 3d). The
        // PREVIOUS champion-phase bundle still has predictions covering
        // today's hours — those rows live in input.Predictions thanks to
        // the unfiltered scan in PredictionsRepository — they just no
        // longer match the strict-equality champion-version check.
        // Falling back to version-equality when the phase metadata is
        // missing keeps test fixtures (which don't always populate
        // PhaseByVersion) working unchanged.
        string ChampionForLead(int lead)
        {
            if (input.ChampionByLead.TryGetValue(lead, out var perLead) && !string.IsNullOrEmpty(perLead))
                return perLead;
            return input.CurrentVersion;
        }
        bool MatchesChampionPhase(string rowVersion, int lead)
        {
            var champVersion = ChampionForLead(lead);
            if (string.Equals(rowVersion, champVersion, StringComparison.Ordinal)) return true;
            return input.PhaseByVersion.TryGetValue(rowVersion, out var rowPhase)
                && input.PhaseByVersion.TryGetValue(champVersion, out var champPhase)
                && !string.IsNullOrEmpty(rowPhase)
                && string.Equals(rowPhase, champPhase, StringComparison.Ordinal);
        }
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => MatchesChampionPhase(p.ModelVersion, p.LeadHours));

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
        // mirroring the temperature pick. Headline gauge resolves per rendered
        // location — see OverviewPwetStation.
        //
        // The home tiles show the precipitation CHAMPION — phases.yaml's
        // champion phase (3a), resolved per station to its newest Active
        // version (input.PrecipCurrentByStation, from
        // ModelMetadataRepository.GetChampionsByStation → ResolveStation-
        // ChampionVersion). 3a predicts hourly (.NET predict, 2026-05-04
        // hourly-spread change) so it covers every hourly tile. This is
        // deliberately the champion phase, not a 6-hourly challenger: 4a/4b
        // predict only at exact leads {24,48,72,96,120} (a 6-hourly valid
        // grid) and would leave the hourly tiles with P(wet) gaps. The
        // challengers lead the per-lead forecast + Models pages instead.
        var pwetStation = OverviewPwetStation(input);
        input.PrecipCurrentByStation.TryGetValue(pwetStation, out var pwetChampion);
        // Match by champion PHASE for the same retrain-window reason as the
        // temperature tiles above — yesterday's still-3a bundle covers
        // today's hours that today's freshly-minted bundle doesn't.
        bool MatchesPwetPhase(string rowVersion)
        {
            if (string.IsNullOrEmpty(pwetChampion)) return false;
            if (string.Equals(rowVersion, pwetChampion, StringComparison.Ordinal)) return true;
            return input.PhaseByVersion.TryGetValue(rowVersion, out var rowPhase)
                && input.PhaseByVersion.TryGetValue(pwetChampion, out var champPhase)
                && !string.IsNullOrEmpty(rowPhase)
                && string.Equals(rowPhase, champPhase, StringComparison.Ordinal);
        }
        var pwetByValid = string.IsNullOrEmpty(pwetChampion)
            ? new Dictionary<DateTime, PrecipForecastPoint>()
            : input.PrecipPredictions
                .Where(r => r.Station == pwetStation && MatchesPwetPhase(r.Version))
                .GroupBy(r => r.ValidTimeUtc)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderBy(r => r.LeadHours)
                          .ThenByDescending(r => r.PredictedAtUtc)
                          .First());

        // input.FeelsLikePredictions + LowCloudByValid are already scoped to
        // this page's location by RenderSiteCommand (single-location SiteInputs
        // invariant) — use directly.
        var feelsLikeByValid = input.FeelsLikePredictions
            .GroupBy(u => u.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(u => u.LeadHours)
                      .ThenByDescending(u => u.PredictedAtUtc)
                      .First());

        var lowCloudByValid = input.LowCloudByValid;

        // Day summary line above the tile grid: temp range + mean P(wet) +
        // driest hour. Falls back to a temp-only line when no P(wet) is
        // available for the headline gauge this day.
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
                daySummary = string.Create(Ci, $"""<p class="skill-line"><strong>{minT:0.0}°C → {maxT:0.0}°C</strong> · no P(wet) for this day</p>""");
            }
            var tiles = new StringBuilder();
            int popoverId = 0;
            foreach (var p in dayPreds)
                tiles.Append(RenderHourTile(p, feelsLikeByValid, pwetByValid, lowCloudByValid, popoverId++));
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
        return WrapPage(input, "Overview", "overview", body.ToString());
    }

    /// <summary>Legacy default UTC hour at which the home tile grid starts.
    /// Phase D made this per-location — see
    /// <c>LocationConfig.Overview.FirstVisibleHourUtc</c>. Retained here for
    /// test fixtures / callers that don't set <c>SiteInputs.RenderingFor</c>.
    /// Bonehill's config.yaml carries 5 (matches the pre-Phase-D constant);
    /// Membury wants the full 24h day so its config carries 0.</summary>
    public const int HomeFirstVisibleHourUtc = 5;

    /// <summary>Legacy default UTC hour at which the home tile grid ends
    /// (exclusive). Phase D made this per-location — see
    /// <c>LocationConfig.Overview.LastVisibleHourUtcExclusive</c>. Bonehill 20,
    /// Membury 24.</summary>
    public const int HomeLastVisibleHourUtcExclusive = 20;

    /// <summary>
    /// Phase D — resolve the Overview tile-grid hour window from the per-loc
    /// descriptor when <see cref="SiteInputs.RenderingFor"/> is set; fall back
    /// to the legacy constants for test fixtures + pre-Phase-D callers.
    /// </summary>
    private static (int FirstHour, int LastHourExcl) OverviewWindow(SiteInputs input)
    {
        if (input.RenderingFor is { } loc)
            return (loc.OverviewFirstVisibleHourUtc, loc.OverviewLastVisibleHourUtcExclusive);
        return (HomeFirstVisibleHourUtc, HomeLastVisibleHourUtcExclusive);
    }

    /// <summary>
    /// Headline P(wet) gauge for the Overview tiles — the rendered location's
    /// primary rainfall station: the first entry in its
    /// <see cref="LocationDescriptor.RainStationSlugs"/> (config order — Bonehill
    /// lists Bellever first, Membury lists Chards Snowdon Hill first). Prefer the
    /// first slug that actually has a champion version in
    /// <c>PrecipCurrentByStation</c> so a not-yet-trained gauge can't blank the
    /// tile grid; fall back to the first slug, then to Bellever for pre-Phase-D /
    /// test callers with no <see cref="SiteInputs.RenderingFor"/>. Hardcoding a
    /// single gauge here was the cause of the Membury Overview showing no
    /// P(wet) — see memory note feedback_avoid_hardcoded_phase_station_lists.
    /// </summary>
    private static string OverviewPwetStation(SiteInputs input)
    {
        var slugs = input.RenderingFor?.RainStationSlugs;
        if (slugs is null || slugs.Count == 0)
            return "ea_bellever_dartmoor";
        return slugs.FirstOrDefault(s => input.PrecipCurrentByStation.ContainsKey(s))
               ?? slugs[0];
    }

    /// <summary>
    /// Day sub-nav for the home page — pill links labelled "Mon 4/5"
    /// style (day-of-week short + day/month numeric, e.g. "Tue 5/5"). Today
    /// is offset 0 (file index.html); each forward day gets index-{n}.html.
    /// Days with zero tiles after the outdoor-window + future-only filter
    /// are skipped so the sub-nav can't link to a blank page. The
    /// "keep the active offset visible even if empty" exception (added
    /// 2026-05-07) was removed 2026-05-13 — the user found "Today"
    /// showing as a highlighted-but-empty tab worse than no Today tab at
    /// all when the page was visited after the outdoor window.
    /// </summary>
    private static string RenderHomeDaySubNav(SiteInputs input, int activeOffset)
    {
        var today = input.GeneratedAtUtc.Date;
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        for (int n = 0; n <= MaxHomeDayOffset; n++)
        {
            if (CountHomeDayTiles(input, n) == 0) continue;

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
        var (firstHour, lastHourExcl) = OverviewWindow(input);
        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(firstHour);
        var dayWindowEnd = dayUtc.AddHours(lastHourExcl);

        string ChampionForLead(int lead)
        {
            if (input.ChampionByLead.TryGetValue(lead, out var perLead) && !string.IsNullOrEmpty(perLead))
                return perLead;
            return input.CurrentVersion;
        }
        // Phase-matching (with version-equality fallback) mirrors the
        // RenderIndex filter so the sub-nav doesn't disagree with the day
        // body. See the longer comment in RenderIndex for the rationale.
        bool MatchesChampionPhase(string rowVersion, int lead)
        {
            var champVersion = ChampionForLead(lead);
            if (string.Equals(rowVersion, champVersion, StringComparison.Ordinal)) return true;
            return input.PhaseByVersion.TryGetValue(rowVersion, out var rowPhase)
                && input.PhaseByVersion.TryGetValue(champVersion, out var champPhase)
                && !string.IsNullOrEmpty(rowPhase)
                && string.Equals(rowPhase, champPhase, StringComparison.Ordinal);
        }
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => MatchesChampionPhase(p.ModelVersion, p.LeadHours));

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
            // Band label moved to its own line under the UTCI value (and
            // wrapped in double-quotes) — the ⓘ toggle stays inline with
            // the UTCI value so the click target is always next to the
            // number it explains. Splits the previous one-liner into two
            // <div>s.
            feelsCell = string.Create(Ci, $"""
                <div class="feels">
                  <div>Feels like <strong style="color: {apparentColor}">{fl.ApparentC:0.0}°C</strong></div>
                  <div>UTCI <strong style="color: {utciColor}">{fl.UtciC:0.0}°C</strong> {toggle}</div>
                  <div class="utci-band"><small>"{Escape(PrettyUtciBand(fl.Band))}"</small></div>
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
