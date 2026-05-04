using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Home page: forward forecast grouped by UTC day for a 5-day window.
    /// Each day gets a one-line summary header (min/max temp, mean P(wet),
    /// driest hour) followed by per-hour tiles. Each tile carries a UTCI
    /// info button that pops out the four element-blender values
    /// (temperature / humidity / wind / shortwave / cloud) that fed UTCI
    /// for that hour, so "why is UTCI this value?" is reachable without
    /// leaving the page.
    /// </summary>
    public static string RenderIndex(SiteInputs input)
    {
        // Champion-only filter (mirrors the Models page's "active phase" gate)
        // so a challenger version doesn't leak onto the headline cards. Empty
        // CurrentVersion = no manifest, fall back to "any".
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion)
            ? input.Predictions
            : input.Predictions.Where(p => p.ModelVersion == input.CurrentVersion);

        // For each future valid_time, take the smallest lead (most recent
        // cycle); within ties, freshest PredictionMadeAt wins. Cap at 5 days
        // forward so the page stays scannable — temp + rain max lead is 120h
        // anyway, and the dry-window page covers the per-day outlook beyond
        // that scope.
        var horizonEnd = input.GeneratedAtUtc.AddDays(5);
        var futurePredictions = cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc
                        && p.ValidTimeUtc <= horizonEnd)
            .GroupBy(p => p.ValidTimeUtc)
            .Select(g => g
                .OrderBy(p => p.LeadHours)
                .ThenByDescending(p => p.PredictionMadeAtUtc)
                .First())
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        // P(wet) lookup keyed by valid_time across all leads, smallest-lead-wins
        // mirroring the temperature pick. Bellever as the headline gauge.
        const string PwetStation = "ea_bellever_dartmoor";
        input.PrecipCurrentByStation.TryGetValue(PwetStation, out var pwetChampion);
        var pwetByValid = input.PrecipPredictions
            .Where(r => r.Station == PwetStation
                        && !string.IsNullOrEmpty(pwetChampion)
                        && r.Version == pwetChampion)
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

        var dayBlocks = new StringBuilder();
        var byDay = futurePredictions
            .GroupBy(p => p.ValidTimeUtc.Date)
            .OrderBy(g => g.Key)
            .ToList();

        // Sequential popover ID — Pico's <details> is the simplest no-JS
        // pop-out mechanism; each tile gets a unique id so a future feature
        // can deep-link to a specific hour's panel.
        int popoverId = 0;

        foreach (var day in byDay)
        {
            var dayPreds = day.OrderBy(p => p.ValidTimeUtc).ToList();
            var minT = dayPreds.Min(p => p.BlendTemperature);
            var maxT = dayPreds.Max(p => p.BlendTemperature);

            // Day-level rain summary: mean of available P(wet), plus the
            // hour with the lowest P(wet) ("best go-now hour"). Pulled from
            // the same Bellever pwetByValid map; tolerate ±1h drift per
            // tile.
            var dayPwets = dayPreds
                .Select(p => TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw) ? pw : null)
                .Where(pw => pw is not null)
                .Select(pw => pw!)
                .ToList();
            string daySummary;
            if (dayPwets.Count > 0)
            {
                var meanP = dayPwets.Average(p => p.ProbWet);
                var driest = dayPwets.OrderBy(p => p.ProbWet).First();
                daySummary = Ci.ToString() == ""
                    ? "" : ""; // (placeholder — formatted below)
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
            foreach (var p in dayPreds)
            {
                tiles.Append(RenderHourTile(p, feelsLikeByValid, pwetByValid, popoverId++));
            }

            // Day-of-week + date in the section heading; readers can scan
            // "Mon 5 May" without doing UTC arithmetic. UTC label retained
            // because every value on the page is still UTC.
            dayBlocks.Append(Ci, $"""
                <section class="day-block">
                  <h3>{day.Key:dddd dd MMMM}</h3>
                  {daySummary}
                  <div class="forecast-grid">
                {tiles}  </div>
                </section>
                """);
        }

        if (byDay.Count == 0)
        {
            dayBlocks.Append("""
                <section class="day-block">
                  <p><em>No forward predictions available</em></p>
                </section>
                """);
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Forward forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m. Five days forward, grouped by UTC day. Each tile is the champion-blender forecast at the shortest available lead. Click the ⓘ on a tile to see the wind / humidity / cloud / radiation values that drove its UTCI.</p>
              </hgroup>
              {dayBlocks}
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
    }

    /// <summary>
    /// One forward-hour tile — temperature headline, optional Feels-like /
    /// UTCI / P(wet) chips, and a Pico-styled <c>&lt;details&gt;</c> pop-out
    /// listing the four element-blender values that fed UTCI for this hour.
    /// </summary>
    private static string RenderHourTile(
        Models.TempPredictionRow p,
        IReadOnlyDictionary<DateTime, FeelsLikeForecastPoint> feelsLikeByValid,
        IReadOnlyDictionary<DateTime, PrecipForecastPoint> pwetByValid,
        int popoverId)
    {
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
            if (fl.TemperatureC is not null || fl.RelativeHumidityPct is not null
                || fl.WindSpeed10mMs is not null || fl.ShortwaveDownWm2 is not null
                || fl.CloudCoverPct is not null)
            {
                var rows = new StringBuilder();
                if (fl.TemperatureC is double t)
                    rows.Append(Ci, $"<tr><td>Temperature</td><td class=\"num\">{t:0.0} °C</td></tr>");
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
              <header><h4>{p.ValidTimeUtc:HH:mm}Z</h4></header>
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
