using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Home page: one card per future valid hour, showing the blender's best
    /// forecast for that hour. "Best" = the smallest available lead — shorter
    /// leads see more recent NWP runs and score better, so for any given hour
    /// we surface the most-recent cycle's prediction. Cards flow left-to-right
    /// in the existing forecast-grid; valid times wrap to as many rows as the
    /// viewport needs.
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
        // cycle); within ties, freshest PredictionMadeAt wins. This collapses
        // the {24, 48, 72, 96, 120}h-lead matrix to a single forward sequence.
        var futurePredictions = cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc)
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

        var cards = new StringBuilder();
        foreach (var p in futurePredictions)
        {
            // Feels-like and P(wet) tolerate a ±1h drift between the temperature
            // anchor and the precip / feels-like cycle's anchor — predict cycles
            // sometimes land an hour off, and we'd rather show a chip than a gap.
            string feelsCell = "";
            if (TryNearest(feelsLikeByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var fl))
            {
                var apparentColor = TemperatureColor(fl!.ApparentC);
                var utciColor = TemperatureColor(fl.UtciC);
                feelsCell =
                    "<div class=\"feels\">"
                    + $"<div>Feels like <strong style=\"color: {apparentColor}\">{fl.ApparentC.ToString("0.0", Ci)}°C</strong></div>"
                    + $"<div>UTCI <strong style=\"color: {utciColor}\">{fl.UtciC.ToString("0.0", Ci)}°C</strong> <small>{Escape(PrettyUtciBand(fl.Band))}</small></div>"
                    + "</div>";
            }

            string pwetCell = "";
            if (TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw))
            {
                // ☔ when the displayed P(wet) rounds to ≥ 25% — anything below that
                // would print "0% ☔" which reads as a contradiction. Threshold
                // of 0.245 picks up genuine signal without flagging noise.
                var rain = pw!.ProbWet >= 0.245 ? " <span class=\"rain\">&#x2614;</span>" : "";
                pwetCell = $"<div class=\"pwet\">P(wet) <strong>{(pw.ProbWet * 100).ToString("0", Ci)}%</strong>{rain}</div>";
            }

            var tempColor = TemperatureColor(p.BlendTemperature);
            cards.Append(Ci, $"""
                <article class="forecast-card">
                  <header><h3>{p.ValidTimeUtc:ddd HH:mm}Z</h3><small>{p.ValidTimeUtc:yyyy-MM-dd}</small></header>
                  <div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature.ToString("0.0", Ci)}°C</div>
                  {feelsCell}
                  {pwetCell}
                  <footer><small>+{p.LeadHours}h · made {p.PredictionMadeAtUtc:MM-dd HH:mm}Z</small></footer>
                </article>
                """);
        }

        if (futurePredictions.Count == 0)
        {
            cards.Append("""
                <article class="forecast-card forecast-card-empty">
                  <div class="temp">—</div>
                  <footer><small>No forward predictions available</small></footer>
                </article>
                """);
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Forward forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m. Each card is the blended forecast at the shortest available lead — most-recent NWP cycle wins.</p>
              </hgroup>
              <div class="forecast-grid">
            {cards}  </div>
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
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
