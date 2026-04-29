using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderIndex(SiteInputs input)
    {
        // Cards are "what's the blender saying now?" — always the champion, never a
        // challenger that happens to have also written the same (lead, valid_time).
        // Empty CurrentVersion means no manifest read (legacy), so fall back to "any".
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion)
            ? input.Predictions
            : input.Predictions.Where(p => p.ModelVersion == input.CurrentVersion).ToList();

        var latestByLead = cardSource
            .GroupBy(p => p.LeadHours)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(p => p.PredictionMadeAtUtc).First());

        // Phase 3a P(wet) companion: Bellever champion only. Precip predictions are
        // emitted on their own NWP-run grid (typically :00Z of each run), which rarely
        // lines up exactly with the temperature card's ValidTime. Match on the same
        // lead bucket and pick the precip ValidTime closest to the temp card's, within
        // ±12h so a random 24h-lead row from the opposite side of the day can't leak in.
        const string PwetStation = "ea_bellever_dartmoor";
        input.PrecipCurrentByStation.TryGetValue(PwetStation, out var pwetChampion);
        var pwetByLead = input.PrecipPredictions
            .Where(r => r.Station == PwetStation
                        && !string.IsNullOrEmpty(pwetChampion)
                        && r.Version == pwetChampion)
            // Deduplicate: same (lead, valid_time) appears twice when two runs pick
            // the same hour — keep the freshest.
            .GroupBy(r => (r.LeadHours, r.ValidTimeUtc))
            .Select(g => g.OrderByDescending(r => r.PredictedAtUtc).First())
            .GroupBy(r => r.LeadHours)
            .ToDictionary(g => g.Key, g => g.ToList());

        // Feels-like (UTCI + Steadman) per lead bucket. We had assumed the predict
        // pipeline emits on the same anchor as the temperature blender (so exact
        // (lead, valid) would match), but predict cycles drift in practice — temp
        // can be promoted on one anchor hours before feels-like rebuilds against
        // the new champion. So match by lead bucket and pick the row whose
        // ValidTime is closest to the temp card's, within ±12h — same fudge
        // factor the P(wet) chip uses below.
        var feelsLikeByLead = input.FeelsLikePredictions
            .GroupBy(u => (u.LeadHours, u.ValidTimeUtc))
            .Select(g => g.OrderByDescending(u => u.PredictedAtUtc).First())
            .GroupBy(u => u.LeadHours)
            .ToDictionary(g => g.Key, g => g.ToList());

        var cards = new StringBuilder();
        foreach (var lead in PocLeads)
        {
            if (latestByLead.TryGetValue(lead, out var p))
            {
                string feelsCell = "";
                if (feelsLikeByLead.TryGetValue(lead, out var uRows))
                {
                    var u = uRows
                        .Select(r => (Row: r, Delta: Math.Abs((r.ValidTimeUtc - p.ValidTimeUtc).TotalHours)))
                        .Where(x => x.Delta <= 12)
                        .OrderBy(x => x.Delta)
                        .FirstOrDefault().Row;
                    if (u is not null)
                    {
                        // Two-line chip: Steadman 1994 apparent-temperature first (the
                        // BBC/BoM-style "feels like" the public knows), UTCI underneath
                        // with its band name (the rigorous biothermal index). Both
                        // numbers take the temperature gradient so cold/hot reads at a
                        // glance; the band label only attaches to UTCI.
                        var apparentColor = TemperatureColor(u.ApparentC);
                        var utciColor = TemperatureColor(u.UtciC);
                        feelsCell =
                            "<div class=\"feels\">"
                            + $"<div>Feels like <strong style=\"color: {apparentColor}\">{u.ApparentC.ToString("0.0", Ci)}°C</strong></div>"
                            + $"<div>UTCI <strong style=\"color: {utciColor}\">{u.UtciC.ToString("0.0", Ci)}°C</strong> <small>{Escape(PrettyUtciBand(u.Band))}</small></div>"
                            + "</div>";
                    }
                }

                string pwetCell = "";
                if (pwetByLead.TryGetValue(lead, out var pwRows))
                {
                    var closest = pwRows
                        .Select(r => (Row: r, Delta: Math.Abs((r.ValidTimeUtc - p.ValidTimeUtc).TotalHours)))
                        .Where(x => x.Delta <= 12)
                        .OrderBy(x => x.Delta)
                        .FirstOrDefault();
                    if (closest.Row is not null)
                    {
                        pwetCell = $"<div class=\"pwet\">P(wet) <strong>{(closest.Row.ProbWet * 100).ToString("0", Ci)}%</strong> <small>Bellever {closest.Row.ValidTimeUtc:HH:mm}Z</small></div>";
                    }
                }

                var tempColor = TemperatureColor(p.BlendTemperature);
                cards.Append(Ci, $"""
                    <article class="forecast-card">
                      <header><h3>+{lead}h · {p.ValidTimeUtc:ddd}</h3><small>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</small></header>
                      <div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature.ToString("0.0", Ci)}°C</div>
                      {feelsCell}
                      {pwetCell}
                      <footer>
                        <small>Made {p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</small><br/>
                        <small>Model: <code>{Escape(p.ModelVersion)}</code></small>
                      </footer>
                    </article>
                    """);
            }
            else
            {
                var missingDay = input.GeneratedAtUtc.AddHours(lead);
                cards.Append(Ci, $"""
                    <article class="forecast-card forecast-card-empty">
                      <header><h3>+{lead}h · {missingDay:ddd}</h3></header>
                      <div class="temp">—</div>
                      <footer><small>No prediction available</small></footer>
                    </article>
                    """);
            }
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Latest blended forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m</p>
              </hgroup>
              <div class="forecast-grid">
            {cards}  </div>
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
    }
}
