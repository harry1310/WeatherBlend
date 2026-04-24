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

        var cards = new StringBuilder();
        foreach (var lead in new[] { 24, 48, 72 })
        {
            if (latestByLead.TryGetValue(lead, out var p))
            {
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
                      <header><h3>+{lead}h</h3><small>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</small></header>
                      <div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature.ToString("0.0", Ci)}°C</div>
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
                cards.Append(Ci, $"""
                    <article class="forecast-card forecast-card-empty">
                      <header><h3>+{lead}h</h3></header>
                      <div class="temp">—</div>
                      <footer><small>No prediction available</small></footer>
                    </article>
                    """);
            }
        }

        var skill = ComputeHeadlineSkill(input);
        var versionNote = string.IsNullOrEmpty(input.CurrentVersion)
            ? "No champion pinned in MANIFEST — cards may drift between active versions."
            : $"Champion version: <code>{Escape(input.CurrentVersion)}</code>. Charts comparing every active version against truth live on the <a href=\"skill.html\">skill page</a>.";

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Latest blended forecast</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m</p>
              </hgroup>
              <div class="forecast-grid">
            {cards}  </div>
              <p class="skill-line">{versionNote}</p>
              <p class="skill-line">{Escape(skill)}</p>
            </section>
            """);

        return WrapPage(input, "Home", "index", body.ToString());
    }

    private static string ComputeHeadlineSkill(SiteInputs input)
    {
        var scored = input.Predictions
            .Where(p => p.LeadHours == 24 && input.TruthByTime.ContainsKey(p.ValidTimeUtc))
            .ToList();
        if (scored.Count == 0) return "No scored predictions yet — skill headline will appear once ERA5 catches up.";

        var truth = scored.Select(p => input.TruthByTime[p.ValidTimeUtc]).ToArray();
        var blend = scored.Select(p => p.BlendTemperature).ToArray();
        var blendMae = Mae(blend, truth);

        // Best single across the six per-model columns.
        (string name, double mae)[] singles =
        {
            ("GFS",   MaeWithGaps(scored.Select(p => p.TempGfs).ToArray(),   truth)),
            ("ECMWF", MaeWithGaps(scored.Select(p => p.TempEcmwf).ToArray(), truth)),
            ("ICON",  MaeWithGaps(scored.Select(p => p.TempIcon).ToArray(),  truth)),
            ("MF",    MaeWithGaps(scored.Select(p => p.TempMf).ToArray(),    truth)),
            ("UKMO",  MaeWithGaps(scored.Select(p => p.TempUkmo).ToArray(),  truth)),
            ("GEM",   MaeWithGaps(scored.Select(p => p.TempGem).ToArray(),   truth)),
        };
        var best = singles.Where(s => !double.IsNaN(s.mae)).OrderBy(s => s.mae).FirstOrDefault();
        if (best.name is null || double.IsNaN(best.mae))
        {
            return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C.";
        }

        var pct = (best.mae - blendMae) / best.mae * 100.0;
        var verb = blendMae < best.mae ? "beat" : "trailed";
        return $"Over {scored.Count} scored 24h predictions, blend MAE is {blendMae.ToString("0.00", Ci)}°C — {verb} best single ({best.name}, {best.mae.ToString("0.00", Ci)}°C) by {Math.Abs(pct).ToString("0.0", Ci)}%.";
    }

    private static double Mae(double[] pred, double[] truth)
    {
        double sum = 0;
        int n = Math.Min(pred.Length, truth.Length);
        for (int i = 0; i < n; i++) sum += Math.Abs(pred[i] - truth[i]);
        return n == 0 ? double.NaN : sum / n;
    }

    private static double MaeWithGaps(double?[] pred, double[] truth)
    {
        double sum = 0;
        int n = 0;
        for (int i = 0; i < pred.Length && i < truth.Length; i++)
        {
            if (pred[i] is double v) { sum += Math.Abs(v - truth[i]); n++; }
        }
        return n == 0 ? double.NaN : sum / n;
    }
}
