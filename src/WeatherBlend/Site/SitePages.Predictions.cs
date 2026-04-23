using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderPredictions(SiteInputs input)
    {
        var rows = input.Predictions
            .OrderByDescending(p => p.PredictionMadeAtUtc)
            .ThenBy(p => p.LeadHours)
            .ToList();

        var tableBody = new StringBuilder();
        foreach (var p in rows)
        {
            var truth = input.TruthByTime.TryGetValue(p.ValidTimeUtc, out var t) ? t : (double?)null;
            var err = truth.HasValue ? p.BlendTemperature - truth.Value : (double?)null;
            tableBody.Append(Ci, $"""
                <tr>
                  <td><time>{p.PredictionMadeAtUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td><time>{p.ValidTimeUtc:yyyy-MM-dd HH:mm}Z</time></td>
                  <td>{p.LeadHours}h</td>
                  <td class="num"><strong>{p.BlendTemperature.ToString("0.00", Ci)}</strong></td>
                  <td class="num">{FmtNullable(truth)}</td>
                  <td class="num">{FmtNullable(err, "+0.00;-0.00;0.00")}</td>
                  <td class="num">{FmtNullable(p.TempGfs)}</td>
                  <td class="num">{FmtNullable(p.TempEcmwf)}</td>
                  <td class="num">{FmtNullable(p.TempIcon)}</td>
                  <td class="num">{FmtNullable(p.TempMf)}</td>
                  <td class="num">{FmtNullable(p.TempUkmo)}</td>
                  <td class="num">{FmtNullable(p.TempGem)}</td>
                  <td><small><code>{Escape(p.ModelVersion)}</code></small></td>
                </tr>
                """);
        }

        var body = new StringBuilder();
        body.Append(Ci, $"""
            <section>
              <hgroup>
                <h2>Predictions</h2>
                <p>{rows.Count} rows, sorted by prediction time (newest first).</p>
              </hgroup>
              <figure>
                <table>
                  <thead>
                    <tr>
                      <th>Made at</th>
                      <th>Valid time</th>
                      <th>Lead</th>
                      <th class="num">Blend</th>
                      <th class="num">ERA5</th>
                      <th class="num">Err</th>
                      <th class="num">GFS</th>
                      <th class="num">ECMWF</th>
                      <th class="num">ICON</th>
                      <th class="num">MF</th>
                      <th class="num">UKMO</th>
                      <th class="num">GEM</th>
                      <th>Version</th>
                    </tr>
                  </thead>
                  <tbody>
            {tableBody}      </tbody>
                </table>
              </figure>
            </section>
            """);

        return WrapPage(input, "Predictions", "predictions", body.ToString());
    }
}
