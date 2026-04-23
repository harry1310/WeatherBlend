using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderVerify(SiteInputs input)
    {
        var content = new StringBuilder();

        content.Append("""
            <section>
              <hgroup>
                <h2>Verification</h2>
                <p>Rolling blend-vs-ERA5 MAE by lead time, stratified by model version.</p>
              </hgroup>
            """);

        if (input.RollingMae.Count == 0)
        {
            content.Append("<p><em>No rolling MAE points computed — the window is too short or there's no matching ERA5 truth yet.</em></p>");
        }
        else
        {
            // One chart per lead, with one line per model version.
            foreach (var lead in new[] { 24, 48, 72 })
            {
                var versions = input.RollingMae.Where(r => r.LeadHours == lead)
                    .Select(r => r.ModelVersion).Distinct().OrderBy(v => v, StringComparer.Ordinal).ToList();

                var series = new List<LineSeries>();
                var palette = new[] { "#7c4dff", "#26a69a", "#ef5350", "#ffa726", "#42a5f5" };
                for (int i = 0; i < versions.Count; i++)
                {
                    var v = versions[i];
                    var pts = input.RollingMae
                        .Where(r => r.LeadHours == lead && r.ModelVersion == v)
                        .OrderBy(r => r.WindowEndUtc)
                        .Select(r => (X: r.WindowEndUtc.ToOADate(), Y: r.BlendMae))
                        .ToList();
                    if (pts.Count > 0)
                    {
                        series.Add(new LineSeries(v, palette[i % palette.Length], pts));
                    }
                }

                if (series.Count == 0)
                {
                    content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                    content.Append(RenderEmptyChart($"Rolling MAE — lead {lead}h", "No scored predictions at this lead."));
                    continue;
                }

                content.Append(Ci, $"<h3>Lead {lead}h</h3>");
                content.Append(LineChartRenderer.Render(new LineChartSpec
                {
                    Title = $"Rolling MAE — lead {lead}h",
                    XLabel = "Window end (UTC)",
                    YLabel = "MAE (°C)",
                    Series = series,
                    FormatX = v => DateTime.FromOADate(v).ToString("MM-dd", Ci),
                    FormatY = v => v.ToString("0.00", Ci),
                }));
            }
        }

        content.Append("</section>");
        return WrapPage(input, "Verification", "verify", content.ToString());
    }
}
