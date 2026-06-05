using System.Text;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Top-level home page (<c>/index.html</c>) — minimal location picker.
    /// One large button per configured location linking into its
    /// <c>/{slug}/index.html</c> Overview. Phase D landing page; the
    /// per-loc Overview keeps the rich day-tab UTCI grid (renamed from
    /// "Home" pre-Phase-D).
    /// </summary>
    public static string RenderHomePicker(SiteInputs input)
    {
        // Only list locations that actually render pages. A tab-less location
        // (e.g. a data-only site like sennen_cove that's collecting/backfilling
        // but has no models/tabs yet) would otherwise link to a non-existent
        // {slug}/index.html — a 404. It surfaces here once it's given tabs.
        var locs = input.Locations.Where(l => l.Tabs.Count > 0).ToList();
        var body = new StringBuilder();
        body.Append("<section>");
        body.Append("""
              <hgroup>
                <h2>Choose a location</h2>
                <p>Multi-model blended forecasts for two Devon locations.</p>
              </hgroup>
            """);

        body.Append("<div class=\"loc-picker\">");
        if (locs.Count == 0)
        {
            body.Append("<p><em>No locations configured.</em></p>");
        }
        else
        {
            foreach (var loc in locs)
            {
                var tabsLine = loc.Tabs.Count == 0
                    ? "no tabs configured"
                    : string.Join(" · ", loc.Tabs.Select(PrettyTab));
                body.Append(Ci, $"""
                      <a class="loc-card" href="{loc.Name}/index.html">
                        <h3>{Escape(loc.DisplayName)}</h3>
                        <p class="loc-tabs">{Escape(tabsLine)}</p>
                      </a>
                    """);
            }
        }
        body.Append("</div>");

        body.Append("</section>");
        return WrapPage(input, "WeatherBlend", "home", body.ToString());
    }

    private static string PrettyTab(string tabId) => tabId.ToLowerInvariant() switch
    {
        "overview"    => "Overview",
        "temperature" => "Temperature",
        "rain"        => "Rain",
        "dry_window"  => "Dry window",
        "skill"       => "Skill",
        "models"      => "Models",
        _ => tabId,
    };
}
