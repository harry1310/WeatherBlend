namespace WeatherBlend.Site;

public static partial class SitePages
{
    public static string RenderAbout(SiteInputs input)
    {
        var body = $"""
            <section>
              <h2>About WeatherBlend</h2>
              <p>
                WeatherBlend is a multi-model weather-forecast blending proof of concept for
                <strong>{Escape(input.LocationDisplay)}</strong>
                ({input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m).
                It combines forecasts from six numerical weather-prediction (NWP) models —
                GFS, ECMWF IFS, DWD ICON, Météo-France, UK Met Office, and Environment Canada GEM —
                via a LightGBM blender trained against ERA5 reanalysis.
              </p>

              <h3>Scope</h3>
              <p>
                This PoC targets <strong>temperature</strong>, <strong>hourly precipitation occurrence
                P(wet ≥ 0.1 mm/h)</strong>, and <strong>per-day dry-window probability</strong>
                (P(∃ contiguous N-hour dry block in target UTC day) for N = 3, 4, or 6 hours) at lead times
                <strong>24h, 48h, and 72h</strong>. Shorter and longer horizons are out of scope.
                Quantitative precipitation intensity (mm) is deferred indefinitely — the dry-window
                framing covers the user-facing question without the calibration headaches of
                conditional intensity regression.
              </p>

              <h3>Data sources</h3>
              <ul>
                <li><strong>Forecasts:</strong> Open-Meteo (live + historical-forecast API).</li>
                <li><strong>Training truth (temperature):</strong> ERA5 reanalysis via Open-Meteo (gapless, quantitative).</li>
                <li><strong>Training + verification truth (precipitation, dry window):</strong> Environment Agency Hydrology rainfall gauges (Bellever Dartmoor, Princetown), 15-min tips aggregated to hourly with a 4-of-4 reading gate.</li>
                <li><strong>Verification truth (temperature):</strong> METAR from aviationweather.gov and OGIMET, used as a real-observation sanity check.</li>
              </ul>

              <h3>Caveats</h3>
              <ul>
                <li>ERA5 is a 0.25° gridded reanalysis; it represents a grid-cell average near Bonehill Rocks, not the tor itself.</li>
                <li>Lowland METAR (Exeter, Yeovilton) has systematic biases against a 393m moorland tor — useful as a cross-check, not ground truth.</li>
                <li>Blender predictions currently cover one model version and a short verification window.</li>
              </ul>

              <p><small>Source: <a href="https://github.com/">WeatherBlend repo</a>. Rendered {input.GeneratedAtUtc:yyyy-MM-dd HH:mm}Z.</small></p>
            </section>
            """;

        return WrapPage(input, "About", "about", body);
    }
}
