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
                It pulls forecasts from eight numerical weather-prediction (NWP) models —
                GFS, ECMWF IFS, DWD ICON, Météo-France, UK Met Office, Environment Canada GEM,
                ECMWF AIFS (the GraphCast-style AI model), and JMA Global —
                and blends them with LightGBM regressors trained against ERA5 reanalysis
                or per-station EA Hydrology rainfall, depending on the target.
              </p>

              <h3>What it predicts</h3>
              <ul>
                <li>
                  <strong>Temperature</strong> at 2 m — leads 24 / 48 / 72 / 96 / 120 h, blended
                  against ERA5 reanalysis at the Bonehill grid cell. Two flavours run side by
                  side: a <em>lean</em> 13-feature champion (Phase 2b) and a <em>rich</em>
                  88-feature challenger (Phase 2c) that adds per-NWP humidity / cloud / wind /
                  pressure secondaries.
                </li>
                <li>
                  <strong>Precipitation occurrence</strong> P(wet ≥ 0.1 mm/h) — one classifier per
                  EA gauge (Bellever, Princetown, Dartmoor near Hexworthy) at the same five
                  leads. Phase 3a (lean, 27 features) and Phase 3c (rich, 55 features — adds
                  per-NWP humidity, surface pressure, and EA trailing-rainfall persistence).
                  Truth from EA Hydrology 15-min tip readings, hourly-aggregated with a 4-of-4
                  reading gate.
                </li>
                <li>
                  <strong>Dry-window probability</strong> per UTC day — P(∃ contiguous N-hour
                  dry block in 09:00–18:00 local time) for N ∈ &#123;3, 4, 6&#125; hours at
                  leads 24 / 48 / 72 h, per station. Phase 3b (53-feature LightGBM champion)
                  and Phase 3g (parameter-free MC over the Phase 3a hourly P(wet) marginals)
                  ship side-by-side; 3g guarantees cross-window monotonicity by construction.
                </li>
                <li>
                  <strong>Feels-like</strong> — Bröde 2012 UTCI <em>and</em> Steadman 1994
                  shade-form apparent temperature, both derived (no model training) at
                  predict time from the temperature blender plus four element blenders
                  (humidity, wind, shortwave radiation, cloud cover). UTCI is the rigorous
                  biothermal index; Steadman is the BBC/BoM-style "feels like" the public knows.
                </li>
              </ul>

              <h3>Data sources</h3>
              <ul>
                <li><strong>Forecasts:</strong> Open-Meteo (live + historical-forecast API) — provides every NWP listed above through one consistent JSON interface.</li>
                <li><strong>Training truth (temperature + element blenders):</strong> ERA5 reanalysis via Open-Meteo (gapless, quantitative, ~5–7 day publication lag).</li>
                <li><strong>Training + verification truth (precipitation, dry window):</strong> Environment Agency Hydrology rainfall gauges (Bellever Dartmoor, Princetown, Dartmoor near Hexworthy), 15-min tips aggregated to hourly with a 4-of-4 reading gate.</li>
                <li><strong>Verification truth (temperature):</strong> METAR from aviationweather.gov + OGIMET historical, used as a real-observation cross-check rather than primary truth (lowland stations have systematic biases vs the moorland tor).</li>
              </ul>

              <h3>Pipeline</h3>
              <p>
                <code>collect</code> pulls fresh NWP forecasts + observations 4× daily aligned
                to NWP publication times. <code>predict</code> runs the blenders on that data
                and writes per-target Parquet trees to Cloudflare R2. <code>render-site</code>
                regenerates this static HTML on the same cycle plus a 2-hourly cron so the
                rolling MAE charts and observed wet-period bands stay current.
                <code>verify</code> runs weekly to flag rolling-MAE drift &gt; 1.5× training-test
                MAE per (model version, lead).
              </p>

              <h3>Caveats</h3>
              <ul>
                <li>ERA5 is a 0.25° gridded reanalysis. It represents a grid-cell average near Bonehill, not the tor itself — the blender learns this systematic offset.</li>
                <li>Lowland METAR (Exeter ~30 km, Yeovilton ~55 km) is a sanity check against a 393 m moorland tor, not ground truth. Useful as a cross-check, not the metric we tune to.</li>
                <li>Rolling-MAE charts and Models-page Δ-vs-best-single numbers warm up over the first 14 days post-train.</li>
                <li>Open-Meteo's historical-forecast endpoint returns best-available-per-valid-time, not rigorous "as-issued" forecasts. Good for PoC training; not for publication-grade reverification.</li>
              </ul>

              <p><small>Source: <a href="https://github.com/harry1310/WeatherBlend">github.com/harry1310/WeatherBlend</a>. Rendered {input.GeneratedAtUtc:yyyy-MM-dd HH:mm}Z.</small></p>
            </section>
            """;

        return WrapPage(input, "About", "about", body);
    }
}
