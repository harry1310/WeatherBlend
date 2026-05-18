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
                Eight numerical weather-prediction (NWP) models feed in via Open-Meteo —
                NOAA GFS, ECMWF IFS, DWD ICON, Météo-France, UK Met Office (UKV+UM Global blended), Environment
                Canada GEM, ECMWF AIFS (the GraphCast-style AI model), and JMA Global —
                and the predictions are blended with LightGBM trained against ERA5 reanalysis or
                per-station EA Hydrology rainfall, depending on the target. The Met Office DataHub
                Spot product also ships as a sanity check on the temp + rain skill pages, plotted
                alongside the blenders.
              </p>

              <h3>What it predicts</h3>
              <ul>
                <li>
                  <strong>Temperature</strong> at 2 m — leads 24 / 48 / 72 / 96 / 120 h, blended
                  against ERA5 reanalysis at the Bonehill grid cell. Two flavours ship side by
                  side: <em>Phase 2b lean</em> (13-feature champion) and <em>Phase 2c rich</em>
                  (88-feature challenger, adds per-NWP humidity / cloud / wind / pressure secondaries).
                </li>
                <li>
                  <strong>Precipitation occurrence</strong> P(wet ≥ 0.1 mm/h), per hour — one
                  classifier per EA gauge (Bellever Dartmoor, Bovey Tracey, Dartmoor nr Hexworthy)
                  at the same five leads. <em>Phase 3a lean</em> (27 features) and <em>Phase 3c
                  rich</em> (55 features, adds per-NWP humidity, surface pressure, and EA
                  trailing-rainfall persistence). Truth from EA Hydrology 15-min tip readings,
                  hourly-aggregated with a 4-of-4 reading gate.
                </li>
                <li>
                  <strong>Dry-window probability</strong> per UTC day — P(∃ contiguous N-hour
                  dry block in 09:00–18:00 local time) for N ∈ &#123;3, 4, 6&#125; hours at
                  leads 24 / 48 / 72 h, per station. <em>Phase 3b</em> (53-feature LightGBM
                  champion) and <em>Phase 3g</em> (parameter-free Monte Carlo over Phase 3a's
                  hourly P(wet) marginals — 10,000 Bernoulli draws per row) ship side-by-side.
                  3g guarantees cross-window monotonicity P(N=3) ≥ P(N=4) ≥ P(N=6) by construction
                  (single MC pass, three indicators read off the same Bernoulli sequence).
                </li>
                <li>
                  <strong>Start-hour curve</strong> — for each (station, window, lead, day),
                  P(an N-hour dry block runs from each candidate start hour within the daytime
                  window). Derived from the same 3g MC pass as the dry-window prob — each hour
                  is its own marginal probability (overlapping windows, so the curve need not
                  sum to the daily "any block" figure). Sits alongside each window's
                  dry-window cards.
                </li>
                <li>
                  <strong>Feels-like</strong> — Bröde 2012 <abbr title="Universal Thermal Climate Index">UTCI</abbr>
                  <em>and</em> Steadman 1994 shade-form apparent temperature. Both derived at predict
                  time (no separate model training) from the temperature blender plus four element
                  blenders (humidity, wind, shortwave radiation, cloud cover). UTCI is the rigorous
                  biothermal index; Steadman is the BBC/BoM "feels like" the public knows.
                </li>
                <li>
                  <strong>Confidence tags.</strong> Conformal calibrators (split-conformal, α = 0.10
                  → 90% coverage) wrap every active P(wet) and dry-window blender. They're auto-fit
                  on the validation slice the moment a champion or challenger is promoted, so a
                  freshly-retrained version always ships with calibrators in place. Each forecast
                  hour or window carries a "confident wet" / "ambiguous" / "confident dry" tag based
                  on which prediction-set the calibrator places it in.
                </li>
              </ul>

              <h3>Data sources</h3>
              <ul>
                <li><strong>Forecasts:</strong> Open-Meteo (live + historical-forecast API) provides every NWP listed above through one consistent JSON interface; Met Office DataHub Spot adds a ninth deterministic forecast as a comparator (not a blender input).</li>
                <li><strong>Training truth (temperature + element blenders):</strong> ERA5 reanalysis via Open-Meteo (gapless, quantitative, ~5-day publication lag).</li>
                <li><strong>Training + verification truth (precipitation, dry window):</strong> Environment Agency Hydrology rainfall gauges (Bellever Dartmoor, Bovey Tracey, Dartmoor nr Hexworthy), 15-min tips aggregated to hourly with a 4-of-4 reading gate.</li>
                <li><strong>Verification cross-checks (temperature):</strong> METAR EGTE from aviationweather.gov (Exeter Airport, ~30 km E of Bonehill, 31 m elevation), and Met Office DataHub Land Observations at geohash gcj0z3 (Cocktree Throat / Taw Green near North Wyke, ~22 km NNW, ~120-150 m elevation). Both sit well below Bonehill's 393 m so carry a systematic warm bias — Taw Green's is smaller (~1.6 °C lapse-rate estimate vs ~2.4 °C for EGTE). Used as cross-checks, not metrics we tune to.</li>
              </ul>

              <h3>Pipeline</h3>
              <p>
                A Cloudflare Worker fires four GitHub Actions workflows on cron schedules:
                <code>collect</code> pulls fresh NWP forecasts + observations every 6 h (08:30 / 14:30
                / 20:30 / 02:30 UTC); <code>predict-and-render</code> runs 30 min later on the same
                cycle, executing every blender against the freshest inputs and regenerating this
                static site on Cloudflare Pages; <code>era5-refresh</code> backfills the daily ERA5
                truth window at 12:00 UTC; <code>verify</code> runs Mon + Thu at 09:30 UTC and flags
                rolling-MAE / Brier drift &gt; 1.5× training-test score per (model version, lead),
                emitting JSON sidecars that feed the Models-page verify-history tables.
              </p>

              <h3>Caveats</h3>
              <ul>
                <li>ERA5 is a 0.25° gridded reanalysis. It represents a grid-cell average near Bonehill, not the tor itself — the blender learns the systematic offset.</li>
                <li>METAR EGTE and Met Office obs (Taw Green) are both lowland sources well below Bonehill's 393 m. Useful as cross-checks; lapse-rate bias is a few °C and the blender doesn't try to predict either.</li>
                <li>Rolling-MAE / Brier charts and Models-page verify-history tables warm up over the first 5-9 days post-retrain (one verify cycle plus 5-day ERA5 latency before a fresh champion's predictions reach the verify window).</li>
                <li>Open-Meteo's historical-forecast endpoint returns best-available-per-valid-time, not rigorous "as-issued" forecasts. Good for PoC training; not publication-grade re-verification.</li>
                <li>Met Office Spot's PoP threshold is "any measurable precip", looser than our 0.1 mm/h training label; its line on the rain skill chart reads as direction-of-effect, not like-for-like.</li>
              </ul>

              <p><small>Source: <a href="https://github.com/harry1310/WeatherBlend">github.com/harry1310/WeatherBlend</a>. Rendered {input.GeneratedAtUtc:yyyy-MM-dd HH:mm}Z.</small></p>
            </section>
            """;

        return WrapPage(input, "About", "about", body);
    }
}
