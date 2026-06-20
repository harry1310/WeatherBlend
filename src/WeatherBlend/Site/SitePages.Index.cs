using System.Text;
using WeatherBlend.Predict.Surface;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Home day-window: 0 = today's UTC date (offset 0), 1 = tomorrow, … up
    /// to <see cref="MaxHomeDayOffset"/>. The home page now renders one file
    /// per day with a sub-nav at the top to switch between them — the old
    /// stacked-day layout meant scrolling past today before reaching tomorrow.
    /// 5 forward days matches the temp + rain blender's 120h max lead; the
    /// dry-window forecasts page covers per-day outlook beyond that scope.
    /// </summary>
    public const int MaxHomeDayOffset = 5;

    /// <summary>
    /// One forward-day's home page. Tiles are the champion-blender forecast
    /// at the shortest lead, filtered to "outdoor" hours
    /// <c>[<see cref="ClimbingRangeFirstHourUtc"/>, <see cref="ClimbingRangeLastHourExclusiveUtc"/>)</c>
    /// — 21:00-03:59 UTC are dropped because they're irrelevant for climbing
    /// or walking trip planning. Each tile carries a UTCI ⓘ pop-out with the
    /// element-blender values that fed it.
    /// </summary>
    public static string RenderIndex(SiteInputs input, int dayOffset)
    {
        if (dayOffset < 0 || dayOffset > MaxHomeDayOffset)
            throw new ArgumentOutOfRangeException(nameof(dayOffset),
                $"Home dayOffset must be in [0, {MaxHomeDayOffset}].");

        var (firstHour, lastHourExcl) = OverviewWindow(input);
        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(firstHour);
        var dayWindowEnd   = dayUtc.AddHours(lastHourExcl);

        // Champion-PHASE filter — see ChampionMatcher for the full rationale
        // (phase-over-version matching after a retrain, test-fixture
        // version-equality fallback). Empty CurrentVersion = no manifest, fall
        // back to "any" so a freshly-deployed environment still renders cards.
        var champion = new ChampionMatcher(input);
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion)
            ? input.Predictions
            : input.Predictions.Where(p => champion.MatchesChampionPhase(p.ModelVersion));

        // For each future valid_time, take the smallest lead (most recent
        // cycle); within ties, freshest PredictionMadeAt wins. Restrict to
        // this day's outdoor window and drop any past hour (today's tab only).
        var dayPreds = cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc
                        && p.ValidTimeUtc >= dayWindowStart
                        && p.ValidTimeUtc <  dayWindowEnd)
            .LatestPerValid(p => p.ValidTimeUtc, p => p.LeadHours, p => p.PredictionMadeAtUtc)
            .OrderBy(p => p.ValidTimeUtc)
            .ToList();

        // P(wet) lookup keyed by valid_time across all leads, smallest-lead-wins
        // mirroring the temperature pick. Headline gauge resolves per rendered
        // location — see OverviewPwetStation.
        //
        // The home tiles show the precipitation CHAMPION — phases.yaml's
        // champion phase (3a), resolved per station to its newest Active
        // version (input.PrecipCurrentByStation, from
        // ModelMetadataRepository.GetChampionsByStation → ResolveStation-
        // ChampionVersion). 3a predicts hourly (.NET predict, 2026-05-04
        // hourly-spread change) so it covers every hourly tile. This is
        // deliberately the champion phase, not a 6-hourly challenger: 4a/4b
        // predict only at exact leads {24,48,72,96,120} (a 6-hourly valid
        // grid) and would leave the hourly tiles with P(wet) gaps. The
        // challengers lead the per-lead forecast + Models pages instead.
        var pwetStation = OverviewPwetStation(input);
        input.PrecipCurrentByStation.TryGetValue(pwetStation, out var pwetChampion);
        // Match by champion PHASE for the same retrain-window reason as the
        // temperature tiles above — yesterday's still-3a bundle covers
        // today's hours that today's freshly-minted bundle doesn't.
        var pwetByValid = string.IsNullOrEmpty(pwetChampion)
            ? new Dictionary<DateTime, PrecipForecastPoint>()
            : input.PrecipPredictions
                .Where(r => r.Station == pwetStation
                            && champion.MatchesChampionVersion(r.Version, pwetChampion))
                .LatestPerValid(r => r.ValidTimeUtc, r => r.LeadHours, r => r.PredictedAtUtc)
                .ToDictionary(r => r.ValidTimeUtc);

        // input.FeelsLikePredictions + LowCloudByValid are already scoped to
        // this page's location by RenderSiteCommand (single-location SiteInputs
        // invariant) — use directly.
        var feelsLikeByValid = input.FeelsLikePredictions
            .LatestPerValid(u => u.ValidTimeUtc, u => u.LeadHours, u => u.PredictedAtUtc)
            .ToDictionary(u => u.ValidTimeUtc);

        var lowCloudByValid = input.LowCloudByValid;

        // Rock surface / condensation (Phase P1) — smallest-lead-per-valid,
        // freshest made on ties, PER FACE, then collapsed to the BEST (driest,
        // furthest-from-dew) face per hour — the climbable aspect a one-chip tile
        // should summarise; the temp tab charts every face. Whole-crag locations
        // have a single empty face, so this is the old behaviour.
        var rockByValid = CollapseRockToBestFace(input.RockSurfacePredictions);

        // Sea-state badge inputs (marine locations with a seaStateBadge
        // config block — Sennen). Wave rows arrive pre-collapsed to one row
        // per valid by RenderSiteCommand's SQL but get the defensive
        // version-dedup anyway (newest version, freshest made — the
        // straddling-retrain rule, see SeriesDedup); wind speed + direction
        // collapse smallest-lead-per-valid like every other tile lookup.
        // Direction lives in its own wind_mvn tree and can lag the speed
        // tree (Sennen direction rows are new) — the evaluator skips the
        // wind trigger for hours with no direction rather than guessing.
        var seaSpec = input.RenderingFor?.SeaStateBadge;
        var waveByValid = seaSpec is null
            ? new Dictionary<DateTime, WaveForecastPoint>()
            : input.WavePredictions
                .GroupBy(w => w.ValidTimeUtc)
                .Select(g => g
                    .OrderByDescending(w => w.Version, StringComparer.Ordinal)
                    .ThenByDescending(w => w.PredictedAtUtc)
                    .First())
                .ToDictionary(w => w.ValidTimeUtc);
        // Built unconditionally (was sea-state-only): the climbing-conditions
        // strip needs wind at every location, not just marine ones. Empty when
        // a location has no wind blender (Membury) — harmless.
        var windSpeedMsByValid = input.WindPredictions
            .LatestPerValid(w => w.ValidTimeUtc, w => w.LeadHours, w => w.PredictedAtUtc)
            .ToDictionary(w => w.ValidTimeUtc, w => w.SpeedMs);
        var windDirDegByValid = seaSpec is null
            ? new Dictionary<DateTime, double>()
            : input.WindDirectionPredictions
                .LatestPerValid(w => w.ValidTimeUtc, w => w.LeadHours, w => w.PredictedAtUtc)
                .ToDictionary(w => w.ValidTimeUtc, w => w.DirectionDeg);

        // Day summary line above the tile grid: temp range + mean P(wet) +
        // driest hour. Falls back to a temp-only line when no P(wet) is
        // available for the headline gauge this day.
        string daySummary, tilesHtml;
        if (dayPreds.Count == 0)
        {
            daySummary = "<p class=\"skill-line\"><em>No forward predictions in this day's outdoor window.</em></p>";
            tilesHtml = "";
        }
        else
        {
            var minT = dayPreds.Min(p => p.BlendTemperature);
            var maxT = dayPreds.Max(p => p.BlendTemperature);
            var dayPwets = dayPreds
                .Select(p => TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw) ? pw : null)
                .Where(pw => pw is not null)
                .Select(pw => pw!)
                .ToList();
            if (dayPwets.Count > 0)
            {
                var meanP = dayPwets.Average(p => p.ProbWet);
                var driest = dayPwets.OrderBy(p => p.ProbWet).First();
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
                daySummary = string.Create(Ci, $"""<p class="skill-line"><strong>{minT:0.0}°C → {maxT:0.0}°C</strong> · no P(wet) for this day</p>""");
            }
            // Climbing-conditions verdict (idea #1) — Bonehill-first, gated by
            // the per-location config flag. Computed here where the location's
            // lat/lon (for daylight) and every per-hour input are in scope, then
            // passed into the tile as a ready Result so RenderHourTile stays a
            // pure renderer.
            var showConditions = input.RenderingFor?.ShowClimbingConditions == true;

            // Drying-model gate (rain-wet rock → Off with a dry-by ETA), per
            // location. Off by default until a location is calibrated + flipped;
            // when on, precompute each rain-wet hour's "climbable from" time once
            // from the whole rock series so each tile's gate can name it.
            var surfaceWaterGate = input.RenderingFor?.SurfaceWaterEnabled == true;
            // "climbable from" = the first hour the rain film drops below the Off
            // gate (the rock stops being too wet to climb), not when it's bone dry —
            // it comes good gradually after that. So key the ETA on the gate level.
            var rainDryByUtc = surfaceWaterGate
                ? ClimbingConditions.RainDryByMap(rockByValid.Values, ClimbingConditions.RainOffGateMm)
                : new Dictionary<DateTime, DateTime?>();

            var tiles = new StringBuilder();
            int popoverId = 0;
            foreach (var p in dayPreds)
            {
                ClimbingConditions.Result? conditions = null;
                if (showConditions)
                {
                    double? pWet = TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pwc)
                        ? pwc!.ProbWet : null;
                    double? windMph = TryNearest(windSpeedMsByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var wms)
                        ? wms * 2.23694 : null;
                    var rk = TryNearest(rockByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var rkv) ? rkv : null;

                    // Sea-state quality factor (tide / run-up / onshore spray) —
                    // sea-cliff locations with a seaStateBadge config block only
                    // (Sennen). Reuses the badge's nearest-hour wave + wind lookups.
                    ClimbingConditions.Factor? sea = null;
                    if (seaSpec is not null
                        && TryNearest(waveByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var swv))
                    {
                        double? sWindMph = windMph;
                        double? sWindDir = TryNearest(windDirDegByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var sdir)
                            ? sdir : null;
                        sea = SeaConditions.Evaluate(
                            swv!.WaveHeightM, sWindMph, sWindDir, seaSpec);
                    }

                    DateTime? rainDryBy = rk is { } rkp
                        && rainDryByUtc.TryGetValue(rkp.ValidTimeUtc, out var dby) ? dby : null;
                    // Sea-cliff SALT greasiness: feed ambient RH for marine locations
                    // (those with a sea-state block — Sennen) only. Inland crags pass
                    // null → the salt term is inert. Skipped when RH is unknown (NaN).
                    double? ambientRh = seaSpec is not null && rk is { } rkRh && !double.IsNaN(rkRh.AmbientRhPct)
                        ? rkRh.AmbientRhPct : null;
                    conditions = ClimbingConditions.Evaluate(
                        p.ValidTimeUtc, input.Latitude, input.Longitude,
                        p.BlendTemperature, pWet, windMph, rk, sea,
                        surfaceWaterGate: surfaceWaterGate,
                        rainDryByUtc: rainDryBy,
                        greasyMarginC: input.RenderingFor?.GreasyMarginC ?? ClimbingConditions.DefaultGreasyMarginC,
                        windExposure: input.RenderingFor?.WindExposure ?? 1.0,
                        ambientRhPct: ambientRh);
                }
                tiles.Append(RenderHourTile(p, feelsLikeByValid, pwetByValid, lowCloudByValid, rockByValid,
                    seaSpec, waveByValid, windSpeedMsByValid, windDirDegByValid,
                    input.WindGustByValidMs, conditions,
                    input.RenderingFor?.CloudFeatureNoun ?? "the hilltops",
                    input.Latitude, input.Longitude, popoverId++));
            }
            tilesHtml = string.Create(Ci, $"<div class=\"forecast-grid\">{tiles}</div>");
        }

        var body = new StringBuilder();
        body.Append("<section>");
        body.Append(RenderHomeDaySubNav(input, dayOffset));
        body.Append(Ci, $"""
              <hgroup>
                <h2>Forward forecast — {dayUtc:dddd dd MMMM}</h2>
                <p>{Escape(input.LocationDisplay)} — {input.Latitude.ToString("0.0000", Ci)}°, {input.Longitude.ToString("0.0000", Ci)}°, {input.ElevationMeters.ToString("0", Ci)}m.</p>
              </hgroup>
            """);
        // Climbing-mode toggle: only on climbing locations (Bonehill/Sennen).
        // Default OFF → the climbing index (verdict pill + Conditions drawer) is
        // hidden by CSS and the page reads like a plain weather page; ON reveals
        // it. Placed before the tiles so its inline script sets body.climbing-on
        // ahead of the tiles parsing (no flash for opted-in readers).
        if (input.RenderingFor?.ShowClimbingConditions == true)
            body.Append(ClimbingModeToggle());
        body.Append(daySummary);
        body.Append(tilesHtml);
        body.Append(RenderDryWindowCalculator(input, dayUtc, pwetStation));
        body.Append("</section>");

        // Page filename is set by RenderSiteCommand (index.html for today,
        // index-N.html for day-N). The pageId we pass to WrapPage is purely
        // for the top-nav highlight, so it must always be "index" — using
        // "index-N" silently broke the Home highlight on every forward-day
        // tab because NavActive("index-1", "index") returns false.
        return WrapPage(input, "Overview", "overview", body.ToString());
    }

    /// <summary>The shared "climbing range" — lowest UTC hour the Overview tile
    /// grid shows (inclusive), 05:00Z. THE single definition of the climbing
    /// range: every climbing location uses it (Bonehill, Sennen); a non-climbing
    /// location opts into the full day via <c>OverviewConfig.AllHours</c>. Also
    /// the fallback for callers without <see cref="SiteInputs.RenderingFor"/>
    /// (test fixtures).</summary>
    public const int ClimbingRangeFirstHourUtc = 5;

    /// <summary>The climbing range's exclusive upper bound — 20Z, so the grid
    /// renders through 19:00Z. See <see cref="ClimbingRangeFirstHourUtc"/>.</summary>
    public const int ClimbingRangeLastHourExclusiveUtc = 20;

    /// <summary>
    /// Resolve the Overview tile-grid hour window from the per-location
    /// descriptor (RenderSiteCommand sets it to the climbing range, or 0..24 for
    /// an all-hours location). Falls back to the climbing range for callers with
    /// no <see cref="SiteInputs.RenderingFor"/> (test fixtures).
    /// </summary>
    private static (int FirstHour, int LastHourExcl) OverviewWindow(SiteInputs input)
    {
        if (input.RenderingFor is { } loc)
            return (loc.OverviewFirstVisibleHourUtc, loc.OverviewLastVisibleHourUtcExclusive);
        return (ClimbingRangeFirstHourUtc, ClimbingRangeLastHourExclusiveUtc);
    }

    /// <summary>
    /// Headline P(wet) gauge for the Overview tiles — the rendered location's
    /// primary rainfall station: the first entry in its
    /// <see cref="LocationDescriptor.RainStationSlugs"/> (config order — Bonehill
    /// lists Bellever first, Membury lists Chards Snowdon Hill first). Prefer the
    /// first slug that actually has a champion version in
    /// <c>PrecipCurrentByStation</c> so a not-yet-trained gauge can't blank the
    /// tile grid; fall back to the first slug, then to Bellever for pre-Phase-D /
    /// test callers with no <see cref="SiteInputs.RenderingFor"/>. Hardcoding a
    /// single gauge here was the cause of the Membury Overview showing no
    /// P(wet) — see memory note feedback_avoid_hardcoded_phase_station_lists.
    /// </summary>
    private static string OverviewPwetStation(SiteInputs input)
    {
        var slugs = input.RenderingFor?.RainStationSlugs;
        if (slugs is null || slugs.Count == 0)
            return "ea_bellever_dartmoor";
        return slugs.FirstOrDefault(s => input.PrecipCurrentByStation.ContainsKey(s))
               ?? slugs[0];
    }

    /// <summary>
    /// "Will it stay dry?" calculator for the overview day page. Two dropdowns
    /// — set-off time + window length — driven by a small client-side grid of
    /// the Phase 3p Gaussian-copula MC probability that the chosen window is
    /// entirely dry (no hour ≥ 0.1 mm/h). The grid value is the start-hour
    /// curve's <c>RawProduct</c>, which despite the legacy name IS the
    /// per-(start, length) fraction of copula draws in which that window was
    /// all-dry (see DryWindow3pPredictor.ProbDryWindowWithStartHours) — NOT an
    /// iid product. So it answers the user's exact window and carries the
    /// within-day wet/dry autocorrelation 3p models.
    ///
    /// 3p start-hour curves exist only at leads 24/48/72 (target = tomorrow →
    /// +3), so this renders "" on today / +4 / +5. Length options are the
    /// trained windows {3,4,6}h, and the length menu is filtered per start
    /// (client-side, from the grid) so a window can't run past the end of the
    /// daytime span. Times are shown in UTC with a Z to match the overview
    /// tiles, the start-hour chart and every forecast page (the whole site
    /// labels displayed clock times in UTC). The daytime window itself is still
    /// defined as 09:00–18:00 local, so the UTC start options shift by an hour
    /// between BST and GMT — same as the existing start-hour chart axis.
    /// </summary>
    private static string RenderDryWindowCalculator(SiteInputs input, DateTime dayUtc, string station)
    {
        // Pick the dry-window MC phase whose start-hour curve actually exists for
        // this station — 3p at Bonehill, 3q at Sennen (copula MC over 3c). Was
        // hardcoded to 3p, which blanked the calculator at Sennen.
        // Freshest forecast per (window, start) for the target day: smallest
        // lead, newest made — mirrors the overview tiles' smallest-lead pick.
        var rows = DryWindowPhases.All
            .Where(p => !string.IsNullOrEmpty(p.StartHourCurveVersion))
            .Select(p => input.StartHourPredictions
                .Where(s => s.Station == station
                            && string.Equals(s.Version, p.StartHourCurveVersion, StringComparison.Ordinal)
                            && s.TargetDateUtc.Date == dayUtc.Date)
                .LatestPerValid(s => (s.WindowHours, s.StartHourUtc), s => s.LeadHours, s => s.PredictedAtUtc)
                .ToList())
            .FirstOrDefault(r => r.Count > 0);
        if (rows is null || rows.Count == 0) return "";

        // All displayed clock times on the site are UTC with a Z (overview
        // tiles, start-hour chart, every forecast page) — match that so the
        // calculator agrees with the tiles right above it. The daytime window
        // is still DEFINED as 09:00–18:00 local; in UTC that's 08:00–17:00Z in
        // BST / 09:00–18:00Z in GMT, so the start options shift seasonally
        // (exactly like the existing start-hour chart axis).
        string Label(int startHourUtc, int addHours) =>
            string.Create(Ci, $"{startHourUtc + addHours:00}:00Z");

        // start (UTC, ascending) -> ordered list of {length, end-label, prob}.
        var byStart = rows
            .GroupBy(r => r.StartHourUtc)
            .OrderBy(g => g.Key)
            .Select(g => new
            {
                t = Label(g.Key, 0),
                L = g.OrderBy(r => r.WindowHours)
                     .Select(r => new { n = r.WindowHours, e = Label(g.Key, r.WindowHours), p = r.RawProduct })
                     .ToList(),
            })
            .ToList();

        var json = System.Text.Json.JsonSerializer.Serialize(new { starts = byStart });

        var sb = new StringBuilder();
        sb.Append("<section class=\"drywin-calc\">");
        sb.Append("<h3>Will it stay dry?</h3>");
        sb.Append("<p class=\"drywin-intro\">Pick when you'd set off and for how long — the chance it stays dry (no hour with ≥&#8201;0.1&#8201;mm of rain) right through that window, from the Phase&#160;3p copula simulation.</p>");
        sb.Append("<div class=\"drywin-controls\">");
        sb.Append("<label>Set off at <select class=\"dw-start\">");
        for (int i = 0; i < byStart.Count; i++)
            sb.Append(Ci, $"<option value=\"{i}\">{byStart[i].t}</option>");
        sb.Append("</select></label>");
        sb.Append("<label>For <select class=\"dw-len\"></select></label>");
        sb.Append("</div>");
        sb.Append("<p class=\"drywin-result\" aria-live=\"polite\"></p>");
        sb.Append("<script type=\"application/json\" class=\"dw-data\">").Append(json).Append("</script>");
        sb.Append(DryWindowCalcScript);
        sb.Append("</section>");
        return sb.ToString();
    }

    /// <summary>
    /// Vanilla-JS for the dry-window calculator. Scoped to its own
    /// <c>.drywin-calc</c> section via <c>document.currentScript</c>, so it is
    /// self-contained per page and independent of the shared <c>chart.js</c>.
    /// Populates the length menu from the selected start (so invalid
    /// start+length pairs never appear) and writes the live result line.
    /// </summary>
    private const string DryWindowCalcScript = """
        <script>
        (function () {
          var root = document.currentScript.closest('.drywin-calc');
          if (!root) return;
          var data;
          try { data = JSON.parse(root.querySelector('.dw-data').textContent); } catch (e) { return; }
          var startSel = root.querySelector('.dw-start');
          var lenSel = root.querySelector('.dw-len');
          var out = root.querySelector('.drywin-result');
          function colour(p) { return p >= 0.7 ? '#2e7d32' : (p >= 0.4 ? '#b8860b' : '#c62828'); }
          function fillLengths() {
            var s = data.starts[startSel.value];
            lenSel.innerHTML = '';
            for (var i = 0; i < s.L.length; i++) {
              var o = s.L[i];
              var opt = document.createElement('option');
              opt.value = i;
              opt.textContent = o.n + ' hours (to ' + o.e + ')';
              lenSel.appendChild(opt);
            }
          }
          function update() {
            var s = data.starts[startSel.value];
            var o = s.L[lenSel.value];
            if (!o) { out.textContent = ''; return; }
            var pct = Math.round(o.p * 100);
            out.innerHTML = 'Set off at <strong>' + s.t + '</strong> for <strong>' + o.n +
              ' hours</strong>: <strong style="color:' + colour(o.p) + '">' + pct +
              '%</strong> chance it stays dry.';
          }
          startSel.addEventListener('change', function () { fillLengths(); update(); });
          lenSel.addEventListener('change', update);
          fillLengths();
          update();
        })();
        </script>
        """;

    /// <summary>
    /// Day sub-nav for the home page — pill links labelled "Mon 4/5"
    /// style (day-of-week short + day/month numeric, e.g. "Tue 5/5"). Today
    /// is offset 0 (file index.html); each forward day gets index-{n}.html.
    /// Days with zero tiles after the outdoor-window + future-only filter
    /// are skipped so the sub-nav can't link to a blank page. The
    /// "keep the active offset visible even if empty" exception (added
    /// 2026-05-07) was removed 2026-05-13 — the user found "Today"
    /// showing as a highlighted-but-empty tab worse than no Today tab at
    /// all when the page was visited after the outdoor window.
    /// </summary>
    private static string RenderHomeDaySubNav(SiteInputs input, int activeOffset)
    {
        var today = input.GeneratedAtUtc.Date;
        var s = new StringBuilder();
        s.Append("<nav class=\"lead-nav\"><ul>");
        for (int n = 0; n <= MaxHomeDayOffset; n++)
        {
            if (CountHomeDayTiles(input, n) == 0) continue;

            var date = today.AddDays(n);
            var file = n == 0 ? "index.html" : $"index-{n}.html";
            var label = n == 0 ? "Today" : $"{date:ddd} {date.Day}/{date.Month}";
            var cls = n == activeOffset ? " class=\"active\"" : "";
            s.Append(Ci, $"<li><a href=\"{file}\"{cls}>{Escape(label)}</a></li>");
        }
        s.Append("</ul></nav>");
        return s.ToString();
    }

    /// <summary>
    /// Count of tiles that the home page would render for the given
    /// day offset. Mirrors the filter chain at the top of
    /// <see cref="RenderIndex"/> — champion-only per lead, future-of-now,
    /// inside the outdoor visible-hour window, smallest-lead wins per
    /// valid_time. Used by the day sub-nav to suppress days that would
    /// link to an empty page.
    /// </summary>
    private static int CountHomeDayTiles(SiteInputs input, int dayOffset)
    {
        var (firstHour, lastHourExcl) = OverviewWindow(input);
        var todayUtc = input.GeneratedAtUtc.Date;
        var dayUtc = todayUtc.AddDays(dayOffset);
        var dayWindowStart = dayUtc.AddHours(firstHour);
        var dayWindowEnd = dayUtc.AddHours(lastHourExcl);

        // Phase-matching (with version-equality fallback) mirrors the
        // RenderIndex filter so the sub-nav doesn't disagree with the day
        // body. See ChampionMatcher for the rationale.
        var champion = new ChampionMatcher(input);
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion)
            ? input.Predictions
            : input.Predictions.Where(p => champion.MatchesChampionPhase(p.ModelVersion));

        return cardSource
            .Where(p => p.ValidTimeUtc > input.GeneratedAtUtc
                        && p.ValidTimeUtc >= dayWindowStart
                        && p.ValidTimeUtc < dayWindowEnd)
            .Select(p => p.ValidTimeUtc)
            .Distinct()
            .Count();
    }

    /// <summary>
    /// One forward-hour tile — temperature headline, optional Feels-like /
    /// UTCI / P(wet) chips, and a Pico-styled <c>&lt;details&gt;</c> pop-out
    /// listing the four element-blender values that fed UTCI for this hour.
    /// </summary>
    /// <summary>Visibility-signal firing threshold — number of vis-publishing
    /// NWPs (out of 6) that must forecast sub-1km vis for the badge to fire
    /// on visibility alone.</summary>
    private const int LowCloudVisFireThreshold = 3;

    /// <summary>Cloud-base-signal firing threshold — number of NWPs (out of
    /// up to 11) that must forecast T-Td &lt; 1.5°C for the badge to fire
    /// on cloud-base alone. Higher absolute count than vis because the
    /// denominator is much larger.</summary>
    private const int LowCloudBaseFireThreshold = 6;

    /// <summary>Tile-badge lookup for rock surface rows: dedup smallest-lead /
    /// freshest-made PER (valid, face), then keep the BEST face per valid — the
    /// driest wall (least surface water), and among equally-dry walls the one
    /// furthest from its dew point. On a sea cliff you climb the dry aspect, so
    /// the badge reflects what's climbable; the wetter faces stay in the temp-tab
    /// chart + drawer (Harry 2026-06-20, was worst-face). Single-face locations
    /// (Bonehill, empty face) reduce to one-row-per-valid. Internal for tests.</summary>
    internal static Dictionary<DateTime, RockSurfaceForecastPoint> CollapseRockToBestFace(
        IReadOnlyList<RockSurfaceForecastPoint> rows)
        => rows
            .GroupBy(r => r.Face)
            .SelectMany(g => g.LatestPerValid(r => r.ValidTimeUtc, r => r.LeadHours, r => r.PredictedAtUtc))
            .GroupBy(r => r.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(r => r.SurfaceWaterMm)                 // driest face first
                      .ThenByDescending(r => r.CondensationMarginC)   // then furthest from the dew point
                      .ThenBy(r => r.Face, StringComparer.Ordinal)    // deterministic on ties
                      .First());

    private static string RenderHourTile(
        Models.TempPredictionRow p,
        IReadOnlyDictionary<DateTime, FeelsLikeForecastPoint> feelsLikeByValid,
        IReadOnlyDictionary<DateTime, PrecipForecastPoint> pwetByValid,
        IReadOnlyDictionary<DateTime, LowCloudSignal> lowCloudByValid,
        IReadOnlyDictionary<DateTime, RockSurfaceForecastPoint> rockByValid,
        SeaStateBadgeSpec? seaSpec,
        IReadOnlyDictionary<DateTime, WaveForecastPoint> waveByValid,
        IReadOnlyDictionary<DateTime, double> windSpeedMsByValid,
        IReadOnlyDictionary<DateTime, double> windDirDegByValid,
        IReadOnlyDictionary<DateTime, double> windGustByValidMs,
        ClimbingConditions.Result? conditions,
        string cloudFeatureNoun,
        double latitude,
        double longitude,
        int popoverId)
    {
        // ---- Alerts (low-cloud / rock / sea) collected as drawer lines ----
        // Each fired family contributes one <div class="alert-line"> to the
        // Alerts drawer body, plus to the family count and the overall severity
        // (any red → red, else any amber → amber). Replaces the old per-badge
        // <details> pop-outs that overflowed the tile (2026-06-16 redesign).
        var alertLines = new StringBuilder();
        int alertCount = 0;
        bool anyRed = false, anyAmber = false;
        void AddAlert(BadgeSeverity sev, string icon, string text)
        {
            alertCount++;
            if (sev == BadgeSeverity.Red) anyRed = true;
            else if (sev == BadgeSeverity.Amber) anyAmber = true;
            var color = sev == BadgeSeverity.Red ? "var(--alert-red)" : "var(--alert-amber)";
            alertLines.Append(Ci,
                $"""<div class="alert-line"><span class="pip" style="color:{color}">{icon}</span><span>{text}</span></div>""");
        }

        // Low-cloud / mist — amber when ONE signal hits its threshold, red when
        // BOTH do (unified severity, 2026-06-12). Both fired signals are listed.
        if (lowCloudByValid.TryGetValue(p.ValidTimeUtc, out var lc))
        {
            var visFired = lc.VisFiredCount >= LowCloudVisFireThreshold;
            var cbFired  = lc.CloudBaseFiredCount >= LowCloudBaseFireThreshold;
            var severity = LowCloudBadge.Evaluate(visFired, cbFired);
            if (severity != BadgeSeverity.None)
            {
                var parts = new List<string>();
                if (visFired)
                    parts.Add(string.Create(Ci, $"{lc.VisFiredCount}/{lc.VisTotalCount} NWPs: mist (vis &lt; 1 km)"));
                if (cbFired)
                    parts.Add(string.Create(Ci, $"{lc.CloudBaseFiredCount}/{lc.CloudBaseTotalCount} NWPs: cloud base below {cloudFeatureNoun} (T−Td &lt; 1.5°C)"));
                AddAlert(severity, "☁", $"Low cloud — {string.Join("; ", parts)}");
            }
        }

        // Rock greasy/wet is NO LONGER an alert (de-dup 2026-06-16): it's carried
        // wholly by the climbing index now — the friction factor's greasy/wet
        // penalty + the verdict reason. Removing the badge avoids saying the same
        // thing twice (Conditions verdict + a red/amber pill).

        // Sea-state (marine locations with a seaStateBadge block — Sennen):
        // tide + run-up ACCESS triggers only now — onshore wind moved to the
        // climbing index's "Spray" quality factor (de-dup 2026-06-16).
        if (seaSpec is not null
            && TryNearest(waveByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var wv))
        {
            double? windMph = TryNearest(windSpeedMsByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var windMs)
                ? windMs * 2.23694
                : null;
            double? windDir = TryNearest(windDirDegByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var dirDeg)
                ? dirDeg
                : null;
            var sea = SeaStateBadge.Evaluate(
                tideHeightMsl: wv!.TideHeightMsl,
                waveHeightM: wv.WaveHeightM,
                swellPeriodS: wv.SwellPeriodS,
                windMph: windMph,
                windDirDeg: windDir,
                spec: seaSpec);
            if (sea.Severity != BadgeSeverity.None)
                AddAlert(sea.Severity, "🌊", $"Sea state — {Escape(string.Join("; ", sea.FiredTriggers))}");
        }

        // Feels-like / UTCI face line + the element rows that explain UTCI.
        // The "why is UTCI this value" inputs now live in the Detail drawer
        // (detailRows) rather than an inline ⓘ pop-out. ±1h drift tolerance —
        // predict cycles can land an hour off and we'd rather show a value.
        // P(wet) for this hour, looked up once: it feeds BOTH the sky glyph (rain
        // is now folded into the tile's weather icon) and the P(wet) line below.
        PrecipForecastPoint? pwPoint =
            TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pwHit) ? pwHit : null;

        string feelsCell = "";
        string skyCell = "";
        var detailRows = new StringBuilder();
        if (TryNearest(feelsLikeByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var fl))
        {
            // Sky glyph (top-right of the tile) from this hour's cloud + radiation,
            // now also showing rain when P(wet) is high (the umbrella moved here off
            // the P(wet) line, Harry 2026-06-20).
            skyCell = SkyGlyph(fl!.CloudCoverPct, fl.ShortwaveDownWm2, p.ValidTimeUtc, latitude, longitude, pwPoint?.ProbWet);
            var apparentColor = TemperatureColor(fl!.ApparentC);
            var utciColor = TemperatureColor(fl.UtciC);
            // Feels-like and UTCI on their OWN lines (Harry 2026-06-20: UTCI was
            // wrapping onto the wrong line when it shared a row with feels-like);
            // band as a quiet caption under.
            feelsCell = string.Create(Ci, $"""
                <div class="feels">
                  <div>Feels like <strong style="color: {apparentColor}">{fl.ApparentC:0.0}°C</strong></div>
                  <div>UTCI <strong style="color: {utciColor}">{fl.UtciC:0.0}°C</strong></div>
                  <div class="utci-band"><small>"{Escape(PrettyUtciBand(fl.Band))}"</small></div>
                </div>
                """);

            // Element inputs for the Detail drawer. Temperature deliberately
            // omitted — it's the tile headline. Stored m/s shown as mph (site
            // unit decision 2026-05-28); gust appended only when materially
            // above mean wind so calm rows don't read "Wind 3 · gust 3".
            if (fl.RelativeHumidityPct is double rh)
                detailRows.Append(Ci, $"<tr><td>Humidity</td><td class=\"num\">{rh:0} %</td></tr>");
            if (fl.WindSpeed10mMs is double ws)
            {
                var windMph = ws * 2.23694;
                string gustSuffix = "";
                if (windGustByValidMs.TryGetValue(p.ValidTimeUtc, out var gustMs))
                {
                    var gustMph = gustMs * 2.23694;
                    if (gustMph > windMph + 1.0)
                        gustSuffix = string.Create(Ci, $" · gust {gustMph:0.0} mph");
                }
                detailRows.Append(Ci, $"<tr><td>Wind 10 m</td><td class=\"num\">{windMph:0.0} mph{gustSuffix}</td></tr>");
            }
            if (fl.ShortwaveDownWm2 is double sw)
                detailRows.Append(Ci, $"<tr><td>Shortwave down</td><td class=\"num\">{sw:0} W/m²</td></tr>");
            if (fl.CloudCoverPct is double cc)
                detailRows.Append(Ci, $"<tr><td>Cloud cover</td><td class=\"num\">{cc:0} %</td></tr>");
        }

        string pwetCell = "";
        if (pwPoint is { } pw)
        {
            // Umbrella removed (Harry 2026-06-20): rain now lives in the sky glyph,
            // so the P(wet) line is just the number.
            var pwColor = PrecipProbColor(pw.ProbWet);
            pwetCell = string.Create(Ci, $"<div class=\"pwet\">P(wet) <strong style=\"color: {pwColor}\">{(pw.ProbWet * 100):0}%</strong></div>");
        }

        // ---- Status pill + Conditions drawer (the climbing index) ----
        // The verdict shows as a small coloured pill on the face (at-a-glance),
        // with the per-factor breakdown tucked into a tier-tinted drawer.
        string statusPill = "", conditionsDrawer = "";
        if (conditions is { } c)
        {
            // "<24h estimate" marker: near-term hours run the rock calc on raw NWP
            // wind/sun/cloud (air temp stays on the blend) because those element
            // blenders don't predict inside 24h. Flag it so the reader knows the
            // verdict here is a touch rawer than the +24h-onward tiles (Harry 2026-06-20).
            var nearTerm = TryNearest(rockByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var rkTier)
                && rkTier?.ForcingTier == RockSurfacePredictPipeline.ForcingTierNowcast;
            var nowcastNote = nearTerm
                ? "<div class=\"why nowcast-note\">&#8776; &lt;24h estimate — wind, sun &amp; cloud from raw NWP (blends start +24h)</div>"
                : "";
            var nowcastTag = nearTerm
                ? "<span class=\"nowcast-tag\" title=\"&lt;24h estimate — wind/sun/cloud from raw NWP\">&#8776;</span>"
                : "";
            statusPill = string.Create(Ci,
                $"""<span class="status" style="--cond-color: {c.TierColor}">{Escape(c.TierLabel)}{nowcastTag}</span>""");
            var factorRows = new StringBuilder();
            foreach (var f in c.Factors)
                factorRows.Append(Ci, $"<tr><td>{Escape(f.Name)}</td><td class=\"num\">{(f.Score * 100):0}</td><td class=\"det\">{Escape(f.Detail)}</td></tr>");
            conditionsDrawer = string.Create(Ci, $"""
                <details class="drawer cond-drawer" style="--cond-color: {c.TierColor}">
                  <summary class="s-cond"><span class="dot"></span>Conditions<span class="chev">&#x25B6;</span></summary>
                  <div class="drawer-body">
                    <div class="why">{Escape(c.Reason)}</div>
                    {nowcastNote}
                    <table class="drawer-table">{factorRows}</table>
                  </div>
                </details>
                """);
        }

        // ---- Alerts drawer ----
        // On climbing pages (conditions present) it's ALWAYS rendered — greyed
        // to "0" when nothing fired — so every tile carries the same drawer set
        // and the grid stays aligned. Also shown anywhere an alert fired.
        string alertsDrawer = "";
        if (conditions is not null || alertCount > 0)
        {
            string sumCls, body, countStyle;
            if (alertCount == 0)
            {
                sumCls = "s-none";
                countStyle = "background: var(--pico-muted-border-color); color: var(--pico-muted-color)";
                body = "<div class=\"why\">Nothing firing this hour.</div>";
            }
            else
            {
                sumCls = anyRed ? "s-red" : "s-amber";
                countStyle = anyRed ? "background: var(--alert-red)" : "background: var(--alert-amber)";
                body = alertLines.ToString();
            }
            alertsDrawer = string.Create(Ci, $"""
                <details class="drawer alert-drawer">
                  <summary class="{sumCls}"><span class="dot"></span>Alerts<span class="count" style="{countStyle}">{alertCount}</span></summary>
                  <div class="drawer-body">{body}</div>
                </details>
                """);
        }

        // ---- Detail drawer (UTCI element inputs + provenance) ----
        // Always rendered: even with no element values it carries the
        // lead / made-time line, so the drawer set is uniform across tiles.
        var detailDrawer = string.Create(Ci, $"""
            <details class="drawer detail-drawer">
              <summary class="s-none"><span class="dot"></span>Detail<span class="chev">&#x25B6;</span></summary>
              <div class="drawer-body">
                <table class="drawer-table">{detailRows}</table>
                <div class="why made-line">+{p.LeadHours}h · made {p.PredictionMadeAtUtc:MM-dd HH:mm}Z</div>
              </div>
            </details>
            """);

        var tempColor = TemperatureColor(p.BlendTemperature);
        return string.Create(Ci, $"""
            <article class="forecast-card">
              <div class="card-head"><span class="time">{p.ValidTimeUtc:HH:mm}Z</span>{statusPill}</div>
              <div class="hero">
                <div class="temp-line"><div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature:0.0}°C</div>{skyCell}</div>
                {feelsCell}
                {pwetCell}
              </div>
              {conditionsDrawer}
              {alertsDrawer}
              {detailDrawer}
            </article>
            """);
    }

    /// <summary>
    /// Sky glyph for the tile header from the hour's cloud cover + shortwave.
    /// Daytime folds the reported cloud fraction with the radiation "clearness
    /// index" (actual SW ÷ clear-sky SW for the sun's elevation), so a
    /// bright-but-cloudy hour reads sunnier than the cloud % alone would say — the
    /// same SW-vs-cloud disagreement the rock model hit under near-total cloud.
    /// Night (sun below the horizon) shows moon vs cloud on the cloud fraction
    /// alone. Returns "" when there's no cloud value to show.
    /// </summary>
    /// <summary>At/above this hourly P(wet) the sky glyph shows steady rain (🌧️);
    /// between <see cref="SkyShowerProb"/> and here it shows showers (🌦️ by day).
    /// Mirrors the P(wet)-line umbrella threshold that used to live on the P(wet)
    /// line (0.245) — the umbrella was folded into the weather icon here so the
    /// P(wet) line could lose it (Harry 2026-06-20).</summary>
    private const double SkyShowerProb = 0.245;
    private const double SkyRainLikelyProb = 0.45;

    internal static string SkyGlyph(double? cloudPct, double? swWm2, DateTime validUtc, double lat, double lon, double? pWet = null)
    {
        if (cloudPct is not double cloud) return "";
        var elevDeg = SolarGeometry.SolarPosition(validUtc, lat, lon).ElevationDeg;
        var cloudFrac = Math.Clamp(cloud / 100.0, 0.0, 1.0);

        // Rain takes visual priority over the sun/cloud split — a wet hour reads as
        // rain first. By day a shower keeps a hint of sun (🌦️); steady rain or a
        // night shower is a plain rain cloud (🌧️).
        if (pWet is double pw && pw >= SkyShowerProb)
        {
            var (rainGlyph, rainLabel) = pw >= SkyRainLikelyProb
                ? ("\U0001F327️", "Rain likely")                                  // 🌧️
                : elevDeg > 0 ? ("\U0001F326️", "Showers possible")               // 🌦️ (day)
                              : ("\U0001F327️", "Showers possible");              // 🌧️ (night)
            return string.Create(Ci, $"""<span class="sky" title="{rainLabel} · cloud {cloud:0}% · P(wet) {pw * 100:0}%">{rainGlyph}</span>""");
        }

        string glyph, label;
        if (elevDeg <= 0)
        {
            (glyph, label) = cloudFrac < 0.5 ? ("\U0001F319", "Clear night") : ("☁️", "Cloudy night");
        }
        else
        {
            // Effective cloudiness blends the reported cloud fraction with how much
            // sun actually reaches the ground (1 − clearness index). The radiation
            // term is only trusted when the sun is reasonably up, where the
            // clear-sky reference is stable.
            var eff = cloudFrac;
            var sinElev = Math.Sin(elevDeg * Math.PI / 180.0);
            if (swWm2 is double sw && elevDeg > 10.0 && sinElev > 0.0)
            {
                var clearSky = 1098.0 * sinElev * Math.Exp(-0.057 / sinElev);   // Haurwitz clear-sky GHI
                var k = Math.Clamp(sw / clearSky, 0.0, 1.0);                    // clearness index
                eff = 0.5 * cloudFrac + 0.5 * (1.0 - k);
            }
            (glyph, label) =
                eff < 0.20 ? ("☀️", "Sunny") :
                eff < 0.55 ? ("\U0001F324️", "Mostly sunny") :
                eff < 0.80 ? ("⛅", "Partly cloudy") :
                             ("☁️", "Cloudy");
        }
        return string.Create(Ci, $"""<span class="sky" title="{label} · cloud {cloud:0}%">{glyph}</span>""");
    }

    /// <summary>
    /// "Climbing mode" toggle, rendered only on climbing locations
    /// (Bonehill/Sennen). A checkbox that flips <c>body.climbing-on</c> and
    /// remembers the choice in localStorage. Default OFF: the CSS hides the
    /// climbing index (verdict pill + Conditions drawer) so the page reads like
    /// a plain weather page. The inline script applies the saved state during
    /// parse (before the tiles) so opted-in readers don't see a flash.
    /// </summary>
    private static string ClimbingModeToggle() => """
        <div class="climbing-toggle">
          <label class="ct-label">
            <input type="checkbox" class="ct-input" aria-label="Climbing mode">
            <span>Climbing mode</span>
          </label>
        </div>
        <script>
        (function () {
          var KEY = 'wb-climbing-mode';
          var on = false;
          try { on = localStorage.getItem(KEY) === 'on'; } catch (e) {}
          function apply(v) { if (document.body) document.body.classList.toggle('climbing-on', v); }
          apply(on);
          function wire() {
            document.querySelectorAll('.ct-input').forEach(function (cb) {
              cb.checked = on;
              cb.addEventListener('change', function () {
                on = cb.checked;
                apply(on);
                document.querySelectorAll('.ct-input').forEach(function (o) { o.checked = on; });
                try { localStorage.setItem(KEY, on ? 'on' : 'off'); } catch (e) {}
              });
            });
          }
          if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire);
          else wire();
        })();
        </script>
        """;

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
