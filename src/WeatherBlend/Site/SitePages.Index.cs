using System.Text;

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
    /// <c>[<see cref="HomeFirstVisibleHourUtc"/>, <see cref="HomeLastVisibleHourUtcExclusive"/>)</c>
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

        // Champion-PHASE filter, per lead — see ChampionMatcher for the
        // full rationale (per-lead pins, phase-over-version matching after
        // a retrain, test-fixture version-equality fallback). Empty
        // CurrentVersion + no per-lead pins = no manifest, fall back to
        // "any" so a freshly-deployed environment still renders cards.
        var champion = new ChampionMatcher(input);
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => champion.MatchesChampionPhase(p.ModelVersion, p.LeadHours));

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
        // freshest made on ties, PER FACE, then collapsed to the WORST face
        // per hour (smallest condensation margin — the conservative summary a
        // one-chip tile can carry; the temp tab charts every face). Whole-crag
        // locations have a single empty face, so this is the old behaviour.
        var rockByValid = CollapseRockToWorstFace(input.RockSurfacePredictions);

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
                    conditions = ClimbingConditions.Evaluate(
                        p.ValidTimeUtc, input.Latitude, input.Longitude,
                        p.BlendTemperature, pWet, windMph, rk);
                }
                tiles.Append(RenderHourTile(p, feelsLikeByValid, pwetByValid, lowCloudByValid, rockByValid,
                    seaSpec, waveByValid, windSpeedMsByValid, windDirDegByValid,
                    input.WindGustByValidMs, conditions, popoverId++));
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

    /// <summary>Legacy default UTC hour at which the home tile grid starts.
    /// Phase D made this per-location — see
    /// <c>LocationConfig.Overview.FirstVisibleHourUtc</c>. Retained here for
    /// test fixtures / callers that don't set <c>SiteInputs.RenderingFor</c>.
    /// Bonehill's config.yaml carries 5 (matches the pre-Phase-D constant);
    /// Membury wants the full 24h day so its config carries 0.</summary>
    public const int HomeFirstVisibleHourUtc = 5;

    /// <summary>Legacy default UTC hour at which the home tile grid ends
    /// (exclusive). Phase D made this per-location — see
    /// <c>LocationConfig.Overview.LastVisibleHourUtcExclusive</c>. Bonehill 20,
    /// Membury 24.</summary>
    public const int HomeLastVisibleHourUtcExclusive = 20;

    /// <summary>
    /// Phase D — resolve the Overview tile-grid hour window from the per-loc
    /// descriptor when <see cref="SiteInputs.RenderingFor"/> is set; fall back
    /// to the legacy constants for test fixtures + pre-Phase-D callers.
    /// </summary>
    private static (int FirstHour, int LastHourExcl) OverviewWindow(SiteInputs input)
    {
        if (input.RenderingFor is { } loc)
            return (loc.OverviewFirstVisibleHourUtc, loc.OverviewLastVisibleHourUtcExclusive);
        return (HomeFirstVisibleHourUtc, HomeLastVisibleHourUtcExclusive);
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
        var version = DryWindowPhases.Phase3p.StartHourCurveVersion;
        if (string.IsNullOrEmpty(version)) return "";

        // Freshest forecast per (window, start) for this target day: smallest
        // lead, newest made — mirrors the overview tiles' smallest-lead pick.
        var rows = input.StartHourPredictions
            .Where(s => s.Station == station
                        && string.Equals(s.Version, version, StringComparison.Ordinal)
                        && s.TargetDateUtc.Date == dayUtc.Date)
            .LatestPerValid(s => (s.WindowHours, s.StartHourUtc), s => s.LeadHours, s => s.PredictedAtUtc)
            .ToList();
        if (rows.Count == 0) return "";

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
        var cardSource = string.IsNullOrEmpty(input.CurrentVersion) && input.ChampionByLead.Count == 0
            ? input.Predictions
            : input.Predictions.Where(p => champion.MatchesChampionPhase(p.ModelVersion, p.LeadHours));

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

    /// <summary>CSS severity class for a badge pill — unified amber/red rule
    /// (one trigger amber, two-plus red). Callers only invoke this for
    /// severities that render, so None maps to empty.</summary>
    private static string BadgeSeverityClass(BadgeSeverity severity) => severity switch
    {
        BadgeSeverity.Red => "badge-red",
        BadgeSeverity.Amber => "badge-amber",
        _ => "",
    };

    /// <summary>Tile-badge lookup for rock surface rows: dedup smallest-lead /
    /// freshest-made PER (valid, face), then keep the WORST face per valid —
    /// the smallest condensation margin, i.e. the wall closest to sweating.
    /// Single-face locations (Bonehill, empty face) reduce to the original
    /// one-row-per-valid behaviour. Internal for tests.</summary>
    internal static Dictionary<DateTime, RockSurfaceForecastPoint> CollapseRockToWorstFace(
        IReadOnlyList<RockSurfaceForecastPoint> rows)
        => rows
            .GroupBy(r => r.Face)
            .SelectMany(g => g.LatestPerValid(r => r.ValidTimeUtc, r => r.LeadHours, r => r.PredictedAtUtc))
            .GroupBy(r => r.ValidTimeUtc)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(r => r.CondensationMarginC)
                      .ThenBy(r => r.Face, StringComparer.Ordinal) // deterministic on margin ties
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
        int popoverId)
    {
        // Low-cloud / mist warning — amber when ONE signal hits its
        // threshold, red when BOTH do (unified badge severity, 2026-06-12).
        // Sits at the top of the tile as a Pico <details>/<summary>
        // pop-out so the trigger details work on touch devices (the earlier
        // title="..." tooltip was unreachable from mobile).
        string lowCloudBadge = "";
        if (lowCloudByValid.TryGetValue(p.ValidTimeUtc, out var lc))
        {
            var visFired = lc.VisFiredCount >= LowCloudVisFireThreshold;
            var cbFired  = lc.CloudBaseFiredCount >= LowCloudBaseFireThreshold;
            var severity = LowCloudBadge.Evaluate(visFired, cbFired);
            if (severity != BadgeSeverity.None)
            {
                var rows = new StringBuilder();
                if (visFired)
                    rows.Append(Ci, $"<li>{lc.VisFiredCount}/{lc.VisTotalCount} NWPs: mist (vis &lt; 1 km)</li>");
                if (cbFired)
                    rows.Append(Ci, $"<li>{lc.CloudBaseFiredCount}/{lc.CloudBaseTotalCount} NWPs: cloud base below tor (T−Td &lt; 1.5°C)</li>");
                lowCloudBadge = $"""
                    <details class="badge-pop low-cloud-pop">
                      <summary class="tile-badge low-cloud-badge {BadgeSeverityClass(severity)}">☁ low cloud</summary>
                      <ul>{rows}</ul>
                    </details>
                    """;
            }
        }

        // Rock surface / condensation badge (Phase P1). Fires when the rock is
        // at or near dew point: red "rock wet" for condensation (margin ≤ 0),
        // amber "rock greasy?" for the marginal band (0 < margin ≤ greasyMargin,
        // already encoded in GreasinessStatus). No badge when dry. Same ±1h
        // tolerance + Pico <details> pop-out idiom as the low-cloud badge.
        // Trigger logic untouched by the 2026-06-12 badge unification — its
        // semantics were already two-tier; only the styling moved onto the
        // shared badge-amber / badge-red classes.
        string rockBadge = "";
        if (TryNearest(rockByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var rk)
            && rk!.GreasinessStatus != Predict.Surface.RockSurfacePhysics.StatusDry)
        {
            var isCond = rk.GreasinessStatus == Predict.Surface.RockSurfacePhysics.StatusCondensation;
            var severityCls = BadgeSeverityClass(isCond ? BadgeSeverity.Red : BadgeSeverity.Amber);
            var cls = $"tile-badge rock-badge {(isCond ? "rock-wet" : "rock-greasy")} {severityCls}";
            var label = isCond ? "rock wet" : "rock greasy?";
            // Cliff-face mode: rk is the WORST face this hour — name it so the
            // pop-out says which wall is sweating (other faces may be drier;
            // the temp tab chart shows all of them).
            var subject = rk.Face.Length == 0 ? "Rock" : $"{char.ToUpperInvariant(rk.Face[0])}{rk.Face[1..]} face";
            rockBadge = string.Create(Ci, $"""
                <details class="badge-pop rock-pop">
                  <summary class="{cls}">&#x26A0;&#xFE0E; {label}</summary>
                  <ul><li>{subject} {rk.RockSurfaceTempC:0.0}°C vs dew point {rk.DewPointC:0.0}°C — margin {rk.CondensationMarginC:+0.0;-0.0;0.0}°C</li></ul>
                </details>
                """);
        }

        // Sea-state badge (marine locations with a seaStateBadge config
        // block — Sennen). Tide / run-up / onshore-wind triggers, amber for
        // one fired, red for two-plus; pop-out lists each FIRED trigger with
        // its live values. Wave + wind lookups tolerate ±1h like the other
        // tile chips. A missing input skips its trigger inside the evaluator
        // (e.g. no wind_mvn direction row yet → tide + waves only) — missing
        // data never fires a trigger.
        string seaBadge = "";
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
            {
                var rows = new StringBuilder();
                foreach (var trigger in sea.FiredTriggers)
                    rows.Append(Ci, $"<li>{Escape(trigger)}</li>");
                seaBadge = $"""
                    <details class="badge-pop sea-pop">
                      <summary class="tile-badge sea-badge {BadgeSeverityClass(sea.Severity)}">🌊 sea state</summary>
                      <ul>{rows}</ul>
                    </details>
                    """;
            }
        }

        // Feels-like and P(wet) tolerate ±1h drift — predict cycles can land
        // an hour off, and we'd rather show a chip than a gap.
        string feelsCell = "";
        if (TryNearest(feelsLikeByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var fl))
        {
            var apparentColor = TemperatureColor(fl!.ApparentC);
            var utciColor = TemperatureColor(fl.UtciC);

            // UTCI pop-out: only emit the toggle when at least one element
            // value is present (older parquets pre-dating the element-input
            // persistence have all five null and the panel would be empty).
            string toggle = "";
            // Temperature deliberately excluded from the gate too — only show
            // the toggle when at least one of the four "why is UTCI this value"
            // element fields is present.
            if (fl.RelativeHumidityPct is not null
                || fl.WindSpeed10mMs is not null || fl.ShortwaveDownWm2 is not null
                || fl.CloudCoverPct is not null)
            {
                // Temperature row deliberately omitted — it's already the
                // headline value on the tile, repeating it in the pop-out is
                // noise. The other four are the "why is UTCI this value?"
                // story the reader can't get elsewhere.
                var rows = new StringBuilder();
                if (fl.RelativeHumidityPct is double rh)
                    rows.Append(Ci, $"<tr><td>Humidity</td><td class=\"num\">{rh:0} %</td></tr>");
                if (fl.WindSpeed10mMs is double ws)
                {
                    // Stored as m/s; display as mph (× 2.23694) — site-wide
                    // unit decision 2026-05-28. Gust appended inline when
                    // materially above wind (gust mph > wind mph + 1) so
                    // calm-wind rows don't sprout a redundant
                    // "Wind 3, gust 3" suffix.
                    var windMph = ws * 2.23694;
                    string gustSuffix = "";
                    if (windGustByValidMs.TryGetValue(p.ValidTimeUtc, out var gustMs))
                    {
                        var gustMph = gustMs * 2.23694;
                        if (gustMph > windMph + 1.0)
                            gustSuffix = string.Create(Ci, $" · gust {gustMph:0.0} mph");
                    }
                    rows.Append(Ci, $"<tr><td>Wind 10 m</td><td class=\"num\">{windMph:0.0} mph{gustSuffix}</td></tr>");
                }
                if (fl.ShortwaveDownWm2 is double sw)
                    rows.Append(Ci, $"<tr><td>Shortwave down</td><td class=\"num\">{sw:0} W/m²</td></tr>");
                if (fl.CloudCoverPct is double cc)
                    rows.Append(Ci, $"<tr><td>Cloud cover</td><td class=\"num\">{cc:0} %</td></tr>");
                toggle = $"""
                    <details class="utci-pop">
                      <summary title="Element values that fed UTCI for this hour">ⓘ</summary>
                      <table class="utci-pop-table">{rows}</table>
                    </details>
                    """;
            }
            // Band label moved to its own line under the UTCI value (and
            // wrapped in double-quotes) — the ⓘ toggle stays inline with
            // the UTCI value so the click target is always next to the
            // number it explains. Splits the previous one-liner into two
            // <div>s.
            feelsCell = string.Create(Ci, $"""
                <div class="feels">
                  <div>Feels like <strong style="color: {apparentColor}">{fl.ApparentC:0.0}°C</strong></div>
                  <div>UTCI <strong style="color: {utciColor}">{fl.UtciC:0.0}°C</strong> {toggle}</div>
                  <div class="utci-band"><small>"{Escape(PrettyUtciBand(fl.Band))}"</small></div>
                </div>
                """);
        }

        string pwetCell = "";
        if (TryNearest(pwetByValid, p.ValidTimeUtc, TimeSpan.FromHours(1), out var pw))
        {
            // ☔ when the displayed P(wet) rounds to ≥ 25%
            var rain = pw!.ProbWet >= 0.245 ? " <span class=\"rain\">&#x2614;</span>" : "";
            var pwColor = PrecipProbColor(pw.ProbWet);
            pwetCell = string.Create(Ci, $"<div class=\"pwet\">P(wet) <strong style=\"color: {pwColor}\">{(pw.ProbWet * 100):0}%</strong>{rain}</div>");
        }

        // Badges sit in their own stacked block directly under the time at the
        // TOP of the tile — vertical column, left-aligned, one pill per line —
        // rather than as flex children of the <header> row (which laid the
        // time + both pills out left-to-right and overflowed onto the next
        // tile once both fired). ALWAYS emitted (even empty): the block carries
        // a fixed reserved min-height in CSS so the temperature — the tile's
        // main value — lines up across every tile in the grid regardless of how
        // many badges (0/1/2) fired. Badge-free tiles show that reserved space
        // as a small blank gap, by design (user call 2026-06-06).
        string badgeBlock = $"""<div class="tile-badges">{lowCloudBadge}{rockBadge}{seaBadge}</div>""";

        // Climbing-conditions strip (idea #1) — the tile headline when present.
        // Coloured by tier; the reason (limiting factor, or the gate that
        // fired) is always shown, with the full per-factor breakdown in a
        // pop-out so the verdict is never a bare unexplained colour.
        string conditionsStrip = "";
        if (conditions is { } c)
        {
            var factorRows = new StringBuilder();
            foreach (var f in c.Factors)
                factorRows.Append(Ci, $"<tr><td>{Escape(f.Name)}</td><td class=\"num\">{(f.Score * 100):0}</td><td>{Escape(f.Detail)}</td></tr>");
            var breakdown = c.Factors.Count > 0
                ? $"""
                    <details class="badge-pop conditions-pop">
                      <summary class="conditions-why">why?</summary>
                      <table class="conditions-table"><thead><tr><th>factor</th><th>/100</th><th></th></tr></thead>{factorRows}</table>
                    </details>
                    """
                : "";
            conditionsStrip = string.Create(Ci, $"""
                <div class="conditions-strip" style="--cond-color: {c.TierColor}">
                  <span class="conditions-tier">{Escape(c.TierLabel)}</span>
                  <span class="conditions-reason">{Escape(c.Reason)}</span>
                  {breakdown}
                </div>
                """);
        }

        var tempColor = TemperatureColor(p.BlendTemperature);
        return string.Create(Ci, $"""
            <article class="forecast-card">
              <header><h4>{p.ValidTimeUtc:HH:mm}Z</h4></header>
              {conditionsStrip}
              {badgeBlock}
              <div class="temp" style="--temp-color: {tempColor}">{p.BlendTemperature:0.0}°C</div>
              {feelsCell}
              {pwetCell}
              <footer><small>+{p.LeadHours}h · made {p.PredictionMadeAtUtc:MM-dd HH:mm}Z</small></footer>
            </article>
            """);
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
