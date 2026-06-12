using System.Text;
using WeatherBlend.Models;

namespace WeatherBlend.Site;

public static partial class SitePages
{
    /// <summary>
    /// Sea-state Models page (<c>models-sea-state.html</c>) — locations with
    /// the <c>sea_state</c> tab only (Sennen today). One bespoke card for the
    /// <c>wave_height_lgb</c> bundle rather than the generic per-lead blender
    /// card: the model is ANY-LEAD v1 (no PerLead test metrics exist), so the
    /// card leads with the bundle's calibration-slice MAE / band coverage and
    /// the training row counts + window from <c>training_metadata.json</c>.
    ///
    /// Below the card sits the buoy verify history (Feature: wave verify
    /// rail) — per-run rows of rolling Hs MAE + empirical 90%-band coverage
    /// per 24h actual-lead bucket from the <c>verify_wave_height_*.json</c>
    /// sidecars. Degrades to a "no runs yet" hint until the first wave verify
    /// lands a sidecar.
    /// </summary>
    public static string RenderModelsSeaState(SiteInputs input)
    {
        var content = new StringBuilder();
        content.Append("<section>");
        content.Append(RenderModelsSubNav("sea-state", input.RenderingFor));
        content.Append("""
              <hgroup>
                <h2>Sea state models</h2>
                <p>Significant wave height — quantile-LGB over the five wave models with a
                   conformal-calibrated (CQR) 90% band. Hs MAE in metres, lower better;
                   band coverage nominal 0.90.</p>
              </hgroup>
            """);

        var card = input.WaveModelCard;
        if (card is null)
        {
            content.Append("<p><em>No wave_height bundle metadata on disk for this location. " +
                           "Run the wave trainer (WeatherProbabilistic train_wave_pi.py), then rclone " +
                           "<code>data/models/wave_height</code> from R2.</em></p>");
            content.Append("</section>");
            return WrapPage(input, "Sea state models", "models", content.ToString());
        }

        content.Append(RenderWaveBlenderCard(card));
        content.Append(RenderWaveVerifyHistory(input, card));

        content.Append("</section>");
        return WrapPage(input, "Sea state models", "models", content.ToString());
    }

    private static string RenderWaveBlenderCard(WaveModelCard card)
    {
        var calRows = new StringBuilder();
        if (card.CalMae.HasValue)
            calRows.Append(Ci, $"""
                <tr>
                  <td>Hs MAE (cal slice, vs {Escape(card.TruthSource)})</td>
                  <td class="num"><strong>{card.CalMae.Value.ToString("0.000", Ci)} m</strong></td>
                </tr>
                """);
        if (card.CalBandCoverage.HasValue)
        {
            var nominal = card.NominalCoverage ?? 0.90;
            calRows.Append(Ci, $"""
                <tr>
                  <td>90% band coverage (cal slice)</td>
                  <td class="num"><strong>{card.CalBandCoverage.Value.ToString("0.000", Ci)}</strong> <small>(nominal {nominal.ToString("0.00", Ci)})</small></td>
                </tr>
                """);
        }
        calRows.Append(Ci, $"""
            <tr>
              <td>Training / val / cal rows</td>
              <td class="num">{card.RowsTrain:n0} / {card.RowsVal:n0} / {card.RowsCal:n0}</td>
            </tr>
            <tr>
              <td>Training window</td>
              <td class="num">{Escape(TrimWindowStamp(card.WindowStart))} → {Escape(TrimWindowStamp(card.WindowEnd))}</td>
            </tr>
            <tr>
              <td>Features</td>
              <td class="num">{card.FeatureCount}</td>
            </tr>
            """);

        // The any-lead note: prefer the bundle's own calibration.json note
        // (it carries the per-lead-calibration ETA) with a static fallback.
        var anyLeadNote = string.IsNullOrWhiteSpace(card.CalNote)
            ? "Any-lead v1 — trained on the lead-unlabelled marine hindcast; one model serves every lead."
            : card.CalNote!;

        return $"""
            <h3>Significant wave height — Sennen marine cell</h3>
            <article class="blender-card">
              <header>
                <hgroup>
                  <h4>Phase {Escape(card.Phase)} <small>· <code>{Escape(card.Version)}</code></small></h4>
                  <p class="skill-line">Quantile-LGB (q05/q50/q95) over the five wave models, split-CQR
                     calibrated to a 90% band. Trained {card.TrainedAtUtc:yyyy-MM-dd} on
                     <code>{Escape(card.TruthSource)}</code> reanalysis (training truth; the Sevenstones
                     buoy is the verification truth below). {Escape(anyLeadNote)}</p>
                </hgroup>
              </header>
              <table>
                <tbody>
            {calRows}    </tbody>
              </table>
            </article>
            """;
    }

    /// <summary>
    /// Per-run buoy verify history for the wave card: one row per
    /// <c>verify_wave_height</c> sidecar, with "MAE / coverage" cells per 24h
    /// actual-lead bucket (column label = bucket upper edge, matching the
    /// sidecar's LeadHours). Custom rather than the shared
    /// <c>RenderVerifyHistorySection</c> because the wave rows carry a second
    /// headline metric (band coverage) the generic table has no column for.
    /// </summary>
    private static string RenderWaveVerifyHistory(SiteInputs input, WaveModelCard card)
    {
        var matchingFiles = input.VerifyHistory
            .Where(f => string.Equals(f.Target, "wave_height", StringComparison.Ordinal))
            .Select(f =>
            {
                var rows = f.Rows.Where(r =>
                        string.Equals(r.Station, input.ActiveLocationName, StringComparison.Ordinal)
                        && string.Equals(r.Phase, card.Phase, StringComparison.Ordinal))
                    .ToList();
                return (File: f, Rows: rows);
            })
            .Where(x => x.Rows.Count > 0)
            .OrderByDescending(x => x.File.AsOfUtc)
            .ToList();

        if (matchingFiles.Count == 0)
        {
            return $"""
                <div class="verify-history">
                  <h5>Buoy verify history <small>(no runs yet)</small></h5>
                  <p class="skill-line">No wave verify rows yet for phase <code>{Escape(card.Phase)}</code>.
                     The wave rail joined the Mon/Thu 09:30 UTC verify cadence — rows appear after its
                     first run scores predictions against the Sevenstones buoy.</p>
                </div>
                """;
        }

        // Bucket columns = union of bucket labels across runs, ascending.
        var allBuckets = matchingFiles
            .SelectMany(x => x.Rows.Select(r => r.LeadHours))
            .Distinct()
            .OrderBy(b => b)
            .ToList();

        var tbody = new StringBuilder();
        foreach (var (file, rows) in matchingFiles)
        {
            // Latest version per bucket (lex-sort works: vYYYY-MM-DD_HHmmss).
            var byBucket = rows
                .GroupBy(r => r.LeadHours)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderByDescending(r => r.ModelVersion, StringComparer.Ordinal).First());

            var versionsScored = byBucket.Values
                .Select(r => r.ModelVersion)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(v => v, StringComparer.Ordinal)
                .ToList();
            var versionCellHtml = string.Join(
                "<br>", versionsScored.Select(v => $"<code>{Escape(v)}</code>"));
            var maxN = byBucket.Values.Max(r => r.N);

            var cells = new StringBuilder();
            foreach (var bucket in allBuckets)
            {
                if (byBucket.TryGetValue(bucket, out var r))
                {
                    var driftCls = r.DriftFlag ? "delta-bad" : "";
                    var covLine = r.BandCoverage.HasValue
                        ? $"<small>cov {r.BandCoverage.Value.ToString("0.00", Ci)}</small>"
                        : "<small>cov —</small>";
                    cells.Append(Ci, $"""<td class="num"><span class="{driftCls}">{r.BlendMetric.ToString("0.000", Ci)}</span><br>{covLine}</td>""");
                }
                else
                {
                    cells.Append("""<td class="num">—</td>""");
                }
            }

            tbody.Append(Ci, $"""
                <tr>
                  <td><time datetime="{file.AsOfUtc:yyyy-MM-dd}">{file.AsOfUtc:ddd yyyy-MM-dd}</time></td>
                  <td><small>{versionCellHtml}</small></td>
                  <td class="num"><span title="Max matched buoy hours per bucket in this run">{maxN}</span></td>
                  {cells}
                </tr>
                """);
        }

        var headers = new StringBuilder();
        foreach (var bucket in allBuckets)
            headers.Append(Ci, $"""<th class="num">≤{bucket}h</th>""");

        return $"""
            <div class="verify-history">
              <h5>Buoy verify history <small>({matchingFiles.Count} run{(matchingFiles.Count == 1 ? "" : "s")})</small></h5>
              <p class="skill-line">Mon + Thu rolling Hs MAE (m) + empirical 90%-band coverage vs the
                 Sevenstones Light Vessel buoy, per 24h actual-lead bucket (≤24h = leads 0-23h, …).
                 The buoy sits ~28 km W of the forecast cell, so absolute MAE carries a location
                 mismatch — read the trend, not the level. Red cells flag drift: band coverage
                 well below the nominal 0.90 (MAE is never a drift trigger here, for the same
                 location-mismatch reason).</p>
              <table>
                <thead>
                  <tr>
                    <th>Run (UTC)</th>
                    <th>Version</th>
                    <th class="num" title="Max matched buoy hours per bucket in this run">N</th>
                    {headers}
                  </tr>
                </thead>
                <tbody>
            {tbody}    </tbody>
              </table>
            </div>
            """;
    }

    /// <summary>"2022-01-01 00:00:00" → "2022-01-01" (the hour part of the
    /// training window adds noise to a card that reads in days/years).</summary>
    private static string TrimWindowStamp(string stamp)
    {
        if (string.IsNullOrWhiteSpace(stamp)) return "—";
        var idx = stamp.IndexOf(' ');
        return idx > 0 ? stamp[..idx] : stamp;
    }
}
