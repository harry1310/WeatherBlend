namespace WeatherBlend.Train.Common;

/// <summary>
/// Lead-0 "nowcast" sourcing rules, shared by the temperature and precipitation
/// feature builders so train and (later) predict agree on exactly one set of
/// conventions.
///
/// A 0h-lead model answers "what's happening now / in the next hour", so it is
/// trained on Open-Meteo's <b>historical-forecast</b> archive — best-available
/// per valid-time, i.e. ≈analysis quality at lead 0 (RunTimeSource=hist_forecast)
/// — rather than the <c>offset_day</c> Previous Runs tree, whose shortest lead is
/// 24h. The surface fields for this come from the same widened hist-forecast
/// backfill that lands them on disk; upper-air (…hPa) for the oro phase at lead 0
/// also comes from those hist_forecast rows (not the lead-matched 'exact' tree,
/// which has nothing at lead 0).
///
/// <para><b>Persistence anti-leak.</b> The EA rainfall persistence window is
/// <c>(runTime − N, runTime]</c> — runTime INCLUSIVE. At lead 0, run-time =
/// valid-time, so anchoring at valid would fold the predicted hour's own gauge
/// reading into the features. We therefore anchor persistence at
/// <c>valid − max(lead, <see cref="MinPersistenceLagHours"/>)</c>: a no-op for
/// the ≥24h leads, and a strict back-shift at lead 0. The lag is set
/// conservatively to the worst-case gauge availability a LIVE nowcast run would
/// face, so training never uses fresher persistence than predict can — erring
/// large only under-uses recent rain (suboptimal), erring small would leak.</para>
/// </summary>
public static class NowcastSource
{
    /// <summary>The single lead value that denotes a nowcast model.</summary>
    public const int LeadHours = 0;

    /// <summary>Minimum hours before valid-time to anchor rainfall persistence
    /// at lead 0 (see class remarks). Conservative w.r.t. live EA gauge latency;
    /// revisit once predict-time gauge freshness is measured.</summary>
    public const double MinPersistenceLagHours = 3.0;

    public static bool IsNowcast(int leadHours) => leadHours == LeadHours;

    /// <summary>Forecast-tree <c>RunTimeSource</c> the surface feature pivot
    /// reads for this lead: <c>hist_forecast</c> at lead 0, else
    /// <c>offset_day</c>.</summary>
    public static string SurfaceRunTimeSource(int leadHours)
        => IsNowcast(leadHours) ? "hist_forecast" : "offset_day";

    /// <summary>Forecast-tree <c>RunTimeSource</c> the upper-air pivot reads for
    /// this lead: <c>hist_forecast</c> at lead 0 (the hist rows carry …hPa), else
    /// the lead-matched <c>exact</c> tree.</summary>
    public static string UpperAirRunTimeSource(int leadHours)
        => IsNowcast(leadHours) ? "hist_forecast" : "exact";

    /// <summary>Hours before valid-time to anchor rainfall persistence for this
    /// lead. Identity for ≥24h leads; back-shifts lead 0 off the predicted hour.</summary>
    public static double PersistenceAnchorHours(int leadHours)
        => System.Math.Max(leadHours, MinPersistenceLagHours);
}
