"""
Dry-window start-hour PoC — option #1 from the menu.

Per (target_date, station, N), we already publish a daily P(any N-hour dry
block exists in the 09–18 Europe/London window) from the Phase 3b/3d
blender. We want a *within-day* curve answering "and when in the day is
that block most likely to be?".

This script tests a cheap derivation:

  1. Pull hourly per-NWP precipitation forecasts for the daytime UTC hours
     of the target day, picking the freshest run per (Model, ValidTime).
  2. Compute q_h = fraction of NWPs predicting wet (≥ 0.1 mm/h) at hour h.
     Treat q_h as an ensemble proxy for P(wet@h).
  3. For each candidate start hour s, compute p_s = ∏_{h=s..s+N-1} (1-q_h)
     under hourly independence.
  4. Normalise to a conditional distribution π_s = p_s / Σ p_s — "given a
     block exists today, where is it?".
  5. Compare against 3b's daily P(any block) for the same (station, N,
     lead, date). The calibrated marginal P(block starts at s) is
     π_s × P_3b — magnitude from 3b, shape from the ensemble.
  6. Score against truth: compute the actual hourly wet/dry sequence from
     EA gauge rainfall and flag which start-hours were truly dry-for-N.

Independence is a strong assumption (NWP errors are correlated) so the raw
M = Σ p_s overstates daily P(any block). Calibrating back to 3b's number
side-steps the magnitude bias and only trusts the *shape* of the curve.

Usage: python scripts/DryWindowStartHour/start_hour_v0.py
"""

import io
import sys
from datetime import date, datetime, timedelta

import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---- Knobs ----
STATION_SLUG = "ea_bellever_dartmoor"
STATION_NAME = "Bellever Dartmoor"
LOCATION_NAME = "bonehill_rocks"
N = 6                       # dry-block length (hours)
LEAD = 24                   # lead bucket from the 3b blender
WET_THRESHOLD_MM = 0.1
DAYTIME_LOCAL_START = 9
DAYTIME_LOCAL_END = 18

FCAST_GLOB     = "data/forecasts/**/*.parquet"
DRY_WINDOW_GLOB = "data/predictions/dry_window/**/*.parquet"
RAINFALL_GLOB  = "data/truth/rainfall/**/*.parquet"

# Constituent NWPs — same set as the precip blender (3a/3b/3c). Keeping the
# set explicit so the q_h ensemble matches what 3b sees, modulo per-row
# missingness handling.
NWPS = [
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless",
    "meteofrance_seamless", "ukmo_seamless", "gem_seamless",
    "ecmwf_aifs025_single", "jma_seamless",
]


def _last_sunday(year: int, month: int) -> date:
    """Last Sunday of (year, month) — used for the BST/GMT switchover."""
    d = date(year, month, 31)
    while d.month == month and d.weekday() != 6:
        d -= timedelta(days=1)
    # If we walked off the month boundary, scan from the start.
    if d.month != month:
        d = date(year, month, 1)
        while d.weekday() != 6:
            d += timedelta(days=1)
        # Walk to the last one.
        while True:
            nxt = d + timedelta(days=7)
            if nxt.month != month:
                return d
            d = nxt
    return d


def is_bst(d: date) -> bool:
    """True if `d` falls inside Europe/London BST. BST runs from the last
    Sunday of March 01:00 UTC to the last Sunday of October 01:00 UTC.
    Day-resolution check is good enough — we never resolve at the exact
    01:00 transition hour."""
    start = _last_sunday(d.year, 3)
    end = _last_sunday(d.year, 10)
    return start <= d < end


def daytime_utc_range(d: date) -> tuple[int, int]:
    """Map (date, local 9–18 Europe/London) to a [start, end) UTC hour
    range on that UTC day. BST → 8–17, GMT → 9–18."""
    offset = 1 if is_bst(d) else 0
    return DAYTIME_LOCAL_START - offset, DAYTIME_LOCAL_END - offset


def hourly_q(con: duckdb.DuckDBPyConnection, target_date: date,
             lead: int) -> dict[int, float]:
    """
    For each daytime UTC hour, return q_h = mean across NWPs of
    (precip ≥ 0.1 mm/h). One row per (Model, ValidTime) — the freshest run
    whose RunTime is at most LEAD+12h ago and at most LEAD-12h ago, i.e.
    the run that lands in the [LEAD-12, LEAD+12] lead bucket. Loose
    bracketing because exact-lead-only would drop too many models.
    """
    s_utc, e_utc = daytime_utc_range(target_date)
    day_start = datetime(target_date.year, target_date.month, target_date.day)
    day_end = day_start + timedelta(days=1)
    lead_lo = lead - 12
    lead_hi = lead + 12

    sql = f"""
WITH src AS (
  SELECT Model, RunTimeUtc, ValidTimeUtc, Precipitation,
         row_number() OVER (
           PARTITION BY Model, ValidTimeUtc
           ORDER BY RunTimeUtc DESC
         ) AS rn
  FROM read_parquet('{FCAST_GLOB}', hive_partitioning=false, union_by_name=true)
  WHERE LocationName = '{LOCATION_NAME}'
    AND Model IN ({", ".join("'" + m + "'" for m in NWPS)})
    AND ValidTimeUtc >= TIMESTAMP '{day_start:%Y-%m-%d %H:%M:%S}'
    AND ValidTimeUtc <  TIMESTAMP '{day_end:%Y-%m-%d %H:%M:%S}'
    AND Precipitation IS NOT NULL
    AND date_diff('hour', RunTimeUtc, ValidTimeUtc) BETWEEN {lead_lo} AND {lead_hi}
)
SELECT extract(hour from ValidTimeUtc) AS h,
       avg(CASE WHEN Precipitation >= {WET_THRESHOLD_MM} THEN 1.0 ELSE 0.0 END) AS q
FROM src WHERE rn = 1
GROUP BY 1 ORDER BY 1
"""
    rows = con.execute(sql).fetchall()
    return {int(h): float(q) for h, q in rows if s_utc <= int(h) < e_utc}


def daily_p_any_3b(con: duckdb.DuckDBPyConnection, target_date: date,
                   lead: int) -> float | None:
    """Latest 3b/3d champion P(any block) for (station, N, lead, date)."""
    sql = f"""
SELECT ProbHasDryWindow
FROM read_parquet('{DRY_WINDOW_GLOB}', hive_partitioning=false, union_by_name=true)
WHERE LocationName = '{LOCATION_NAME}'
  AND TruthStation = '{STATION_SLUG}'
  AND WindowHours = {N}
  AND LeadHours = {lead}
  AND TargetDateUtc = TIMESTAMP '{target_date:%Y-%m-%d 00:00:00}'
ORDER BY PredictionMadeAtUtc DESC
LIMIT 1
"""
    rows = con.execute(sql).fetchall()
    return float(rows[0][0]) if rows else None


def truth_dry_pattern(con: duckdb.DuckDBPyConnection,
                      target_date: date) -> dict[int, bool] | None:
    """For each daytime UTC hour, was that hour dry (< 0.1 mm) per the EA
    gauge? Same 4-of-4 rule as PrecipVerify so a partial hour drops the
    day. Returns None if any daytime hour is missing."""
    s_utc, e_utc = daytime_utc_range(target_date)
    day_start = datetime(target_date.year, target_date.month, target_date.day)
    day_end = day_start + timedelta(days=1)
    sql = f"""
SELECT extract(hour from valid_time) AS h, mm_hour FROM (
  SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
         SUM(Value15MinMm) AS mm_hour
  FROM read_parquet('{RAINFALL_GLOB}', hive_partitioning=false, union_by_name=true)
  WHERE LocationName = '{LOCATION_NAME}'
    AND StationName  = '{STATION_NAME}'
    AND Value15MinMm IS NOT NULL
    AND ObservedTimeUtc >= TIMESTAMP '{day_start:%Y-%m-%d %H:%M:%S}'
    AND ObservedTimeUtc <  TIMESTAMP '{day_end:%Y-%m-%d %H:%M:%S}'
  GROUP BY 1
  HAVING COUNT(*) = 4
) ORDER BY 1
"""
    rows = {int(h): float(mm) for h, mm in con.execute(sql).fetchall()}
    pattern = {}
    for h in range(s_utc, e_utc):
        if h not in rows:
            return None
        pattern[h] = rows[h] < WET_THRESHOLD_MM
    return pattern


def candidate_starts(s_utc: int, e_utc: int, n: int) -> list[int]:
    """Start hours s such that [s, s+n) ⊆ [s_utc, e_utc)."""
    return list(range(s_utc, e_utc - n + 1))


def block_probs(q: dict[int, float], starts: list[int],
                n: int) -> dict[int, float]:
    """p_s = ∏_{h=s..s+n-1} (1 - q_h). Missing hours treated as q=0
    (defensive — caller should ensure full coverage)."""
    out = {}
    for s in starts:
        p = 1.0
        for h in range(s, s + n):
            p *= 1.0 - q.get(h, 0.0)
        out[s] = p
    return out


def truth_starts(pattern: dict[int, bool], starts: list[int],
                 n: int) -> set[int]:
    """Start hours where every hour in [s, s+n) was dry per truth."""
    out = set()
    for s in starts:
        if all(pattern.get(h, False) for h in range(s, s + n)):
            out.add(s)
    return out


def main() -> None:
    con = duckdb.connect()

    # Probe what target_dates have a 3b prediction at lead = LEAD.
    sql = f"""
SELECT DISTINCT TargetDateUtc
FROM read_parquet('{DRY_WINDOW_GLOB}', hive_partitioning=false, union_by_name=true)
WHERE LocationName = '{LOCATION_NAME}'
  AND TruthStation = '{STATION_SLUG}'
  AND WindowHours = {N}
  AND LeadHours = {LEAD}
ORDER BY TargetDateUtc
"""
    candidate_dates = [r[0].date() for r in con.execute(sql).fetchall()]
    print(f"Found {len(candidate_dates)} target dates with "
          f"({STATION_SLUG}, N={N}, lead={LEAD}h) 3b predictions.")
    print(f"Range: {candidate_dates[0]} → {candidate_dates[-1]}")
    print()

    for d in candidate_dates:
        s_utc, e_utc = daytime_utc_range(d)
        starts = candidate_starts(s_utc, e_utc, N)

        q = hourly_q(con, d, LEAD)
        if len(q) < (e_utc - s_utc):
            covered = sorted(q.keys())
            print(f"{d} {s_utc:02d}-{e_utc:02d}Z  "
                  f"sparse hourly forecast ({len(q)}/{e_utc - s_utc} hours: {covered})")
            print()
            continue

        p3b = daily_p_any_3b(con, d, LEAD)
        truth = truth_dry_pattern(con, d)

        ps = block_probs(q, starts, N)
        m_raw = sum(ps.values())  # upper bound on P(any) under independence
        total = m_raw if m_raw > 0 else 1.0
        pi = {s: ps[s] / total for s in starts}
        cal = {s: pi[s] * p3b for s in starts} if p3b is not None else None

        # ---- Pretty rows ---------------------------------------------------
        hours = list(range(s_utc, e_utc))
        hour_axis = " ".join(f"{h:02d}" for h in hours)
        q_row     = " ".join(f"{int(round(q[h] * 100)):>2d}" for h in hours)
        truth_row = (" ".join(("D " if truth[h] else "W ")[0:2] for h in hours)
                     if truth is not None else "(incomplete truth)")
        cal_row = (" ".join(f"{s:02d}={cal[s]:>4.0%}" for s in starts)
                   if cal is not None else "(no 3b)")
        if truth is not None:
            t_starts = truth_starts(truth, starts, N)
            t_marks = " ".join(("✓ " if s in t_starts else "· ")[0:2] for s in starts)
        else:
            t_marks = ""

        p3b_str = f"{p3b:>4.0%}" if p3b is not None else "  — "
        print(f"{d}  3b P(any)={p3b_str}  M_raw={m_raw:>5.2f}")
        print(f"  hour:   {hour_axis}")
        print(f"  q% wet: {q_row}")
        print(f"  truth:  {truth_row}")
        print(f"  P(start@s): {cal_row}")
        if t_marks:
            print(f"  truth dry@s: {t_marks}")
        print()


if __name__ == "__main__":
    main()
