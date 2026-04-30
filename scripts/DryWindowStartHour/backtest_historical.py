"""
Historical shape-skill backtest for the start-hour curve (option #1:
ensemble-fraction q_h).

Idea recap: per UTC daytime hour h, q_h = fraction of NWPs predicting
≥ 0.1 mm/h precip; per candidate start s, p_s = ∏(1 - q_{s..s+N-1});
π_s = p_s / Σp_s is the curve's *shape* over start hours, conditional on
"a dry block exists today". We want to know if that shape lines up with
the start hours that were actually dry-for-N in EA truth, on the days
where the question is non-trivial.

Score only on **informative** days: those with ≥ 1 valid truth start AND
< all-starts valid (i.e. some 6h windows were dry, others weren't). Days
with no dry block at all, or where the entire daytime was dry, carry no
shape signal — the curve has nothing to peak at.

Span: 2025-02-01 → 2026-04-15 (8-NWP era, before the live window).

Metrics:
  - top-1 accuracy   : 1 if argmax_s π_s ∈ truth_starts else 0
  - top-1 vs uniform : how often top-1 beats picking a uniform random start
  - Brier            : Σ_s (π_s - τ_s)² where τ_s = 1/|truth| if s ∈ truth else 0
  - log-loss         : -Σ τ_s log(π_s + ε) — vs uniform-π baseline.
"""

import math
import sys
from datetime import date, datetime, timedelta

import duckdb

# Reuse helpers from v0
sys.path.insert(0, "scripts/DryWindowStartHour")
from start_hour_v0 import (  # noqa: E402
    NWPS, WET_THRESHOLD_MM, FCAST_GLOB, RAINFALL_GLOB,
    LOCATION_NAME, candidate_starts, daytime_utc_range,
    block_probs, truth_starts, truth_dry_pattern,
)

STATION_NAME = "Bellever Dartmoor"
N = 6
LEAD = 24
START_DATE = date(2025, 2, 1)
END_DATE = date(2026, 4, 15)
EPS = 1e-6


REPLAY_GLOB = "data/predictions/precipitation_replay/ea_bellever_dartmoor/**/lead_24h.parquet"


def preload_q_3a(con: duckdb.DuckDBPyConnection) -> dict[date, dict[int, float]]:
    """Phase 3a champion's per-row P(wet) from the replay parquet — that's
    option 2: the actual trained model's hourly probability for the whole
    span. Lead pinned to 24h for now (matches the backtest target)."""
    sql = f"""
SELECT CAST(date_trunc('day', ValidTimeUtc) AS DATE) AS d,
       CAST(extract(hour from ValidTimeUtc) AS INTEGER) AS h,
       ProbWet
FROM read_parquet('{REPLAY_GLOB}', hive_partitioning=false, union_by_name=true)
WHERE ValidTimeUtc >= TIMESTAMP '{START_DATE:%Y-%m-%d 00:00:00}'
  AND ValidTimeUtc <  TIMESTAMP '{END_DATE:%Y-%m-%d 00:00:00}'
"""
    out: dict[date, dict[int, float]] = {}
    for d, h, p in con.execute(sql).fetchall():
        out.setdefault(d, {})[int(h)] = float(p)
    return out


def preload_q_tables(con: duckdb.DuckDBPyConnection, lead: int
                    ) -> tuple[dict[date, dict[int, float]],
                               dict[date, dict[int, float]]]:
    """Single SQL pass over the whole span. Returns two q tables per
    (date, hour):

      q_binary   : mean across NWPs of (precip ≥ 0.1 mm/h) — option 1
      q_smooth   : mean across NWPs of clamp(precip / 0.5, 0, 1) — a
                   "magnitude-aware" proxy that rewards confident-wet
                   forecasts (≥ 0.5 mm/h) more than borderline drizzle.
                   Cheaper stand-in for full Phase 3a P(wet) hourly.

    Both use the same per-(Model, ValidTime) latest-run-in-bracket pick.
    """
    lead_lo = lead - 12
    lead_hi = lead + 12
    sql = f"""
WITH src AS (
  SELECT Model,
         CAST(date_trunc('day', ValidTimeUtc) AS DATE) AS d,
         CAST(extract(hour from ValidTimeUtc) AS INTEGER) AS h,
         Precipitation,
         row_number() OVER (
           PARTITION BY Model, ValidTimeUtc
           ORDER BY RunTimeUtc DESC
         ) AS rn
  FROM read_parquet('{FCAST_GLOB}', hive_partitioning=false, union_by_name=true)
  WHERE LocationName = '{LOCATION_NAME}'
    AND Model IN ({", ".join("'" + m + "'" for m in NWPS)})
    AND ValidTimeUtc >= TIMESTAMP '{START_DATE:%Y-%m-%d 00:00:00}'
    AND ValidTimeUtc <  TIMESTAMP '{END_DATE:%Y-%m-%d 00:00:00}'
    AND Precipitation IS NOT NULL
    AND date_diff('hour', RunTimeUtc, ValidTimeUtc) BETWEEN {lead_lo} AND {lead_hi}
)
SELECT d, h,
       avg(CASE WHEN Precipitation >= {WET_THRESHOLD_MM} THEN 1.0 ELSE 0.0 END) AS q_bin,
       avg(LEAST(GREATEST(Precipitation / 0.5, 0.0), 1.0)) AS q_smooth
FROM src WHERE rn = 1
GROUP BY 1, 2
"""
    q_bin: dict[date, dict[int, float]] = {}
    q_sm:  dict[date, dict[int, float]] = {}
    for d, h, qb, qs in con.execute(sql).fetchall():
        q_bin.setdefault(d, {})[int(h)] = float(qb)
        q_sm.setdefault(d, {})[int(h)] = float(qs)
    return q_bin, q_sm


def main() -> None:
    con = duckdb.connect()

    # Pre-load truth for the whole span in one shot — avoids per-day SQL.
    sql = f"""
SELECT date_trunc('day', valid_time)::DATE AS d,
       extract(hour from valid_time) AS h,
       mm
FROM (
  SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
         SUM(Value15MinMm) AS mm
  FROM read_parquet('{RAINFALL_GLOB}', hive_partitioning=false, union_by_name=true)
  WHERE LocationName = '{LOCATION_NAME}'
    AND StationName = '{STATION_NAME}'
    AND Value15MinMm IS NOT NULL
    AND ObservedTimeUtc >= TIMESTAMP '{START_DATE:%Y-%m-%d 00:00:00}'
    AND ObservedTimeUtc <  TIMESTAMP '{END_DATE:%Y-%m-%d 00:00:00}'
  GROUP BY 1
  HAVING COUNT(*) = 4
)
"""
    truth_by_date: dict[date, dict[int, bool]] = {}
    for d, h, mm in con.execute(sql).fetchall():
        truth_by_date.setdefault(d, {})[int(h)] = float(mm) < WET_THRESHOLD_MM

    print("Preloading hourly q tables from forecasts tree...", flush=True)
    q_bin_by_date, q_sm_by_date = preload_q_tables(con, LEAD)
    print(f"  ensemble q tables cover {len(q_bin_by_date)} dates.", flush=True)
    print("Preloading Phase 3a replay P(wet)...", flush=True)
    q_3a_by_date = preload_q_3a(con)
    print(f"  Phase 3a replay covers {len(q_3a_by_date)} dates.", flush=True)

    def score_variant(name: str, q_by_date: dict[date, dict[int, float]],
                      collect_examples: bool = False):
        """Walk every informative day, score π built from this q table.
        Returns (n_total, n_complete, n_informative, n_top1, brier_sum,
                 ll_sum, ll_uniform_sum, examples)."""
        n_total = n_complete = n_inf = n_top1 = 0
        sum_brier = sum_ll = sum_ll_uniform = 0.0
        examples: list[tuple[date, dict[int, float], set[int]]] = []

        d = START_DATE
        while d < END_DATE:
            n_total += 1
            s_utc, e_utc = daytime_utc_range(d)
            starts = candidate_starts(s_utc, e_utc, N)

            truth_pattern = truth_by_date.get(d)
            if truth_pattern is None or not all(h in truth_pattern for h in range(s_utc, e_utc)):
                d += timedelta(days=1); continue
            n_complete += 1

            t_set = truth_starts(truth_pattern, starts, N)
            if len(t_set) == 0 or len(t_set) == len(starts):
                d += timedelta(days=1); continue
            n_inf += 1

            q_full = q_by_date.get(d, {})
            q = {h: q_full[h] for h in range(s_utc, e_utc) if h in q_full}
            if len(q) < (e_utc - s_utc):
                d += timedelta(days=1); continue

            ps = block_probs(q, starts, N)
            total = sum(ps.values())
            pi = ({s: 1 / len(starts) for s in starts} if total <= 0
                  else {s: ps[s] / total for s in starts})

            argmax_s = max(starts, key=lambda s: pi[s])
            if argmax_s in t_set: n_top1 += 1

            tau = {s: (1 / len(t_set) if s in t_set else 0.0) for s in starts}
            sum_brier += sum((pi[s] - tau[s]) ** 2 for s in starts)
            sum_ll += -sum(tau[s] * math.log(max(pi[s], EPS)) for s in starts)
            sum_ll_uniform += -sum(tau[s] * math.log(1 / len(starts)) for s in starts)

            if collect_examples and len(examples) < 8:
                examples.append((d, dict(pi), t_set))

            d += timedelta(days=1)

        return (n_total, n_complete, n_inf, n_top1,
                sum_brier, sum_ll, sum_ll_uniform, examples)

    print(f"\nWindow: {START_DATE} .. {END_DATE}")

    # Binary first (collects examples), then the other two variants on the
    # exact same informative date set so the columns line up.
    n_total, n_complete, n_inf, top1_b, brier_b, ll_b, ll_u, examples = \
        score_variant("binary 0.1mm",  q_bin_by_date, collect_examples=True)
    _, _, _, top1_s, brier_s, ll_s, _, _ = \
        score_variant("smooth /0.5",   q_sm_by_date, collect_examples=False)
    _, _, _, top1_3a, brier_3a, ll_3a, _, examples_3a = \
        score_variant("phase 3a P(wet)", q_3a_by_date, collect_examples=True)

    print(f"  candidate / complete-truth / informative dates : "
          f"{n_total} / {n_complete} / {n_inf}")
    print()

    def fmt(top1, brier, ll):
        skill = 1 - (ll / ll_u) if ll_u > 0 else 0.0
        return (f"top-1 {top1}/{n_inf}={top1/n_inf:.1%}  "
                f"Brier {brier/n_inf:.4f}  "
                f"log-loss {ll/n_inf:.4f}  skill {skill:+.3f}")

    print(f"  uniform baseline               : log-loss {ll_u/n_inf:.4f}  skill 0.000")
    print(f"  option 1 — binary (>=0.1mm)    : {fmt(top1_b, brier_b, ll_b)}")
    print(f"  option 1 — smooth (precip/0.5) : {fmt(top1_s, brier_s, ll_s)}")
    print(f"  option 2 — phase 3a hourly P   : {fmt(top1_3a, brier_3a, ll_3a)}")
    print()

    print("Sample informative days — option 1 (binary) curves, first 8:")
    for dd, pi, t_set in examples:
        s_utc, e_utc = daytime_utc_range(dd)
        starts = candidate_starts(s_utc, e_utc, N)
        pi_str = " ".join(f"{s:02d}={pi[s]:>4.0%}" for s in starts)
        t_str = ",".join(f"{s:02d}" for s in sorted(t_set))
        argmax_s = max(starts, key=lambda s: pi[s])
        hit = "HIT" if argmax_s in t_set else "MISS"
        print(f"  {dd}  curve={pi_str}   truth={t_str}  argmax={argmax_s:02d} {hit}")

    print()
    print("Sample informative days — option 2 (Phase 3a) curves, first 8:")
    for dd, pi, t_set in examples_3a:
        s_utc, e_utc = daytime_utc_range(dd)
        starts = candidate_starts(s_utc, e_utc, N)
        pi_str = " ".join(f"{s:02d}={pi[s]:>4.0%}" for s in starts)
        t_str = ",".join(f"{s:02d}" for s in sorted(t_set))
        argmax_s = max(starts, key=lambda s: pi[s])
        hit = "HIT" if argmax_s in t_set else "MISS"
        print(f"  {dd}  curve={pi_str}   truth={t_str}  argmax={argmax_s:02d} {hit}")


if __name__ == "__main__":
    main()
