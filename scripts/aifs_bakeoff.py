"""
One-shot bake-off: blend (AIFS optional) vs blend (AIFS required) vs AIFS alone.

For each (artefact, lead) we:
1. Parse the artefact's per-lead test date range from training_metadata.json.
2. Query the forecasts tree with the same dedup/filter the spec used at train time.
3. Compute MAE of temp_aifs vs era5 on those test rows — that's the AIFS-only
   baseline against the same data the blender was scored against.

Usage: python scripts/aifs_bakeoff.py
"""
import io
import json
import re
import sys
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ARTEFACTS = [
    ("LEAN  optional (NEW post-fix)",  "data/models/temperature/v2026-04-28_220927",
     ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "gem_seamless"],
     {120: ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]}),
    ("RICH  optional (NEW post-fix)",  "data/models/temperature/v2026-04-28_220958_phase2c",
     ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "meteofrance_seamless", "ukmo_seamless", "gem_seamless"],
     {120: ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless"]}),
]

LOCATION = "bonehill_rocks"
FCAST_GLOB = "data/forecasts/**/*.parquet"
ERA5_GLOB  = "data/truth/era5/**/*.parquet"

def parse_range(s: str):
    # "2025-12-20 01:00Z → 2026-04-19 22:00Z"
    parts = re.split(r"\s*→\s*", s)
    a = parts[0].rstrip("Z").strip()
    b = parts[1].rstrip("Z").strip()
    return a, b

def aifs_mae(con, lead: int, required: list[str], start: str, end: str) -> tuple[float, int]:
    # Always pull AIFS into the pivot — even if the spec didn't require it,
    # we need its values to compute the baseline. The HAVING clause keeps the
    # spec's required-model gate so the row population matches the artefact's
    # test slice.
    required_clause = " AND ".join(
        [f"MAX(CASE WHEN Model = '{m}' THEN Temperature2m END) IS NOT NULL" for m in required]
    )
    in_models = list(dict.fromkeys(required + ["ecmwf_aifs025_single"]))  # dedup, AIFS in pivot
    in_clause = "(" + ",".join(f"'{m}'" for m in in_models) + ")"
    sql = f"""
    WITH latest AS (
        SELECT ValidTimeUtc, Model, Temperature2m,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{FCAST_GLOB}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead}
          AND Temperature2m IS NOT NULL
          AND Model IN {in_clause}
    ),
    pivoted AS (
        SELECT ValidTimeUtc,
               MAX(CASE WHEN Model = 'ecmwf_aifs025_single' THEN Temperature2m END) AS aifs_t
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc
        HAVING {required_clause}
    ),
    era5 AS (
        SELECT ValidTimeUtc, Temperature2m AS era5_t
        FROM read_parquet('{ERA5_GLOB}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}' AND Temperature2m IS NOT NULL
    )
    SELECT AVG(ABS(p.aifs_t - e.era5_t)) AS mae,
           COUNT(*) AS n
    FROM pivoted p JOIN era5 e USING (ValidTimeUtc)
    WHERE p.aifs_t IS NOT NULL
      AND p.ValidTimeUtc >= TIMESTAMP '{start}'
      AND p.ValidTimeUtc <= TIMESTAMP '{end}'
    """
    row = con.execute(sql).fetchone()
    return (row[0] or float("nan")), int(row[1])

con = duckdb.connect()

print()
print(f"{'Artefact':<28} {'Lead':>4} {'Test rows':>9} {'Blend test MAE':>14} {'AIFS test MAE':>13} {'Δ blend-vs-AIFS':>17}")
print("-" * 90)

for label, dirpath, required_default, lead_overrides in ARTEFACTS:
    with open(f"{dirpath}/training_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    for lead_str, s in sorted(meta["PerLead"].items(), key=lambda x: int(x[0])):
        lead = int(lead_str)
        required = lead_overrides.get(lead, required_default)
        start, end = parse_range(s["DataRangeTest"])
        blend = s["BlendTestMae"]
        aifs, n = aifs_mae(con, lead, required, start, end)
        delta = (blend - aifs) / aifs * 100
        flag = "  BLEND LOSES" if blend > aifs else "  blend wins"
        print(f"{label:<28} {lead:>3}h {n:>9d} {blend:>13.3f}°C {aifs:>12.3f}°C {delta:>+15.1f}% {flag}")
    print()
