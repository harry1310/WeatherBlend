"""One-shot fix: multiply MO Global Precipitation column by 1000 in every
existing backfilled parquet.

Pre-2026-04-27 the backfill script's m/s → mm/h conversion was off by 1000×
(used kg/m²/s formula × 3600, but the MO Global var is actually m/s = need
× 3,600,000). All historical Precipitation values are therefore 1000× too
small. The script + future writes are now fixed; this script patches the
already-written parquets so we don't have to re-download 1461 cycles.

Idempotency: writes a marker key in parquet metadata so re-running this
script doesn't double-multiply. If the marker is present, file is skipped.

After running, push to R2:
    rclone copy "data/forecasts/.../model=met_office_global/" "r2:..." --progress
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TREE = ROOT / "data" / "forecasts" / "location=bonehill_rocks" / "model=met_office_global"
MARKER_KEY = b"met_office_precip_units_fixed_2026_04_27"
MULTIPLIER = 1000.0


def main() -> int:
    if not TREE.exists():
        print(f"FATAL: tree not found at {TREE}")
        return 1
    files = sorted(TREE.rglob("*.parquet"))
    print(f"Found {len(files):,} parquet files under {TREE}")
    patched = skipped = 0
    for fp in files:
        # Read parquet metadata to check marker
        pf = pq.ParquetFile(fp)
        existing_meta = pf.schema_arrow.metadata or {}
        if MARKER_KEY in existing_meta:
            skipped += 1
            continue
        df = pd.read_parquet(fp)
        if "Precipitation" not in df.columns:
            print(f"  {fp.relative_to(TREE)}: no Precipitation column; skipping")
            continue
        df["Precipitation"] = df["Precipitation"] * MULTIPLIER
        # Write back with marker
        new_meta = {**(existing_meta or {}), MARKER_KEY: b"1"}
        df.to_parquet(fp, index=False)
        # Re-open and add the marker — pandas to_parquet doesn't pass metadata
        # through directly; use pyarrow to add it.
        table = pq.read_table(fp)
        new_schema = table.schema.with_metadata(new_meta)
        table = table.cast(new_schema)
        pq.write_table(table, fp)
        patched += 1
        if patched % 100 == 0:
            print(f"  patched {patched}/{len(files):,}")
    print(f"\nDone. Patched {patched:,}, skipped {skipped:,} (already patched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
