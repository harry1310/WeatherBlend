"""One-shot patch: divide AIFS Precipitation column by 1000 to correct the
units bug found 2026-05-05.

Background: EcmwfClient was multiplying tp × 1000 for both IFS and AIFS,
assuming both publish in metres. Per ECMWF Open Data docs, IFS does publish
in metres but AIFS publishes in kg/m² (= mm) directly, so AIFS values were
1000× too high (modulated empirically to ~430× by the model's slightly
lower rainfall predictions vs IFS).

Parser is fixed in commit-after-this. This script back-corrects the
3163 existing parquets on disk so we don't have to re-run the ~7h of
AIFS backfill CI to regenerate them.

Idempotent guard: only patches rows where Precipitation > 100 (AIFS
post-bug values are 100s-1000s of mm, never legit; if a partial run
already had some patched rows, those would be < 100 and stay untouched).
Re-running the script is safe.

Usage:
    py scripts/patch_aifs_precip_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa


def patch_file(path: Path) -> tuple[bool, int, int]:
    """Returns (patched_bool, n_rows, n_changed)."""
    table = pq.read_table(path)
    if "Precipitation" not in table.column_names:
        return False, len(table), 0

    precip = table.column("Precipitation")
    # Mask: rows where Precipitation > 100 (the "post-bug" regime).
    # Genuine precip rates max ~30 mm/h so anything > 100 is the units bug.
    over_100_mask = pc.greater(precip, pa.scalar(100.0))
    n_changed = pc.sum(pc.cast(over_100_mask, pa.int32())).as_py() or 0
    if n_changed == 0:
        return False, len(table), 0

    # Divide everything by 1000 if ANY row in the file is bug-affected.
    # Mixed files shouldn't exist (each parquet = one cycle, all rows
    # pulled together with same parser) but be defensive.
    new_precip = pc.divide(precip, pa.scalar(1000.0))
    new_table = table.set_column(
        table.column_names.index("Precipitation"),
        "Precipitation",
        new_precip,
    )
    pq.write_table(new_table, path)
    return True, len(table), n_changed


def main() -> int:
    root = Path("data/forecasts/location=bonehill_rocks/model=ecmwf_aifs_oper")
    if not root.exists():
        print(f"FATAL: {root} not found")
        return 2

    files = sorted(root.rglob("*.parquet"))
    print(f"Found {len(files)} AIFS parquets under {root}")

    patched = 0
    skipped = 0
    total_rows_changed = 0
    for i, f in enumerate(files):
        was_patched, n_rows, n_changed = patch_file(f)
        if was_patched:
            patched += 1
            total_rows_changed += n_changed
        else:
            skipped += 1
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(files)}: patched={patched} skipped={skipped}")

    print(
        f"Done. patched={patched} files (rows changed = {total_rows_changed}), "
        f"skipped={skipped} (already-correct or no Precipitation column)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
