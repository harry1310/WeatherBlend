"""Phase B backfill: populate StationEntry.Location in every station-keyed
MANIFEST.json on R2.

Phase B commit 1 added a ``Location`` field to MANIFEST.json's per-station
entries; new promotes pin it from the freshly-trained bundle's
``training_metadata.json`` LocationName. This one-shot fills the field in
for entries that predate Phase B.

For each station-keyed target (precipitation, dry_window) it downloads
MANIFEST.json and, for every ``Stations`` entry whose ``Location`` is
empty/absent, looks the station slug up in STATION_TO_LOCATION and writes
it. Idempotent — entries already carrying a non-empty Location are left
untouched. dry_window keys are composite (``{slug}/window_{N}h``); the
slug is taken from the segment before the first slash.

Defaults to dry-run (prints the planned changes without touching R2).
Pass ``--apply`` to write back. Run when no retrain / predict is touching
the same manifest — it's a read-modify-write, not atomic against a
concurrent .NET writer.

Usage::

    python scripts/backfill_manifest_locations.py            # dry-run
    python scripts/backfill_manifest_locations.py --apply     # write to R2
    python scripts/backfill_manifest_locations.py --apply --target precipitation

Pre-req: rclone configured with the ``r2:`` remote (see
``C:/Projects/Weather/R2Bucket.txt``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from station_locations import (
    PER_STATION_TARGETS,
    STATION_TO_LOCATION,
    station_slug_from_key,
)

R2_PREFIX = "r2:weatherblend/data/models"


def download_manifest(target: str, dest: Path) -> bool:
    """rclone the target's MANIFEST.json down to ``dest``. Returns False
    when the manifest doesn't exist on R2 (target never trained)."""
    src = f"{R2_PREFIX}/{target}/MANIFEST.json"
    result = subprocess.run(
        ["rclone", "copyto", src, str(dest)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and dest.exists()


def upload_manifest(target: str, src: Path) -> None:
    """Push the patched MANIFEST.json back. --s3-no-check-bucket avoids
    the missing-CreateBucket-permission bite for the IAM token."""
    dst = f"{R2_PREFIX}/{target}/MANIFEST.json"
    subprocess.run(
        ["rclone", "copyto", str(src), dst, "--s3-no-check-bucket"],
        check=True, capture_output=True, text=True,
    )


def plan_target(manifest: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Inspect a manifest's Stations dict. Returns (changes, skipped):
    changes is [(station_key, location)] to write, skipped is
    [(station_key, reason)] left alone.
    """
    changes: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    stations = manifest.get("Stations") or {}
    for key, entry in stations.items():
        existing = entry.get("Location")
        if isinstance(existing, str) and existing.strip():
            skipped.append((key, f"already pinned to {existing!r}"))
            continue
        slug = station_slug_from_key(key)
        loc = STATION_TO_LOCATION.get(slug)
        if loc is None:
            skipped.append((key, f"unknown station slug {slug!r}; add it to STATION_TO_LOCATION"))
            continue
        changes.append((key, loc))
    return changes, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to R2; default is dry-run.")
    ap.add_argument("--target", default=None,
                    help="Restrict to one station-keyed target "
                         "(precipitation | dry_window). Default: all.")
    args = ap.parse_args()

    if args.target and args.target not in PER_STATION_TARGETS:
        print(f"ERROR: --target must be one of {sorted(PER_STATION_TARGETS)}; "
              f"got {args.target!r}.")
        return 2

    targets = [args.target] if args.target else sorted(PER_STATION_TARGETS)
    work_root = Path(tempfile.mkdtemp(prefix="wb-manifest-loc-"))
    total_changes = 0
    total_skipped = 0
    failed = 0

    try:
        for target in targets:
            local = work_root / f"{target}.json"
            print(f"--- {target} ---")
            if not download_manifest(target, local):
                print(f"  no MANIFEST.json on R2 for {target!r} — skipping.\n")
                continue

            manifest = json.loads(local.read_text())
            changes, skipped = plan_target(manifest)
            total_skipped += len(skipped)

            for key, reason in skipped:
                print(f"  skip  {key}  ({reason})")
            for key, loc in changes:
                print(f"  set   {key}  -> Location={loc}")

            if not changes:
                print("  nothing to change.\n")
                continue

            total_changes += len(changes)
            if not args.apply:
                print(f"  DRY-RUN — {len(changes)} entr(ies) would be patched.\n")
                continue

            # Apply: mutate the in-memory dict, write back, push to R2.
            stations = manifest["Stations"]
            for key, loc in changes:
                stations[key]["Location"] = loc
            local.write_text(json.dumps(manifest, indent=2))
            try:
                upload_manifest(target, local)
                print(f"  OK — patched {len(changes)} entr(ies) and pushed to R2.\n")
            except subprocess.CalledProcessError as e:
                failed += 1
                tail = e.stderr.strip().splitlines()[-1] if e.stderr else "rclone failed"
                print(f"  ERR — upload failed: {tail}\n")
    finally:
        import shutil
        shutil.rmtree(work_root, ignore_errors=True)

    verb = "applied" if args.apply else "planned"
    print(f"Done. {verb} {total_changes} change(s); {total_skipped} skipped; "
          f"{failed} upload failure(s).")
    if not args.apply and total_changes:
        print("DRY-RUN — re-run with --apply to write to R2.")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
