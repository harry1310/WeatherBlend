"""One-shot Phase A backfill: pin LocationName into every training_metadata.json
on R2 that predates the 2026-05-12 multi-location safety work.

Walks `data/models/{target}/.../training_metadata.json`, infers the location:

  * precipitation + dry_window: read the station slug from the bundle path
    (`data/models/precipitation/ea_chards_snowdon_hill/...`) and look it up
    in STATION_TO_LOCATION. Membury slugs map to membury_devon; Bonehill
    slugs (Bellever / Bovey / Hexworthy / Princetown) map to bonehill_rocks.
  * temperature / element: bonehill_rocks. These targets are single-location
    today (no Membury equivalents trained), and the trainer would write
    `_cfg.Location.Name = bonehill_rocks` for new bundles, so the inference
    is unambiguous.

Defaults to dry-run (prints the inferred {file -> location} pairs without
touching R2). Pass `--apply` to actually write the JSON + push back. The
edit is in-place: download -> patch -> upload, leaving every other field
untouched. Idempotent — bundles that already have a non-empty LocationName
are skipped without warning.

Usage::

    python scripts/backfill_location_metadata.py            # dry-run, prints plan
    python scripts/backfill_location_metadata.py --apply    # actually write to R2
    python scripts/backfill_location_metadata.py --apply --target precipitation
                                                            # scope to one target

Pre-req: rclone configured with the `r2:` remote (see
``C:/Projects/Weather/R2Bucket.txt``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from station_locations import STATION_TO_LOCATION, SINGLE_LOCATION_TARGETS

R2_PREFIX = "r2:weatherblend/data/models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_metadata_files(target: str | None) -> list[str]:
    """Run rclone to enumerate every training_metadata.json under the models
    tree, optionally scoped to a single target. Returns relative paths
    (under data/models/).
    """
    prefix = R2_PREFIX
    if target:
        prefix = f"{R2_PREFIX}/{target}"
    result = subprocess.run(
        ["rclone", "lsf", "-R", "--files-only", prefix],
        capture_output=True, text=True, check=True,
    )
    files = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.endswith("training_metadata.json"):
            # Re-prefix with target if we filtered, so downstream logic always
            # sees a path of the form "{target}/.../training_metadata.json".
            if target:
                files.append(f"{target}/{line}")
            else:
                files.append(line)
    return sorted(files)


def infer_location(rel_path: str) -> tuple[str | None, str]:
    """Given e.g. "precipitation/ea_chards_snowdon_hill/v.../training_metadata.json",
    return (location_name | None, reason_string). None means we can't infer
    confidently — the bundle gets skipped with the reason logged.
    """
    parts = rel_path.split("/")
    if len(parts) < 2:
        return (None, f"path too short: {rel_path!r}")
    target = parts[0]
    # Per-station targets carry the station slug as the second segment.
    if target in {"precipitation", "dry_window"}:
        if len(parts) < 3:
            return (None, f"per-station target {target!r} but no station segment in {rel_path!r}")
        station_slug = parts[1]
        if station_slug not in STATION_TO_LOCATION:
            return (None, f"unknown station slug {station_slug!r}; add it to STATION_TO_LOCATION")
        return (STATION_TO_LOCATION[station_slug],
                f"station {station_slug} -> {STATION_TO_LOCATION[station_slug]}")
    # Single-location targets: the version dir sits directly under target/.
    if target in SINGLE_LOCATION_TARGETS:
        return ("bonehill_rocks", f"single-location target {target!r}")
    return (None, f"unknown target {target!r}; add to SINGLE_LOCATION_TARGETS or expand inference")


def patch_metadata(local_path: Path, location_name: str) -> tuple[bool, str]:
    """Read the JSON at local_path, set LocationName to location_name unless
    already non-empty, and write it back. Returns (was_changed, note)."""
    data = json.loads(local_path.read_text())
    existing = data.get("LocationName")
    if isinstance(existing, str) and existing.strip():
        return (False, f"already pinned to {existing!r}")
    # Insert at the top of the dict for human readability.
    new_data: dict = {}
    new_data["Version"] = data.get("Version", "")
    new_data["Target"] = data.get("Target", "")
    new_data["Phase"] = data.get("Phase", "")
    new_data["LocationName"] = location_name
    for k, v in data.items():
        if k in {"Version", "Target", "Phase", "LocationName"}:
            continue
        new_data[k] = v
    local_path.write_text(json.dumps(new_data, indent=2))
    return (True, f"patched -> {location_name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to R2; default is dry-run.")
    ap.add_argument("--target", default=None,
                    help="Restrict to one target (e.g. precipitation, dry_window, "
                         "temperature). Default: all.")
    args = ap.parse_args()

    print(f"Listing training_metadata.json under {R2_PREFIX}"
          + (f"/{args.target}" if args.target else "")
          + "...")
    files = list_metadata_files(args.target)
    print(f"  found {len(files)} files\n")

    plan: list[tuple[str, str, str]] = []   # (rel_path, location, reason)
    skipped: list[tuple[str, str]] = []     # (rel_path, reason)

    for rel in files:
        loc, reason = infer_location(rel)
        if loc is None:
            skipped.append((rel, reason))
        else:
            plan.append((rel, loc, reason))

    print(f"Inferred location for {len(plan)} files; skipped {len(skipped)}.\n")
    if skipped:
        print("--- skipped ---")
        for rel, reason in skipped:
            print(f"  {rel}  ({reason})")
        print()
    print("--- planned changes (sample, up to 25) ---")
    for rel, loc, reason in plan[:25]:
        print(f"  [{loc:>14}]  {rel}  ({reason})")
    if len(plan) > 25:
        print(f"  ... and {len(plan) - 25} more")
    print()

    if not args.apply:
        print("DRY-RUN — re-run with --apply to write to R2.")
        return 0

    if not plan:
        print("Nothing to apply.")
        return 0

    print(f"Applying {len(plan)} changes...")
    written = 0
    unchanged = 0
    failed = 0
    work_root = Path(tempfile.mkdtemp(prefix="wb-loc-backfill-"))
    try:
        for rel, loc, _ in plan:
            local_path = work_root / rel.replace("/", os.sep)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            r2_src = f"r2:weatherblend/data/models/{rel}"
            try:
                subprocess.run(
                    ["rclone", "copyto", r2_src, str(local_path)],
                    check=True, capture_output=True, text=True,
                )
                changed, note = patch_metadata(local_path, loc)
                if not changed:
                    unchanged += 1
                    continue
                # Push back. --s3-no-check-bucket avoids the missing
                # CreateBucket permission bite for the IAM token.
                subprocess.run(
                    ["rclone", "copyto", str(local_path), r2_src,
                     "--s3-no-check-bucket"],
                    check=True, capture_output=True, text=True,
                )
                print(f"  OK {rel}  -> {loc}")
                written += 1
            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"  ERR {rel}  ({e.stderr.strip().splitlines()[-1] if e.stderr else 'rclone failed'})")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"\nDone. Written: {written}, already-pinned: {unchanged}, failed: {failed}.")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
