"""P3 migration: retire the flat element manifest layout.

WeatherBlend used to keep the four element_* single-model blenders
(wind / humidity / shortwave_radiation / cloud_cover) in a *flat*
manifest layout:

    data/models/{target}/MANIFEST.json   {Target, Current, Versions, Active}
    data/models/{target}/v{ts}/...       bundle dirs directly under target/

Every other target (temperature, precipitation, dry_window) is
Stations-keyed: the manifest carries a ``Stations`` dict and bundles
live under ``{target}/{station}/v{ts}/``. P3 removes the flat layout
from the .NET code (ModelArtifact no longer has the flat helpers), so
the on-disk element data must be migrated to match:

    data/models/{target}/MANIFEST.json
        {Target, Stations: {bonehill_rocks: {Versions, Active,
                                             ChampionByLead, Location}}}
    data/models/{target}/bonehill_rocks/v{ts}/...

The element blenders are bonehill-only, so every version moves under a
single station key — the location slug (``bonehill_rocks``). The stale
``Current`` pointer (retired project-wide earlier) is dropped.

This script does TWO things per target:
  1. rewrites MANIFEST.json flat -> Stations-keyed
  2. moves every ``v*`` bundle directory into the new station subdir

Idempotent: a manifest that already has a non-empty ``Stations`` dict,
and version dirs already sitting under the station subdir, are left
alone — safe to re-run.

Two backends:
  * local filesystem (default) — operates on a ``data/models`` tree on
    disk. This is what you run to migrate the gitignored local data.
  * ``--r2`` — operates on the R2 bucket via rclone, for the one-shot
    deploy-time migration. Mirrors backfill_manifest_locations.py.

Default is DRY-RUN — prints the plan, touches nothing. Pass ``--apply``
to execute. Run when nothing else is training/predicting against the
same tree (read-modify-write, not atomic against a concurrent writer).

Usage::

    # local (gitignored data tree) — what you run now
    python scripts/migrate_element_manifests.py                       # dry-run
    python scripts/migrate_element_manifests.py --apply
    python scripts/migrate_element_manifests.py --apply --root <path>

    # R2 (deploy-time only — do NOT run during a commit freeze)
    python scripts/migrate_element_manifests.py --r2                  # dry-run
    python scripts/migrate_element_manifests.py --r2 --apply

Pre-req for --r2: rclone configured with the ``r2:`` remote
(see C:/Projects/Weather/R2Bucket.txt).
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

# The four element_* single-model targets, by their on-disk ModelDirName
# (see WeatherBlend/src/WeatherBlend/Train/Element/ElementTargets.cs).
ELEMENT_TARGETS = ["wind", "humidity", "shortwave_radiation", "cloud_cover"]

# Element blenders are trained on a single location's NWP. bonehill_rocks
# is the only one (config.yaml locations[0]); it becomes the manifest's
# sole station key and the StationEntry.Location pin.
DEFAULT_LOCATION = "bonehill_rocks"

R2_MODELS_PREFIX = "r2:weatherblend/data/models"
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "models"


def is_flat(manifest: dict) -> bool:
    """A manifest still needs migrating when it carries no populated
    ``Stations`` dict. An already-migrated manifest has Stations non-empty."""
    return not (manifest.get("Stations") or {})


def build_station_manifest(manifest: dict, location: str) -> dict:
    """Flat manifest -> Stations-keyed manifest. The flat Versions/Active
    lists move wholesale into a single StationEntry under ``location``;
    ``Current`` is dropped (retired project-wide). ChampionByLead is carried
    over if the flat manifest had one, else an empty dict."""
    return {
        "Target": manifest.get("Target", ""),
        "Stations": {
            location: {
                "Versions": list(manifest.get("Versions", [])),
                "Active": list(manifest.get("Active", [])),
                "ChampionByLead": dict(manifest.get("ChampionByLead", {})),
                "Location": location,
            }
        },
    }


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

def migrate_local(root: Path, location: str, apply: bool) -> tuple[int, int]:
    """Migrate the element targets under a local ``data/models`` tree.
    Returns (targets_changed, failures)."""
    changed = 0
    failures = 0
    for target in ELEMENT_TARGETS:
        target_dir = root / target
        manifest_path = target_dir / "MANIFEST.json"
        print(f"--- {target} ---")
        if not manifest_path.exists():
            print(f"  no MANIFEST.json at {manifest_path} — skipping.\n")
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not is_flat(manifest):
            print("  already Stations-keyed — skipping.\n")
            continue

        # Version dirs sitting directly under target/ (the flat layout).
        version_dirs = sorted(
            d.name for d in target_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )
        new_manifest = build_station_manifest(manifest, location)
        station_dir = target_dir / location

        print(f"  manifest: flat -> Stations[{location!r}] "
              f"({len(new_manifest['Stations'][location]['Versions'])} versions, "
              f"{len(new_manifest['Stations'][location]['Active'])} active)")
        for v in version_dirs:
            print(f"  move    {target}/{v}  ->  {target}/{location}/{v}")
        if not version_dirs:
            print("  (no v* bundle dirs to move)")

        if not apply:
            print("  DRY-RUN — nothing written.\n")
            changed += 1
            continue

        try:
            station_dir.mkdir(parents=True, exist_ok=True)
            for v in version_dirs:
                src = target_dir / v
                dst = station_dir / v
                if dst.exists():
                    print(f"  WARN {dst} already exists — leaving {src} in place.")
                    continue
                shutil.move(str(src), str(dst))
            # Manifest last: if a move failed above we'd rather leave the
            # flat manifest pointing at a half-moved tree than claim success.
            manifest_path.write_text(
                json.dumps(new_manifest, indent=2), encoding="utf-8")
            print(f"  OK — migrated {target}.\n")
            changed += 1
        except OSError as e:
            failures += 1
            print(f"  ERR — {target}: {e}\n")
    return changed, failures


# ---------------------------------------------------------------------------
# R2 backend (rclone) — deploy-time only
# ---------------------------------------------------------------------------

def _rclone(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True)


def migrate_r2(location: str, apply: bool) -> tuple[int, int]:
    """Migrate the element targets on R2 via rclone. Returns
    (targets_changed, failures). Deploy-time one-shot — do NOT run during a
    commit freeze."""
    changed = 0
    failures = 0
    work_root = Path(tempfile.mkdtemp(prefix="wb-element-migrate-"))
    try:
        for target in ELEMENT_TARGETS:
            print(f"--- {target} (R2) ---")
            local_manifest = work_root / f"{target}.json"
            src = f"{R2_MODELS_PREFIX}/{target}/MANIFEST.json"
            got = _rclone(["copyto", src, str(local_manifest)])
            if got.returncode != 0 or not local_manifest.exists():
                print(f"  no MANIFEST.json on R2 for {target!r} — skipping.\n")
                continue

            manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
            if not is_flat(manifest):
                print("  already Stations-keyed — skipping.\n")
                continue

            # rclone lsf with --dirs-only enumerates the version dirs that
            # sit directly under target/ in the flat layout.
            lsf = _rclone([
                "lsf", f"{R2_MODELS_PREFIX}/{target}/", "--dirs-only",
            ])
            version_dirs = sorted(
                d.rstrip("/") for d in lsf.stdout.splitlines()
                if d.startswith("v")
            )
            new_manifest = build_station_manifest(manifest, location)

            print(f"  manifest: flat -> Stations[{location!r}]")
            for v in version_dirs:
                print(f"  move    {target}/{v}  ->  {target}/{location}/{v}")

            if not apply:
                print("  DRY-RUN — nothing written.\n")
                changed += 1
                continue

            try:
                for v in version_dirs:
                    move = _rclone([
                        "move",
                        f"{R2_MODELS_PREFIX}/{target}/{v}",
                        f"{R2_MODELS_PREFIX}/{target}/{location}/{v}",
                        "--s3-no-check-bucket",
                    ])
                    if move.returncode != 0:
                        raise RuntimeError(
                            move.stderr.strip().splitlines()[-1]
                            if move.stderr else "rclone move failed")
                local_manifest.write_text(
                    json.dumps(new_manifest, indent=2), encoding="utf-8")
                up = _rclone([
                    "copyto", str(local_manifest), src, "--s3-no-check-bucket",
                ])
                if up.returncode != 0:
                    raise RuntimeError(
                        up.stderr.strip().splitlines()[-1]
                        if up.stderr else "rclone manifest upload failed")
                print(f"  OK — migrated {target} on R2.\n")
                changed += 1
            except RuntimeError as e:
                failures += 1
                print(f"  ERR — {target}: {e}\n")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return changed, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Execute the migration; default is dry-run.")
    ap.add_argument("--r2", action="store_true",
                    help="Operate on the R2 bucket via rclone instead of a "
                         "local tree. Deploy-time one-shot.")
    ap.add_argument("--root", type=Path, default=DEFAULT_LOCAL_ROOT,
                    help=f"Local data/models root (default: {DEFAULT_LOCAL_ROOT}). "
                         "Ignored with --r2.")
    ap.add_argument("--location", default=DEFAULT_LOCATION,
                    help=f"Station/location key for the element manifests "
                         f"(default: {DEFAULT_LOCATION}).")
    args = ap.parse_args()

    where = "R2" if args.r2 else f"local {args.root}"
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"P3 element-manifest migration — {where} — {mode}\n")

    if args.r2:
        changed, failures = migrate_r2(args.location, args.apply)
    else:
        if not args.root.is_dir():
            print(f"ERROR: --root {args.root} is not a directory.")
            return 2
        changed, failures = migrate_local(args.root, args.location, args.apply)

    verb = "migrated" if args.apply else "would migrate"
    print(f"Done. {verb} {changed} target(s); {failures} failure(s).")
    if not args.apply and changed:
        print("DRY-RUN — re-run with --apply to execute.")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
