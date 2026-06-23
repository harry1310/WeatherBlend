#!/usr/bin/env python3
"""
Prune stale model versions from R2 to shrink the `data/models` tree.

WHY: every predict/render run (≈20×/day on fresh runners) re-pulls the whole models
tree as Class B reads — and most of it is OLD versions nothing uses (a manifest may
list 24 versions but mark only 1 Active). Pruning the stale ones cuts the read count
(the R2 bill) AND storage. Date-window scoping can't help here because models are
version-partitioned, not date-partitioned, so this is the lever (2026-06-21).

SAFE BY DESIGN:
  * DRY-RUN by default — prints the plan, deletes NOTHING. Requires --apply to act.
  * KEEP set per (target, station) = Active ∪ ChampionByLead values ∪ every version
    newer than --buffer-days (default 45 — covers verify's 30d SCORING window plus the
    ~15d a version's predictions outlive its training; the 90d sync is a superset, not
    the metadata need). Champions and anything recent are never touched.
  * DELETE = the manifest's Versions list MINUS the keep set — i.e. only versions the
    manifest itself already considers superseded AND older than the buffer.
  * On --apply: back up each MANIFEST.json on R2 first (.bak_<date>_pre_oldver_prune,
    matching the existing convention), purge each stale version dir, THEN rewrite the
    manifest's Versions list to match (no dangling references the render could read).
  * Reads the CANONICAL R2 manifests (not the possibly-stale local copy).
  * Touches ONLY data/models. Predictions/forecasts/truth are out of scope (verify
    needs prediction history — a separate, more careful job).
  * Orphan version dirs not tracked by any manifest (e.g. legacy pre-station-keyed
    temperature/v.../) are REPORTED, never auto-deleted.

Usage:
  python scripts/prune_model_versions.py                 # dry run (default)
  python scripts/prune_model_versions.py --buffer-days 21
  python scripts/prune_model_versions.py --apply         # actually prune
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile

R2 = "r2:weatherblend/data/models"
VERSION_RE = re.compile(r"^v(\d{4})-(\d{2})-(\d{2})_(\d{6})")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def version_date(v):
    """Parse the yyyy-MM-dd from a version name 'v2026-06-09_172445'. None if unparseable."""
    m = VERSION_RE.match(v)
    if not m:
        return None
    return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def list_manifests():
    out = run(["rclone", "lsf", R2 + "/", "-R", "--include", "MANIFEST.json"])
    return [line.strip() for line in out.splitlines() if line.strip().endswith("MANIFEST.json")]


def cat_manifest(rel):
    return json.loads(run(["rclone", "cat", f"{R2}/{rel}"]))


def dir_object_count(path):
    """How many objects under an R2 prefix (for the savings report)."""
    try:
        out = run(["rclone", "lsf", path, "-R", "--files-only"])
        return sum(1 for ln in out.splitlines() if ln.strip())
    except subprocess.CalledProcessError:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Prune stale model versions from R2 (dry-run default).")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run, deletes nothing)")
    # MUST cover verify's SCORING window (not its 90d prediction sync, which is a generous
    # superset). VerifyCommand only scores --window-days and only reads metadata for
    # versions in that window: 14d (temp) / 30d (precip, dry-window, element). A version's
    # predictions outlive its training by ~its champion tenure (≈7d, weekly retrain) + lead
    # (≤3d), so a version trained ~40d ago can still have in-window predictions. 30d window
    # + ~15d that tail = 45d. (Missing metadata only degrades to a null label / blank drift
    # for SUPERSEDED versions — never a crash — but 45d keeps everything still scored.)
    ap.add_argument("--buffer-days", type=int, default=45, help="keep every version newer than this many days (default 45 — covers verify's 30d scoring window + prediction tail)")
    ap.add_argument("--count-objects", action="store_true", help="count objects per stale version (slower; shows exact savings)")
    ap.add_argument("--target", default=None, help="limit to one target (e.g. cloud_cover) — stage the first apply on one tree")
    args = ap.parse_args()

    today = dt.datetime.now(dt.timezone.utc).date()
    cutoff = today - dt.timedelta(days=args.buffer_days)
    mode = "APPLY (DELETING)" if args.apply else "DRY RUN (no deletes)"
    print(f"== model-version prune — {mode}; keep Active + champions + versions newer than {cutoff} ({args.buffer_days}d) ==\n")

    manifests = list_manifests()
    total_stale_versions = 0
    total_stale_objects = 0
    plan = []   # (target, station, [stale versions], keep_count)

    for rel in manifests:
        target = rel.split("/")[0]
        if args.target and target != args.target:
            continue
        try:
            man = cat_manifest(rel)
        except Exception as e:
            print(f"  !! {rel}: could not read ({e}) — SKIPPING")
            continue
        stations = man.get("Stations", {})
        for station, body in stations.items():
            versions = list(body.get("Versions", []))
            active = set(body.get("Active", []))
            champ_by_lead = set()
            for v in (body.get("ChampionByLead") or {}).values():
                champ_by_lead.add(v)
            keep = set(active) | champ_by_lead
            # also keep anything recent (buffer) — even non-champions.
            for v in versions:
                d = version_date(v)
                if d is None or d >= cutoff:
                    keep.add(v)
            stale = [v for v in versions if v not in keep]
            if not stale:
                continue
            plan.append((target, station, stale, len(keep)))
            total_stale_versions += len(stale)

    if not plan:
        print("Nothing to prune — every version is Active, a champion, or within the buffer.")
        return 0

    for target, station, stale, keep_count in plan:
        objs = ""
        if args.count_objects:
            n = sum(dir_object_count(f"{R2}/{target}/{station}/{v}") for v in stale)
            total_stale_objects += n
            objs = f", ~{n} objects"
        print(f"  {target}/{station}: keep {keep_count}, PRUNE {len(stale)} stale version(s){objs}")
        for v in stale:
            print(f"      - {v}")

    print(f"\nTOTAL: {total_stale_versions} stale version dirs across {len(plan)} (target, station) cells"
          + (f", ~{total_stale_objects} objects" if args.count_objects else ""))

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply to prune (manifests are backed up first).")
        return 0

    print("\n== APPLYING ==")
    stamp = today.isoformat()
    # Group the plan by manifest so we back up + rewrite each manifest once.
    by_manifest = {}
    for target, station, stale, _ in plan:
        by_manifest.setdefault(target, []).append((station, stale))

    for rel in manifests:
        target = rel.split("/")[0]
        if target not in by_manifest:
            continue
        # 1) back up the manifest on R2 (matches the existing .bak_<date>_pre_oldver_prune convention).
        bak = f"{R2}/{rel}.bak_{stamp}_pre_oldver_prune"
        run(["rclone", "copyto", f"{R2}/{rel}", bak])
        print(f"  backed up {rel} -> {bak.split('/')[-1]}")
        # 2) purge each stale version dir.
        man = cat_manifest(rel)
        for station, stale in by_manifest[target]:
            for v in stale:
                run(["rclone", "purge", f"{R2}/{target}/{station}/{v}"])
                print(f"      purged {target}/{station}/{v}")
            # 3) rewrite the manifest's Versions list to drop the pruned versions.
            keep_set = set(man["Stations"][station]["Versions"]) - set(stale)
            man["Stations"][station]["Versions"] = [v for v in man["Stations"][station]["Versions"] if v in keep_set]
        # 4) push the rewritten manifest back (write to a temp file, re-parse to
        #    prove it's valid JSON, then upload — never push a half-written manifest).
        fd, tmp = tempfile.mkstemp(suffix="_MANIFEST.json")
        os.close(fd)
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=2)
        with open(tmp) as fh:
            json.load(fh)  # raises if we somehow produced invalid JSON
        run(["rclone", "copyto", tmp, f"{R2}/{rel}"])
        os.remove(tmp)
        print(f"  rewrote {rel}")

    print(f"\nDONE — pruned {total_stale_versions} stale version dirs. Manifests backed up with .bak_{stamp}_pre_oldver_prune.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
