#!/usr/bin/env bash
#
# scripts/sync_train_data.sh
#
# Single source of truth for "which data trees does each WeatherBlend train
# phase consume?". Called by:
#   - .github/workflows/retrain-blenders.yml  (Sunday auto-retrain + ad-hoc)
#   - tests/WeatherBlend.Tests/Smoke/...      (via Process.Start with a
#                                              local fake-R2 source)
#
# The whole point of having one script for both is to make the smoke tests
# share the workflow's pull logic. When a new train phase is added to
# phases.yaml, updating the PER-PHASE NEEDS TABLE below tells the workflow
# AND the smoke fixtures what to expect — adding a phase without updating
# its data pulls fails the smoke at PR time rather than silently producing
# "Loaded 0 rows" in CI (the failure mode 2026-05-28 wind_speed_lgb hit
# when MIDAS was on R2 but the workflow's pull step missed it).
#
# Usage:
#   sync_train_data.sh <location> <phases-csv>
#
# Args:
#   location    Single location slug, e.g. "bonehill_rocks". Selects the
#               per-location partitions of forecasts + ERA5.
#   phases-csv  Comma-separated phase ids, e.g. "wind_speed_lgb,wind_gust_lgb".
#               The script computes the UNION of each phase's data needs and
#               issues exactly the rclone copies required.
#
# Required env:
#   R2_SOURCE   Either "r2:<bucket-name>" (production via the rclone "r2:"
#               remote) or a local path like "/tmp/fake-r2" (smoke). rclone
#               handles both source forms natively — only the protocol
#               prefix changes; the rest of the per-tree paths are identical.
#
# Optional env:
#   LOCAL_ROOT  Destination root (default: current working directory). Smoke
#               tests set this to their tempdir so the pull lands inside the
#               test scope rather than polluting the dev tree.
#
# Exit codes:
#   0  success
#   2  bad args
#   3  unknown phase id (declared in phases.yaml but missing from this
#      script's case table — fail loudly so the gap is visible at PR time)
#   *  rclone exit code on real network / FS error
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <location> <phases-csv>"
  echo "  e.g. $0 bonehill_rocks wind_speed_lgb,wind_gust_lgb"
  exit 2
fi

location="$1"
phases_csv="$2"

: "${R2_SOURCE:?R2_SOURCE env var required (e.g. r2:weatherblend or /tmp/fake-r2)}"
LOCAL_ROOT="${LOCAL_ROOT:-./data}"

# Source paths always include the leading "data/" — that's the R2 bucket
# layout and the smoke's fake-R2 dir mirrors it. Destination paths are
# anchored at LOCAL_ROOT (no "data/" prefix re-added at the destination).
# This asymmetry lets:
#   - the workflow set LOCAL_ROOT=./data → files land under ./data/...
#     (production layout, repo CWD)
#   - the smoke set LOCAL_ROOT={scope.Root} → files land under
#     {scope.Root}/... (SmokeScope's storage convention; no "data/" subdir).

mkdir -p "$LOCAL_ROOT"
cd "$LOCAL_ROOT"

# ----------------------------------------------------------------------------
# PER-PHASE NEEDS TABLE
# ----------------------------------------------------------------------------
# Each phase id (matching phases.yaml's `id:` field) declares which top-level
# trees it consumes. Workflows + smoke tests call this script to materialise
# the union — a single declarative list of dependencies per phase keeps the
# two contexts in lockstep.
#
# When adding a new train phase:
#   1. Add its id to phases.yaml under the appropriate target.
#   2. Add a case here mapping its id to the trees it reads.
#   3. The workflow + smoke automatically pull the right data on the next run.

need_forecasts=0
need_era5=0
need_metar=0
need_rainfall=0
need_midas=0
need_orographic=0
need_models=0

IFS=',' read -ra phases <<< "$phases_csv"
for p in "${phases[@]}"; do
  case "$p" in
    # Temperature blenders + the four single-truth element blenders +
    # wind_gust_lgb all share the same dependency set: per-location
    # forecast tree + per-location ERA5 truth + the existing models tree
    # for RetrainGuard's previous-summary read.
    2b|2c|2d|wind|humidity|shortwave_radiation|cloud_cover|wind_gust_lgb)
      need_forecasts=1; need_era5=1; need_models=1 ;;

    # Precipitation / dry-window blenders score against EA rainfall truth.
    3a|3c|3d|3b)
      need_forecasts=1; need_rainfall=1; need_models=1 ;;

    # 3o adds the per-station static orographic JSON on top of 3c's inputs.
    3o)
      need_forecasts=1; need_rainfall=1; need_orographic=1; need_models=1 ;;

    # 3p (copula MC) + 4b (mint) consume previously-trained bundles + EA truth.
    # Orographic is technically only needed by the 3p smoke for its 3o
    # source-binding consistency check, but cheap so pull it.
    3p|4b)
      need_forecasts=1; need_rainfall=1; need_orographic=1; need_models=1 ;;

    # 3q (copula MC over 3c) — Sennen's coastal dry-window challenger. Like
    # 3p but bound to 3c, which has NO orographic features, so no orographic
    # pull. Σ-fit needs EA rainfall truth; the 3c champion version is read
    # from the (already-pulled) manifest.
    3q)
      need_forecasts=1; need_rainfall=1; need_models=1 ;;

    # wind_speed_lgb moved to Python 2026-06-10 (Option B CQR cutover) — its
    # pull declaration now lives in WeatherProbabilistic's
    # scripts/sync_train_data.sh. A request here exits 3 loudly below.

    all)
      need_forecasts=1; need_era5=1; need_metar=1; need_rainfall=1
      need_midas=1; need_orographic=1; need_models=1 ;;

    *)
      echo "::error::sync_train_data: unknown phase id '$p' — add it to the case table in scripts/sync_train_data.sh."
      exit 3 ;;
  esac
done

# ----------------------------------------------------------------------------
# Pull helper
# ----------------------------------------------------------------------------
# rclone --fast-list / --transfers / --checkers tuned for many-small-files
# trees (forecasts is the worst case at tens of thousands of parquet files).
# Caller's RCLONE_TIMEOUT / RCLONE_RETRIES env (set by the workflow's job
# env block) flow through to rclone untouched.
rcl_flags=(--fast-list --transfers 16 --checkers 32)

# --s3-no-check-bucket is required on first writes to fresh R2 prefixes
# (the IAM token doesn't carry CreateBucket). It's a no-op for reads but
# safe to always pass — keeps the script symmetric whether source is R2
# or a local path.
#
# Args:
#   $1 src_subpath   path under R2_SOURCE (includes "data/" prefix)
#   $2 dest_subpath  path under LOCAL_ROOT (no "data/" prefix —
#                    LOCAL_ROOT is already the data root)
#   $@ extra rclone flags
pull() {
  local src_sub="$1"; shift
  local dest_sub="$1"; shift
  echo "::group::sync_train_data: pull ${src_sub} -> ${dest_sub}"
  mkdir -p "$dest_sub"
  rclone copy "${R2_SOURCE%/}/${src_sub}" "$dest_sub" "${rcl_flags[@]}" --s3-no-check-bucket "$@"
  echo "::endgroup::"
}

# ----------------------------------------------------------------------------
# Tree-by-tree pulls (in dependency order — forecasts heaviest first)
# ----------------------------------------------------------------------------

if [ "$need_forecasts" -gt 0 ]; then
  pull "data/forecasts/location=$location" "forecasts/location=$location"
fi

if [ "$need_era5" -gt 0 ]; then
  pull "data/truth/era5/location=$location" "truth/era5/location=$location"
fi

if [ "$need_metar" -gt 0 ]; then
  pull "data/truth/metar" "truth/metar"
fi

if [ "$need_rainfall" -gt 0 ]; then
  pull "data/truth/rainfall" "truth/rainfall"
fi

if [ "$need_midas" -gt 0 ]; then
  # Station 01383 = Dunkeswell aerodrome. Glob matches production MIDAS
  # Open filenames (midas-open_uk-hourly-weather-obs_dv-{ver}_devon_01383_*.csv)
  # AND smoke-fixture filenames (midas-open_..._dv-smoke_..._01383_*.csv).
  pull "data/truth/midas/raw" "truth/midas/raw" --include 'midas-open*_01383_*.csv'
fi

if [ "$need_orographic" -gt 0 ]; then
  pull "data/static/orographic" "static/orographic" --include '*.json'
fi

if [ "$need_models" -gt 0 ]; then
  # Whole tree — RetrainGuard reads previous training_summary.json for
  # every station/phase combination it might compare against, so blanket
  # pull is simpler than enumerating per-target subtrees.
  #
  # SOFT pull: RetrainGuard handles "no previous summary" gracefully
  # (skips delta-vs-previous checks on first-deploy / fresh stations),
  # so a missing data/models source isn't fatal. The hard pulls above
  # (forecasts, era5, midas, etc.) ARE required because the trainers
  # genuinely cannot produce rows without their truth/feature inputs —
  # those failures stay loud.
  pull "data/models" "models" || echo "::notice::sync_train_data: data/models absent at source — RetrainGuard will treat this as a fresh-deploy (no previous-summary comparison)."
fi

echo "sync_train_data: done. Pulled trees:"
echo "  forecasts=$need_forecasts era5=$need_era5 metar=$need_metar rainfall=$need_rainfall"
echo "  midas=$need_midas orographic=$need_orographic models=$need_models"
