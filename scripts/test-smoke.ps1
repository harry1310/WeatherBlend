# Integration smoke run: ONLY the [Trait("Category","Smoke")] tests — the
# end-to-end train -> sync_train_data.sh -> predict -> parquet/manifest chains
# that catch the hard-to-spot integration bugs (e.g. the 2026-05-28 "trained on
# 0 rows" sync-path bug). Slow (~20min): the cost is structural — the bash sync
# script + rclone round-trip, DuckDB/parquet IO and native cold-start, NOT model
# math (LightGBM iterations are already capped via WB_SMOKE_ITER in EnvScope).
# Run this as the FINAL check before pushing a major piece of work; use
# test-fast.ps1 for the inner loop.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
dotnet test "$root" -c Release --filter "Category=Smoke" @args
