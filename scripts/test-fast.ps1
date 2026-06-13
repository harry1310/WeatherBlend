# Fast inner-loop test run: everything EXCEPT the [Trait("Category","Smoke")]
# integration smokes. ~891 tests in ~20s vs ~22min for the full suite — the
# entire wall-time of the full run is ~13 model-training smoke tests (see
# scripts/test-smoke.ps1). Run this constantly during development; run the
# smokes (test-smoke.ps1) as the final check before pushing a major piece.
#
# Extra args pass through to dotnet test, e.g.:
#   ./scripts/test-fast.ps1 --filter "FullyQualifiedName~Climbing"
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
dotnet test "$root" -c Release --filter "Category!=Smoke" @args
