"""One-shot cleanup: drop PAV bundles + phase3n_k3 from each dry_window
cell's Active list, preserving Current, Versions, ChampionByLead.

A bundle is "PAV" if its local dir has calibrator_24h.json. We don't
delete bundle dirs — only the Active references.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Projects\Weather\WeatherBlend")
MANIFEST = ROOT / "data" / "models" / "dry_window" / "MANIFEST.json"
DRY_WINDOW_ROOT = ROOT / "data" / "models" / "dry_window"


def is_pav(station_key: str, version: str) -> bool:
    # station_key looks like "ea_bellever_dartmoor/window_3h"
    p = DRY_WINDOW_ROOT / station_key / version / "calibrator_24h.json"
    return p.exists()


def main() -> int:
    m = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    summary = []
    for station_key, entry in m["Stations"].items():
        active = entry.get("Active", [])
        if not active:
            continue
        current = entry.get("Current", "")
        keep, drop = [], []
        for v in active:
            # Never drop the Current 3b champion — it carries an intentional
            # calibrator from Phase 3d-calibrated. Only drop bundles whose
            # calibrator was the PAV-on-MC experiment (3g/3j/3n with calibrators)
            # or the K=3 bake-off.
            if v == current:
                keep.append(v)
                continue
            if v.endswith("_phase3n_k3"):
                drop.append((v, "K=3 bake-off"))
            elif is_pav(station_key, v) and any(v.endswith(t) for t in ("_phase3g", "_phase3j", "_phase3n")):
                drop.append((v, "PAV-on-MC"))
            else:
                keep.append(v)
        entry["Active"] = keep
        if drop:
            summary.append((station_key, drop))

    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")

    print(f"Cleaned MANIFEST.json. Dropped versions per cell:")
    print("-" * 80)
    total_dropped = 0
    for station_key, drop in summary:
        print(f"\n{station_key}: dropped {len(drop)}")
        for v, why in drop:
            print(f"  - {v}  ({why})")
        total_dropped += len(drop)
    print(f"\nTotal versions removed from Active: {total_dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
