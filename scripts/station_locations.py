"""Shared station -> location lookup for the multi-location backfills.

Both ``backfill_location_metadata.py`` (Phase A — per-bundle
``training_metadata.json``) and ``backfill_manifest_locations.py``
(Phase B — per-station ``MANIFEST.json`` entries) need the same
authoritative mapping from an EA rainfall station slug to the
configured location it belongs to. Keeping it in one module means a
new station is added in exactly one place.

Derived from ``WeatherBlend/src/WeatherBlend/config.yaml``
(``locations[].rainfall.stations``). DO NOT widen
``STATION_TO_LOCATION`` without checking the bundle's actual
training-data partition.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Inference table — DO NOT widen without checking the bundle's actual
# training-data partition.
# ---------------------------------------------------------------------------

STATION_TO_LOCATION: dict[str, str] = {
    # Bonehill rainfall stations
    "ea_bellever_dartmoor":      "bonehill_rocks",
    "ea_bovey_tracey":           "bonehill_rocks",
    "ea_dartmoor_nr_hexworthy":  "bonehill_rocks",
    "ea_princetown":             "bonehill_rocks",  # historical, dropped 2026-05-04
    # Membury rainfall stations (added 2026-05-11)
    "ea_chards_snowdon_hill":    "membury_devon",
    "ea_goren":                  "membury_devon",
    "ea_raymonds_hill":          "membury_devon",
}

# Single-location targets — flat-layout manifests, version dirs sit
# directly under target/. Anything matching these whose path doesn't
# carry a station slug is assumed bonehill_rocks (the only location
# trained for these targets to date).
SINGLE_LOCATION_TARGETS: set[str] = {
    "temperature",
    "wind",
    "cloud_cover",
    "humidity",
    "shortwave_radiation",
    "feels_like",      # joined output, not strictly trained per-location
    "start_hour",      # ditto
}

# Station-keyed targets — MANIFEST.json carries a `Stations` dict, one
# entry per (station[/window]) blender.
PER_STATION_TARGETS: set[str] = {
    "precipitation",
    "dry_window",
}


def station_slug_from_key(key: str) -> str:
    """Extract the bare station slug from a MANIFEST.json ``Stations`` key.

    precipitation keys ARE the slug (``ea_bellever_dartmoor``).
    dry_window keys are composite (``ea_bellever_dartmoor/window_3h``) —
    the slug is the segment before the first slash.
    """
    return key.split("/", 1)[0]
