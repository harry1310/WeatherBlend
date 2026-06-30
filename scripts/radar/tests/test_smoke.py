"""SMOKE TEST — the whole live nowcast pipeline end-to-end on synthetic ODIM frames (no network, no R2):
load ODIM -> advection engine -> calibration -> the bonehill.json the site card consumes.

This is the integration guard for the radar product: if the ODIM parse, georef, advection, calibration,
JSON schema, or the start/stop ETA logic breaks, this fails. Run: pytest scripts/radar/tests/
"""
import numpy as np
import nowcast

# stub calibration (so the smoke doesn't depend on data/radar/calibration_bonehill.json, which is built in CI)
CAL = {"a": -1.36, "b": 1.30, "wet": 0.1}


def test_smoke_pipeline_end_to_end(moving_blob_frames):
    out = nowcast.compute(moving_blob_frames, CAL)

    # JSON schema the site card reads
    for key in ("frame_valid", "computed_at", "frame_age_min", "lead_min", "motion_kmh", "attribution", "sites"):
        assert key in out, f"missing top-level key {key}"
    assert "Met Office" in out["attribution"]
    assert out["frame_valid"] == "2026-06-30T12:45"          # newest frame

    # all four sites present, each with the card's fields
    assert set(out["sites"]) == set(nowcast.SITES)
    crag = out["sites"]["Bonehill crag"]
    for key in ("accum_mm", "p_wet", "max_rate_mmh", "rain_from", "rain_until", "onset_in_min"):
        assert key in crag

    # the blob marching east reaches the crag -> calibrated, wet, with an onset time
    assert 0.0 <= crag["p_wet"] <= 1.0
    assert crag["accum_mm"] > 0
    assert crag["rain_from"] is not None
    assert crag["onset_in_min"] is not None


def test_smoke_dry_field_stays_dry(moving_blob_frames, tmp_path):
    """A pair of empty frames -> no rain advected -> low P(wet), no onset."""
    from conftest import make_odim
    dry = [make_odim(str(tmp_path / f"d{i}.h5"), np.zeros((200, 200), "float32"), "20260630" + "09" + f"{i*15:02d}")
           for i in range(2)]
    out = nowcast.compute(dry, CAL)
    crag = out["sites"]["Bonehill crag"]
    assert crag["accum_mm"] == 0
    assert crag["rain_from"] is None
    assert crag["p_wet"] < 0.3                                # base rate only
