"""Unit tests for the radar advection engine (_engine.py)."""
import numpy as np
import _engine
from conftest import make_odim, CRAG, CRAG_PIXEL


def test_load_odim_parses_grid_and_georef(tmp_path):
    field = np.zeros((200, 200), "float32")
    field[100, 100] = 3.0
    field[0, 0] = -1.0  # nodata
    rate, georef, valid = _engine.load_odim(make_odim(str(tmp_path / "f.h5"), field, "202606301230"))
    assert rate[100, 100] == 3.0
    assert np.isnan(rate[0, 0])               # nodata -> NaN
    assert str(valid) == "2026-06-30T12:30"
    assert georef["xs"] == 1000.0


def test_pixel_of_roundtrips_the_crag(tmp_path):
    rate, georef, _ = _engine.load_odim(make_odim(str(tmp_path / "f.h5"), np.zeros((200, 200), "float32")))
    assert _engine.pixel_of(georef, *CRAG) == CRAG_PIXEL


def test_to_u8_treats_nan_as_dry():
    R = np.array([[np.nan, 0.0], [0.05, 10.0]], "float32")
    u = _engine.to_u8(R)
    assert u[0, 0] == u[0, 1]                  # NaN and dry map the same
    assert u[1, 1] > u[1, 0]                   # heavier rain -> brighter


def test_advect_translates_a_blob_east():
    field = np.zeros((120, 120), "float32")
    field[60, 40] = 10.0
    flow = np.zeros((120, 120, 2), "float32")
    flow[..., 0] = 5.0                          # +5 px east per frame-interval
    out = _engine.advect(field, flow, 2)        # 2 intervals -> +10 px east
    assert out[60, 50] > out[60, 40]            # mass moved toward col 50
    assert out[60, 50] == out.max()


def test_nowcast_blob_advects_onto_crag(moving_blob_frames):
    frames = [_engine.load_odim(p)[0] for p in moving_blob_frames]
    _, georef, _ = _engine.load_odim(moving_blob_frames[-1])
    res = _engine.nowcast(frames, georef, {"crag": CRAG}, lead_min=60, dt_min=15, nbhd_km=2)
    assert res["crag"]["accum_mm"] > 0          # approaching blob reaches the crag within the hour (needs motion)
    assert res["crag"]["wet"] is True
    # eastward motion is detected where the rain is (median over the mostly-dry field is ~0 by design)
    flow = _engine.flow_between(frames[-2], frames[-1])
    assert flow[..., 0].max() > 3               # strong +x (east) displacement somewhere
