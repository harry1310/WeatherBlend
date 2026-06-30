"""Test fixtures for the radar nowcast pipeline. Builds SYNTHETIC ODIM HDF5 frames (no network), so the
unit + smoke tests run fully offline. Mirrors the real file's structure (the recipe proven by the spike)."""
import os
import sys

import numpy as np
import h5py
import pytest
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/radar on path

PROJ = "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.999601 +x_0=400000 +y_0=-100000 +ellps=airy +units=m"
CRAG = (50.5831, -3.7931)
CRAG_PIXEL = (100, 100)  # where we place the crag in the synthetic grid


def make_odim(path, field, valid="202606301200"):
    """Write a minimal ODIM HDF5 with the crag at CRAG_PIXEL. `field` is a (H,W) mm/h grid; valid=YYYYMMDDHHMM."""
    H, W = field.shape
    tf = Transformer.from_crs(4326, PROJ, always_xy=True)
    inv = Transformer.from_crs(PROJ, 4326, always_xy=True)
    xb, yb = tf.transform(CRAG[1], CRAG[0])
    x_ul = xb - CRAG_PIXEL[1] * 1000.0          # UL corner so the crag lands at CRAG_PIXEL
    y_ul = yb + CRAG_PIXEL[0] * 1000.0
    ul_lon, ul_lat = inv.transform(x_ul, y_ul)
    with h5py.File(path, "w") as f:
        w = f.create_group("where")
        w.attrs["projdef"] = np.bytes_(PROJ)
        w.attrs["xscale"] = 1000.0; w.attrs["yscale"] = 1000.0
        w.attrs["xsize"] = W; w.attrs["ysize"] = H
        w.attrs["UL_lat"] = ul_lat; w.attrs["UL_lon"] = ul_lon
        wh = f.create_group("what")
        wh.attrs["date"] = np.bytes_(valid[:8]); wh.attrs["time"] = np.bytes_(valid[8:12])
        d = f.create_group("dataset1/data1")
        d.create_dataset("data", data=field.astype("float32"))
        dw = d.create_group("what")
        dw.attrs["gain"] = 1.0; dw.attrs["offset"] = 0.0
        dw.attrs["nodata"] = -1.0; dw.attrs["undetect"] = 0.0
    return path


@pytest.fixture
def moving_blob_frames(tmp_path):
    """4 frames (15-min apart) with a rain blob marching EAST toward the crag — so advection carries it
    onto the crag in the next hour. Returns paths oldest..newest."""
    H = W = 200
    paths = []
    for i, col in enumerate([60, 70, 80, 90]):     # +10 px/frame eastward; latest at col 90, crag at 100
        field = np.zeros((H, W), "float32")
        field[95:106, col - 5:col + 6] = 5.0       # ~5 mm/h blob spanning the crag's row band
        valid = "20260630" + "12" + f"{i * 15:02d}"  # 12:00, 12:15, 12:30, 12:45
        paths.append(make_odim(str(tmp_path / f"frame_{i}.h5"), field, valid))
    return paths
