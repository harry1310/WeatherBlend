"""Fetch Met Office UK 1km composite radar (ODIM HDF5) from the free AWS Open Data bucket.

Bucket `met-office-radar-obs-data` (eu-west-2), anonymous HTTPS. Keys:
  radar/YYYY/MM/DD/YYYYMMDDhhmm_ODIM_ng_radar_rainrate_composite_1km_UK.h5
15-min frames, ~20-min publish latency, 2-year rolling archive. CC BY-SA (attribution required).
"""
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta

BASE = "https://met-office-radar-obs-data.s3.eu-west-2.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _list(prefix, delimiter=None, maxkeys=1000):
    url = f"{BASE}?list-type=2&prefix={prefix}&max-keys={maxkeys}"
    if delimiter:
        url += f"&delimiter={delimiter}"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
    keys = [e.text for e in root.iter(NS + "Key")]
    cps = [e.find(NS + "Prefix").text for e in root.iter(NS + "CommonPrefixes")]
    return keys, cps


def latest_keys(n=4):
    """The newest `n` frame keys (walks year/month/day prefixes — date-agnostic, robust to the rolling window).
    Crosses the DAY boundary: just after midnight the newest day holds < n frames, so top up from the
    previous day(s) — otherwise `nowcast` gets < 2 frames and crashes (the 2026-07-01 00:10 midnight failure)."""
    p = "radar/"
    for _ in range(3):                      # year, month, day
        _, cps = _list(p, delimiter="/")
        p = max(c for c in cps if c)
    keys, _ = _list(p, maxkeys=400)         # a day has ~96 frames
    guard = 0
    while len(keys) < n and guard < 3:      # newest day too sparse (near midnight) — pull the previous day
        y, m, d = (int(x) for x in p.strip("/").split("/")[1:4])
        prev = date(y, m, d) - timedelta(days=1)
        pk, _ = _list(f"radar/{prev:%Y/%m/%d}/", maxkeys=400)
        keys = pk + keys
        p = f"radar/{prev:%Y/%m/%d}/"
        guard += 1
    return sorted(keys)[-n:]                # oldest..newest, across the boundary if needed


def download(key, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, os.path.basename(key))
    if not os.path.exists(local):
        urllib.request.urlretrieve(BASE + key, local)
    return local


def latest_frames(n, dest_dir):
    """Download the newest `n` frames; return local paths oldest..newest."""
    return [download(k, dest_dir) for k in latest_keys(n)]


if __name__ == "__main__":
    import sys
    dest = sys.argv[1] if len(sys.argv) > 1 else "data/radar/live"
    for p in latest_frames(4, dest):
        print(p)
