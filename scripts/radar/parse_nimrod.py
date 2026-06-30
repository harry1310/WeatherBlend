#!/usr/bin/env python
"""Parse Met Office NIMROD 1km UK rainfall-radar tars into compact per-region data.

Each daily tar holds 288 five-minute NIMROD frames of the full UK 1km composite (2175x1725,
OSGB grid, rain-rate stored as int16 = mm/hr * 32). This decodes them, crops to a SW-England
box covering Bonehill / Membury / Sennen + an upwind margin (so advection has the incoming
field), and writes:
  data/radar/extracted/field/YYYY/YYYYMMDD.npz  - compressed int16 field [n,h,w] (raw, /32 = mm/hr),
                                                  frame timestamps, and the crop georef + location px.
  data/radar/extracted/points.parquet           - per-5min rain-rate (mm/hr) at the 3 locations.

This is everything the radar-advection probe needs (the regional field + the exact location
series), so the 40 GB of raw national composite can be deleted once this is verified complete.

Usage:
  python scripts/radar/parse_nimrod.py --day 20260515     # one day (validation)
  python scripts/radar/parse_nimrod.py --year 2025        # one year
  python scripts/radar/parse_nimrod.py                    # all years present
"""
import gzip, struct, tarfile, sys, glob, os, argparse
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from pyproj import Transformer

RAW = "data/radar/nimrod/uk-1km"
OUT_FIELD = "data/radar/extracted/field"
OUT_POINTS = "data/radar/extracted/points.parquet"

# NIMROD UK 1km composite grid (validated against the file headers)
NROWS, NCOLS = 2175, 1725
Y0, X0, DXY = 1549500.0, -404500.0, 1000.0   # OSGB northing of top row, easting of left col, 1 km
SCALE = 32.0                                  # units header = 'mm/h*32'
MDI = -32767                                  # integer missing-data value
MARGIN_KM = 128                               # upwind fetch margin (km) around the location bbox — covers the full 0-2h advection horizon from any flow direction

LOCS = {  # exact config.yaml coords (WGS84 lat, lon)
    "bonehill": (50.5831, -3.7931),
    "membury":  (50.8254, -3.0000),
    "sennen":   (50.0786, -5.7044),
}


def location_pixels():
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    px = {}
    for nm, (lat, lon) in LOCS.items():
        e, n = t.transform(lon, lat)
        col = int(round((e - X0) / DXY))
        row = int(round((Y0 - n) / DXY))
        px[nm] = (row, col)
    return px


def crop_box(px):
    rows = [r for r, c in px.values()]
    cols = [c for r, c in px.values()]
    r0 = max(0, min(rows) - MARGIN_KM); r1 = min(NROWS, max(rows) + MARGIN_KM + 1)
    c0 = max(0, min(cols) - MARGIN_KM); c1 = min(NCOLS, max(cols) + MARGIN_KM + 1)
    return r0, r1, c0, c1


def parse_frame(raw):
    """Return (datetime-tuple, int16 ndarray [NROWS,NCOLS]) for one decompressed NIMROD file."""
    if struct.unpack('>i', raw[0:4])[0] != 512:
        raise ValueError("bad 512-byte header marker")
    gi = np.frombuffer(raw[4:4 + 62], dtype='>i2')
    dt = (int(gi[0]), int(gi[1]), int(gi[2]), int(gi[3]), int(gi[4]))   # Y, M, D, h, m
    if (int(gi[15]), int(gi[16])) != (NROWS, NCOLS):
        raise ValueError(f"unexpected grid {int(gi[15])}x{int(gi[16])}")
    off = 4 + 512 + 4
    off += 4  # data record length marker
    arr = np.frombuffer(raw[off:off + NROWS * NCOLS * 2], dtype='>i2').reshape(NROWS, NCOLS)
    return dt, arr


def process_day(tar_path, r0, r1, c0, c1, locpx_crop):
    """Crop every frame in a daily tar; return (field int16 [n,h,w], times int64, points dict)."""
    tf = tarfile.open(tar_path)
    members = sorted((m for m in tf.getmembers() if m.name.endswith('.gz')), key=lambda m: m.name)
    fields, times = [], []
    pts = {nm: [] for nm in LOCS}
    for m in members:
        raw = gzip.decompress(tf.extractfile(m).read())
        dt, arr = parse_frame(raw)
        sub = arr[r0:r1, c0:c1].copy()              # int16, raw (mm/hr*32); MDI preserved
        fields.append(sub)
        ts = np.datetime64(f"{dt[0]:04d}-{dt[1]:02d}-{dt[2]:02d}T{dt[3]:02d}:{dt[4]:02d}", 's')
        times.append(ts)
        for nm, (rr, cc) in locpx_crop.items():
            v = sub[rr, cc]
            pts[nm].append(np.nan if v == MDI else v / SCALE)
    tf.close()
    return np.stack(fields), np.array(times), pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day"); ap.add_argument("--year")
    args = ap.parse_args()

    px = location_pixels()
    r0, r1, c0, c1 = crop_box(px)
    locpx_crop = {nm: (r - r0, c - c0) for nm, (r, c) in px.items()}
    print(f"locations (full-grid row,col): {px}")
    print(f"crop rows[{r0}:{r1}] cols[{c0}:{c1}] = {r1-r0} x {c1-c0} px; in-crop loc px: {locpx_crop}")

    if args.day:
        tars = glob.glob(f"{RAW}/*/metoffice-c-band-rain-radar_uk_{args.day}_1km-composite.dat.gz.tar")
    elif args.year:
        tars = sorted(glob.glob(f"{RAW}/{args.year}/*.tar"))
    else:
        tars = sorted(glob.glob(f"{RAW}/*/*.tar"))
    print(f"{len(tars)} daily tars to process")

    point_rows = {"time": [], **{nm: [] for nm in LOCS}}
    done = skip = 0
    for tp in tars:
        ymd = os.path.basename(tp).split('_')[2]
        year = ymd[:4]
        outdir = f"{OUT_FIELD}/{year}"; os.makedirs(outdir, exist_ok=True)
        outnpz = f"{outdir}/{ymd}.npz"
        if os.path.exists(outnpz) and os.path.getsize(outnpz) > 1000:
            skip += 1
            d = np.load(outnpz)  # still gather points for the parquet
            point_rows["time"].extend(d["times"].astype('datetime64[s]').tolist())
            for nm in LOCS:
                point_rows[nm].extend(d[f"pt_{nm}"].tolist())
            continue
        field, times, pts = process_day(tp, r0, r1, c0, c1, locpx_crop)
        tmp = outnpz + ".tmp"   # atomic write: an interrupted run leaves a .tmp (ignored), never a skippable corrupt .npz
        with open(tmp, "wb") as _fh:
            np.savez_compressed(
                _fh, field=field.astype(np.int16), times=times.astype('datetime64[s]'),
                r0=r0, c0=c0, y0=Y0, x0=X0, dxy=DXY, scale=SCALE, mdi=MDI,
                **{f"pt_{nm}": np.array(pts[nm], dtype=np.float32) for nm in LOCS},
                loc_rows=np.array([locpx_crop[nm][0] for nm in LOCS]),
                loc_cols=np.array([locpx_crop[nm][1] for nm in LOCS]),
                loc_names=np.array(list(LOCS.keys())))
        os.replace(tmp, outnpz)
        point_rows["time"].extend(times.astype('datetime64[s]').tolist())
        for nm in LOCS:
            point_rows[nm].extend(pts[nm])
        done += 1
        if done <= 3 or done % 50 == 0:
            wet = sum(1 for v in pts["bonehill"] if v and v > 0.1)
            print(f"  {ymd}  {field.shape}  npz {os.path.getsize(outnpz)/1e6:.2f} MB  "
                  f"bonehill wet-frames {wet}/288")
    print(f"fields: {done} written, {skip} already present")

    # combined point series parquet (tiny; rebuilt each run)
    if point_rows["time"]:
        os.makedirs(os.path.dirname(OUT_POINTS), exist_ok=True)
        order = np.argsort(np.array(point_rows["time"]))
        cols = {"time_utc": np.array(point_rows["time"])[order]}
        for nm in LOCS:
            cols[f"{nm}_mmhr"] = np.array(point_rows[nm], dtype=np.float32)[order]
        pq.write_table(pa.table(cols), OUT_POINTS, compression="zstd")
        print(f"points: {len(cols['time_utc'])} rows -> {OUT_POINTS} "
              f"({os.path.getsize(OUT_POINTS)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
