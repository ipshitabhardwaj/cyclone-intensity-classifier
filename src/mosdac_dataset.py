"""
Preprocessing for real MOSDAC INSAT-3D archive orders (3DIMG_L1C_SGP, AOI-cropped
GeoTIFF, TIR1 + WV bands, brightness temperature) matched against real IMD/RSMC
best-track data.

This is the "scale up classification" pipeline: instead of the 136-image Kaggle
set, we order full storm lifecycles directly from MOSDAC and label every frame
by linearly interpolating the official 3-hourly IMD best track to that frame's
exact timestamp. That gives many more labeled frames per storm than best-track
points alone, since MOSDAC delivers imagery every 30 minutes.

Expected inputs:
  - A directory of MOSDAC GeoTIFF files named like
    3DIMG_15MAY2020_0000_L1C_SGP_V01R00_IMG_TIR1_TEMP.tif (exact suffix pattern
    may vary -- see parse_mosdac_filename, adjust the regex once real files
    are in hand).
  - A best-track CSV (see data/besttrack/amphan_2020.csv) with columns:
    date,time_utc,lat,lon,pressure_hpa,wind_kt

Usage (once the MOSDAC order has downloaded):
    python src/mosdac_dataset.py \
        --mosdac_dir data/mosdac/amphan_2020 \
        --besttrack data/besttrack/amphan_2020.csv \
        --storm_name AMPHAN_2020 \
        --out_dir data/real_storms/amphan_2020 \
        --window_km 600 \
        --out_size 128
"""
import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from utils import knots_to_kmph, wind_speed_to_category, category_label
from hursat_dataset import band_to_uint8, IRWIN_T_MIN, IRWIN_T_MAX

# MOSDAC's water-vapor channel reads much colder than the IR window channel,
# even in relatively clear air (it's sensing upper-tropospheric moisture at
# altitude, not surface/cloud-top temperature) -- so it gets its own clip
# range rather than reusing IRWIN_T_MIN/MAX. An initial guess of 190-250K
# (typical published WV-channel figures) turned out wrong for real delivered
# data: sampling actual Amphan WV crops across its lifecycle showed values
# consistently in ~178-210K, with almost everything piled up near 178-182K.
# A 190-250K clip range was clipping essentially the entire real image to
# its floor, producing flat, textureless output (caught by inspecting an
# actual cropped image, not just checking the code ran without error).
WV_T_MIN, WV_T_MAX = 178.0, 210.0

# Matches e.g. 3DIMG_15MAY2020_0000_L1C_SGP_V01R00 (date+time embedded in name).
# MOSDAC's actual delivered filenames may add a band suffix (e.g. _TIR1) --
# this regex only needs the date/time part, so it still matches with a suffix.
FNAME_RE = re.compile(
    r"3DIMG_(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{4})_(?P<hm>\d{4})"
)
MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_mosdac_filename(fname):
    """Extract the scene's UTC datetime from a MOSDAC 3DIMG filename.

    Returns None if the filename doesn't match the expected pattern (caller
    should skip/warn rather than crash -- real delivered filenames sometimes
    differ slightly from the order-page listing).
    """
    m = FNAME_RE.search(os.path.basename(fname))
    if not m:
        return None
    day = int(m.group("day"))
    mon = MONTHS.get(m.group("mon").upper())
    year = int(m.group("year"))
    hm = m.group("hm")
    if mon is None:
        return None
    hour, minute = int(hm[:2]), int(hm[2:])
    return datetime(year, mon, day, hour, minute)


def load_best_track(csv_path):
    """Load a best-track CSV into a sorted list of (datetime, lat, lon, pressure_hpa, wind_kt)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            hm = r["time_utc"].zfill(4)
            dt = datetime.strptime(r["date"] + hm, "%Y-%m-%d%H%M")
            rows.append((dt, float(r["lat"]), float(r["lon"]),
                         float(r["pressure_hpa"]), float(r["wind_kt"])))
    rows.sort(key=lambda x: x[0])
    return rows


def interpolate_track(dt, track):
    """Linearly interpolate lat/lon/pressure/wind at `dt` from best-track points.

    Returns None if dt falls outside [track[0].dt, track[-1].dt] -- i.e. the
    storm didn't exist yet / had already dissipated at that timestamp, so
    there is no honest label for that frame (drop it, don't guess).
    """
    if dt < track[0][0] or dt > track[-1][0]:
        return None
    for (t0, lat0, lon0, p0, w0), (t1, lat1, lon1, p1, w1) in zip(track, track[1:]):
        if t0 <= dt <= t1:
            if t1 == t0:
                frac = 0.0
            else:
                frac = (dt - t0).total_seconds() / (t1 - t0).total_seconds()
            lat = lat0 + frac * (lat1 - lat0)
            lon = lon0 + frac * (lon1 - lon0)
            pres = p0 + frac * (p1 - p0)
            wind = w0 + frac * (w1 - w0)
            return {"lat": lat, "lon": lon, "pressure_hpa": pres, "wind_kt": wind}
    return None


def crop_around_latlon(src_path, lat, lon, window_km, out_size):
    """Crop a georeferenced raster to a window_km x window_km box centered on
    (lat, lon), then resize to (out_size, out_size). Returns the raw (float32,
    physical-unit) crop, or None if the requested window falls (partly)
    outside the raster's extent.

    MOSDAC's delivered L1C_SGP GeoTIFFs are NOT in plain lat/lon degrees --
    real files inspected here are EPSG:3857 (Web Mercator, meters). Comparing
    a degree-sized window directly against the raster's bounds (as an
    earlier version of this function did, before real files were in hand)
    would silently miss the storm on every single frame, since a handful of
    degrees is a tiny number next to Web Mercator's multi-million-meter
    coordinates. The fix: reproject the storm-center point into the
    raster's own CRS first, then build the window in that CRS's units.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform as warp_transform

    with rasterio.open(src_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        cx, cy = xs[0], ys[0]
        # window_km is real-world distance regardless of the raster's CRS
        # units; for a projected (metric) CRS like EPSG:3857 this is a
        # direct meter offset, which is what real delivered files need.
        half_m = (window_km * 1000.0) / 2.0
        min_x, max_x = cx - half_m, cx + half_m
        min_y, max_y = cy - half_m, cy + half_m
        raster_left, raster_bottom, raster_right, raster_top = src.bounds
        if (min_x < raster_left or max_x > raster_right or
                min_y < raster_bottom or max_y > raster_top):
            return None  # storm center too close to the AOI edge for a full crop
        window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
        data = src.read(1, window=window)
        if data.size == 0:
            return None
        return data.astype("float32")


def _resize_uint8(u8, out_size):
    from PIL import Image
    return Image.fromarray(u8, mode="L").resize((out_size, out_size), Image.BILINEAR)


def build_manifest(mosdac_dir, besttrack_csv, storm_name, out_dir, window_km=600.0, out_size=128):
    """Walk a MOSDAC download directory, match each (TIR1, WV) scene pair to
    an interpolated best-track label, crop both bands storm-centered, and
    write JPEGs + a manifest.csv in the same layout hursat_dataset.py uses
    (see load_mosdac_samples), so this, the Kaggle set, and the HURSAT-B1
    set can all be concatenated into one training pool with zero per-source
    special-casing in train_combined.py.

    TIR1 is the IR window channel -- physically the same measurement as
    HURSAT-B1's IRWIN -- so it reuses hursat_dataset.py's IRWIN_T_MIN/MAX
    and cold=bright convention for visual consistency across every real
    dataset in this project. WV (water vapor) gets its own clip range
    (see WV_T_MIN/MAX above) since it reads colder even in clear air.
    """
    track = load_best_track(besttrack_csv)
    ir_dir = os.path.join(out_dir, "ir")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(os.listdir(mosdac_dir))
    by_time = {}
    for fn in files:
        dt = parse_mosdac_filename(fn)
        if dt is None:
            continue
        by_time.setdefault(dt, {})
        low = fn.lower()
        if "tir1" in low:
            by_time[dt]["tir1"] = os.path.join(mosdac_dir, fn)
        elif "wv" in low:
            by_time[dt]["wv"] = os.path.join(mosdac_dir, fn)

    rows = []
    n_no_label, n_edge, n_missing_band = 0, 0, 0
    for dt, bands in sorted(by_time.items()):
        label = interpolate_track(dt, track)
        if label is None:
            n_no_label += 1
            continue
        if "tir1" not in bands or "wv" not in bands:
            n_missing_band += 1
            continue

        ir_crop = crop_around_latlon(bands["tir1"], label["lat"], label["lon"],
                                      window_km, out_size)
        wv_crop = crop_around_latlon(bands["wv"], label["lat"], label["lon"],
                                      window_km, out_size)
        if ir_crop is None or wv_crop is None:
            n_edge += 1
            continue

        ir_u8 = band_to_uint8(ir_crop, None, IRWIN_T_MIN, IRWIN_T_MAX, invert=True)
        wv_u8 = band_to_uint8(wv_crop, None, WV_T_MIN, WV_T_MAX, invert=True)
        ir_img = _resize_uint8(ir_u8, out_size)
        raw_img = _resize_uint8(wv_u8, out_size)

        stamp = dt.strftime("%Y%m%dT%H%M")
        img_name = f"{storm_name}_{stamp}.jpg"
        ir_img.save(os.path.join(ir_dir, img_name), quality=95)
        raw_img.save(os.path.join(raw_dir, img_name), quality=95)

        wind_kmph = knots_to_kmph(label["wind_kt"])
        cat_idx = wind_speed_to_category(wind_kmph)
        rows.append({
            "img_name": img_name, "storm": storm_name, "datetime_utc": dt.isoformat(),
            "lat": f"{label['lat']:.2f}", "lon": f"{label['lon']:.2f}",
            "pressure_mb": f"{label['pressure_hpa']:.1f}", "wind_kt": f"{label['wind_kt']:.1f}",
            "wind_kmph": f"{wind_kmph:.1f}", "category": category_label(cat_idx),
            "cat_idx": cat_idx, "has_raw": True,
        })

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = ["img_name", "storm", "datetime_utc", "lat", "lon", "pressure_mb",
                  "wind_kt", "wind_kmph", "category", "cat_idx", "has_raw"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[mosdac_dataset] {storm_name}: {len(rows)} labeled frames written, "
          f"{n_no_label} outside best-track window, {n_missing_band} missing a band, "
          f"{n_edge} too close to AOI edge for a full crop -> {manifest_path}")


def load_mosdac_samples(out_dir):
    """Same sample-dict shape as hursat_dataset.load_hursat_samples(), so a
    MOSDAC-derived storm concatenates directly with the Kaggle and
    HURSAT-B1 sample lists in train_combined.py."""
    manifest_path = os.path.join(out_dir, "manifest.csv")
    samples = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ir_path = os.path.join(out_dir, "ir", row["img_name"])
            raw_path = os.path.join(out_dir, "raw", row["img_name"])
            if not os.path.isfile(ir_path):
                continue
            samples.append({
                "img_name": row["img_name"],
                "storm": row["storm"],
                "ir_path": ir_path,
                "raw_path": raw_path if os.path.isfile(raw_path) else ir_path,
                "has_raw": row["has_raw"] == "True",
                "knots": float(row["wind_kt"]),
                "kmph": float(row["wind_kmph"]),
                "cat_idx": int(row["cat_idx"]),
            })
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosdac_dir", required=True)
    ap.add_argument("--besttrack", required=True)
    ap.add_argument("--storm_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_km", type=float, default=600.0)
    ap.add_argument("--out_size", type=int, default=128)
    args = ap.parse_args()
    build_manifest(args.mosdac_dir, args.besttrack, args.storm_name,
                    args.out_dir, args.window_km, args.out_size)
