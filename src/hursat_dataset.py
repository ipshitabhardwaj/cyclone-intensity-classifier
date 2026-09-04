"""
Preprocessing for real NOAA HURSAT-B1 v06 archives (Hurricane Satellite data --
per-storm NetCDF stacks, already storm-centered, 301x301 px, ~8km, 3-hourly).

This is the second "scale up training data" source (alongside the 136-image
Kaggle set): 10 real North Indian Ocean storms (2008-2016) downloaded by hand
from https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/
(this host blocks automated/proxy fetches, so the .tar.gz files were fetched
by the user's own browser), specifically chosen to cover IMD categories that
the Kaggle set barely has (D/DD/CS), using real IBTrACS record counts to pick
storms:
  NARGIS(2008) PHET(2010) PHAILIN(2013) MADI(2013) HUDHUD(2014) NILOFAR(2014)
  KOMEN(2015) CHAPALA(2015) KYANT(2016) VARDAH(2016)

Each archive unpacks to many NetCDF files, one per (timestamp, satellite)
pair -- several geostationary/polar satellites can all see the same storm at
the same synoptic hour, so this module dedupes to one frame per timestamp
(lowest viewing-zenith-angle = most nadir view = least parallax distortion)
before extracting.

Filename layout inside each tarball (confirmed by direct inspection):
  <SID>.<NAME>.<YYYY>.<MM>.<DD>.<HHMM>.<VZA>.<SAT-ID>.<seq>.hursat-b1.v06.nc
e.g. 2008117N11090.NARGIS.2008.04.25.1200.21.FY2-C.018.hursat-b1.v06.nc

Key channel used: IRWIN (IR window brightness temperature, Kelvin, ~180-310K
real range). The existing Kaggle pipeline (kaggle_dataset.py) was trained on
processed INSAT3D JPEGs where cold cloud tops render bright/white (the
standard enhanced-IR convention forecasters use -- the coldest, highest
cloud tops around the eye wall are the most diagnostic pixels). IRWIN is
converted to that same visual convention here (see irwin_to_uint8), then
written out as plain 8-bit grayscale JPEGs so the *exact* same
KaggleINSAT3DDataset class / img_size=128 / 0-255 pipeline can load either
dataset with zero changes -- load_hursat_samples() below returns the same
sample-dict shape as kaggle_dataset.load_samples(), meant to be concatenated
with it before building the Dataset.

A second channel (VSCHN, visible reflectance) was evaluated as a stand-in for
the Kaggle dataset's "raw" image, but spot-checking real frames found VSCHN
quality inconsistent across satellites in this archive (e.g. one MET-7 frame
came back uniformly ~0.9 reflectance across the whole 301x301 scene -- no
cloud texture at all, almost certainly a calibration artifact rather than a
real reading). Rather than risk feeding that into the model's second input
channel, every HURSAT-B1 sample reuses the IR frame for "raw" (has_raw=False)
-- the exact same fallback kaggle_dataset.py already uses for its own frames
without a real raw counterpart, so this is a data-quality choice already
built into the pipeline, not a workaround.

Usage (run where the .tar.gz files live -- this needs the `netCDF4` package:
    pip install netCDF4 --break-system-packages):
    python src/hursat_dataset.py \
        --raw_dir hursat_raw \
        --out_dir data/hursat_processed \
        --img_size 128
"""
import argparse
import csv
import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils import knots_to_kmph, wind_speed_to_category, category_label

# IRWIN brightness-temperature clip range (Kelvin). Real tropical-cyclone IR
# scenes: ~185K at the coldest overshooting convective tops, ~300K for clear
# ocean/land background. Values are inverted when mapped to 0-255 so cold
# (deep convection, most diagnostic) renders bright/white -- the same visual
# convention as the processed INSAT3D JPEGs already in the Kaggle set.
IRWIN_T_MIN, IRWIN_T_MAX = 185.0, 300.0

# NetCDF filenames look like:
#   2008117N11090.NARGIS.2008.04.25.1200.21.FY2-C.018.hursat-b1.v06.nc
# i.e. exactly 12 '.'-separated fields (storm/satellite names never contain
# a literal '.', so a plain split is reliable and avoids a fragile regex).
_EXPECTED_SUFFIX = ("hursat-b1", "v06", "nc")


def parse_hursat_filename(fname: str):
    """Return {sid, name, year, month, day, hm, vza, sat_id, seq} or None if
    `fname` doesn't match the expected HURSAT-B1 naming pattern."""
    parts = os.path.basename(fname).split(".")
    if len(parts) != 12 or tuple(parts[-3:]) != _EXPECTED_SUFFIX:
        return None
    sid, name, year, month, day, hm, vza, sat_id, seq = parts[:9]
    try:
        return {
            "sid": sid, "name": name,
            "year": year, "month": month, "day": day, "hm": hm,
            "vza": float(vza), "sat_id": sat_id, "seq": seq,
        }
    except ValueError:
        return None


def _masked_to_float(val):
    """netCDF4 returns numpy masked scalars/arrays for missing data; convert
    a fully-masked or NaN scalar to None, otherwise a plain float."""
    if val is None:
        return None
    if np.ma.is_masked(val):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def read_hursat_frame(nc_path: str):
    """Open one HURSAT-B1 NetCDF file and pull out the fields this pipeline
    needs. Returns None if the file can't be read or has no usable IRWIN
    data (a handful of frames are all-fill near the scan edge)."""
    import netCDF4

    try:
        ds = netCDF4.Dataset(nc_path, "r")
    except OSError:
        return None
    try:
        if "IRWIN" not in ds.variables:
            return None
        irwin = ds.variables["IRWIN"][0]  # (1, 301, 301) -> (301, 301), Kelvin
        if np.ma.is_masked(irwin) and irwin.mask.all():
            return None
        # VSCHN (visible reflectance) deliberately not used -- see module
        # docstring on inconsistent per-satellite VSCHN quality.
        wind_kt = _masked_to_float(ds.variables["WindSpd"][0]) if "WindSpd" in ds.variables else None
        pres_mb = _masked_to_float(ds.variables["CentPrs"][0]) if "CentPrs" in ds.variables else None
        lat = _masked_to_float(ds.variables["CentLat"][0]) if "CentLat" in ds.variables else None
        lon = _masked_to_float(ds.variables["CentLon"][0]) if "CentLon" in ds.variables else None
        return {"irwin": np.array(irwin), "irwin_mask": np.ma.getmaskarray(irwin),
                "wind_kt": wind_kt, "pres_mb": pres_mb, "lat": lat, "lon": lon}
    finally:
        ds.close()


def band_to_uint8(arr, mask, t_min, t_max, invert):
    """Clip+normalize a physical-unit band to 0-255. `invert=True` maps low
    values (cold Kelvin) to high pixel values (bright), matching the
    enhanced-IR convention; `invert=False` is a plain linear stretch (used
    for VSCHN reflectance, where more sunlight reflected = brighter is
    already the natural mapping)."""
    a = np.clip(arr, t_min, t_max)
    norm = (a - t_min) / (t_max - t_min)  # 0..1
    if invert:
        norm = 1.0 - norm
    out = (norm * 255.0).astype(np.uint8)
    if mask is not None and mask.any():
        fill = 0 if invert else 0  # masked-out pixels (scan edge) -> black, not a fake reading
        out = np.where(mask, fill, out)
    return out


def irwin_to_uint8(irwin, mask):
    return band_to_uint8(irwin, mask, IRWIN_T_MIN, IRWIN_T_MAX, invert=True)


def _dedupe_best_view(frames):
    """frames: list of (meta, nc_path). Groups by exact timestamp and keeps
    only the lowest-VZA (most nadir, least parallax-distorted) file per
    group -- HURSAT-B1 bundles every satellite that saw the storm at each
    3-hourly synoptic time, and they're not all equally usable."""
    by_ts = {}
    for meta, path in frames:
        key = (meta["year"], meta["month"], meta["day"], meta["hm"])
        best = by_ts.get(key)
        if best is None or meta["vza"] < best[0]["vza"]:
            by_ts[key] = (meta, path)
    return sorted(by_ts.values(), key=lambda mp: (mp[0]["year"], mp[0]["month"], mp[0]["day"], mp[0]["hm"]))


def process_storm_tarball(tar_path, out_dir, img_size, storm_label=None):
    """Extract one HURSAT-B1 storm archive, dedupe, convert, and write JPEGs
    + manifest rows. Returns a list of manifest row dicts."""
    storm_label = storm_label or os.path.basename(tar_path).split(".tar")[0]
    tmp_dir = tempfile.mkdtemp(prefix="hursat_")
    rows = []
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp_dir)

        nc_files = []
        for root, _dirs, files in os.walk(tmp_dir):
            for fn in files:
                if fn.endswith(".nc"):
                    nc_files.append(os.path.join(root, fn))

        parsed = []
        n_unparsable = 0
        for path in nc_files:
            meta = parse_hursat_filename(path)
            if meta is None:
                n_unparsable += 1
                continue
            parsed.append((meta, path))

        best = _dedupe_best_view(parsed)

        ir_dir = os.path.join(out_dir, "ir")
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(ir_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        n_no_irwin, n_no_wind, n_written = 0, 0, 0
        for meta, path in best:
            frame = read_hursat_frame(path)
            if frame is None:
                n_no_irwin += 1
                continue
            if frame["wind_kt"] is None or frame["wind_kt"] <= 0:
                n_no_wind += 1
                continue

            ir_u8 = irwin_to_uint8(frame["irwin"], frame["irwin_mask"])
            ir_img = Image.fromarray(ir_u8, mode="L").resize((img_size, img_size), Image.BILINEAR)
            has_raw = False
            raw_img = ir_img  # same fallback convention as kaggle_dataset.py

            stamp = f"{meta['year']}{meta['month']}{meta['day']}_{meta['hm']}"
            img_name = f"{storm_label}_{stamp}.jpg"
            ir_img.save(os.path.join(ir_dir, img_name), quality=95)
            raw_img.save(os.path.join(raw_dir, img_name), quality=95)

            knots = frame["wind_kt"]
            kmph = knots_to_kmph(knots)
            cat_idx = wind_speed_to_category(kmph)
            rows.append({
                "img_name": img_name, "storm": storm_label,
                "datetime_utc": f"{meta['year']}-{meta['month']}-{meta['day']}T{meta['hm'][:2]}:{meta['hm'][2:]}",
                "lat": frame["lat"], "lon": frame["lon"], "pressure_mb": frame["pres_mb"],
                "wind_kt": knots, "wind_kmph": kmph, "category": category_label(cat_idx),
                "cat_idx": cat_idx, "has_raw": has_raw, "sat_id": meta["sat_id"], "vza": meta["vza"],
            })
            n_written += 1

        print(f"[hursat_dataset] {storm_label}: {len(nc_files)} nc files "
              f"({n_unparsable} unparsable) -> {len(best)} deduped timestamps -> "
              f"{n_written} written ({n_no_irwin} no usable IRWIN, {n_no_wind} no wind label)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return rows


def build_manifest(raw_dir, out_dir, img_size=128):
    """Process every *.tar.gz in raw_dir and write out_dir/manifest.csv."""
    os.makedirs(out_dir, exist_ok=True)
    tarballs = sorted(f for f in os.listdir(raw_dir) if f.endswith(".tar.gz"))
    if not tarballs:
        print(f"[hursat_dataset] no .tar.gz files found in {raw_dir}")
        return

    all_rows = []
    for fn in tarballs:
        # storm label from the filename, e.g. HURSAT_b1_v06_2008117N11090_NARGIS_c20170721 -> NARGIS
        # (joined with "_" rather than taking a single field, since a few storm names
        # themselves contain an underscore, e.g. ..._HUD_HUD_c20170721 -> HUDHUD not HUD)
        stem = fn.replace(".tar.gz", "")
        parts = stem.split("_")
        storm_label = "".join(parts[4:-1]) if len(parts) > 5 else (parts[4] if len(parts) > 4 else stem)
        rows = process_storm_tarball(os.path.join(raw_dir, fn), out_dir, img_size, storm_label)
        all_rows.extend(rows)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = ["img_name", "storm", "datetime_utc", "lat", "lon", "pressure_mb",
                  "wind_kt", "wind_kmph", "category", "cat_idx", "has_raw", "sat_id", "vza"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    counts = {}
    for r in all_rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"\n[hursat_dataset] TOTAL: {len(all_rows)} labeled frames from {len(tarballs)} storms -> {manifest_path}")
    print(f"[hursat_dataset] category counts: {counts}")


def load_hursat_samples(out_dir):
    """Read manifest.csv back into the same sample-dict shape as
    kaggle_dataset.load_samples(), so the two can be concatenated before
    building a KaggleINSAT3DDataset."""
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
    ap.add_argument("--raw_dir", required=True, help="directory containing the downloaded .tar.gz files")
    ap.add_argument("--out_dir", required=True, help="where to write ir/, raw/, manifest.csv")
    ap.add_argument("--img_size", type=int, default=128)
    args = ap.parse_args()
    build_manifest(args.raw_dir, args.out_dir, args.img_size)
