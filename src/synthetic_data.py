"""
Generates a small synthetic dataset that MIRRORS the CyINSAT schema
(details.csv, parameters.csv, images/<cyclone_id>/<channel>/<frame>.png)
so the rest of the pipeline (dataset loader, model, training, Grad-CAM)
can be built and smoke-tested WITHOUT real MOSDAC/CyINSAT access.

This is NOT real satellite data — it draws a Gaussian "eye/eyewall" blob
whose size, contrast and symmetry scale with a synthetic wind-speed
trajectory, so a CNN trained on it should show falling loss and
non-trivial accuracy, proving the training loop actually learns something.

Swap `data.root` in configs/config.yaml to a real CyINSAT export (same
folder layout) and nothing else needs to change.
"""
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from utils import wind_speed_to_category, load_config

CHANNELS = ["IR1", "IR2", "MIR", "WV"]


def _lifecycle_curve(n_frames: int, peak_kmph: float) -> np.ndarray:
    """A smooth ramp-up/ramp-down wind-speed trajectory peaking at peak_kmph."""
    t = np.linspace(0, 1, n_frames)
    # skewed bump: fast intensification, slower decay (roughly storm-like)
    shape = np.exp(-((t - 0.35) ** 2) / (2 * 0.18 ** 2))
    shape = shape / shape.max()
    noise = np.random.normal(0, 2.5, n_frames)
    curve = 25 + shape * (peak_kmph - 25) + noise
    return np.clip(curve, 20, 260)


def _make_frame(img_size: int, wind_kmph: float, rng: np.random.Generator) -> dict:
    """Render one synthetic 4-channel frame for a given wind speed."""
    cat_idx = wind_speed_to_category(wind_kmph)
    intensity_frac = np.clip((wind_kmph - 20) / (240 - 20), 0, 1)

    yy, xx = np.mgrid[0:img_size, 0:img_size]
    cy, cx = img_size / 2 + rng.normal(0, 3), img_size / 2 + rng.normal(0, 3)

    # eyewall radius shrinks and sharpens as storm strengthens
    radius = img_size * (0.30 - 0.16 * intensity_frac) + rng.normal(0, 1.5)
    sharpness = 1.5 + 3.5 * intensity_frac
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    eyewall = np.exp(-((dist - radius) ** 2) / (2 * sharpness ** 2))

    # a visible "eye" (warm/clear center) only appears at higher intensity
    eye = np.zeros_like(eyewall)
    if intensity_frac > 0.45:
        eye_r = radius * 0.35
        eye = np.exp(-(dist ** 2) / (2 * (eye_r * 0.6) ** 2)) * (intensity_frac - 0.45) * 2

    base_cloud = np.exp(-dist / (img_size * 0.55)) * (0.4 + 0.3 * intensity_frac)

    channels = {}
    for ch in CHANNELS:
        field = base_cloud + eyewall * (0.5 + 0.5 * intensity_frac)
        if ch == "IR1" or ch == "IR2":
            field = field - eye * 0.8  # cold cloud tops bright, warm eye dark
        elif ch == "MIR":
            field = field * 0.8 + rng.normal(0, 0.03, field.shape)
        elif ch == "WV":
            field = field * 0.6 + base_cloud * 0.3  # broader, blurrier signature
        field = field + rng.normal(0, 0.04, field.shape)
        field = np.clip(field, 0, 1)
        channels[ch] = (field * 255).astype(np.uint8)
    return channels, cat_idx


def generate(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    n_storms = cfg["synthetic"]["num_storms"]
    f_min = cfg["synthetic"]["frames_per_storm_min"]
    f_max = cfg["synthetic"]["frames_per_storm_max"]
    seed = cfg["data"]["seed"]

    rng = np.random.default_rng(seed)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    images_root = os.path.join(root, "images")

    details_rows = []
    param_rows = []
    basins = ["BOB", "AS"]  # Bay of Bengal / Arabian Sea, matching North Indian Ocean scope

    for storm_idx in range(n_storms):
        cyclone_id = f"SYN{storm_idx:03d}"
        peak_kmph = rng.uniform(35, 235)
        n_frames = rng.integers(f_min, f_max + 1)
        curve = _lifecycle_curve(n_frames, peak_kmph)

        storm_dir = os.path.join(images_root, cyclone_id)
        for ch in CHANNELS:
            os.makedirs(os.path.join(storm_dir, ch), exist_ok=True)

        lat = rng.uniform(8, 20)
        lon = rng.uniform(75, 90)
        for f_idx, wind in enumerate(curve):
            channels, cat_idx = _make_frame(img_size, wind, rng)
            for ch, arr in channels.items():
                Image.fromarray(arr, mode="L").save(
                    os.path.join(storm_dir, ch, f"{f_idx:03d}.png")
                )
            pressure = 1010 - wind * 0.62 + rng.normal(0, 2)
            lat_t = lat + f_idx * rng.normal(0.05, 0.02)
            lon_t = lon + f_idx * rng.normal(0.08, 0.03)
            param_rows.append({
                "cyclone_id": cyclone_id,
                "frame_idx": f_idx,
                "wind_speed_kmph": round(float(wind), 1),
                "pressure_hpa": round(float(pressure), 1),
                "category_idx": cat_idx,
                "lat": round(float(lat_t), 2),
                "lon": round(float(lon_t), 2),
                "ir1_path": f"images/{cyclone_id}/IR1/{f_idx:03d}.png",
                "ir2_path": f"images/{cyclone_id}/IR2/{f_idx:03d}.png",
                "mir_path": f"images/{cyclone_id}/MIR/{f_idx:03d}.png",
                "wv_path": f"images/{cyclone_id}/WV/{f_idx:03d}.png",
            })

        details_rows.append({
            "cyclone_id": cyclone_id,
            "name": f"SYNTHETIC-{storm_idx:03d}",
            "basin": rng.choice(basins),
            "max_wind_kmph": round(float(curve.max()), 1),
            "min_pressure_hpa": round(float(1010 - curve.max() * 0.62), 1),
            "num_frames": int(n_frames),
        })

    pd.DataFrame(details_rows).to_csv(os.path.join(root, "details.csv"), index=False)
    pd.DataFrame(param_rows).to_csv(os.path.join(root, "parameters.csv"), index=False)
    print(f"Synthetic dataset written to {root}/")
    print(f"  {n_storms} storms, {len(param_rows)} frames total")
    return root


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    generate(cfg_path)
