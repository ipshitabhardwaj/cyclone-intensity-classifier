"""
PyTorch Dataset for the real Kaggle dataset actually being used right now:
  "INSAT3D Infrared & Raw Cyclone Imagery (2012-2021)" by Sshubam Verma
  https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021

Real folder layout (verified against the downloaded archive):
  <root>/insat_3d_ds - Sheet.csv                                  img_name,label(knots)
  <root>/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/<img>.jpg  processed IR view
  <root>/insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/<img>.jpg    raw view (~131/136 match by filename)

Known, disclosed limitations of this dataset (see README):
  - Only 136 labeled images total, all single frames -- no cyclone_id/timestamp,
    so there is no way to group frames by storm. A storm-level split (used for
    the CyINSAT-schema pipeline in dataset.py) is impossible here; this file
    falls back to a plain random split and logs the resulting class counts so
    the imbalance is visible rather than hidden.
  - No pressure label -- only wind speed (in knots). The model trained on this
    data therefore has a 1-output regression head (wind only), not the
    2-output [wind, pressure] head used by the synthetic/CyINSAT pipeline.
  - A handful of "raw" images don't have a filename match for their "ir"
    counterpart (odd names like "59_LUBAN.jpg"); those rows fall back to
    reusing the IR image for both channels, flagged via meta["has_raw"].
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

from utils import knots_to_kmph, kmph_to_category

IR_SUBDIR = os.path.join("insat3d_ir_cyclone_ds", "CYCLONE_DATASET_INFRARED")
RAW_SUBDIR = os.path.join("insat3d_raw_cyclone_ds", "CYCLONE_DATASET_FINAL")
CSV_NAME = "insat_3d_ds - Sheet.csv"

# Normalization range for the wind regression target (km/h). Real data in this
# set ranges ~46-237 km/h (25-128 knots); padded a bit on both sides.
WIND_MIN, WIND_MAX = 40.0, 250.0

# Normalization range for the pressure regression target (mb). Real labeled
# data (HURSAT-B1 + MOSDAC/Amphan, from IBTrACS/best-track matching) ranges
# 920-1010 mb; padded a bit on both sides. The 136 Kaggle images have no
# pressure label at all -- those samples get pressure_mb=None and are masked
# out of the pressure loss/metric (see y_reg_mask below), not dropped.
PRESSURE_MIN, PRESSURE_MAX = 900.0, 1015.0


def load_samples(root: str):
    """Read the CSV and resolve image paths. Returns a list of dict rows."""
    csv_path = os.path.join(root, CSV_NAME)
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        img_name = row["img_name"]
        knots = float(row["label"])
        kmph = knots_to_kmph(knots)
        ir_path = os.path.join(root, IR_SUBDIR, img_name)
        raw_path = os.path.join(root, RAW_SUBDIR, img_name)
        has_raw = os.path.isfile(raw_path)
        if not os.path.isfile(ir_path):
            continue  # shouldn't happen (CSV matches the IR folder 1:1), but be safe
        samples.append({
            "img_name": img_name,
            "ir_path": ir_path,
            "raw_path": raw_path if has_raw else ir_path,
            "has_raw": has_raw,
            "knots": knots,
            "kmph": kmph,
            "cat_idx": kmph_to_category(kmph),
            "pressure_mb": None,  # not labeled in this dataset -- see module docstring
        })
    return samples


def random_split(samples, val_frac: float, seed: int):
    """Plain random split (no storm ids to group by -- see module docstring)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


class KaggleINSAT3DDataset(Dataset):
    """
    train=True enables light augmentation (random h/v flip, 90-degree rotation,
    brightness jitter) -- applied IDENTICALLY to the ir and raw image of a pair
    since they're the same physical scene. This matters a lot at this dataset's
    size: with only ~109 training images and no ImageNet pretraining fallback,
    the model was memorizing individual images (train acc 95%+, val acc noisy
    20-40%) rather than learning storm structure. Cyclones have no fixed
    orientation in these crops, so flips/rotations are physically valid, not
    just generic image-augmentation boilerplate.
    """

    def __init__(self, samples: list, img_size: int, train: bool = False):
        self.samples = samples
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.samples)

    def _load(self, path, aug: dict):
        img = Image.open(path).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))
        if aug["hflip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if aug["vflip"]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if aug["rot"]:
            img = img.rotate(aug["rot"])
        if aug["brightness"] != 1.0:
            img = ImageEnhance.Brightness(img).enhance(aug["brightness"])
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.train:
            aug = {
                "hflip": random.random() < 0.5,
                "vflip": random.random() < 0.5,
                "rot": random.choice([0, 90, 180, 270]),
                "brightness": random.uniform(0.85, 1.15),
            }
        else:
            aug = {"hflip": False, "vflip": False, "rot": 0, "brightness": 1.0}
        ir = self._load(s["ir_path"], aug)
        raw = self._load(s["raw_path"], aug)
        x = torch.from_numpy(np.stack([ir, raw], axis=0))  # (2, H, W)

        wind_norm = (s["kmph"] - WIND_MIN) / (WIND_MAX - WIND_MIN)
        pressure_mb = s.get("pressure_mb")
        has_pressure = pressure_mb is not None
        pressure_norm = (pressure_mb - PRESSURE_MIN) / (PRESSURE_MAX - PRESSURE_MIN) if has_pressure else 0.0
        y_reg = torch.tensor([wind_norm, pressure_norm], dtype=torch.float32)
        y_reg_mask = torch.tensor([1.0, 1.0 if has_pressure else 0.0], dtype=torch.float32)
        y_cls = torch.tensor(s["cat_idx"], dtype=torch.long)

        meta = {
            "img_name": s["img_name"],
            "knots": s["knots"],
            "kmph": s["kmph"],
            # float('nan'), not None -- default_collate can't batch a mix of
            # None and float across samples (Kaggle has no pressure, HURSAT/
            # MOSDAC do); check has_pressure via y_reg_mask[..., 1], not this.
            "pressure_mb": pressure_mb if has_pressure else float("nan"),
            "has_raw": s["has_raw"],
        }
        return x, y_cls, y_reg, y_reg_mask, meta


def denormalize_wind(y_reg: torch.Tensor) -> torch.Tensor:
    return y_reg[..., 0] * (WIND_MAX - WIND_MIN) + WIND_MIN


def denormalize_pressure(y_reg: torch.Tensor) -> torch.Tensor:
    return y_reg[..., 1] * (PRESSURE_MAX - PRESSURE_MIN) + PRESSURE_MIN
