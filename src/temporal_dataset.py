"""
Sequence dataset for temporal intensity FORECASTING (not classification of a
single frame): given the last SEQ_LEN observed frames of a storm, predict its
wind speed at +6h / +12h / +24h ahead.

Only two of the three real data sources have the timestamps + storm identity
a sequence needs:
  - NOAA HURSAT-B1 (10 storms, ~3-hourly cadence, 680 frames total)
  - MOSDAC / Cyclone Amphan (1 storm, ~30-minute cadence, 216 frames)
The 136-image Kaggle set has no storm id or timestamp at all (see
kaggle_dataset.py's docstring) and is therefore NOT usable here -- it stays
single-frame-classification-only.

This is deliberately built directly off each manifest.csv (rather than
reusing hursat_dataset.load_hursat_samples / mosdac_dataset.load_mosdac_samples,
which drop the datetime_utc column) because the whole point of this module
is the timestamps those loaders throw away.

Validation is the SAME storm-held-out split as train_combined.py (PHET,
NILOFAR) so a held-out number here means the same thing it does for the
classifier -- performance on 2 whole storms never touched during training,
not a random slice of frames that could leak near-duplicate neighbors across
train/val.

Horizon tolerance: rather than hardcoding "3-hourly" or "30-minute", each
storm's own median frame spacing is used to decide how close a candidate
frame must be to exactly t0+h to count as that horizon's label. This lets
HURSAT-B1 (3h cadence) and Amphan (30min cadence) share one pipeline without
one starving the other of valid horizon labels.
"""
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import wind_speed_to_category

SEQ_LEN = 4                      # number of past frames fed to the model
HORIZONS_HOURS = (6, 12, 24)     # forecast lead times
WIND_MIN, WIND_MAX = 40.0, 250.0  # same normalization range as kaggle_dataset.py

VAL_STORMS = {"PHET", "NILOFAR"}  # identical to train_combined.py's held-out set


def _parse_dt(s: str) -> datetime:
    # Handles both "2008-04-25T12:00" (HURSAT) and "2020-05-16T00:30:00" (MOSDAC)
    return datetime.fromisoformat(s)


def _read_manifest(out_dir: str):
    """Read one manifest.csv into per-storm sorted frame lists."""
    manifest_path = os.path.join(out_dir, "manifest.csv")
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ir_path = os.path.join(out_dir, "ir", row["img_name"])
            raw_path = os.path.join(out_dir, "raw", row["img_name"])
            if not os.path.isfile(ir_path):
                continue
            rows.append({
                "storm": row["storm"],
                "dt": _parse_dt(row["datetime_utc"]),
                "ir_path": ir_path,
                "raw_path": raw_path if os.path.isfile(raw_path) else ir_path,
                "has_raw": row["has_raw"] == "True",
                "kmph": float(row["wind_kmph"]),
                "cat_idx": int(row["cat_idx"]),
            })
    return rows


def load_storm_timelines(hursat_root: str, mosdac_root: str):
    """Returns {storm_name: [frame_dict, ...]} sorted by time, pooling both
    real timestamped sources."""
    rows = _read_manifest(hursat_root)
    if os.path.exists(os.path.join(mosdac_root, "manifest.csv")):
        rows += _read_manifest(mosdac_root)

    by_storm = {}
    for r in rows:
        by_storm.setdefault(r["storm"], []).append(r)
    for storm in by_storm:
        by_storm[storm].sort(key=lambda r: r["dt"])
    return by_storm


def _median_cadence_hours(frames):
    if len(frames) < 2:
        return 3.0
    diffs = [
        (frames[i + 1]["dt"] - frames[i]["dt"]).total_seconds() / 3600.0
        for i in range(len(frames) - 1)
    ]
    diffs = [d for d in diffs if d > 0]
    return float(np.median(diffs)) if diffs else 3.0


def build_sequences(by_storm: dict, seq_len: int = SEQ_LEN, horizons=HORIZONS_HOURS):
    """For every storm and every valid anchor frame, build one training
    example: the seq_len frames ending at the anchor, plus a wind-speed
    target for each horizon that has a real frame close enough in time to
    count (others are masked out, not fabricated)."""
    sequences = []
    for storm, frames in by_storm.items():
        if len(frames) < seq_len:
            continue
        cadence = _median_cadence_hours(frames)
        tol_hours = max(cadence * 0.5, 0.25)

        for i in range(seq_len - 1, len(frames)):
            window = frames[i - seq_len + 1: i + 1]
            t0 = window[-1]["dt"]

            targets = {}
            for h in horizons:
                target_dt = t0.timestamp() + h * 3600.0
                best_j, best_diff = None, None
                for j in range(i + 1, len(frames)):
                    diff = abs(frames[j]["dt"].timestamp() - target_dt)
                    if best_diff is None or diff < best_diff:
                        best_diff, best_j = diff, j
                    if frames[j]["dt"].timestamp() - target_dt > tol_hours * 3600.0:
                        break  # frames are sorted; no closer match further ahead
                if best_j is not None and best_diff <= tol_hours * 3600.0:
                    targets[h] = frames[best_j]["kmph"]

            if not targets:
                continue  # no usable supervision at all for this anchor

            sequences.append({
                "storm": storm,
                "window": window,          # list of seq_len frame dicts
                "anchor_kmph": window[-1]["kmph"],
                "anchor_dt": t0,
                "targets": targets,        # {horizon_hours: wind_kmph}
            })
    return sequences


def split_sequences(sequences, val_storms=VAL_STORMS):
    train = [s for s in sequences if s["storm"] not in val_storms]
    val = [s for s in sequences if s["storm"] in val_storms]
    return train, val


class CycloneSequenceDataset(Dataset):
    """Yields (frames_tensor[SEQ_LEN,2,H,W], target_vec[3], target_mask[3], meta)."""

    def __init__(self, sequences, img_size: int = 128, train: bool = False,
                 horizons=HORIZONS_HOURS):
        self.sequences = sequences
        self.img_size = img_size
        self.train = train
        self.horizons = horizons

    def __len__(self):
        return len(self.sequences)

    def _load_img(self, path, aug):
        img = Image.open(path).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))
        if aug["hflip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if aug["vflip"]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        s = self.sequences[idx]
        aug = {"hflip": False, "vflip": False}
        if self.train:
            aug = {"hflip": random.random() < 0.5, "vflip": random.random() < 0.5}

        frames = []
        t_first = s["window"][0]["dt"].timestamp()
        delta_hours = []
        for f in s["window"]:
            ir = self._load_img(f["ir_path"], aug)
            raw = self._load_img(f["raw_path"], aug)
            frames.append(np.stack([ir, raw], axis=0))  # (2,H,W)
            delta_hours.append((f["dt"].timestamp() - t_first) / 3600.0)
        x = torch.from_numpy(np.stack(frames, axis=0))  # (SEQ_LEN,2,H,W)
        # normalized so it stays a small, well-scaled feature regardless of
        # whether the window spans ~1.5h (Amphan, 30min cadence) or ~9h
        # (HURSAT-B1, 3h cadence)
        dt_feat = torch.tensor(delta_hours, dtype=torch.float32) / 24.0

        target_vec = torch.zeros(len(self.horizons), dtype=torch.float32)
        target_mask = torch.zeros(len(self.horizons), dtype=torch.float32)
        for k, h in enumerate(self.horizons):
            if h in s["targets"]:
                target_vec[k] = (s["targets"][h] - WIND_MIN) / (WIND_MAX - WIND_MIN)
                target_mask[k] = 1.0

        meta = {
            "storm": s["storm"],
            "anchor_kmph": s["anchor_kmph"],
            "anchor_dt": s["anchor_dt"].isoformat(),
            "targets_kmph": s["targets"],
        }
        return x, dt_feat, target_vec, target_mask, meta


def denormalize_wind(y: torch.Tensor) -> torch.Tensor:
    return y * (WIND_MAX - WIND_MIN) + WIND_MIN


if __name__ == "__main__":
    # quick sanity check
    by_storm = load_storm_timelines("data/hursat_processed", "data/mosdac_processed")
    print("storms:", {k: len(v) for k, v in by_storm.items()})
    seqs = build_sequences(by_storm)
    train, val = split_sequences(seqs)
    print(f"total sequences: {len(seqs)}  train: {len(train)}  val: {len(val)}")
    for h in HORIZONS_HOURS:
        n = sum(1 for s in seqs if h in s["targets"])
        print(f"  horizon +{h}h: {n} sequences have a label ({100*n/len(seqs):.0f}%)")
