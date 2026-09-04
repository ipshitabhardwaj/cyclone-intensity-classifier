"""
PyTorch Dataset for CyINSAT-schema data:
  <root>/details.csv       one row per cyclone (name, basin, max intensity, ...)
  <root>/parameters.csv    one row per frame (wind, pressure, lat/lon, image paths)
  <root>/images/<cyclone_id>/<channel>/<frame>.png

Splits are done at the STORM level (not the frame level) so frames from the
same cyclone never appear in both train and val — otherwise the model can
"cheat" by memorizing a storm's cloud texture rather than learning the
wind-speed <-> structure relationship.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils import wind_speed_to_category, NUM_CATEGORIES

# Normalization constants for the regression targets (rough IMD-scale ranges).
WIND_MIN, WIND_MAX = 20.0, 260.0
PRES_MIN, PRES_MAX = 870.0, 1015.0


def storm_train_val_split(details_df: pd.DataFrame, val_frac: float, seed: int):
    ids = details_df["cyclone_id"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


class CycloneDataset(Dataset):
    def __init__(self, root: str, cyclone_ids: set, channels, img_size: int):
        self.root = root
        self.channels = channels
        self.img_size = img_size
        params = pd.read_csv(os.path.join(root, "parameters.csv"), dtype={"cyclone_id": str})
        self.df = params[params["cyclone_id"].isin(cyclone_ids)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                "No frames found for the given cyclone_ids. Check that "
                "parameters.csv exists and cyclone_id values match details.csv."
            )
        self._path_cols = {
            "IR1": "ir1_path", "IR2": "ir2_path", "MIR": "mir_path", "WV": "wv_path",
        }

    def __len__(self):
        return len(self.df)

    def _load_channel(self, path: str) -> np.ndarray:
        img = Image.open(os.path.join(self.root, path)).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arrs = []
        for ch in self.channels:
            col = self._path_cols.get(ch)
            if col is None or col not in row or pd.isna(row[col]):
                raise KeyError(f"No path column for channel '{ch}' in parameters.csv")
            arrs.append(self._load_channel(row[col]))
        x = torch.from_numpy(np.stack(arrs, axis=0))  # (C, H, W)

        wind = float(row["wind_speed_kmph"])
        pressure = float(row["pressure_hpa"])
        cat_idx = int(row["category_idx"]) if "category_idx" in row and not pd.isna(row["category_idx"]) \
            else wind_speed_to_category(wind)

        wind_norm = (wind - WIND_MIN) / (WIND_MAX - WIND_MIN)
        pres_norm = (pressure - PRES_MIN) / (PRES_MAX - PRES_MIN)
        y_reg = torch.tensor([wind_norm, pres_norm], dtype=torch.float32)
        y_cls = torch.tensor(cat_idx, dtype=torch.long)

        meta = {
            "cyclone_id": row["cyclone_id"],
            "frame_idx": int(row["frame_idx"]),
            "wind_speed_kmph": wind,
            "pressure_hpa": pressure,
        }
        return x, y_cls, y_reg, meta


def denormalize_reg(y_reg: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalization above -> (wind_kmph, pressure_hpa)."""
    wind = y_reg[..., 0] * (WIND_MAX - WIND_MIN) + WIND_MIN
    pres = y_reg[..., 1] * (PRES_MAX - PRES_MIN) + PRES_MIN
    return torch.stack([wind, pres], dim=-1)
