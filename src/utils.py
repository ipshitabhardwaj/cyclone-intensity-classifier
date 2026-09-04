"""Shared utilities: config loading, IMD intensity-category mapping, seeding.

torch is imported lazily (inside set_seed) rather than at module level, so
lightweight preprocessing scripts (hursat_dataset.py, mosdac_dataset.py) that
only need the plain category/conversion helpers below can import this module
on machines that don't have torch installed at all.
"""
import random
import yaml
import numpy as np

# IMD tropical cyclone intensity scale (approximate standard thresholds, km/h sustained wind).
# (label, short_code, lower_bound_kmph_inclusive)
IMD_CATEGORIES = [
    ("Low Pressure Area",              "LPA",  0),
    ("Depression",                     "D",    31),
    ("Deep Depression",                "DD",   50),
    ("Cyclonic Storm",                 "CS",   62),
    ("Severe Cyclonic Storm",          "SCS",  89),
    ("Very Severe Cyclonic Storm",     "VSCS", 118),
    ("Extremely Severe Cyclonic Storm","ESCS", 167),
    ("Super Cyclonic Storm",           "SuCS", 222),
]
NUM_CATEGORIES = len(IMD_CATEGORIES)


KNOTS_TO_KMPH = 1.852


def knots_to_kmph(knots: float) -> float:
    return knots * KNOTS_TO_KMPH


def wind_speed_to_category(wind_kmph: float) -> int:
    """Map a sustained wind speed (km/h) to an IMD category index (0..7)."""
    idx = 0
    for i, (_, _, lower) in enumerate(IMD_CATEGORIES):
        if wind_kmph >= lower:
            idx = i
        else:
            break
    return idx


def category_label(idx: int) -> str:
    return IMD_CATEGORIES[idx][0]


# alias -- same mapping, name matches how kaggle_dataset.py calls it
kmph_to_category = wind_speed_to_category


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
