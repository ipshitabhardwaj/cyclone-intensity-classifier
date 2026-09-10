"""
MC-Dropout confidence calibration check: does the dashboard's "80%
confidence" actually mean the prediction is right about 80% of the time?

Uses the SAME storm-held-out validation set as train_combined.py (PHET,
NILOFAR -- 136 frames from 2 storms never seen in training) and the SAME
predict_with_uncertainty() the "Assess a Storm" tab uses, so the numbers
here describe exactly what a user of the dashboard is actually seeing --
not a separately-computed, possibly-rosier calibration metric.

Reports the standard reliability-diagram breakdown (confidence bin -> mean
predicted confidence vs. empirical accuracy in that bin) and the Expected
Calibration Error (ECE): the accuracy-weighted average gap between
confidence and empirical accuracy across bins. ECE=0 is perfect
calibration; there is no universal "good" threshold, but under ~0.05-0.10
is generally considered reasonably calibrated for an 8-way classifier.

Usage:
    python src/calibration_check.py --config configs/config_combined.yaml --checkpoint checkpoints_combined/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_dataset import KaggleINSAT3DDataset
from hursat_dataset import load_hursat_samples
from model import CycloneNet
from uncertainty import predict_with_uncertainty
from train_combined import VAL_STORMS
from utils import load_config


def compute_calibration(model, val_ds, n_mc: int = 30, n_bins: int = 10, device: str = "cpu"):
    """Runs MC-Dropout over every sample in val_ds once. Returns per-bin
    reliability-diagram data plus the overall Expected Calibration Error."""
    confidences, corrects = [], []
    for i in range(len(val_ds)):
        x, y_cls, _y_reg, _y_reg_mask, _meta = val_ds[i]
        result = predict_with_uncertainty(model, x, n_samples=n_mc, device=device)
        confidences.append(result["pred_category_confidence"])
        corrects.append(int(result["pred_category_idx"] == int(y_cls)))
    confidences = np.array(confidences)
    corrects = np.array(corrects)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_conf_mean, bin_acc, bin_count = [], [], []
    ece, n = 0.0, len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & ((confidences < hi) if hi < 1.0 else (confidences <= hi))
        cnt = int(mask.sum())
        if cnt == 0:
            bin_conf_mean.append(None)
            bin_acc.append(None)
            bin_count.append(0)
            continue
        conf_mean, acc = float(confidences[mask].mean()), float(corrects[mask].mean())
        bin_conf_mean.append(conf_mean)
        bin_acc.append(acc)
        bin_count.append(cnt)
        ece += (cnt / n) * abs(acc - conf_mean)

    return {
        "bin_edges": bin_edges.tolist(), "bin_conf_mean": bin_conf_mean,
        "bin_acc": bin_acc, "bin_count": bin_count, "ece": ece, "n_samples": n,
        "overall_acc": float(corrects.mean()), "overall_conf": float(confidences.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_combined.yaml")
    ap.add_argument("--checkpoint", default="checkpoints_combined/best.pt")
    ap.add_argument("--n_mc", type=int, default=30)
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hursat_samples = load_hursat_samples(cfg["data"]["hursat_root"])
    val_s = [s for s in hursat_samples if s["storm"] in VAL_STORMS]
    val_ds = KaggleINSAT3DDataset(val_s, cfg["data"]["img_size"], train=False)

    model = CycloneNet(in_channels=2, num_classes=cfg["model"]["num_classes"], pretrained=False,
                        num_reg_outputs=cfg["model"]["num_reg_outputs"])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    print(f"[calibration_check] {len(val_ds)} held-out images ({sorted(VAL_STORMS)}), "
          f"{args.n_mc} MC-Dropout passes each...")
    r = compute_calibration(model, val_ds, n_mc=args.n_mc, n_bins=args.n_bins)

    print(f"\n[calibration_check] Overall accuracy={r['overall_acc']:.3f}  "
          f"mean confidence={r['overall_conf']:.3f}")
    print(f"[calibration_check] Expected Calibration Error (ECE) = {r['ece']:.3f}")
    print(f"\n{'confidence bin':<16}{'mean conf':<12}{'empirical acc':<16}{'n':<6}")
    for lo, hi, conf, acc, cnt in zip(r["bin_edges"][:-1], r["bin_edges"][1:],
                                      r["bin_conf_mean"], r["bin_acc"], r["bin_count"]):
        if cnt == 0:
            print(f"[{lo:.1f}, {hi:.1f})      --          --              0")
        else:
            print(f"[{lo:.1f}, {hi:.1f})      {conf:.3f}       {acc:.3f}          {cnt}")


if __name__ == "__main__":
    main()
