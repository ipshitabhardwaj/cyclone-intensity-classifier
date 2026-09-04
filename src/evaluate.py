"""
Evaluates a trained checkpoint: confusion matrix, per-class accuracy,
wind/pressure MAE, and a handful of Grad-CAM sample images for the report/demo.

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CycloneDataset, storm_train_val_split, denormalize_reg
from model import CycloneNet
from gradcam import GradCAM, overlay_heatmap
from utils import load_config, IMD_CATEGORIES

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--gradcam_samples", type=int, default=6)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data"]["root"]
    channels = cfg["data"]["channels"]
    img_size = cfg["data"]["img_size"]

    details = pd.read_csv(os.path.join(root, "details.csv"), dtype={"cyclone_id": str})
    _, val_ids = storm_train_val_split(details, cfg["data"]["val_split"], cfg["data"]["seed"])
    val_ds = CycloneDataset(root, val_ids, channels, img_size)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CycloneNet(in_channels=len(channels), num_classes=cfg["model"]["num_classes"],
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")

    all_true, all_pred = [], []
    wind_ae, pres_ae = [], []
    with torch.no_grad():
        for x, y_cls, y_reg, _meta in val_loader:
            x = x.to(device)
            cls_out, reg_out = model(x)
            pred = cls_out.argmax(dim=1).cpu().numpy()
            all_true.extend(y_cls.numpy().tolist())
            all_pred.extend(pred.tolist())

            pred_phys = denormalize_reg(reg_out.cpu())
            true_phys = denormalize_reg(y_reg)
            wind_ae.extend((pred_phys[:, 0] - true_phys[:, 0]).abs().tolist())
            pres_ae.extend((pred_phys[:, 1] - true_phys[:, 1]).abs().tolist())

    labels_present = sorted(set(all_true) | set(all_pred))
    names = [IMD_CATEGORIES[i][1] for i in labels_present]
    print("\n[eval] Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_true, all_pred, labels=labels_present)
    print(pd.DataFrame(cm, index=names, columns=names))
    print("\n[eval] Classification report:")
    print(classification_report(all_true, all_pred, labels=labels_present, target_names=names, zero_division=0))
    print(f"[eval] Wind speed MAE: {np.mean(wind_ae):.2f} km/h")
    print(f"[eval] Pressure MAE:   {np.mean(pres_ae):.2f} hPa")

    # ---- Grad-CAM samples ----
    os.makedirs(os.path.join(args.out_dir, "gradcam"), exist_ok=True)
    cam = GradCAM(model)
    idxs = np.linspace(0, len(val_ds) - 1, min(args.gradcam_samples, len(val_ds)), dtype=int)
    for i in idxs:
        x, y_cls, y_reg, meta = val_ds[i]
        x_in = x.unsqueeze(0).to(device)
        heatmap, pred_idx, probs, reg_pred = cam(x_in)
        base = x[0].numpy()  # IR1 channel as the visual background
        overlay = overlay_heatmap(base, heatmap)
        pred_phys = denormalize_reg(torch.tensor(reg_pred))
        fname = f"{meta['cyclone_id']}_f{meta['frame_idx']:03d}_true-{IMD_CATEGORIES[y_cls.item()][1]}_pred-{IMD_CATEGORIES[pred_idx][1]}.png"
        Image.fromarray(overlay).save(os.path.join(args.out_dir, "gradcam", fname))
        print(f"[eval] Grad-CAM saved: {fname}  "
              f"(true wind={meta['wind_speed_kmph']:.0f}kmph, pred wind={pred_phys[0]:.0f}kmph)")
    cam.remove()
    print(f"\n[eval] Grad-CAM samples written to {args.out_dir}/gradcam/")


if __name__ == "__main__":
    main()
