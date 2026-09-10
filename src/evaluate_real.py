"""
Evaluates a checkpoint trained by train_real.py on the real Kaggle dataset:
confusion matrix, per-class report, wind MAE, and Grad-CAM sample images.

Usage:
    python src/evaluate_real.py --config configs/config_real.yaml --checkpoint checkpoints_real/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_dataset import KaggleINSAT3DDataset, load_samples, random_split, denormalize_wind
from model import CycloneNet
from gradcam import GradCAM, overlay_heatmap
from utils import load_config, category_label

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_real.yaml")
    ap.add_argument("--checkpoint", default="checkpoints_real/best.pt")
    ap.add_argument("--gradcam_samples", type=int, default=6)
    ap.add_argument("--out_dir", default="outputs_real")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    samples = load_samples(root)
    _, val_s = random_split(samples, cfg["data"]["val_split"], cfg["data"]["seed"])
    val_ds = KaggleINSAT3DDataset(val_s, img_size)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CycloneNet(in_channels=2, num_classes=cfg["model"]["num_classes"], pretrained=False,
                        num_reg_outputs=cfg["model"]["num_reg_outputs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval_real] loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")
    print(f"[eval_real] val set size: {len(val_ds)} images (tiny -- read metrics as indicative, not final)")

    all_true, all_pred, wind_ae = [], [], []
    with torch.no_grad():
        for x, y_cls, y_reg, _y_reg_mask, _meta in val_loader:
            x = x.to(device)
            cls_out, reg_out = model(x)
            pred = cls_out.argmax(dim=1).cpu().numpy()
            all_true.extend(y_cls.numpy().tolist())
            all_pred.extend(pred.tolist())

            pred_wind = denormalize_wind(reg_out.cpu())
            true_wind = denormalize_wind(y_reg[..., :1])
            wind_ae.extend((pred_wind - true_wind).abs().tolist())

    labels_present = sorted(set(all_true) | set(all_pred))
    names = [category_label(i) for i in labels_present]
    print("\n[eval_real] Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_true, all_pred, labels=labels_present)
    import pandas as pd
    print(pd.DataFrame(cm, index=names, columns=names))
    print("\n[eval_real] Classification report:")
    print(classification_report(all_true, all_pred, labels=labels_present, target_names=names, zero_division=0))
    print(f"[eval_real] Wind speed MAE: {np.mean(wind_ae):.2f} km/h")

    os.makedirs(os.path.join(args.out_dir, "gradcam"), exist_ok=True)
    cam = GradCAM(model)
    idxs = np.linspace(0, len(val_ds) - 1, min(args.gradcam_samples, len(val_ds)), dtype=int)
    for i in idxs:
        x, y_cls, y_reg, _y_reg_mask, meta = val_ds[int(i)]
        x_in = x.unsqueeze(0).to(device)
        heatmap, pred_idx, probs, reg_pred = cam(x_in)
        base = x[0].numpy()  # IR channel as the visual background
        overlay = overlay_heatmap(base, heatmap)
        pred_wind = denormalize_wind(torch.tensor(reg_pred))
        safe_name = meta["img_name"].replace("/", "_")
        fname = f"{safe_name}_true-{category_label(y_cls.item())}_pred-{category_label(pred_idx)}.png"
        Image.fromarray(overlay).save(os.path.join(args.out_dir, "gradcam", fname))
        print(f"[eval_real] Grad-CAM saved: {fname} "
              f"(true wind={meta['kmph']:.0f}kmph, pred wind={float(pred_wind):.0f}kmph)")
    cam.remove()
    print(f"\n[eval_real] Grad-CAM samples written to {args.out_dir}/gradcam/")


if __name__ == "__main__":
    main()
