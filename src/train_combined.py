"""
Trains CycloneNet on the combined real-data pool:
  - 136 single-frame images, "INSAT3D Infrared & Raw Cyclone Imagery" (Kaggle,
    src/kaggle_dataset.py) -- no storm id, no pressure label.
  - 680 frames across 10 real North Indian Ocean cyclones (2008-2016) from
    NOAA HURSAT-B1 (src/hursat_dataset.py), chosen specifically to fill in
    the Depression/Deep Depression/Cyclonic Storm categories the Kaggle set
    was thin on -- see src/hursat_dataset.py's module docstring.
  - 216 frames of Cyclone Amphan (2020), a real MOSDAC order matched against
    its official IMD best track (src/mosdac_dataset.py). Amphan reached
    Super Cyclonic Storm -- the one category neither of the other two
    sources has more than a single example of -- so this source matters
    more than its frame count suggests.

Validation is a STORM-HELD-OUT split on the HURSAT-B1 side only (PHET and
NILOFAR -- 136 frames, ~2 whole storms never seen during training), not a
random split. A random split over 3-hourly frames of the same storm would
put near-duplicate frames (taken 3 hours apart, still showing nearly the
same cloud structure) on both sides -- val accuracy would then partly be
memorization, not generalization to a storm the model hasn't seen. Both the
136 Kaggle images and all 216 Amphan frames go into training rather than a
held-out set: Kaggle never had a storm id to hold out by, and Amphan is the
project's only Super Cyclonic Storm source at real volume -- holding it out
would starve training of the category it's needed to fix, for a validation
signal that already exists from HURSAT-B1.

Usage:
    python src/train_combined.py --config configs/config_combined.yaml
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_dataset import KaggleINSAT3DDataset, load_samples as load_kaggle_samples, denormalize_wind
from hursat_dataset import load_hursat_samples
from mosdac_dataset import load_mosdac_samples
from model import CycloneNet
from utils import load_config, set_seed, category_label

# Held out ENTIRELY from training -- chosen to cover all 7 categories present
# in the HURSAT-B1 pool between them (PHET: LPA/D/DD/CS/SCS/VSCS/ESCS,
# NILOFAR: same set) so val accuracy isn't measured on an easy subset.
VAL_STORMS = {"PHET", "NILOFAR"}


def run_epoch(model, loader, optimizer, device, loss_w, train: bool):
    model.train(mode=train)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    total_loss, total_cls_correct, total_n = 0.0, 0, 0
    total_wind_ae = 0.0

    for x, y_cls, y_reg, _meta in loader:
        x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)

        with torch.set_grad_enabled(train):
            cls_out, reg_out = model(x)
            loss_cls = ce(cls_out, y_cls)
            loss_reg = mse(reg_out, y_reg)
            loss = loss_w["cls"] * loss_cls + loss_w["reg"] * loss_reg

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls_correct += (cls_out.argmax(dim=1) == y_cls).sum().item()
        total_n += bs

        pred_wind = denormalize_wind(reg_out.detach().cpu())
        true_wind = denormalize_wind(y_reg.detach().cpu())
        total_wind_ae += (pred_wind - true_wind).abs().sum().item()

    return {
        "loss": total_loss / total_n,
        "acc": total_cls_correct / total_n,
        "wind_mae_kmph": total_wind_ae / total_n,
    }


def class_counts(subset):
    counts = {}
    for s in subset:
        counts[s["cat_idx"]] = counts.get(s["cat_idx"], 0) + 1
    return {category_label(k): v for k, v in sorted(counts.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_combined.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_combined] device = {device}")

    img_size = cfg["data"]["img_size"]

    kaggle_samples = load_kaggle_samples(cfg["data"]["kaggle_root"])
    hursat_samples = load_hursat_samples(cfg["data"]["hursat_root"])
    mosdac_samples = load_mosdac_samples(cfg["data"]["mosdac_root"]) if os.path.exists(
        os.path.join(cfg["data"]["mosdac_root"], "manifest.csv")) else []

    hursat_train = [s for s in hursat_samples if s["storm"] not in VAL_STORMS]
    hursat_val = [s for s in hursat_samples if s["storm"] in VAL_STORMS]

    train_s = kaggle_samples + hursat_train + mosdac_samples
    val_s = hursat_val

    print(f"[train_combined] {len(kaggle_samples)} Kaggle images + "
          f"{len(hursat_samples)} HURSAT-B1 frames + {len(mosdac_samples)} MOSDAC/Amphan frames "
          f"({len(hursat_samples) - len(hursat_val)} HURSAT-B1 storms' worth of frames in train, "
          f"{sorted(VAL_STORMS)} held out entirely for val)")
    print(f"[train_combined] train: {len(train_s)} samples, val: {len(val_s)} samples")
    print(f"[train_combined] train class counts: {class_counts(train_s)}")
    print(f"[train_combined] val class counts:   {class_counts(val_s)}")

    train_ds = KaggleINSAT3DDataset(train_s, img_size, train=True)
    val_ds = KaggleINSAT3DDataset(val_s, img_size, train=False)

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=tcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"])

    model = CycloneNet(in_channels=2, num_classes=cfg["model"]["num_classes"],
                        pretrained=cfg["model"]["pretrained"],
                        num_reg_outputs=cfg["model"]["num_reg_outputs"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(tcfg["log_path"]), exist_ok=True)

    best_val_acc = -1.0
    log_rows = []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, tcfg["loss_weights"], train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, tcfg["loss_weights"], train=False)
        dt = time.time() - t0

        print(f"[epoch {epoch:02d}/{tcfg['epochs']}] "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
              f"val_wind_mae={val_metrics['wind_mae_kmph']:.1f}kmph "
              f"({dt:.1f}s)")

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_rows.append(row)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            ckpt_path = os.path.join(tcfg["checkpoint_dir"], "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_acc": best_val_acc,
            }, ckpt_path)
            print(f"  -> new best (val_acc={best_val_acc:.3f}), saved to {ckpt_path}")

    with open(tcfg["log_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train_combined] done. best val_acc={best_val_acc:.3f}. log -> {tcfg['log_path']}")


if __name__ == "__main__":
    main()
