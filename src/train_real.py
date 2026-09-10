"""
Trains CycloneNet on the REAL Kaggle INSAT-3D dataset (2-channel: IR + raw,
wind-speed regression only -- no pressure label exists in this dataset).

Usage:
    python src/train_real.py --config configs/config_real.yaml
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_dataset import KaggleINSAT3DDataset, load_samples, random_split, denormalize_wind
from model import CycloneNet
from utils import load_config, set_seed, category_label


def run_epoch(model, loader, optimizer, device, loss_w, train: bool):
    model.train(mode=train)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    total_loss, total_cls_correct, total_n = 0.0, 0, 0
    total_wind_ae = 0.0

    for x, y_cls, y_reg, _y_reg_mask, _meta in loader:
        x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
        y_reg = y_reg[..., :1]  # this dataset (Kaggle-only) has no pressure label; wind only

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_real.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_real] device = {device}")

    root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    samples = load_samples(root)
    train_s, val_s = random_split(samples, cfg["data"]["val_split"], cfg["data"]["seed"])
    print(f"[train_real] {len(samples)} labeled images total (NO storm ids in this dataset "
          f"-> plain random split, not storm-level): {len(train_s)} train / {len(val_s)} val")

    def class_counts(subset):
        counts = {}
        for s in subset:
            counts[s["cat_idx"]] = counts.get(s["cat_idx"], 0) + 1
        return {category_label(k): v for k, v in sorted(counts.items())}

    print(f"[train_real] train class counts: {class_counts(train_s)}")
    print(f"[train_real] val class counts:   {class_counts(val_s)}")
    n_no_raw = sum(1 for s in samples if not s["has_raw"])
    print(f"[train_real] {n_no_raw}/{len(samples)} images had no filename-matched raw "
          f"counterpart (IR image reused for both channels for those)")

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
    print(f"[train_real] done. best val_acc={best_val_acc:.3f}. log -> {tcfg['log_path']}")


if __name__ == "__main__":
    main()
