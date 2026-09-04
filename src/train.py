"""
Trains CycloneNet on CyINSAT-schema data (real or synthetic).

Usage:
    python src/train.py --config configs/config.yaml
"""
import argparse
import csv
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CycloneDataset, storm_train_val_split, denormalize_reg
from model import CycloneNet
from utils import load_config, set_seed, NUM_CATEGORIES


def run_epoch(model, loader, optimizer, device, loss_w, train: bool):
    model.train(mode=train)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    total_loss, total_cls_correct, total_n = 0.0, 0, 0
    total_wind_ae, total_pres_ae = 0.0, 0.0

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

        pred_phys = denormalize_reg(reg_out.detach().cpu())
        true_phys = denormalize_reg(y_reg.detach().cpu())
        total_wind_ae += (pred_phys[:, 0] - true_phys[:, 0]).abs().sum().item()
        total_pres_ae += (pred_phys[:, 1] - true_phys[:, 1]).abs().sum().item()

    return {
        "loss": total_loss / total_n,
        "acc": total_cls_correct / total_n,
        "wind_mae_kmph": total_wind_ae / total_n,
        "pressure_mae_hpa": total_pres_ae / total_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    root = cfg["data"]["root"]
    channels = cfg["data"]["channels"]
    img_size = cfg["data"]["img_size"]

    details = pd.read_csv(os.path.join(root, "details.csv"), dtype={"cyclone_id": str})
    train_ids, val_ids = storm_train_val_split(details, cfg["data"]["val_split"], cfg["data"]["seed"])
    print(f"[train] {len(train_ids)} storms train / {len(val_ids)} storms val (storm-level split)")

    train_ds = CycloneDataset(root, train_ids, channels, img_size)
    val_ds = CycloneDataset(root, val_ids, channels, img_size)
    print(f"[train] {len(train_ds)} train frames / {len(val_ds)} val frames")

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=tcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"])

    model = CycloneNet(in_channels=len(channels), num_classes=cfg["model"]["num_classes"],
                        pretrained=cfg["model"]["pretrained"]).to(device)
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
              f"val_pressure_mae={val_metrics['pressure_mae_hpa']:.1f}hpa "
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
    print(f"[train] done. best val_acc={best_val_acc:.3f}. log -> {tcfg['log_path']}")


if __name__ == "__main__":
    main()
