"""
Trains the temporal forecasting head (TemporalCycloneNet) to predict wind
speed at +6h / +12h / +24h from the last SEQ_LEN observed frames of a storm.

The model predicts a DELTA (km/h change from the current, last-observed wind
speed), not an absolute value -- see temporal_model.py's docstring for why
an absolute-value model with no explicit "current state" input under-
performed a trivial persistence baseline at short horizons in an earlier
version of this script.

Every result is still reported against that PERSISTENCE BASELINE (predict
"no change from right now" at every horizon), computed on the exact same
held-out sequences. This is the honest bar for a forecasting model: beating
persistence -- especially at +24h -- is what demonstrates the model learned
real intensity trends rather than just echoing the current frame.

Usage:
    python src/train_temporal.py --config configs/config_temporal.yaml
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_dataset import (
    load_storm_timelines, build_sequences, split_sequences,
    CycloneSequenceDataset, denormalize_wind, HORIZONS_HOURS, SEQ_LEN,
    WIND_MIN, WIND_MAX,
)
from temporal_model import TemporalCycloneNet
from utils import load_config, set_seed


def anchor_norm_tensor(meta):
    anchor = torch.tensor(meta["anchor_kmph"], dtype=torch.float32)
    return (anchor - WIND_MIN) / (WIND_MAX - WIND_MIN), anchor


def masked_huber(pred_delta, target_delta, mask, beta=10.0):
    err = torch.nn.functional.smooth_l1_loss(pred_delta, target_delta, beta=beta, reduction="none")
    err = err * mask
    return err.sum() / mask.sum().clamp(min=1.0)


def masked_mae_from_final(pred_final_kmph, target_kmph, mask):
    abs_err = (pred_final_kmph - target_kmph).abs() * mask
    per_h_sum = abs_err.sum(dim=0)
    per_h_n = mask.sum(dim=0).clamp(min=1.0)
    return (per_h_sum / per_h_n).tolist(), mask.sum(dim=0).tolist()


def persistence_baseline_mae(loader):
    """Predict anchor_kmph (current wind speed, i.e. delta=0) for every
    horizon -- the standard 'no model' forecasting baseline."""
    all_pred, all_target, all_mask = [], [], []
    for _x, _dt, target_vec, target_mask, meta in loader:
        _anchor_norm, anchor_kmph = anchor_norm_tensor(meta)
        target_kmph = denormalize_wind(target_vec)
        pred_final = anchor_kmph.unsqueeze(1).repeat(1, target_vec.shape[1])
        all_pred.append(pred_final)
        all_target.append(target_kmph)
        all_mask.append(target_mask)
    return masked_mae_from_final(torch.cat(all_pred), torch.cat(all_target), torch.cat(all_mask))


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_n = 0.0, 0
    all_pred_final, all_target_kmph, all_mask = [], [], []

    for x, dt_feat, target_vec, target_mask, meta in loader:
        x, dt_feat = x.to(device), dt_feat.to(device)
        target_vec, target_mask = target_vec.to(device), target_mask.to(device)
        anchor_norm, anchor_kmph = anchor_norm_tensor(meta)
        anchor_norm, anchor_kmph = anchor_norm.to(device), anchor_kmph.to(device)

        target_kmph = denormalize_wind(target_vec)
        target_delta = target_kmph - anchor_kmph.unsqueeze(1)

        with torch.set_grad_enabled(train):
            pred_delta = model(x, dt_feat, anchor_norm)
            loss = masked_huber(pred_delta, target_delta, target_mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        pred_final = anchor_kmph.unsqueeze(1) + pred_delta.detach()
        all_pred_final.append(pred_final.cpu())
        all_target_kmph.append(target_kmph.cpu())
        all_mask.append(target_mask.cpu())

    mae_list, n_list = masked_mae_from_final(
        torch.cat(all_pred_final), torch.cat(all_target_kmph), torch.cat(all_mask))
    return {"loss": total_loss / total_n, "mae_per_h": mae_list, "n_per_h": n_list}


def collate_with_meta(batch):
    xs, dts, tvs, tms, metas = zip(*batch)
    x = torch.stack(xs)
    dt = torch.stack(dts)
    tv = torch.stack(tvs)
    tm = torch.stack(tms)
    meta = {
        "storm": [m["storm"] for m in metas],
        "anchor_kmph": [m["anchor_kmph"] for m in metas],
    }
    return x, dt, tv, tm, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_temporal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_temporal] device = {device}")

    by_storm = load_storm_timelines(cfg["data"]["hursat_root"], cfg["data"]["mosdac_root"])
    sequences = build_sequences(by_storm, seq_len=SEQ_LEN, horizons=HORIZONS_HOURS)
    train_seqs, val_seqs = split_sequences(sequences)
    print(f"[train_temporal] {len(by_storm)} storms, {len(sequences)} total sequences "
          f"-> train {len(train_seqs)} / val {len(val_seqs)} "
          f"(held out entirely: {sorted(set(s['storm'] for s in val_seqs))})")

    train_ds = CycloneSequenceDataset(train_seqs, cfg["data"]["img_size"], train=True)
    val_ds = CycloneSequenceDataset(val_seqs, cfg["data"]["img_size"], train=False)

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=tcfg["num_workers"], collate_fn=collate_with_meta)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=tcfg["num_workers"], collate_fn=collate_with_meta)

    baseline_mae, baseline_n = persistence_baseline_mae(val_loader)
    print(f"[train_temporal] PERSISTENCE BASELINE (predict no change) on held-out storms:")
    for h, mae, n in zip(HORIZONS_HOURS, baseline_mae, baseline_n):
        print(f"    +{h:>2}h: MAE={mae:6.2f} km/h  (n={int(n)})")

    model = TemporalCycloneNet(
        backbone_ckpt=cfg["model"]["backbone_ckpt"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        gru_hidden=cfg["model"]["gru_hidden"],
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(tcfg["log_path"]), exist_ok=True)

    best_val_mae_avg = float("inf")
    log_rows = []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        dt = time.time() - t0

        val_mae_avg = float(np.mean(val_metrics["mae_per_h"]))
        mae_str = " ".join(f"+{h}h={m:.1f}" for h, m in zip(HORIZONS_HOURS, val_metrics["mae_per_h"]))
        print(f"[epoch {epoch:02d}/{tcfg['epochs']}] train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_mae(kmph): {mae_str} avg={val_mae_avg:.2f} ({dt:.1f}s)")

        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]}
        for h, m in zip(HORIZONS_HOURS, val_metrics["mae_per_h"]):
            row[f"val_mae_{h}h"] = m
        log_rows.append(row)

        if val_mae_avg < best_val_mae_avg:
            best_val_mae_avg = val_mae_avg
            ckpt_path = os.path.join(tcfg["checkpoint_dir"], "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_mae_per_h": val_metrics["mae_per_h"],
                "val_mae_avg": best_val_mae_avg,
                "baseline_mae_per_h": baseline_mae,
                "horizons_hours": HORIZONS_HOURS,
            }, ckpt_path)
            print(f"  -> new best (val_mae_avg={best_val_mae_avg:.2f}), saved to {ckpt_path}")

    with open(tcfg["log_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[train_temporal] DONE. best val_mae_avg={best_val_mae_avg:.2f} km/h")
    print(f"[train_temporal] FINAL COMPARISON vs persistence baseline (held-out storms):")
    ckpt = torch.load(os.path.join(tcfg["checkpoint_dir"], "best.pt"), map_location="cpu", weights_only=False)
    for h, model_mae, base_mae in zip(HORIZONS_HOURS, ckpt["val_mae_per_h"], baseline_mae):
        verdict = "BEATS baseline" if model_mae < base_mae else "does NOT beat baseline"
        print(f"    +{h:>2}h:  model={model_mae:6.2f} km/h   baseline={base_mae:6.2f} km/h   -> {verdict}")


if __name__ == "__main__":
    main()
