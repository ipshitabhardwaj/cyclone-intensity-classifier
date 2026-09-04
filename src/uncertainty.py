"""
MC-Dropout predictive uncertainty for CycloneNet.

CycloneNet already has a real nn.Dropout(p=0.2) before both heads (see
model.py) -- it exists for training regularization, but it's also exactly the
ingredient MC-Dropout (Gal & Ghahramani, 2016) needs: keep dropout ACTIVE at
inference, run the same input through the network N times, and treat the
spread across those N stochastic passes as the model's uncertainty. No
architecture change and no retraining required -- this runs against the
existing best.pt checkpoint as-is.

Output per prediction:
  - cls_probs_mean / cls_probs_std : per-category probability, mean + std
    across N passes
  - predicted category & its confidence (mean probability of the argmax
    category -- NOT just the single-pass softmax value, which is often
    overconfident)
  - wind_kmph_mean / wind_kmph_std : regression output in the same units the
    dashboard already shows

A high cls_probs_std or wind_kmph_std on a given frame means the network
itself is unsure -- e.g. a storm at a transitional intensity, or an input
that looks unlike anything in the small training set. That's the basis for
the "confidence-scored predictions" differentiator promised in the PPT.
"""
import torch
import torch.nn.functional as F


def enable_mc_dropout(model):
    """Put the whole model in eval() (so BatchNorm uses running stats, not
    batch stats -- essential, since our val batches can be as small as 1)
    EXCEPT any nn.Dropout submodule, which is switched back to train() so it
    keeps sampling a fresh mask on every forward call."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    return model


@torch.no_grad()
def predict_with_uncertainty(model, x, n_samples: int = 30, device: str = "cpu"):
    """x: a single (C, H, W) tensor or a (1, C, H, W) batch of one image.
    Returns a dict of per-category and wind-speed mean/std across n_samples
    stochastic forward passes.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    enable_mc_dropout(model)

    cls_probs, reg_outs = [], []
    for _ in range(n_samples):
        cls_out, reg_out = model(x)
        cls_probs.append(F.softmax(cls_out, dim=1))
        reg_outs.append(reg_out)

    cls_probs = torch.cat(cls_probs, dim=0)   # (N, num_classes)
    reg_outs = torch.cat(reg_outs, dim=0)     # (N, num_reg_outputs)

    mean_probs = cls_probs.mean(dim=0)
    std_probs = cls_probs.std(dim=0)
    pred_idx = int(mean_probs.argmax())

    return {
        "n_samples": n_samples,
        "cls_probs_mean": mean_probs,
        "cls_probs_std": std_probs,
        "pred_category_idx": pred_idx,
        "pred_category_confidence": float(mean_probs[pred_idx]),
        "pred_category_uncertainty": float(std_probs[pred_idx]),
        "reg_mean": reg_outs.mean(dim=0),
        "reg_std": reg_outs.std(dim=0),
    }


if __name__ == "__main__":
    # Demo: run MC-Dropout uncertainty on a handful of real dataset images
    # against your actual trained checkpoint.
    #   python src/uncertainty.py --config configs/config_real.yaml --checkpoint checkpoints_real/best.pt
    import argparse
    import os
    import random
    import sys

    import torch as _torch

    sys.path.insert(0, os.path.dirname(__file__))
    from model import CycloneNet
    from kaggle_dataset import load_samples, KaggleINSAT3DDataset, denormalize_wind
    from utils import category_label, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_real.yaml")
    ap.add_argument("--checkpoint", default="checkpoints_real/best.pt")
    ap.add_argument("--n", type=int, default=6, help="number of random images to test")
    ap.add_argument("--n_samples", type=int, default=30, help="MC-Dropout forward passes per image")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CycloneNet(in_channels=2, num_classes=cfg["model"]["num_classes"],
                        pretrained=False, num_reg_outputs=cfg["model"]["num_reg_outputs"])
    model.load_state_dict(ckpt["model_state"])

    samples = load_samples(cfg["data"]["root"])
    ds = KaggleINSAT3DDataset(samples, cfg["data"]["img_size"], train=False)

    random.seed()
    idxs = random.sample(range(len(ds)), min(args.n, len(ds)))
    print(f"Running {args.n_samples} MC-Dropout passes on {len(idxs)} random images "
          f"from {args.checkpoint}...\n")
    for i in idxs:
        x, y_cls, y_reg, meta = ds[i]
        result = predict_with_uncertainty(model, x, n_samples=args.n_samples)
        pred_cat = category_label(result["pred_category_idx"])
        true_cat = category_label(int(y_cls))
        wind_mean = float(denormalize_wind(result["reg_mean"]))
        wind_std = float(result["reg_std"][0]) * (250.0 - 40.0)
        print(f"{meta['img_name']:20s} true={true_cat:22s} actual_wind={meta['kmph']:.0f}kmph | "
              f"pred={pred_cat:22s} conf={result['pred_category_confidence']*100:.0f}% "
              f"(+/-{result['pred_category_uncertainty']*100:.0f}pp) "
              f"wind_pred={wind_mean:.0f}+/-{wind_std:.0f}kmph")
