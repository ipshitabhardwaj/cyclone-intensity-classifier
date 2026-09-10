"""
Historical storm comparison: "this looks like <storm/image> at similar intensity".

Uses the trained CycloneNet as a feature extractor -- the 1280-dim pooled
vector right before the classification head already encodes cloud structure
(eye, eyewall symmetry, banding) since that's what the model learned to
predict intensity from. We embed every image in the dataset once into a
"feature bank", then for any new prediction, retrieve the nearest neighbours
in that embedding space by cosine similarity. This gives a forecaster an
actual reference case ("closest match: img 84.jpg, Very Severe Cyclonic
Storm, 156 km/h") instead of a bare category label -- a form of case-based
reasoning most hackathon submissions skip because it needs a real trained
model and a real dataset to embed, not just a classifier.

Usage:
    bank = build_feature_bank(model, dataset)                # once, cache it
    matches = find_similar(model, query_x, bank, k=5)
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_embedding(model, x):
    """x: (B, C, H, W). Returns the (B, 1280) pooled feature vector used as
    input to the classification head -- no dropout applied (model.eval()
    makes nn.Dropout a no-op anyway, but we skip it explicitly for clarity).
    """
    model.eval()
    feats = model.backbone_features(x)
    pooled = model.avgpool(feats).flatten(1)
    return pooled


@torch.no_grad()
def build_feature_bank(model, dataset, batch_size: int = 16):
    """Embed every sample in `dataset` once. Returns a dict with stacked
    embeddings and parallel metadata lists, ready for repeated similarity
    queries without re-embedding the whole dataset each time.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings, img_names, kmphs, cat_idxs = [], [], [], []
    for x, y_cls, y_reg, _y_reg_mask, meta in loader:
        emb = extract_embedding(model, x)
        embeddings.append(emb)
        img_names.extend(meta["img_name"])
        kmphs.extend(meta["kmph"].tolist())
        cat_idxs.extend(y_cls.tolist())

    return {
        "embeddings": torch.cat(embeddings, dim=0),  # (N, 1280)
        "img_names": img_names,
        "kmphs": kmphs,
        "cat_idxs": cat_idxs,
    }


@torch.no_grad()
def find_similar(model, query_x, bank, k: int = 5, exclude_name: str = None):
    """query_x: (1, C, H, W) or (C, H, W). Returns the k nearest bank entries
    by cosine similarity, each as a dict with img_name/kmph/cat_idx/similarity/
    percentile, sorted most-similar first. Pass exclude_name (the query's own
    filename, if it's already in the bank) so a sample never matches itself.

    Raw cosine similarity is included for debugging, but `percentile` is the
    number worth showing on a dashboard: how this match ranks against every
    other image in the bank (100 = the single most similar image available).
    A backbone with real pretrained ImageNet features tends to produce a more
    spread-out, discriminative embedding space than a randomly-initialized
    one -- which is a GOOD thing, but it also means raw cosine similarity can
    look unimpressive (e.g. 0.3) even when the match is genuinely the best
    one in the dataset. Percentile stays interpretable either way.
    """
    if query_x.dim() == 3:
        query_x = query_x.unsqueeze(0)
    query_emb = extract_embedding(model, query_x)  # (1, 1280)

    sims = F.cosine_similarity(query_emb, bank["embeddings"], dim=1)  # (N,)
    n_total = sims.numel()

    order = torch.argsort(sims, descending=True)
    # rank 0 = most similar -> percentile 100; rank (n_total-1) -> percentile ~0
    results = []
    for rank, idx in enumerate(order.tolist()):
        name = bank["img_names"][idx]
        if exclude_name is not None and name == exclude_name:
            continue
        percentile = 100.0 * (1.0 - rank / max(1, n_total - 1))
        results.append({
            "img_name": name,
            "kmph": bank["kmphs"][idx],
            "cat_idx": bank["cat_idxs"][idx],
            "similarity": float(sims[idx]),
            "percentile": percentile,
        })
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    # Demo: build the feature bank from all real images and show nearest
    # neighbours for a handful of random queries against your actual
    # trained checkpoint.
    #   python src/historical_comparison.py --config configs/config_real.yaml --checkpoint checkpoints_real/best.pt
    import argparse
    import os
    import random
    import sys

    import torch as _torch

    sys.path.insert(0, os.path.dirname(__file__))
    from model import CycloneNet
    from kaggle_dataset import load_samples, KaggleINSAT3DDataset
    from utils import category_label, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_real.yaml")
    ap.add_argument("--checkpoint", default="checkpoints_real/best.pt")
    ap.add_argument("--n", type=int, default=4, help="number of random query images to test")
    ap.add_argument("--k", type=int, default=3, help="number of nearest neighbours to show")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CycloneNet(in_channels=2, num_classes=cfg["model"]["num_classes"],
                        pretrained=False, num_reg_outputs=cfg["model"]["num_reg_outputs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = load_samples(cfg["data"]["root"])
    ds = KaggleINSAT3DDataset(samples, cfg["data"]["img_size"], train=False)

    print(f"Building feature bank over all {len(ds)} images...")
    bank = build_feature_bank(model, ds)
    print(f"Bank built: {tuple(bank['embeddings'].shape)}\n")

    random.seed()
    idxs = random.sample(range(len(ds)), min(args.n, len(ds)))
    for i in idxs:
        x, y_cls, y_reg, _y_reg_mask, meta = ds[i]
        matches = find_similar(model, x, bank, k=args.k, exclude_name=meta["img_name"])
        print(f"Query: {meta['img_name']} -- true={category_label(int(y_cls))} ({meta['kmph']:.0f}kmph)")
        for m in matches:
            print(f"   similar: {m['img_name']:16s} match={m['percentile']:.0f}th percentile "
                  f"(raw cos sim={m['similarity']:.3f})  {category_label(m['cat_idx']):22s} {m['kmph']:.0f}kmph")
        print()
