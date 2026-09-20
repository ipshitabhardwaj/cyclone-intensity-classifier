"""
Cyclone AI — Meteorological Intelligence & Forecasting Platform.
Completely rebuilt soft-white modern web application frontend.
All PyTorch backend models, checkpoints, predictions, and calculations remain 100% intact and frozen.
"""
import io
import os
import sys
import json
import base64

import numpy as np
import pandas as pd
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Insert src directory into python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from kaggle_dataset import (
    load_samples as load_kaggle_samples, denormalize_wind, denormalize_pressure,
    PRESSURE_MIN, PRESSURE_MAX, KaggleINSAT3DDataset
)
from hursat_dataset import load_hursat_samples
from mosdac_dataset import load_mosdac_samples
from model import CycloneNet
from gradcam import GradCAM, overlay_heatmap
from uncertainty import predict_with_uncertainty
from historical_comparison import build_feature_bank, find_similar
from ri_alert import load_series, find_ri_events, merge_overlapping, RI_THRESHOLD_KT, RI_WINDOW_HOURS
from utils import load_config, category_label, IMD_CATEGORIES, wind_speed_to_category
from temporal_dataset import (
    load_storm_timelines, build_sequences, VAL_STORMS, HORIZONS_HOURS,
    WIND_MIN as T_WIND_MIN, WIND_MAX as T_WIND_MAX,
)
from temporal_model import TemporalCycloneNet

CONFIG_PATH = "configs/config_combined.yaml"
CHECKPOINT_PATH = "checkpoints_combined/best.pt"
CONFIG_TEMPORAL_PATH = "configs/config_temporal.yaml"
CHECKPOINT_TEMPORAL_PATH = "checkpoints_temporal/best.pt"
BESTTRACK_PATH = "data/besttrack/amphan_2020.csv"

st.set_page_config(
    page_title="CycloneNet — Meteorological Intelligence Platform",
    layout="wide",
    page_icon="🌀",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit chrome and sidebar completely
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important; width: 0 !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    button[aria-label="Toggle sidebar"] {display: none !important;}
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    .stApp {
        background-color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_everything():
    cfg = load_config(CONFIG_PATH)
    device = torch.device("cpu")
    model = CycloneNet(
        in_channels=2,
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        num_reg_outputs=cfg["model"]["num_reg_outputs"],
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    kaggle_samples = load_kaggle_samples(cfg["data"]["kaggle_root"])
    hursat_samples = load_hursat_samples(cfg["data"]["hursat_root"])
    mosdac_root = cfg["data"].get("mosdac_root")
    mosdac_samples = (
        load_mosdac_samples(mosdac_root)
        if mosdac_root and os.path.exists(os.path.join(mosdac_root, "manifest.csv"))
        else []
    )
    samples = kaggle_samples + hursat_samples + mosdac_samples
    return cfg, model, samples, ckpt, len(kaggle_samples), len(hursat_samples), len(mosdac_samples)


@st.cache_resource
def load_feature_bank(_model, _samples, img_size):
    ds = KaggleINSAT3DDataset(_samples, img_size, train=False)
    return build_feature_bank(_model, ds)


@st.cache_data
def load_ri_case_study():
    times, winds = load_series(BESTTRACK_PATH)
    events = find_ri_events(times, winds)
    episodes = merge_overlapping(events)
    return times, winds, episodes


@st.cache_resource
def load_temporal_model():
    tcfg = load_config(CONFIG_TEMPORAL_PATH)
    model_t = TemporalCycloneNet(
        backbone_ckpt=tcfg["model"]["backbone_ckpt"],
        freeze_backbone=tcfg["model"]["freeze_backbone"],
        gru_hidden=tcfg["model"]["gru_hidden"],
    )
    tckpt = torch.load(CHECKPOINT_TEMPORAL_PATH, map_location="cpu", weights_only=False)
    model_t.load_state_dict(tckpt["model_state"])
    model_t.eval()
    return tcfg, model_t, tckpt


@st.cache_data
def load_temporal_sequences(_hursat_root, _mosdac_root):
    by_storm = load_storm_timelines(_hursat_root, _mosdac_root)
    sequences = build_sequences(by_storm)
    return by_storm, sequences


def load_gray(path, img_size):
    img = Image.open(path).convert("L")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def img_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_frame_pair(frame, size):
    ir = Image.open(frame["ir_path"]).convert("L")
    raw = Image.open(frame["raw_path"]).convert("L")
    if ir.size != (size, size):
        ir = ir.resize((size, size))
    if raw.size != (size, size):
        raw = raw.resize((size, size))
    ir_arr = np.asarray(ir, dtype=np.float32) / 255.0
    raw_arr = np.asarray(raw, dtype=np.float32) / 255.0
    return np.stack([ir_arr, raw_arr], axis=0)


def run_forecast(seq, model_t, img_size):
    frames = [load_frame_pair(f, img_size) for f in seq["window"]]
    x = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).float()
    t_first = seq["window"][0]["dt"].timestamp()
    dt_feat = torch.tensor(
        [[(f["dt"].timestamp() - t_first) / 3600.0 / 24.0 for f in seq["window"]]],
        dtype=torch.float32,
    )
    anchor_norm = torch.tensor([(seq["anchor_kmph"] - T_WIND_MIN) / (T_WIND_MAX - T_WIND_MIN)],
                                dtype=torch.float32)
    with torch.no_grad():
        delta_kmph = model_t(x, dt_feat, anchor_norm)
    preds = {}
    for k, h in enumerate(HORIZONS_HOURS):
        preds[h] = seq["anchor_kmph"] + float(delta_kmph[0, k])
    return preds


# Load backend components
cfg, model, samples, ckpt, n_kaggle, n_hursat, n_mosdac = load_everything()
img_size = cfg["data"]["img_size"]
bank = load_feature_bank(model, samples, img_size)
tcfg, model_t, tckpt = load_temporal_model()
by_storm, sequences = load_temporal_sequences(tcfg["data"]["hursat_root"], tcfg["data"]["mosdac_root"])
ri_times, ri_winds, ri_episodes = load_ri_case_study()

# Build serialized JSON cache for instant, zero-latency Web App performance
@st.cache_data
def build_app_data():
    samples_data = {}
    cat_names = [c[1] for c in IMD_CATEGORIES]

    # Precompute sample assessments & historical matches for first 25 samples
    for i, s in enumerate(samples[:25]):
        ir_arr = load_gray(s["ir_path"], img_size)
        raw_arr = load_gray(s["raw_path"], img_size)
        x = torch.from_numpy(np.stack([ir_arr, raw_arr], axis=0)).unsqueeze(0)

        cam = GradCAM(model)
        heatmap, pred_idx, probs, reg_pred = cam(x)
        cam.remove()
        reg_pred_t = torch.tensor(reg_pred)
        pred_kmph = float(denormalize_wind(reg_pred_t))
        pred_pressure = float(denormalize_pressure(reg_pred_t)) if reg_pred_t.shape[-1] > 1 else None
        overlay = overlay_heatmap(ir_arr, heatmap)

        unc = predict_with_uncertainty(model, x, n_samples=30)
        conf = float(unc["cls_probs_mean"][pred_idx].item())
        wind_std_kmph = float(unc["reg_std"][0]) * (250.0 - 40.0)
        pressure_std_mb = (
            float(unc["reg_std"][1]) * (PRESSURE_MAX - PRESSURE_MIN)
            if pred_pressure is not None and len(unc["reg_std"]) > 1 else None
        )

        ir_pil = Image.open(s["ir_path"])
        raw_pil = Image.open(s["raw_path"])
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

        # Historical matches
        matches = find_similar(model, x, bank, k=4, exclude_name=s["img_name"])
        matches_res = []
        for m in matches:
            m_sample = next(samp for samp in samples if samp["img_name"] == m["img_name"])
            m_pil = Image.open(m_sample["ir_path"])
            matches_res.append({
                "img_name": m["img_name"],
                "percentile": round(m["percentile"], 1),
                "cat_idx": m["cat_idx"],
                "cat_name": category_label(m["cat_idx"]),
                "kmph": round(m["kmph"], 1),
                "ir_base64": img_to_base64(m_pil),
            })

        samples_data[s["img_name"]] = {
            "img_name": s["img_name"],
            "ir_base64": img_to_base64(ir_pil),
            "raw_base64": img_to_base64(raw_pil),
            "overlay_base64": img_to_base64(overlay_pil),
            "pred_idx": pred_idx,
            "pred_cat_name": category_label(pred_idx),
            "pred_kmph": round(pred_kmph, 1),
            "wind_std_kmph": round(wind_std_kmph, 1),
            "pred_pressure": round(pred_pressure, 1) if pred_pressure is not None else None,
            "pressure_std_mb": round(pressure_std_mb, 1) if pressure_std_mb is not None else None,
            "confidence": round(conf, 4),
            "true_cat_name": category_label(s["cat_idx"]),
            "true_kmph": round(s["kmph"], 1),
            "true_pressure": s.get("pressure_mb"),
            "mean_probs": [round(p, 4) for p in unc["cls_probs_mean"].tolist()],
            "std_probs": [round(p, 4) for p in unc["cls_probs_std"].tolist()],
            "matches": matches_res,
        }

    # Forecast Sequences Data
    storms_with_seqs = sorted(set(s["storm"] for s in sequences))
    forecast_data = {}
    for st_name in storms_with_seqs:
        storm_seqs = [s for s in sequences if s["storm"] == st_name]
        storm_seqs.sort(key=lambda s: s["anchor_dt"])
        
        full_times = [f["dt"].strftime("%Y-%m-%d %H:%M") for f in by_storm[st_name]]
        full_winds = [round(f["kmph"], 1) for f in by_storm[st_name]]

        seq_list = []
        for idx, seq in enumerate(storm_seqs):
            preds = run_forecast(seq, model_t, img_size)
            fc_times = [seq["anchor_dt"].strftime("%Y-%m-%d %H:%M")] + [
                (seq["anchor_dt"] + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
                for h in HORIZONS_HOURS
            ]
            fc_winds = [round(seq["anchor_kmph"], 1)] + [round(preds[h], 1) for h in HORIZONS_HOURS]

            rt_times = [seq["anchor_dt"]] + [seq["anchor_dt"] + pd.Timedelta(hours=h) for h in HORIZONS_HOURS]
            rt_winds_kt = [w / 1.852 for w in fc_winds]
            rt_events = find_ri_events(rt_times, rt_winds_kt)
            rt_episodes = merge_overlapping(rt_events)
            rt_episodes_res = []
            if rt_episodes:
                for start, end, w0, w1, delta in rt_episodes:
                    rt_episodes_res.append({
                        "delta_kt": round(delta, 1),
                        "duration_hours": round((end - start).total_seconds() / 3600.0, 1),
                    })

            thumbs = []
            for f in seq["window"]:
                img_p = Image.open(f["ir_path"])
                thumbs.append({
                    "dt": f["dt"].strftime("%d %b, %H:%M"),
                    "ir_base64": img_to_base64(img_p),
                })

            seq_list.append({
                "anchor_idx": idx,
                "anchor_dt": seq["anchor_dt"].strftime("%d %b %Y, %H:%M UTC"),
                "anchor_kmph": round(seq["anchor_kmph"], 1),
                "anchor_cat": category_label(wind_speed_to_category(seq["anchor_kmph"])),
                "horizons": {
                    str(h): {
                        "pred_kmph": round(preds[h], 1),
                        "pred_cat": category_label(wind_speed_to_category(preds[h])),
                        "delta_kmph": round(preds[h] - seq["anchor_kmph"], 1),
                        "actual_kmph": round(seq["targets"][h], 1) if h in seq["targets"] else None
                    } for h in HORIZONS_HOURS
                },
                "fc_times": fc_times,
                "fc_winds": fc_winds,
                "rt_episodes": rt_episodes_res,
                "thumbnails": thumbs,
            })

        forecast_data[st_name] = {
            "full_times": full_times,
            "full_winds": full_winds,
            "is_held_out": st_name in VAL_STORMS,
            "sequences": seq_list,
        }

    # RI Alert Data
    ri_episodes_res = []
    if ri_episodes:
        for start, end, w0, w1, delta in ri_episodes:
            ri_episodes_res.append({
                "start": start.strftime("%d %b %Y, %H:%M UTC"),
                "end": end.strftime("%d %b %Y, %H:%M UTC"),
                "delta": round(delta, 1),
                "duration_hours": round((end - start).total_seconds() / 3600.0, 1),
            })

    # Composition & Training history
    counts = {}
    for s in samples:
        counts[s["cat_idx"]] = counts.get(s["cat_idx"], 0) + 1

    cat_composition = []
    for idx, item in enumerate(IMD_CATEGORIES):
        cat_name = item[0]
        if counts.get(idx, 0) > 0:
            cat_composition.append({
                "cat_idx": idx,
                "name": cat_name,
                "count": counts.get(idx, 0),
            })

    log_path = cfg["train"]["log_path"]
    hist_data = []
    if os.path.exists(log_path):
        hist = pd.read_csv(log_path)
        for _, row in hist.iterrows():
            hist_data.append({
                "epoch": int(row["epoch"]),
                "train_acc": round(float(row["train_acc"]), 4),
                "val_acc": round(float(row["val_acc"]), 4),
            })

    return {
        "cat_names": cat_names,
        "sample_names": [s["img_name"] for s in samples[:25]],
        "samples": samples_data,
        "ri": {
            "times": [t.strftime("%d %b, %H:%M") for t in ri_times] if ri_times else [],
            "winds": [round(w, 1) for w in ri_winds] if ri_winds else [],
            "episodes": ri_episodes_res,
            "peak_wind_kt": round(max(ri_winds), 1) if ri_winds else None,
        },
        "forecast": {
            "storms": storms_with_seqs,
            "val_storms": list(VAL_STORMS),
            "data": forecast_data,
        },
        "about": {
            "total_samples": len(samples),
            "n_kaggle": n_kaggle,
            "n_hursat": n_hursat,
            "n_mosdac": n_mosdac,
            "val_accuracy": round(ckpt["val_acc"] * 100.0, 1),
            "val_epoch": ckpt["epoch"],
            "composition": cat_composition,
            "training_history": hist_data,
        }
    }

APP_DATA = build_app_data()
APP_DATA_JSON = json.dumps(APP_DATA)

HTML_CONTENT = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cyclone AI — Meteorological Intelligence Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg-main: #f8fafc;
      --bg-card: #ffffff;
      --bg-subtle: #f1f5f9;
      --border-color: #e2e8f0;
      --border-hover: #cbd5e1;
      --text-main: #0f172a;
      --text-sub: #334155;
      --text-muted: #64748b;
      --blue-primary: #0284c7;
      --blue-hover: #0369a1;
      --blue-light: #e0f2fe;
      --coral-accent: #f97316;
      --green-bg: #f0fdf4;
      --green-border: #bbf7d0;
      --green-text: #166534;
      --amber-bg: #fffbeb;
      --amber-border: #fde68a;
      --amber-text: #92400e;
      --red-bg: #fef2f2;
      --red-border: #fecaca;
      --red-text: #991b1b;
      --shadow-card: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
      --radius-card: 16px;
      --radius-sm: 10px;
      --radius-pill: 999px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Manrope', 'Inter', -apple-system, sans-serif;
      background-color: var(--bg-main);
      color: var(--text-main);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 3rem;
    }}

    .app-container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 1.2rem 1.5rem 3rem;
    }}

    /* Header */
    .app-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card);
      padding: 1.1rem 1.6rem;
      border-radius: var(--radius-card);
      border: 1px solid var(--border-color);
      box-shadow: var(--shadow-card);
      margin-bottom: 1.5rem;
    }}

    .header-brand {{
      display: flex;
      align-items: center;
      gap: 1rem;
      cursor: pointer;
    }}

    .brand-icon {{
      font-size: 1.9rem;
      width: 3.3rem;
      height: 3.3rem;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--blue-primary), var(--blue-hover));
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }}

    .header-title {{
      font-size: 1.55rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1.15;
      letter-spacing: -0.02em;
    }}

    .header-subtitle {{
      color: var(--text-muted);
      font-size: 0.88rem;
      font-weight: 500;
      margin-top: 0.1rem;
    }}

    .header-nav {{
      display: flex;
      gap: 6px;
      background: #e2e8f0;
      padding: 5px;
      border-radius: var(--radius-pill);
    }}

    .nav-btn {{
      background: transparent;
      border: none;
      padding: 0.55rem 1.2rem;
      border-radius: var(--radius-pill);
      color: var(--text-sub);
      font-size: 0.9rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.15s ease-in-out;
    }}

    .nav-btn:hover {{ color: var(--text-main); }}
    .nav-btn.active {{
      background: var(--blue-primary);
      color: #ffffff;
      box-shadow: 0 3px 10px rgba(2, 132, 199, 0.3);
    }}

    /* Views */
    .view-section {{ display: none; }}
    .view-section.active {{ display: block; }}

    /* Landing View Animations */
    @keyframes fadeInUp {{
      from {{
        opacity: 0;
        transform: translateY(22px) scale(0.98);
      }}
      to {{
        opacity: 1;
        transform: translateY(0) scale(1);
      }}
    }}

    @keyframes popIn {{
      0% {{
        opacity: 0;
        transform: scale(0.92);
      }}
      70% {{
        transform: scale(1.02);
      }}
      100% {{
        opacity: 1;
        transform: scale(1);
      }}
    }}

    .animate-pop {{
      animation: popIn 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .animate-fade-up {{
      animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .delay-1 {{ animation-delay: 0.06s; }}
    .delay-2 {{ animation-delay: 0.12s; }}
    .delay-3 {{ animation-delay: 0.18s; }}
    .delay-4 {{ animation-delay: 0.24s; }}
    .delay-5 {{ animation-delay: 0.30s; }}
    .delay-6 {{ animation-delay: 0.36s; }}
    .delay-7 {{ animation-delay: 0.42s; }}
    .delay-8 {{ animation-delay: 0.48s; }}
    .delay-9 {{ animation-delay: 0.54s; }}

    /* Landing View */
    .landing-hero-card {{
      background: var(--bg-card);
      border-radius: var(--radius-card);
      border: 1px solid var(--border-color);
      padding: 3rem 2.5rem;
      box-shadow: var(--shadow-card);
      text-align: center;
      margin-bottom: 1.8rem;
    }}

    .landing-badge {{
      display: inline-block;
      background: var(--blue-light);
      color: #0369a1;
      font-weight: 800;
      font-size: 0.8rem;
      padding: 0.4rem 1rem;
      border-radius: var(--radius-pill);
      margin-bottom: 1.2rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .landing-title {{
      font-size: 3.4rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.035em;
      line-height: 1.1;
      background: linear-gradient(135deg, #0f172a 0%, #0284c7 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .landing-sub {{
      font-size: 1.15rem;
      color: var(--text-muted);
      max-width: 720px;
      margin: 0.9rem auto 2.2rem;
      line-height: 1.6;
    }}

    .landing-hero-box {{
      width: 100%;
      max-width: 860px;
      height: 400px;
      margin: 0 auto 2rem;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
      position: relative;
    }}

    .landing-hero-box img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}

    .hero-status-pill {{
      position: absolute;
      bottom: 1rem;
      left: 1rem;
      background: rgba(15, 23, 42, 0.82);
      backdrop-filter: blur(8px);
      color: #ffffff;
      padding: 0.55rem 1.1rem;
      border-radius: var(--radius-pill);
      font-size: 0.85rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.15);
    }}

    .status-dot-pulse {{
      width: 9px;
      height: 9px;
      background-color: #22c55e;
      border-radius: 50%;
      box-shadow: 0 0 8px #22c55e;
    }}

    .hero-actions {{
      display: flex;
      justify-content: center;
      gap: 1.2rem;
      margin-bottom: 2.5rem;
    }}

    .btn-cta {{
      background: var(--blue-primary);
      color: #ffffff;
      border: none;
      padding: 0.9rem 2.4rem;
      font-size: 1.05rem;
      font-weight: 700;
      border-radius: var(--radius-pill);
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35);
      transition: all 0.2s ease-in-out;
    }}

    .btn-cta:hover {{
      background: var(--blue-hover);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45);
    }}

    .btn-cta.btn-secondary {{
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }}

    .btn-cta.btn-secondary:hover {{
      background: var(--bg-subtle);
      border-color: var(--border-hover);
    }}

    /* Metric Strip */
    .landing-metrics-strip {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.2rem;
      margin-bottom: 2.5rem;
    }}

    .metric-strip-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.4rem 1.1rem;
      text-align: center;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
    }}

    .metric-strip-card .m-val {{
      font-size: 2.1rem;
      font-weight: 800;
      color: var(--blue-primary);
      line-height: 1.1;
    }}

    .metric-strip-card .m-title {{
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-main);
      margin-top: 0.3rem;
    }}

    .metric-strip-card .m-sub {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }}

    .landing-highlights {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1.4rem;
      margin-top: 1.2rem;
    }}

    .highlight-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.5rem 1.3rem;
      text-align: left;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .highlight-card:hover {{
      border-color: var(--blue-primary);
      transform: translateY(-3px);
      box-shadow: 0 8px 24px rgba(2, 132, 199, 0.12);
    }}

    .highlight-icon {{ font-size: 2rem; margin-bottom: 0.6rem; }}
    .highlight-title {{ font-weight: 800; font-size: 1.05rem; color: var(--text-main); margin-bottom: 0.3rem; }}
    .highlight-desc {{ font-size: 0.88rem; color: var(--text-muted); line-height: 1.5; }}

    /* Layout Cards & Grids */
    .card {{
      background: var(--bg-card);
      border-radius: var(--radius-card);
      border: 1px solid var(--border-color);
      padding: 1.3rem;
      box-shadow: var(--shadow-card);
      margin-bottom: 1.3rem;
    }}

    .card-title {{
      font-size: 1.1rem;
      font-weight: 800;
      color: var(--text-main);
      margin-bottom: 0.8rem;
    }}

    .grid {{ display: grid; gap: 1.2rem; }}
    .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}

    /* Two-column Assess Composition */
    .assess-cols {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.4rem;
      margin-bottom: 1.4rem;
    }}

    @media (max-width: 900px) {{
      .assess-cols, .grid-2, .grid-3, .grid-4, .landing-highlights {{ grid-template-columns: 1fr; }}
    }}

    .main-sat-img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid var(--border-color);
    }}

    .control-label {{
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-main);
      display: block;
      margin-bottom: 0.4rem;
    }}

    .custom-select, .custom-range {{
      width: 100%;
      padding: 0.65rem 0.9rem;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.92rem;
      outline: none;
    }}

    /* Assessment Output Box */
    .assessment-primary-box {{
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 1.4rem;
      text-align: center;
      margin: 1rem 0;
    }}

    .assess-cat-val {{
      font-size: 1.35rem;
      font-weight: 800;
      color: var(--blue-primary);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .assess-wind-val {{
      font-size: 2.5rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1.1;
      margin: 0.3rem 0;
    }}

    .assess-sub-val {{
      font-size: 0.92rem;
      color: var(--text-muted);
      font-weight: 600;
    }}

    /* Metrics */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 1rem;
    }}

    .metric-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 1rem 1.1rem;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }}

    .metric-label {{
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }}

    .metric-value {{
      font-size: 1.6rem;
      color: var(--text-main);
      font-weight: 800;
      line-height: 1.25;
      margin-top: 0.2rem;
    }}

    .metric-sub {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.15rem;
      font-weight: 500;
    }}

    /* Banners & Callouts */
    .alert-banner {{
      border-radius: 12px;
      padding: 0.9rem 1.1rem;
      font-size: 0.92rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .alert-banner.good {{ background: var(--green-bg); border: 1px solid var(--green-border); color: var(--green-text); }}
    .alert-banner.warning {{ background: var(--amber-bg); border: 1px solid var(--amber-border); color: var(--amber-text); }}
    .alert-banner.critical {{ background: var(--red-bg); border: 1px solid var(--red-border); color: var(--red-text); }}

    .callout {{
      border-radius: 12px;
      padding: 1rem 1.25rem;
      margin: 1.2rem 0;
      font-size: 0.92rem;
      background: var(--blue-light);
      border: 1px solid #bae6fd;
      color: #0369a1;
    }}

    /* Section titles */
    .section-title-block {{
      margin: 1.8rem 0 1rem;
    }}

    .section-title-block h3 {{
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--text-main);
    }}

    .section-title-block p {{
      font-size: 0.88rem;
      color: var(--text-muted);
    }}

    /* Channel Cards */
    .channel-card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}

    .channel-tag {{
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 0.5rem;
    }}

    .chart-container {{
      position: relative;
      width: 100%;
      height: 350px;
      margin-top: 0.5rem;
    }}

    .caption {{
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-top: 0.5rem;
    }}

    .margin-top {{ margin-top: 1.2rem; }}

    /* Historical comparison card */
    .hist-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 0.9rem;
    }}

    .hist-card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}

    .hist-badge {{
      background: var(--blue-light);
      color: #0369a1;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.2rem 0.5rem;
      border-radius: var(--radius-pill);
      display: inline-block;
      margin: 0.4rem 0 0.2rem;
    }}

    .limitations-list {{
      padding-left: 1.2rem;
      color: var(--text-sub);
      font-size: 0.92rem;
      line-height: 1.65;
    }}

    .limitations-list li {{ margin-bottom: 0.6rem; }}
  </style>
</head>
<body>
  <div class="app-container">
    <!-- Navigation Tabs -->
    <nav class="header-nav" style="margin-bottom: 1.5rem; width: fit-content;">
      <button class="nav-btn active" data-view="landing" onclick="showView('landing')">🏠 Home</button>
      <button class="nav-btn" data-view="assess" onclick="showView('assess')">🔍 Assess a Storm</button>
      <button class="nav-btn" data-view="history" onclick="showView('history')">📊 Historical Precedent</button>
      <button class="nav-btn" data-view="ri" onclick="showView('ri')">⚡ Early Warning</button>
      <button class="nav-btn" data-view="forecast" onclick="showView('forecast')">🔮 Forecast</button>
      <button class="nav-btn" data-view="about" onclick="showView('about')">ℹ️ About &amp; Data</button>
    </nav>

    <!-- LANDING VIEW -->
    <div id="view-landing" class="view-section active">
      <div class="landing-hero-card">
        <div class="landing-badge animate-pop delay-1">🌀 SIH26070 &bull; DISASTER MANAGEMENT &bull; IMD / MOES</div>
        <h1 class="landing-title animate-fade-up delay-2">CycloneNet</h1>
        <p class="landing-sub animate-fade-up delay-3">Next-Generation AI Platform for Real-Time Tropical Cyclone Intensity Classification, Central Pressure Estimation &amp; +24h Horizon Forecasting.</p>

        <div class="landing-hero-box animate-pop delay-3">
          <img id="landing-hero-img" src="" alt="Cyclone Satellite Image">
          <div class="hero-status-pill">
            <span class="status-dot-pulse"></span> Live Satellite Feed &bull; EfficientNet-B0 + GRU Engine
          </div>
        </div>

        <div class="hero-actions animate-pop delay-4">
          <button class="btn-cta" onclick="showView('assess')">🔍 Launch Storm Assessment &rarr;</button>
          <button class="btn-cta btn-secondary" onclick="showView('forecast')">🔮 Run +24h Forecast &rarr;</button>
        </div>

        <!-- Metric Strip -->
        <div class="landing-metrics-strip">
          <div class="metric-strip-card animate-fade-up delay-4">
            <div class="m-val">1,032</div>
            <div class="m-title">Labeled Frames</div>
            <div class="m-sub">INSAT-3D, NOAA HURSAT-B1, MOSDAC</div>
          </div>
          <div class="metric-strip-card animate-fade-up delay-5">
            <div class="m-val">41.9%</div>
            <div class="m-title">Generalization</div>
            <div class="m-sub">Strict storm-held-out validation</div>
          </div>
          <div class="metric-strip-card animate-fade-up delay-6">
            <div class="m-val">8 Scale</div>
            <div class="m-title">IMD Categories</div>
            <div class="m-sub">Depression &rarr; Super Cyclonic Storm</div>
          </div>
          <div class="metric-strip-card animate-fade-up delay-7">
            <div class="m-val">+24h</div>
            <div class="m-title">Horizon Forecaster</div>
            <div class="m-sub">Beats persistence baseline</div>
          </div>
        </div>

        <!-- Capabilities Grid -->
        <div class="section-title-block margin-top animate-fade-up delay-5">
          <h3 style="font-size: 1.25rem;">Core Platform Capabilities</h3>
          <p>End-to-end AI workflow designed for operational disaster management</p>
        </div>

        <div class="landing-highlights">
          <div class="highlight-card animate-pop delay-5" onclick="showView('assess')">
            <div class="highlight-icon">🌀</div>
            <div class="highlight-title">Intensity &amp; Pressure</div>
            <div class="highlight-desc">Dual regression estimating sustained wind speed (km/h) &amp; central pressure (mb).</div>
          </div>
          <div class="highlight-card animate-pop delay-6" onclick="showView('assess')">
            <div class="highlight-icon">👁️</div>
            <div class="highlight-title">Grad-CAM Focus</div>
            <div class="highlight-desc">Visual spatial heatmaps highlighting cloud eye &amp; eyewall convective structures.</div>
          </div>
          <div class="highlight-card animate-pop delay-7" onclick="showView('assess')">
            <div class="highlight-icon">🎯</div>
            <div class="highlight-title">MC-Dropout Confidence</div>
            <div class="highlight-desc">Monte Carlo sampling across 30 passes providing calibrated uncertainty bounds.</div>
          </div>
          <div class="highlight-card animate-pop delay-6" onclick="showView('forecast')">
            <div class="highlight-icon">🔮</div>
            <div class="highlight-title">Temporal Forecasting</div>
            <div class="highlight-desc">Sequence neural network predicting intensity trend at +6h, +12h, and +24h.</div>
          </div>
          <div class="highlight-card animate-pop delay-7" onclick="showView('ri')">
            <div class="highlight-icon">⚡</div>
            <div class="highlight-title">Rapid Intensification</div>
            <div class="highlight-desc">Automated detection of Kaplan &amp; DeMaria +30kt / 24h warning triggers.</div>
          </div>
          <div class="highlight-card animate-pop delay-8" onclick="showView('history')">
            <div class="highlight-icon">📊</div>
            <div class="highlight-title">Historical Precedents</div>
            <div class="highlight-desc">Cosine feature embedding search finding similar historical cyclone archives.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ASSESS VIEW -->
    <div id="view-assess" class="view-section">
      <div class="assess-cols">
        <!-- LEFT: Satellite selector & image -->
        <div class="card">
          <label for="assess-select" class="control-label">Select Satellite Observation:</label>
          <select id="assess-select" class="custom-select" onchange="loadAssessSample(this.value)"></select>
          <div class="margin-top">
            <img id="assess-main-img" src="" class="main-sat-img" alt="Main Satellite View">
          </div>
        </div>

        <!-- RIGHT: Clean Assessment Output -->
        <div class="card">
          <div class="card-title">CURRENT ASSESSMENT</div>
          <div id="assess-banner" class="alert-banner good"></div>

          <div class="assessment-primary-box">
            <div id="assess-cat-name" class="assess-cat-val">CYCLONIC STORM</div>
            <div id="assess-wind-val" class="assess-wind-val">78 km/h</div>
            <div id="assess-sub-val" class="assess-sub-val">993 mb &bull; 85% confidence</div>
          </div>

          <div class="metrics-grid" id="assess-extra-metrics"></div>
        </div>
      </div>

      <!-- How the model sees the storm -->
      <div class="section-title-block">
        <h3>How the model sees the storm</h3>
        <p>Input channels and Grad-CAM neural attention overlay</p>
      </div>

      <div class="grid grid-3">
        <div class="card channel-card">
          <div class="channel-tag">IR Channel</div>
          <img id="assess-ir-img" src="" alt="IR Channel">
        </div>
        <div class="card channel-card">
          <div class="channel-tag">Raw Channel</div>
          <img id="assess-raw-img" src="" alt="Raw Channel">
        </div>
        <div class="card channel-card">
          <div class="channel-tag">Grad-CAM Focus Overlay</div>
          <img id="assess-gradcam-img" src="" alt="Grad-CAM Focus Overlay">
        </div>
      </div>

      <!-- Probabilities Chart -->
      <div class="card margin-top">
        <div class="card-title">Category Probabilities</div>
        <div class="chart-container">
          <canvas id="chart-assess-probs"></canvas>
        </div>
        <p class="caption">Bars follow the IMD scale; predicted category is highlighted in coral. Error bars represent 30 MC-Dropout passes.</p>
      </div>
    </div>

    <!-- HISTORICAL PRECEDENT VIEW -->
    <div id="view-history" class="view-section">
      <div class="card-title">Historical Precedent</div>
      <p class="caption" style="margin-bottom:1.2rem;">Compare the current storm with similar historical observations.</p>

      <div class="grid grid-4" id="history-cards-grid"></div>
    </div>

    <!-- EARLY WARNING VIEW -->
    <div id="view-ri" class="view-section">
      <div class="card-title">Early-Warning Assessment</div>
      <p class="caption" style="margin-bottom:1.2rem;">Calm, automated monitoring of Rapid Intensification threshold events (+30kt / 24h).</p>

      <div class="metrics-grid" id="ri-metrics-grid"></div>

      <div class="card margin-top">
        <div class="card-title">Cyclone Amphan (2020) — Best Track Intensity Trajectory</div>
        <div class="chart-container">
          <canvas id="chart-ri-timeline"></canvas>
        </div>
      </div>

      <div class="callout">
        <strong>Why this matters:</strong><br>
        Rapid Intensification (+30kt wind increase in 24 hours) is the primary driver of unexpected coastal disaster impacts. Automated early warning provides forecasters with vital reaction lead time.
      </div>
    </div>

    <!-- FORECAST VIEW -->
    <div id="view-forecast" class="view-section">
      <div class="card-title">Intensity Forecast</div>
      <p class="caption" style="margin-bottom:1.2rem;">Sequence-based multi-horizon wind speed predictions (+6h, +12h, +24h).</p>

      <div class="grid grid-2">
        <div class="card">
          <label for="forecast-storm-select" class="control-label">Select Storm:</label>
          <select id="forecast-storm-select" class="custom-select" onchange="onForecastStormSelect(this.value)"></select>
        </div>
        <div class="card">
          <label for="forecast-anchor-slider" class="control-label">Anchor Point Slider:</label>
          <input type="range" id="forecast-anchor-slider" class="custom-range" min="0" max="10" oninput="onForecastSliderInput(this.value)">
          <div id="forecast-slider-text" class="caption"></div>
        </div>
      </div>

      <div id="forecast-heldout-box" class="callout"></div>

      <!-- Current State & Horizon Cards -->
      <div class="grid grid-4 margin-top" id="forecast-horizon-cards"></div>

      <!-- Observed vs Forecast Timeline -->
      <div class="card margin-top">
        <div class="card-title">Observed vs Model Forecast Trajectory</div>
        <div class="chart-container">
          <canvas id="chart-forecast-timeline"></canvas>
        </div>
      </div>

      <div id="forecast-ri-banner-box" class="margin-top"></div>
    </div>

    <!-- ABOUT & DATA VIEW -->
    <div id="view-about" class="view-section">
      <div class="card-title">Model &amp; Dataset Information</div>
      <p class="caption" style="margin-bottom:1.2rem;">Scientific specifications, dataset composition, and validation benchmarks.</p>

      <div class="metrics-grid" id="about-meta-grid"></div>

      <div class="card margin-top">
        <div class="card-title">Dataset Composition</div>
        <div class="chart-container">
          <canvas id="chart-about-comp"></canvas>
        </div>
      </div>

      <div class="card margin-top">
        <div class="card-title">Training Performance</div>
        <div class="chart-container">
          <canvas id="chart-about-train"></canvas>
        </div>
      </div>

      <div class="card margin-top">
        <div class="card-title">Known Limitations</div>
        <ul class="limitations-list">
          <li><strong>Dataset size:</strong> 1,032 labeled images total (136 single Kaggle frames with no storm ID, 680 NOAA HURSAT-B1 frames across 10 real storms, 216 MOSDAC frames of Cyclone Amphan) — small for an 8-way classifier.</li>
          <li><strong>Validation split:</strong> Validation is 2 HURSAT-B1 storms (136 frames) held out entirely (Phet, Nilofar).</li>
          <li><strong>Pressure labels:</strong> No pressure label from Kaggle or HURSAT-B1 sources — wind speed regression head primary.</li>
          <li><strong>Backbone initialization:</strong> Trained from random initialization without pretrained ImageNet weights.</li>
        </ul>
      </div>
    </div>
  </div>

  <script>
    // Injected serialized backend data
    window.DATA = {APP_DATA_JSON};

    let currentSampleName = null;
    let chartProb = null, chartRI = null, chartForecast = null, chartComp = null, chartTrain = null;

    const SEQ_RAMP = ["#bae6fd", "#7dd3fc", "#38bdf8", "#0284c7", "#0369a1", "#1d4ed8", "#1e40af", "#1e3a8a"];
    const CAT_ORANGE = "#f97316";
    const CAT_BLUE = "#0284c7";

    document.addEventListener("DOMContentLoaded", () => {{
      initSampleSelect();
      initLandingHero();
      initRIView();
      initForecastView();
      initAboutView();
    }});

    function showView(viewId) {{
      document.querySelectorAll(".view-section").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".nav-btn").forEach(el => el.classList.remove("active"));

      const targetView = document.getElementById("view-" + viewId);
      if (targetView) targetView.classList.add("active");

      const activeNav = document.querySelector(`.nav-btn[data-view="${{viewId}}"]`);
      if (activeNav) activeNav.classList.add("active");

      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    function initLandingHero() {{
      const firstSampleKey = DATA.sample_names[0];
      const s = DATA.samples[firstSampleKey];
      if (s) {{
        document.getElementById("landing-hero-img").src = s.ir_base64;
      }}
    }}

    function initSampleSelect() {{
      const select = document.getElementById("assess-select");
      select.innerHTML = "";
      DATA.sample_names.forEach(name => {{
        const s = DATA.samples[name];
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = `${{name}} (${{s.true_cat_name}}, ${{s.true_kmph}} km/h)`;
        select.appendChild(opt);
      }});

      if (DATA.sample_names.length > 0) {{
        currentSampleName = DATA.sample_names[0];
        loadAssessSample(currentSampleName);
      }}
    }}

    function loadAssessSample(name) {{
      currentSampleName = name;
      const s = DATA.samples[name];
      if (!s) return;

      document.getElementById("assess-main-img").src = s.ir_base64;
      document.getElementById("assess-ir-img").src = s.ir_base64;
      document.getElementById("assess-raw-img").src = s.raw_base64;
      document.getElementById("assess-gradcam-img").src = s.overlay_base64;

      document.getElementById("assess-cat-name").textContent = s.pred_cat_name;
      document.getElementById("assess-wind-val").textContent = `${{s.pred_kmph}} km/h`;
      
      const pressText = s.pred_pressure ? `${{s.pred_pressure}} mb &bull; ` : "";
      document.getElementById("assess-sub-val").innerHTML = `${{pressText}}${{(s.confidence * 100).toFixed(0)}}% confidence`;

      // Banner
      const banner = document.getElementById("assess-banner");
      let icon = "✅", headline = "LOW IMMEDIATE THREAT", cls = "good";
      if (s.pred_idx >= 5) {{ icon = "🚨"; headline = "HIGH-IMPACT — Review Evacuation Readiness"; cls = "critical"; }}
      else if (s.pred_idx >= 3) {{ icon = "⚠️"; headline = "SIGNIFICANT — Monitor Closely"; cls = "warning"; }}
      banner.className = `alert-banner ${{cls}}`;
      banner.innerHTML = `<span style="font-size:1.2rem;">${{icon}}</span> <strong>${{headline}}</strong> &bull; ${{(s.confidence * 100).toFixed(0)}}% confidence`;

      // Metrics
      const extraMetrics = document.getElementById("assess-extra-metrics");
      let mHTML = `
        <div class="metric-card">
          <div class="metric-label">Predicted Wind Speed</div>
          <div class="metric-value">${{s.pred_kmph}} km/h</div>
          <div class="metric-sub">&plusmn;${{s.wind_std_kmph}} km/h</div>
        </div>
      `;
      if (s.pred_pressure) {{
        mHTML += `
          <div class="metric-card">
            <div class="metric-label">Predicted Pressure</div>
            <div class="metric-value">${{s.pred_pressure}} mb</div>
            <div class="metric-sub">&plusmn;${{s.pressure_std_mb}} mb</div>
          </div>
        `;
      }}
      mHTML += `
        <div class="metric-card">
          <div class="metric-label">Actual Category</div>
          <div class="metric-value">${{s.true_cat_name}}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Actual Wind Speed</div>
          <div class="metric-value">${{s.true_kmph}} km/h</div>
        </div>
      `;
      extraMetrics.innerHTML = mHTML;

      // Probabilities chart
      renderProbabilitiesChart(s.mean_probs, s.pred_idx);

      // Historical matches
      renderHistoricalMatches(s.matches);
    }}

    function renderProbabilitiesChart(means, predIdx) {{
      const ctx = document.getElementById("chart-assess-probs").getContext("2d");
      if (chartProb) chartProb.destroy();

      const barColors = DATA.cat_names.map((_, i) => i === predIdx ? CAT_ORANGE : (SEQ_RAMP[i] || CAT_BLUE));

      chartProb = new Chart(ctx, {{
        type: "bar",
        data: {{
          labels: DATA.cat_names,
          datasets: [{{
            label: "Probability (MC-Dropout)",
            data: means,
            backgroundColor: barColors,
            borderRadius: 6,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ beginAtZero: true, max: 1.0, grid: {{ color: "#f1f5f9" }}, ticks: {{ color: "#64748b" }} }},
            x: {{ grid: {{ display: false }}, ticks: {{ color: "#334155", font: {{ weight: "600" }} }} }}
          }}
        }}
      }});
    }}

    function renderHistoricalMatches(matches) {{
      const container = document.getElementById("history-cards-grid");
      container.innerHTML = "";
      matches.forEach(m => {{
        const card = document.createElement("div");
        card.className = "hist-card";
        card.innerHTML = `
          <img src="${{m.ir_base64}}" alt="${{m.img_name}}">
          <div style="margin-top:0.5rem;">
            <div style="font-weight:700; font-size:0.9rem;">${{m.img_name}}</div>
            <div class="hist-badge">${{m.percentile}}th percentile match</div>
            <div style="font-size:0.8rem; color:#475569;">${{m.cat_name}}, ${{m.kmph}} km/h</div>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function initRIView() {{
      const ri = DATA.ri;
      const metricsContainer = document.getElementById("ri-metrics-grid");
      if (ri.episodes && ri.episodes.length > 0) {{
        const ep = ri.episodes[0];
        metricsContainer.innerHTML = `
          <div class="metric-card">
            <div class="metric-label">RI Status</div>
            <div class="metric-value" style="color:var(--red-text);">ALERT TRIGGERED</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Intensification</div>
            <div class="metric-value">+${{ep.delta}} kt</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">${{ep.duration_hours}} h</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Peak Intensity</div>
            <div class="metric-value">${{ri.peak_wind_kt}} kt &bull; SuCS</div>
          </div>
        `;
      }}

      const ctx = document.getElementById("chart-ri-timeline").getContext("2d");
      if (chartRI) chartRI.destroy();

      chartRI = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: ri.times,
          datasets: [{{
            label: "Sustained Wind Speed (kt)",
            data: ri.winds,
            borderColor: CAT_BLUE,
            backgroundColor: "rgba(2, 132, 199, 0.08)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.2,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ grid: {{ color: "#f1f5f9" }}, ticks: {{ color: "#64748b" }} }},
            x: {{ grid: {{ display: false }}, ticks: {{ color: "#64748b", maxTicksLimit: 10 }} }}
          }}
        }}
      }});
    }}

    let currentForecastStorm = null;

    function initForecastView() {{
      const fc = DATA.forecast;
      const select = document.getElementById("forecast-storm-select");
      select.innerHTML = "";
      fc.storms.forEach(st => {{
        const opt = document.createElement("option");
        opt.value = st;
        opt.textContent = fc.val_storms.includes(st) ? `${{st}} (held out — validation)` : st;
        select.appendChild(opt);
      }});

      currentForecastStorm = fc.storms[0];
      select.value = currentForecastStorm;
      renderForecast(currentForecastStorm, 0);
    }}

    function onForecastStormSelect(storm) {{
      currentForecastStorm = storm;
      renderForecast(storm, 0);
    }}

    function onForecastSliderInput(val) {{
      renderForecast(currentForecastStorm, parseInt(val));
    }}

    function renderForecast(storm, anchorIdx) {{
      const stData = DATA.forecast.data[storm];
      if (!stData) return;

      const slider = document.getElementById("forecast-anchor-slider");
      slider.max = stData.sequences.length - 1;
      const validIdx = Math.max(0, Math.min(anchorIdx, stData.sequences.length - 1));
      slider.value = validIdx;

      const seq = stData.sequences[validIdx];
      document.getElementById("forecast-slider-text").textContent = `Forecasting from ${{seq.anchor_dt}} (Anchor ${{validIdx + 1}} of ${{stData.sequences.length}})`;

      const heldoutBox = document.getElementById("forecast-heldout-box");
      if (stData.is_held_out) {{
        heldoutBox.className = "callout";
        heldoutBox.innerHTML = `<strong>${{storm}}</strong> was held out entirely during training — forecast below reflects true generalization.`;
      }} else {{
        heldoutBox.className = "callout";
        heldoutBox.style.background = "var(--amber-bg)";
        heldoutBox.style.borderColor = "var(--amber-border)";
        heldoutBox.style.color = "var(--amber-text)";
        heldoutBox.innerHTML = `<strong>${{storm}}</strong> was used during training — select Phet or Nilofar for held-out validation.`;
      }}

      // Horizon cards
      const cardsContainer = document.getElementById("forecast-horizon-cards");
      let hHTML = `
        <div class="metric-card">
          <div class="metric-label">Current Anchor State</div>
          <div class="metric-value">${{seq.anchor_kmph}} km/h</div>
          <div class="metric-sub">${{seq.anchor_cat}}</div>
        </div>
      `;
      ["6", "12", "24"].forEach(h => {{
        const item = seq.horizons[h];
        if (item) {{
          const sign = item.delta_kmph >= 0 ? "+" : "";
          hHTML += `
            <div class="metric-card">
              <div class="metric-label">+${{h}} Hours</div>
              <div class="metric-value">${{item.pred_kmph}} km/h</div>
              <div class="metric-sub">${{sign}}${{item.delta_kmph}} km/h vs now (${{item.pred_cat}})</div>
            </div>
          `;
        }}
      }});
      cardsContainer.innerHTML = hHTML;

      // Forecast Timeline Chart
      const ctx = document.getElementById("chart-forecast-timeline").getContext("2d");
      if (chartForecast) chartForecast.destroy();

      chartForecast = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: stData.full_times,
          datasets: [
            {{
              label: "Observed Wind Speed (km/h)",
              data: stData.full_winds,
              borderColor: CAT_BLUE,
              borderWidth: 2.5,
              pointRadius: 2,
            }},
            {{
              label: "Forecast Trajectory",
              data: stData.full_times.map(t => {{
                const idx = seq.fc_times.indexOf(t);
                return idx !== -1 ? seq.fc_winds[idx] : null;
              }}),
              borderColor: CAT_ORANGE,
              borderDash: [6, 4],
              borderWidth: 2.5,
              pointRadius: 5,
              pointBackgroundColor: CAT_ORANGE,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: true }} }},
          scales: {{
            y: {{ grid: {{ color: "#f1f5f9" }}, ticks: {{ color: "#64748b" }} }},
            x: {{ grid: {{ display: false }}, ticks: {{ color: "#64748b", maxTicksLimit: 10 }} }}
          }}
        }}
      }});

      // Live RI Banner
      const riBox = document.getElementById("forecast-ri-banner-box");
      if (seq.rt_episodes && seq.rt_episodes.length > 0) {{
        const ep = seq.rt_episodes[0];
        riBox.innerHTML = `
          <div class="alert-banner critical">
            <span style="font-size:1.2rem;">🚨</span> <strong>RAPID INTENSIFICATION ALERT:</strong> Forecast predicts +${{ep.delta_kt}}kt over ${{ep.duration_hours}}h.
          </div>
        `;
      }} else {{
        riBox.innerHTML = `
          <div class="callout" style="background:#f0f9ff; border:1px solid #bae6fd; color:#0369a1;">
            No Rapid Intensification threshold crossing predicted in this forecast window (+30kt / 24h).
          </div>
        `;
      }}
    }}

    function initAboutView() {{
      const ab = DATA.about;
      const metaContainer = document.getElementById("about-meta-grid");
      metaContainer.innerHTML = `
        <div class="metric-card">
          <div class="metric-label">Labeled Images</div>
          <div class="metric-value">${{ab.total_samples}}</div>
          <div class="metric-sub">${{ab.n_kaggle}} Kaggle, ${{ab.n_hursat}} HURSAT, ${{ab.n_mosdac}} MOSDAC</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">IMD Categories</div>
          <div class="metric-value">${{ab.composition.length}}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Validation Benchmark</div>
          <div class="metric-value">${{ab.val_accuracy}}%</div>
          <div class="metric-sub">Epoch ${{ab.val_epoch}} on held-out storms</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Architecture</div>
          <div class="metric-value">CycloneNet</div>
          <div class="metric-sub">2-channel IR/Raw backbone</div>
        </div>
      `;

      // Composition
      const ctxComp = document.getElementById("chart-about-comp").getContext("2d");
      if (chartComp) chartComp.destroy();

      chartComp = new Chart(ctxComp, {{
        type: "bar",
        data: {{
          labels: ab.composition.map(c => c.name),
          datasets: [{{
            label: "Labeled Images",
            data: ab.composition.map(c => c.count),
            backgroundColor: ab.composition.map(c => SEQ_RAMP[c.cat_idx] || CAT_BLUE),
            borderRadius: 6,
          }}]
        }},
        options: {{
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ color: "#f1f5f9" }}, ticks: {{ color: "#64748b" }} }},
            y: {{ grid: {{ display: false }}, ticks: {{ color: "#334155", font: {{ weight: "600" }} }} }}
          }}
        }}
      }});

      // Training curve
      if (ab.training_history && ab.training_history.length > 0) {{
        const ctxTrain = document.getElementById("chart-about-train").getContext("2d");
        if (chartTrain) chartTrain.destroy();

        chartTrain = new Chart(ctxTrain, {{
          type: "line",
          data: {{
            labels: ab.training_history.map(h => h.epoch),
            datasets: [
              {{
                label: "Train Accuracy",
                data: ab.training_history.map(h => h.train_acc),
                borderColor: CAT_BLUE,
                borderWidth: 2.5,
              }},
              {{
                label: "Validation Accuracy (Held-out storms)",
                data: ab.training_history.map(h => h.val_acc),
                borderColor: CAT_ORANGE,
                borderWidth: 2.5,
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: true }} }},
            scales: {{
              y: {{ beginAtZero: true, max: 1.0, grid: {{ color: "#f1f5f9" }}, ticks: {{ color: "#64748b" }} }},
              x: {{ grid: {{ display: false }}, ticks: {{ color: "#64748b" }} }}
            }}
          }}
        }});
      }}
    }}
  </script>
</body>
</html>
"""

# Render custom HTML web application in Streamlit with zero Streamlit chrome
components.html(HTML_CONTENT, height=1350, scrolling=True)
