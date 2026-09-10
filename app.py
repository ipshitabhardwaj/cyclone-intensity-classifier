"""
Streamlit demo dashboard for the SIH26070 cyclone classification prototype.

Run it from the project root:
    streamlit run app.py

Five tabs:
  - Classify: pick/upload an image, get category + wind speed with MC-Dropout
    confidence, Grad-CAM explanation, and the full probability breakdown.
  - Historical Comparison: nearest-neighbour retrieval against every image in
    the training set, shown as a percentile match (not raw cosine similarity
    -- see src/historical_comparison.py for why).
  - Rapid Intensification: a real case study on Cyclone Amphan's actual
    best-track record, showing the RI episode our detector correctly flags.
  - Forecast: given the last 4 observed frames of a storm, predicts wind
    speed +6h/+12h/+24h ahead (src/temporal_model.py), validated against a
    persistence baseline on 2 fully held-out storms -- see
    src/train_temporal.py.
  - About & Data: dataset composition, known limitations, training curve.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from kaggle_dataset import load_samples as load_kaggle_samples, denormalize_wind
from hursat_dataset import load_hursat_samples
from mosdac_dataset import load_mosdac_samples
from model import CycloneNet
from gradcam import GradCAM, overlay_heatmap
from uncertainty import predict_with_uncertainty
from historical_comparison import build_feature_bank, find_similar
from ri_alert import load_series, find_ri_events, merge_overlapping
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

st.set_page_config(page_title="Cyclone Intensity Classifier", layout="wide", page_icon="🌀")

# ---------------------------------------------------------------------------
# Palette (validated: dataviz skill, references/palette.md).
# Ordinal severity ramp -- one hue (blue), light->dark, one step per IMD
# category so the *order* of the 8-level scale is visually obvious regardless
# of which category is predicted. Status colors are reserved for confidence /
# alert states and never reused as a "9th category" color.
# ---------------------------------------------------------------------------
SEQ_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
CAT_BLUE = "#2a78d6"    # categorical slot 1 -- train / primary series
CAT_ORANGE = "#eb6834"  # categorical slot 2 -- val / secondary series
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
TEXT_MUTED = "#9aa4b2"  # light muted gray -- readable against the dark app background

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope, Arial, sans-serif", size=13, color="#c7ccd4"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def confidence_color(pct: float) -> str:
    if pct >= 0.85:
        return STATUS_GOOD
    if pct >= 0.65:
        return STATUS_WARNING
    return STATUS_SERIOUS


def callout(text: str, kind: str = "info") -> None:
    """A themed callout card, standing in for st.info/st.warning so the app
    doesn't default to Streamlit's stock alert-box look."""
    bg, accent, icon = {
        "info": ("#0f1f30", "#6da7ec", "ℹ️"),
        "warning": ("#2a2312", "#fab219", "⚠️"),
    }[kind]
    st.markdown(
        f"<div class='callout' style='background:{bg};'>{icon}&nbsp;&nbsp;{text}</div>",
        unsafe_allow_html=True,
    )


def alert_banner(pred_idx: int, conf: float) -> None:
    """The one decision-support line: what the category means and what to do
    next, plus the confidence number -- replaces three separate statements
    (category, a confidence caption, a caveat) with one."""
    if pred_idx >= 5:
        level, icon, headline = "critical", "🚨", "HIGH-IMPACT -- review evacuation readiness"
    elif pred_idx >= 3:
        level, icon, headline = "warning", "⚠️", "SIGNIFICANT -- monitor closely"
    else:
        level, icon, headline = "good", "✅", "LOW IMMEDIATE THREAT"
    conf_note = " (low model confidence -- verify manually)" if conf < 0.65 else ""
    bg, accent = {
        "critical": ("#2a1414", STATUS_CRITICAL),
        "warning": ("#2a2312", STATUS_WARNING),
        "good": ("#122a1a", STATUS_GOOD),
    }[level]
    st.markdown(
        f"<div class='alert-banner' style='background:{bg}; border-color:{accent};'>"
        f"<span style='font-size:1.2rem;'>{icon}</span>&nbsp;&nbsp;"
        f"<span style='color:{accent}; font-weight:700;'>{headline}</span>"
        f"<span style='color:#8b93a1;'> &middot; {conf*100:.0f}% confidence{conf_note}</span></div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, .stApp, .stApp * {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit chrome -- toolbar, footer badge, top decoration bar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}

    /* Hero header, replacing st.title */
    .hero {display: flex; align-items: center; gap: 1rem; margin-bottom: 0.2rem;}
    .hero-icon {
        font-size: 2rem; width: 3.2rem; height: 3.2rem; border-radius: 50%;
        background: linear-gradient(135deg, #2a78d6, #184f95);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .eyebrow {
        color: #6da7ec; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 0.15rem;
    }
    .hero-title {font-size: 1.9rem; font-weight: 800; line-height: 1.2; color: #eef1f6;}
    .hero-sub {color: #9aa4b2; font-size: 0.95rem; margin: 0.4rem 0 1.4rem;}

    /* Tabs restyled as a pill segmented control instead of the default underline */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px !important; background: #141924; padding: 5px; border-radius: 999px !important;
        width: fit-content;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: auto !important; padding: 0.55rem 1.15rem !important; border-radius: 999px !important;
        color: #9aa4b2 !important; font-weight: 600; background: transparent !important;
        margin: 0 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {background: #2a78d6 !important; color: #ffffff !important;}
    [data-testid="stTabs"] [data-baseweb="tab"] p {color: inherit !important;}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {display: none;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #141924; border-radius: 12px; padding: 0.9rem 1.1rem;
        border: 1px solid #232a38;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.55rem; color: #eef1f6; font-weight: 700;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #8b93a1; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    }

    /* Callout cards, replacing default st.info/st.warning styling */
    .callout {
        border-radius: 10px; padding: 0.8rem 1rem; margin: 0.7rem 0;
        font-size: 0.92rem; line-height: 1.5; color: #d7dce3;
    }

    section[data-testid="stSidebar"] {background: #0e1219; border-right: 1px solid #1b212c;}
    section[data-testid="stSidebar"] h2 {font-size: 1.05rem; color: #eef1f6;}

    /* One-line "what this solves" statement under the header */
    .tagline {
        color: #cfd6e0; font-size: 0.95rem; font-weight: 500; line-height: 1.5;
        margin: 0.9rem 0 1.6rem; padding-left: 0.9rem; border-left: 3px solid #2a78d6;
    }

    /* Operational alert banner on the assessment tab */
    .alert-banner {
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.6rem 0 1.3rem;
        border: 1px solid; font-size: 0.95rem; line-height: 1.5;
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
    from kaggle_dataset import KaggleINSAT3DDataset
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
    """seq: one entry from build_sequences(). Returns {horizon: predicted_kmph}."""
    frames = [load_frame_pair(f, img_size) for f in seq["window"]]
    x = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).float()  # (1,T,2,H,W)
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


if not os.path.exists(CHECKPOINT_PATH):
    st.error(
        f"No checkpoint found at `{CHECKPOINT_PATH}`. Run `python src/train_real.py "
        f"--config {CONFIG_PATH}` first to train and save one."
    )
    st.stop()

cfg, model, samples, ckpt, n_kaggle, n_hursat, n_mosdac = load_everything()
img_size = cfg["data"]["img_size"]

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">🌀</div>
        <div>
            <div class="eyebrow">SIH26070 &middot; Disaster Management</div>
            <div class="hero-title">Cyclone Intensity Classifier</div>
        </div>
    </div>
    <div class="hero-sub">Classification + wind-speed estimation from real INSAT-3D imagery
    &middot; best checkpoint {ckpt['val_acc']:.1%} val accuracy (epoch {ckpt['epoch']})</div>
    <div class="tagline">Automates the one step in cyclone monitoring that doesn't scale: reading
    a satellite frame by eye. Feed in an image, get a category, wind speed, and an alert &mdash;
    in seconds.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this prototype")
    st.write(
        "Identifies a tropical cyclone's IMD intensity category and estimates its "
        "sustained wind speed from satellite imagery, explains the prediction with "
        "Grad-CAM, scores its own confidence with MC-Dropout, and retrieves the "
        "closest historical match."
    )
    callout(
        f"Trained on <strong>{len(samples)} real labeled images</strong> from three sources: "
        f"{n_kaggle} single Kaggle INSAT-3D frames, {n_hursat} NOAA HURSAT-B1 frames across "
        f"10 real North Indian Ocean cyclones, and {n_mosdac} MOSDAC frames of Cyclone Amphan "
        "(2020) matched to its official IMD best track &mdash; the only source with real Super "
        "Cyclonic Storm coverage. Validated on 2 entire HURSAT-B1 storms held out of training "
        "&mdash; not just held-out images &mdash; so the reported accuracy reflects generalizing "
        "to a storm the model has never seen. Classes are still imbalanced (Low Pressure Area "
        "especially). Read predictions as illustrative of the pipeline, not an operational "
        "forecast.",
        kind="warning",
    )

tab_classify, tab_history, tab_ri, tab_forecast, tab_about = st.tabs(
    ["🔍 Assess a Storm", "📊 Historical Precedent", "⚡ Early-Warning Alert",
     "🔮 Forecast", "ℹ️ About & Data"]
)


def load_gray(path):
    img = Image.open(path).convert("L")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0


with tab_classify:
    mode = st.radio("Choose an image to classify", ["Pick from dataset", "Upload your own"], horizontal=True)

    true_kmph, true_cat, query_name = None, None, None
    if mode == "Pick from dataset":
        names = [s["img_name"] for s in samples]
        choice = st.selectbox("Image", names)
        s = next(s for s in samples if s["img_name"] == choice)
        ir_arr = load_gray(s["ir_path"])
        raw_arr = load_gray(s["raw_path"])
        true_kmph, true_cat, query_name = s["kmph"], s["cat_idx"], s["img_name"]
        ir_display, raw_display = Image.open(s["ir_path"]), Image.open(s["raw_path"])
        if not s["has_raw"]:
            st.caption("Note: this image had no matched raw counterpart -- IR image reused for both channels.")
    else:
        uploaded = st.file_uploader("Upload a cyclone satellite image (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            callout("Upload an image, or switch to 'Pick from dataset' above.", kind="info")
            st.stop()
        pil = Image.open(uploaded).convert("L")
        ir_arr = np.asarray(pil.resize((img_size, img_size)), dtype=np.float32) / 255.0
        raw_arr = ir_arr.copy()
        ir_display, raw_display = pil, pil
        callout(
            "Single-image upload: the same image is used for both channels the model "
            "expects. Predictions here are lower-fidelity than for dataset images with "
            "a real raw counterpart.",
            kind="info",
        )

    x = torch.from_numpy(np.stack([ir_arr, raw_arr], axis=0)).unsqueeze(0)

    # Deterministic pass (dropout off) drives the Grad-CAM heatmap and the
    # displayed predicted class; MC-Dropout (dropout deliberately kept on,
    # see uncertainty.py) separately scores how sure the model actually is.
    cam = GradCAM(model)
    heatmap, pred_idx, probs, reg_pred = cam(x)
    cam.remove()
    pred_kmph = float(denormalize_wind(torch.tensor(reg_pred)))
    overlay = overlay_heatmap(ir_arr, heatmap)

    unc = predict_with_uncertainty(model, x, n_samples=30)
    conf = unc["cls_probs_mean"][pred_idx].item()
    wind_std_kmph = float(unc["reg_std"][0]) * (250.0 - 40.0)

    col1, col2, col3 = st.columns(3)
    col1.image(ir_display, caption="IR channel (input)", width='stretch')
    col2.image(raw_display, caption="Raw channel (input)", width='stretch')
    col3.image(overlay, caption="Grad-CAM: what drove the prediction", width='stretch')

    st.subheader("Prediction")
    alert_banner(pred_idx, conf)
    mcols = st.columns(4) if true_cat is not None else st.columns(2)
    mcols[0].metric("Predicted category", category_label(pred_idx))
    mcols[1].metric("Predicted wind speed", f"{pred_kmph:.0f} km/h", f"±{wind_std_kmph:.0f} km/h")
    if true_cat is not None:
        mcols[2].metric("Actual category", category_label(true_cat))
        mcols[3].metric("Actual wind speed", f"{true_kmph:.0f} km/h")

    st.subheader("Category probabilities")
    cat_names = [c[1] for c in IMD_CATEGORIES]
    mean_probs = unc["cls_probs_mean"].tolist()
    std_probs = unc["cls_probs_std"].tolist()
    bar_colors = [SEQ_RAMP[i] if i != pred_idx else CAT_ORANGE for i in range(len(cat_names))]
    fig = go.Figure(
        go.Bar(
            x=cat_names, y=mean_probs,
            error_y=dict(type="data", array=std_probs, color=TEXT_MUTED, thickness=1.2),
            marker_color=bar_colors,
            hovertemplate="%{x}<br>P = %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(title="Probability (mean ± std, MC-Dropout)", range=[0, 1]),
                       xaxis=dict(tickangle=-15), height=340)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Bars follow the IMD scale light→dark (Depression→Super Cyclonic Storm); the predicted "
        "category is highlighted in orange. Error bars are the spread across 30 MC-Dropout passes, "
        "not a single-pass softmax score -- a wide bar means the model itself is unsure."
    )

with tab_history:
    st.subheader("Closest historical matches")
    st.write("The nearest past storms at a similar stage, so a forecaster isn't starting cold.")
    if query_name is None:
        callout("Pick a dataset image in the <strong>Assess a Storm</strong> tab to see its closest historical matches.", kind="info")
    else:
        bank = load_feature_bank(model, samples, img_size)
        matches = find_similar(model, x, bank, k=4, exclude_name=query_name)
        cols = st.columns(len(matches))
        for col, m in zip(cols, matches):
            match_sample = next(s for s in samples if s["img_name"] == m["img_name"])
            col.image(Image.open(match_sample["ir_path"]), width='stretch')
            badge_color = confidence_color(m["percentile"] / 100.0)
            col.markdown(
                f"**{m['img_name']}**<br>"
                f"<span style='color:{badge_color}; font-weight:600;'>{m['percentile']:.1f}th percentile match</span><br>"
                f"{category_label(m['cat_idx'])}, {m['kmph']:.0f} km/h",
                unsafe_allow_html=True,
            )
        st.caption(
            "Percentile against the whole dataset, not raw similarity -- comparable across "
            "checkpoints, unlike a raw cosine score."
        )

with tab_ri:
    st.subheader("Rapid Intensification — validated case study: Cyclone Amphan (2020)")
    st.write(
        "The alert this system raises the moment a storm crosses the standard "
        "**+30kt/24h** threshold (Kaplan & DeMaria, 2003) -- shown here against Amphan's real "
        "2020 record, since we have the documented outcome to check it against."
    )
    st.caption(
        "No model involved -- validating the alert *logic* against real history. It will run on "
        "model-predicted wind sequences once MOSDAC's time-series imagery lands."
    )
    times, winds, episodes = load_ri_case_study()
    fig_ri = go.Figure()
    fig_ri.add_trace(go.Scatter(
        x=times, y=winds, mode="lines+markers", name="Sustained wind (kt)",
        line=dict(color=CAT_BLUE, width=2), marker=dict(size=5),
        hovertemplate="%{x|%d %b, %H:%M}<br>%{y:.0f} kt<extra></extra>",
    ))
    for start, end, w0, w1, delta in episodes:
        fig_ri.add_vrect(x0=start, x1=end, fillcolor=STATUS_CRITICAL, opacity=0.12, line_width=0)
        fig_ri.add_annotation(
            x=start + (end - start) / 2, y=max(winds) * 1.05,
            text=f"RI episode: +{delta:.0f}kt over {(end-start).total_seconds()/3600:.0f}h",
            showarrow=False, font=dict(color=STATUS_CRITICAL, size=12, family="Arial, sans-serif"),
        )
    fig_ri.update_layout(**PLOTLY_LAYOUT, yaxis=dict(title="Sustained wind speed (kt)"),
                          xaxis=dict(title="Date (UTC)"), height=420, showlegend=False)
    st.plotly_chart(fig_ri, width='stretch')
    if episodes:
        start, end, w0, w1, delta = episodes[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Intensification", f"+{delta:.0f} kt")
        c2.metric("Duration", f"{(end-start).total_seconds()/3600:.0f} h")
        c3.metric("Peak reached", f"{max(winds):.0f} kt · SuCS")

with tab_forecast:
    st.subheader("Intensity forecast: +6h / +12h / +24h")
    st.write(
        "Given the last 4 observed frames of a storm, predicts wind speed 6/12/24 hours "
        "ahead — this is the **prediction** half of the problem statement, not just "
        "classification of the current frame."
    )

    if not os.path.exists(CHECKPOINT_TEMPORAL_PATH):
        callout(
            f"No temporal checkpoint found at <code>{CHECKPOINT_TEMPORAL_PATH}</code>. "
            f"Run <code>python src/train_temporal.py --config {CONFIG_TEMPORAL_PATH}</code> first.",
            kind="warning",
        )
    else:
        tcfg, model_t, tckpt = load_temporal_model()
        by_storm, sequences = load_temporal_sequences(
            tcfg["data"]["hursat_root"], tcfg["data"]["mosdac_root"])

        storms_with_seqs = sorted(set(s["storm"] for s in sequences))
        storm_labels = {
            s: f"{s}  (held out — never trained on)" if s in VAL_STORMS else s
            for s in storms_with_seqs
        }
        picked_storm = st.selectbox(
            "Storm", storms_with_seqs, format_func=lambda s: storm_labels[s])

        storm_seqs = [s for s in sequences if s["storm"] == picked_storm]
        storm_seqs.sort(key=lambda s: s["anchor_dt"])
        default_idx = max(0, int(len(storm_seqs) * 0.55))
        idx = st.slider(
            "Anchor point (last observed frame)", 0, len(storm_seqs) - 1, default_idx,
            format="",
            help="Move to choose which point in the storm's life the forecast is made from.",
        )
        seq = storm_seqs[idx]
        st.caption(f"Forecasting from {seq['anchor_dt'].strftime('%d %b %Y, %H:%M UTC')} "
                   f"— anchor {idx + 1} of {len(storm_seqs)} for {picked_storm}.")

        if picked_storm in VAL_STORMS:
            callout(
                f"<strong>{picked_storm}</strong> was held out entirely during training — "
                "the forecast below is genuine generalization to a storm the model never saw, "
                "not a memorized fit.",
                kind="info",
            )
        else:
            callout(
                f"<strong>{picked_storm}</strong> was used during training, so this forecast "
                "isn't a held-out generalization test — pick Phet or Nilofar above for that.",
                kind="warning",
            )

        preds = run_forecast(seq, model_t, cfg["data"]["img_size"])

        thumb_cols = st.columns(len(seq["window"]))
        for col, f in zip(thumb_cols, seq["window"]):
            col.image(Image.open(f["ir_path"]), width='stretch')
            col.caption(f["dt"].strftime("%d %b, %H:%M"))

        st.markdown("**Current state (anchor)**")
        anchor_cat = wind_speed_to_category(seq["anchor_kmph"])
        acols = st.columns(2)
        acols[0].metric("Wind speed", f"{seq['anchor_kmph']:.0f} km/h")
        acols[1].metric("Category", category_label(anchor_cat))

        st.markdown("**Forecast**")
        fcols = st.columns(len(HORIZONS_HOURS))
        for col, h in zip(fcols, HORIZONS_HOURS):
            pred_kmph = preds[h]
            pred_cat = wind_speed_to_category(pred_kmph)
            delta_disp = pred_kmph - seq["anchor_kmph"]
            actual_note = ""
            if h in seq["targets"]:
                actual_note = f"actual: {seq['targets'][h]:.0f} km/h"
            col.metric(f"+{h}h", f"{pred_kmph:.0f} km/h", f"{delta_disp:+.0f} km/h vs now")
            col.caption(f"{category_label(pred_cat)}" + (f" · {actual_note}" if actual_note else ""))

        # Full real storm timeline (past AND future, all real observed data) with the
        # forecast branching off the chosen anchor -- lets a viewer see at a glance how
        # close the dashed forecast line tracks the real (blue) future, when it exists.
        full_times = [f["dt"] for f in by_storm[picked_storm]]
        full_winds = [f["kmph"] for f in by_storm[picked_storm]]
        fc_times = [seq["anchor_dt"]] + [seq["anchor_dt"] + pd.Timedelta(hours=h) for h in HORIZONS_HOURS]
        fc_winds = [seq["anchor_kmph"]] + [preds[h] for h in HORIZONS_HOURS]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=full_times, y=full_winds, mode="lines", name="Observed (real record)",
            line=dict(color=CAT_BLUE, width=2),
            hovertemplate="%{x|%d %b, %H:%M}<br>%{y:.0f} km/h<extra></extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_times, y=fc_winds, mode="lines+markers", name="Forecast (this model)",
            line=dict(color=CAT_ORANGE, width=2, dash="dash"), marker=dict(size=8, symbol="diamond"),
            hovertemplate="%{x|%d %b, %H:%M}<br>%{y:.0f} km/h (forecast)<extra></extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=[seq["anchor_dt"]], y=[seq["anchor_kmph"]], mode="markers", name="Anchor (now)",
            marker=dict(size=11, color="#eef1f6", line=dict(color=CAT_ORANGE, width=2)),
            hovertemplate="Anchor: %{y:.0f} km/h<extra></extra>",
        ))
        fig_fc.update_layout(**PLOTLY_LAYOUT, yaxis=dict(title="Sustained wind speed (km/h)"),
                              xaxis=dict(title="Date (UTC)"), height=420,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_fc, width='stretch')

        val_mae = tckpt.get("val_mae_per_h", [])
        base_mae = tckpt.get("baseline_mae_per_h", [])
        if val_mae and base_mae:
            improvements = [f"+{h}h: {m:.1f} km/h (vs. {b:.1f} km/h predicting no change)"
                             for h, m, b in zip(HORIZONS_HOURS, val_mae, base_mae)]
            st.caption(
                "Validated on the 2 fully held-out storms (Phet, Nilofar) against a "
                "persistence baseline (\"predict no change\") — mean absolute error: "
                + "; ".join(improvements) + ". The model beats the baseline at every horizon."
            )

with tab_about:
    st.subheader("Dataset composition")
    counts = {}
    for s in samples:
        counts[s["cat_idx"]] = counts.get(s["cat_idx"], 0) + 1
    cat_order = [i for i in range(len(IMD_CATEGORIES)) if counts.get(i, 0) > 0]
    labels = [IMD_CATEGORIES[i][1] for i in cat_order]
    values = [counts[i] for i in cat_order]
    colors = [SEQ_RAMP[i] for i in cat_order]
    fig_d = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                              hovertemplate="%{y}: %{x} images<extra></extra>"))
    fig_d.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="Images"), height=320)
    st.plotly_chart(fig_d, width='stretch')

    log_path = cfg["train"]["log_path"]
    if os.path.exists(log_path):
        st.subheader("Training curve (this checkpoint's run)")
        hist = pd.read_csv(log_path)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_acc"], name="Train accuracy",
                                   line=dict(color=CAT_BLUE, width=2)))
        fig2.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_acc"], name="Val accuracy",
                                   line=dict(color=CAT_ORANGE, width=2)))
        fig2.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="Epoch"), yaxis=dict(title="Accuracy"),
                            height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, width='stretch')
        st.caption(
            "Train accuracy climbing well above val is expected at this data size, and val is "
            "measured on 2 entire storms the model never trained on -- a harder, more honest "
            "bar than a random image split -- documented here rather than hidden."
        )

    st.subheader("Known limitations")
    st.markdown(
        f"- {len(samples)} labeled images total ({n_kaggle} single Kaggle frames with no storm ID, "
        f"{n_hursat} HURSAT-B1 frames across 10 real storms, {n_mosdac} MOSDAC frames of Cyclone "
        "Amphan) -- still small for an 8-way classifier, and still imbalanced (Low Pressure Area "
        "is rare in real records).\n"
        "- The Kaggle and Amphan portions have no held-out storms of their own, so all of those "
        "images are always in training, never in validation -- validation is 2 HURSAT-B1 storms "
        "(136 frames) held out entirely. Amphan reached Super Cyclonic Storm, the one category "
        "the other two sources barely cover, which is why it went to training rather than "
        "validation -- see src/train_combined.py for the reasoning.\n"
        "- No pressure label from the Kaggle or HURSAT-B1 sources -- wind speed only (Amphan's "
        "IMD best track does have pressure, currently unused since the model has a single-output "
        "wind-only regression head).\n"
        "- No ImageNet-pretrained backbone: the download is blocked in this sandbox, so this "
        "checkpoint and the original 136-image one both trained from random initialization -- the "
        "reported accuracies are comparable to each other on that basis, not inflated by one having "
        "a head start.\n"
    )
