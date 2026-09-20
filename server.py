"""
Python HTTP Server for Cyclone AI Meteorological Dashboard.
Reuses the exact PyTorch models, datasets, and inference logic from src/.
Serves the custom modern Web UI on http://localhost:8501.
"""
import io
import os
import sys
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Insert src directory into python path
SYS_SRC = os.path.join(os.path.dirname(__file__), "src")
if SYS_SRC not in sys.path:
    sys.path.insert(0, SYS_SRC)

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

CACHE = {}

def init_backend():
    print("Loading PyTorch models and datasets...")
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
    img_size = cfg["data"]["img_size"]

    ds = KaggleINSAT3DDataset(samples, img_size, train=False)
    feature_bank = build_feature_bank(model, ds)

    tcfg = load_config(CONFIG_TEMPORAL_PATH)
    model_t = TemporalCycloneNet(
        backbone_ckpt=tcfg["model"]["backbone_ckpt"],
        freeze_backbone=tcfg["model"]["freeze_backbone"],
        gru_hidden=tcfg["model"]["gru_hidden"],
    )
    tckpt = torch.load(CHECKPOINT_TEMPORAL_PATH, map_location="cpu", weights_only=False)
    model_t.load_state_dict(tckpt["model_state"])
    model_t.eval()

    by_storm = load_storm_timelines(tcfg["data"]["hursat_root"], tcfg["data"]["mosdac_root"])
    sequences = build_sequences(by_storm)

    ri_times, ri_winds, ri_episodes = None, None, None
    if os.path.exists(BESTTRACK_PATH):
        ri_times, ri_winds, ri_episodes = load_ri_case_study()

    CACHE.update({
        "cfg": cfg, "model": model, "samples": samples, "ckpt": ckpt,
        "n_kaggle": len(kaggle_samples), "n_hursat": len(hursat_samples), "n_mosdac": len(mosdac_samples),
        "img_size": img_size, "feature_bank": feature_bank,
        "tcfg": tcfg, "model_t": model_t, "tckpt": tckpt,
        "by_storm": by_storm, "sequences": sequences,
        "ri_times": ri_times, "ri_winds": ri_winds, "ri_episodes": ri_episodes,
    })
    print("Backend initialization complete!")

def load_ri_case_study():
    times, winds = load_series(BESTTRACK_PATH)
    events = find_ri_events(times, winds)
    episodes = merge_overlapping(events)
    return times, winds, episodes

def load_gray(path, img_size):
    img = Image.open(path).convert("L")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0

def img_to_base64(pil_img):
    if isinstance(pil_img, np.ndarray):
        if pil_img.dtype != np.uint8:
            pil_img = (np.clip(pil_img, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(pil_img)
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
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


class CycloneRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/samples":
            samples_list = []
            for s in CACHE["samples"]:
                samples_list.append({
                    "img_name": s["img_name"],
                    "kmph": round(float(s["kmph"]), 1),
                    "cat_idx": s["cat_idx"],
                    "cat_name": category_label(s["cat_idx"]),
                    "pressure_mb": s.get("pressure_mb"),
                    "has_raw": s["has_raw"],
                })
            self.send_json({"samples": samples_list, "total": len(samples_list)})

        elif path == "/api/assess":
            img_name = query.get("img_name", [None])[0]
            samples = CACHE["samples"]
            img_size = CACHE["img_size"]
            model = CACHE["model"]

            if not img_name:
                s = samples[0]
            else:
                s = next((sample for sample in samples if sample["img_name"] == img_name), samples[0])

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

            cat_names = [c[1] for c in IMD_CATEGORIES]
            mean_probs = unc["cls_probs_mean"].tolist()
            std_probs = unc["cls_probs_std"].tolist()

            res = {
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
                "true_cat_idx": s["cat_idx"],
                "true_cat_name": category_label(s["cat_idx"]),
                "true_kmph": round(s["kmph"], 1),
                "true_pressure": s.get("pressure_mb"),
                "cat_names": cat_names,
                "mean_probs": [round(p, 4) for p in mean_probs],
                "std_probs": [round(p, 4) for p in std_probs],
            }
            self.send_json(res)

        elif path == "/api/history":
            img_name = query.get("img_name", [None])[0]
            samples = CACHE["samples"]
            img_size = CACHE["img_size"]
            model = CACHE["model"]
            bank = CACHE["feature_bank"]

            if not img_name:
                s = samples[0]
            else:
                s = next((sample for sample in samples if sample["img_name"] == img_name), samples[0])

            ir_arr = load_gray(s["ir_path"], img_size)
            raw_arr = load_gray(s["raw_path"], img_size)
            x = torch.from_numpy(np.stack([ir_arr, raw_arr], axis=0)).unsqueeze(0)

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
            self.send_json({"query_name": s["img_name"], "matches": matches_res})

        elif path == "/api/ri_alert":
            times, winds, episodes = CACHE["ri_times"], CACHE["ri_winds"], CACHE["ri_episodes"]
            times_str = [t.strftime("%Y-%m-%d %H:%M UTC") for t in times] if times else []
            episodes_res = []
            if episodes:
                for start, end, w0, w1, delta in episodes:
                    episodes_res.append({
                        "start": start.strftime("%Y-%m-%d %H:%M UTC"),
                        "end": end.strftime("%Y-%m-%d %H:%M UTC"),
                        "w0": round(w0, 1),
                        "w1": round(w1, 1),
                        "delta": round(delta, 1),
                        "duration_hours": round((end - start).total_seconds() / 3600.0, 1),
                    })
            self.send_json({
                "times": times_str,
                "winds": [round(w, 1) for w in winds] if winds else [],
                "episodes": episodes_res,
                "peak_wind_kt": round(max(winds), 1) if winds else None,
            })

        elif path == "/api/forecast":
            sequences = CACHE["sequences"]
            by_storm = CACHE["by_storm"]
            model_t = CACHE["model_t"]
            img_size = CACHE["img_size"]

            storms_with_seqs = sorted(set(s["storm"] for s in sequences))
            picked_storm = query.get("storm", [storms_with_seqs[0]])[0]
            if picked_storm not in storms_with_seqs:
                picked_storm = storms_with_seqs[0]

            storm_seqs = [s for s in sequences if s["storm"] == picked_storm]
            storm_seqs.sort(key=lambda s: s["anchor_dt"])
            
            anchor_idx_param = query.get("anchor_idx", [None])[0]
            if anchor_idx_param is not None:
                try:
                    idx = int(anchor_idx_param)
                except ValueError:
                    idx = max(0, int(len(storm_seqs) * 0.55))
            else:
                idx = max(0, int(len(storm_seqs) * 0.55))
            
            idx = max(0, min(idx, len(storm_seqs) - 1))
            seq = storm_seqs[idx]
            preds = run_forecast(seq, model_t, img_size)

            full_times = [f["dt"].strftime("%Y-%m-%d %H:%M") for f in by_storm[picked_storm]]
            full_winds = [round(f["kmph"], 1) for f in by_storm[picked_storm]]

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
                        "start": start.strftime("%Y-%m-%d %H:%M"),
                        "end": end.strftime("%Y-%m-%d %H:%M"),
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

            self.send_json({
                "storms": storms_with_seqs,
                "val_storms": list(VAL_STORMS),
                "picked_storm": picked_storm,
                "total_anchors": len(storm_seqs),
                "anchor_idx": idx,
                "anchor_dt": seq["anchor_dt"].strftime("%d %b %Y, %H:%M UTC"),
                "anchor_kmph": round(seq["anchor_kmph"], 1),
                "anchor_cat": category_label(wind_speed_to_category(seq["anchor_kmph"])),
                "is_held_out": picked_storm in VAL_STORMS,
                "horizons": {
                    str(h): {
                        "pred_kmph": round(preds[h], 1),
                        "pred_cat": category_label(wind_speed_to_category(preds[h])),
                        "delta_kmph": round(preds[h] - seq["anchor_kmph"], 1),
                        "actual_kmph": round(seq["targets"][h], 1) if h in seq["targets"] else None
                    } for h in HORIZONS_HOURS
                },
                "full_times": full_times,
                "full_winds": full_winds,
                "fc_times": fc_times,
                "fc_winds": fc_winds,
                "rt_episodes": rt_episodes_res,
                "thumbnails": thumbs,
            })

        elif path == "/api/about":
            samples = CACHE["samples"]
            counts = {}
            for s in samples:
                counts[s["cat_idx"]] = counts.get(s["cat_idx"], 0) + 1

            cat_composition = []
            for idx, item in enumerate(IMD_CATEGORIES):
                cat_name, cat_code = item[0], item[1]
                if counts.get(idx, 0) > 0:
                    cat_composition.append({
                        "cat_idx": idx,
                        "name": cat_name,
                        "count": counts.get(idx, 0),
                    })

            log_path = CACHE["cfg"]["train"]["log_path"]
            hist_data = []
            if os.path.exists(log_path):
                hist = pd.read_csv(log_path)
                for _, row in hist.iterrows():
                    hist_data.append({
                        "epoch": int(row["epoch"]),
                        "train_acc": round(float(row["train_acc"]), 4),
                        "val_acc": round(float(row["val_acc"]), 4),
                    })

            self.send_json({
                "total_samples": len(samples),
                "n_kaggle": CACHE["n_kaggle"],
                "n_hursat": CACHE["n_hursat"],
                "n_mosdac": CACHE["n_mosdac"],
                "val_accuracy": round(CACHE["ckpt"]["val_acc"] * 100.0, 1),
                "val_epoch": CACHE["ckpt"]["epoch"],
                "composition": cat_composition,
                "training_history": hist_data,
            })

        else:
            if path == "/" or path == "":
                filepath = os.path.join(os.path.dirname(__file__), "static", "index.html")
            else:
                rel = path.lstrip("/")
                filepath = os.path.join(os.path.dirname(__file__), "static", rel)

            if os.path.exists(filepath) and os.path.isfile(filepath):
                ext = os.path.splitext(filepath)[1].lower()
                content_types = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".svg": "image/svg+xml",
                    ".ico": "image/x-icon",
                }
                ctype = content_types.get(ext, "text/plain")
                with open(filepath, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")

def run_server(port=8501):
    init_backend()
    server_address = ("", port)
    httpd = HTTPServer(server_address, CycloneRequestHandler)
    print(f"Cyclone AI Server running live at http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
