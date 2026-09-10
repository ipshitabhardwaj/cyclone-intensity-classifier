# Cyclone Intensity Classifier & Forecaster — SIH26070

**SIH26070** · AI/ML system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite data · Ministry of Earth Sciences / IMD · Disaster Management theme

An AI/ML system that classifies tropical cyclone intensity from real multi-channel
satellite imagery on IMD's 8-category scale (Low Pressure Area → Depression → Deep
Depression → Cyclonic Storm → Severe CS → Very Severe CS → Extremely Severe CS →
Super Cyclonic Storm), estimates wind speed **and central pressure**, forecasts
intensity **+6h/+12h/+24h ahead**, flags Rapid Intensification, and explains every
prediction with a Grad-CAM heatmap and an MC-Dropout confidence score.

For the full narrative (methodology, honest results, what's been tried and
rejected, what's left) see **[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)**. This
file is the shorter "what is this / how do I run it" entry point.

## Status: trained end-to-end on real satellite data, not synthetic

Earlier versions of this project used synthetic data and a single small
Kaggle dataset (136 images, no storm ids, no pressure). That stage is done.
The current pipeline trains on **1,032 real labeled images from three
independent sources**, unified into one pool:

| Source | What it is | Volume |
|---|---|---|
| Kaggle "INSAT3D Infrared & Raw Cyclone Imagery" | Public dataset, no storm id, wind-speed label only | 136 images |
| NOAA HURSAT-B1 v06 | 10 real North Indian Ocean cyclones (2008–2016) | 680 frames |
| MOSDAC (Cyclone Amphan, 2020) | Real MOSDAC data order matched to the official IMD best track | 216 frames |

**Model:** CycloneNet — an EfficientNet-B0 backbone taking a 2-channel (IR + raw)
input, with an 8-way classification head and a 2-output regression head (wind
speed + pressure, mb).

**Validation is a storm-held-out split** — 2 entire HURSAT-B1 storms (Phet,
Nilofar) never seen in training at all, not just held-out images — so the
numbers below reflect generalizing to an unseen storm, not memorizing frames
from one already seen 3 hours earlier.

**Current honest results** (held-out storms, `checkpoints_combined/best.pt`):

| Metric | Value |
|---|---|
| Classification accuracy | 41.9% |
| Wind speed MAE | 27.2 km/h |
| Pressure MAE | 10.2 mb |
| MC-Dropout calibration (ECE) | 0.103 (moderately over-confident) |

**Temporal forecaster** (`checkpoints_temporal/best.pt`) — given the last 4
observed frames of a storm, predicts wind speed **+6h/+12h/+24h ahead**,
beating a persistence ("no change") baseline at every horizon on the same
held-out storms:

| Horizon | Persistence baseline (MAE) | Forecaster (MAE) |
|---|---|---|
| +6h  | 9.63 km/h  | 9.05 km/h  |
| +12h | 19.26 km/h | 16.79 km/h |
| +24h | 38.48 km/h | 30.59 km/h |

Backbone is frozen from `checkpoints_combined/best.pt`; only a small GRU head
is trained on top of it. See `PROJECT_OVERVIEW.md` section 1 for why the
model predicts a *delta* from current wind speed rather than an absolute value.

**Tried and honestly rejected:** class-weighted loss for the Low Pressure Area
imbalance (4 training examples) made overall accuracy worse (36.0% vs 41.9%)
and didn't fix LPA recall (still 0.00) — the code is in `train_combined.py`
but was not adopted. See `PROJECT_OVERVIEW.md` section 4 for why.

## Dashboard (Streamlit)

```bash
streamlit run app.py
```

Five tabs, all reading live from the current checkpoints and manifests
(nothing hardcoded):

- **🔍 Assess a Storm** — classify any dataset image or an uploaded one; shows
  predicted category, wind speed, pressure, a Grad-CAM heatmap, and an
  MC-Dropout confidence score.
- **📊 Historical Precedent** — retrieves the closest real historical matches
  by embedding similarity ("this looks like Amphan at similar intensity").
- **⚡ Early-Warning Alert** — Rapid Intensification detection (Kaplan &
  DeMaria 2003 threshold) validated against Amphan's real best-track record.
- **🔮 Forecast** — pick any storm and time point, see the temporal model's
  own live +6h/+12h/+24h forecast plotted against the real outcome, plus an
  RI check run on the model's own predictions (not just historical data).
- **ℹ️ About & Data** — dataset composition, known limitations, honest caveats.

## Project structure

```
cyclone-mvp/
├── app.py                          # Streamlit dashboard (5 tabs, see above)
├── PROJECT_OVERVIEW.md             # full narrative: methodology, results, what's left
├── configs/
│   ├── config_combined.yaml        # the CURRENT pipeline: classifier + pressure head
│   ├── config_temporal.yaml        # temporal forecaster
│   ├── config_real.yaml            # legacy: Kaggle-only, wind-only (superseded)
│   └── config.yaml                 # legacy: synthetic/CyINSAT-schema data (superseded)
├── src/
│   ├── utils.py                    # IMD category thresholds, config/seed helpers
│   ├── model.py                    # EfficientNet-B0, 2-channel input, dual head
│   ├── kaggle_dataset.py           # shared sample format + Dataset used by all 3 sources
│   ├── hursat_dataset.py           # NOAA HURSAT-B1 preprocessing
│   ├── mosdac_dataset.py           # MOSDAC/Amphan preprocessing (GeoTIFF, best-track matching)
│   ├── train_combined.py           # CURRENT training loop: all 3 sources, storm-held-out val
│   ├── temporal_dataset.py         # builds per-storm sequences for the forecaster
│   ├── temporal_model.py           # GRU-over-frozen-backbone forecaster
│   ├── train_temporal.py           # forecaster training loop, persistence-baseline comparison
│   ├── ri_alert.py                 # Rapid Intensification threshold detection (source-agnostic)
│   ├── gradcam.py                  # Grad-CAM explainability
│   ├── uncertainty.py              # MC-Dropout confidence estimation
│   ├── calibration_check.py        # is MC-Dropout confidence actually calibrated? (ECE + reliability table)
│   ├── historical_comparison.py    # embedding-based nearest-neighbour retrieval
│   └── train_real.py / evaluate_real.py / train.py / evaluate.py   # legacy, superseded by train_combined.py
├── data/
│   ├── kaggle_insat3d/             # the 136-image Kaggle set
│   ├── hursat_processed/           # preprocessed HURSAT-B1 frames + manifest.csv
│   ├── mosdac_processed/           # preprocessed MOSDAC/Amphan frames + manifest.csv
│   └── besttrack/                  # IMD best-track records (Amphan)
├── checkpoints_combined/best.pt    # CURRENT production classifier + regression checkpoint
├── checkpoints_temporal/best.pt    # CURRENT production forecaster checkpoint
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# Train the classifier + pressure head (all 3 real data sources)
python src/train_combined.py --config configs/config_combined.yaml

# Train the temporal forecaster (reuses the classifier's frozen backbone --
# retrain this too if you retrain the classifier, since the backbone's
# feature space will have changed)
python src/train_temporal.py --config configs/config_temporal.yaml

# Check MC-Dropout calibration
python src/calibration_check.py --config configs/config_combined.yaml --checkpoint checkpoints_combined/best.pt

# Launch the dashboard
streamlit run app.py
```

`pretrained: true` in `configs/config_combined.yaml` will only actually take
effect where `download.pytorch.org` is reachable — it's currently blocked in
the sandbox this was built in, so the model trains from random
initialization there. Worth re-testing on a machine with normal internet
access; a real ImageNet-pretrained backbone is the single biggest lever left
on accuracy (see `PROJECT_OVERVIEW.md` section 4).

## Storm-held-out validation (why it matters)

Train/val splits happen at the **storm level**, not the frame level —
`VAL_STORMS = {"PHET", "NILOFAR"}` in `train_combined.py` and
`temporal_dataset.py` are held out entirely, never seen in training. A random
split over 3-hourly frames of the same storm would put near-duplicate frames
on both sides and inflate accuracy without proving anything about
generalization to a storm the model hasn't seen — a common mistake in
published work on this exact task. Kaggle and MOSDAC/Amphan samples all go
into training (Kaggle has no storm id to hold out by; Amphan is the only real
Super Cyclonic Storm source, so holding it out would remove the one thing it
was added to fix, for a validation signal HURSAT-B1 already provides).

## What's left

See `PROJECT_OVERVIEW.md` section 4 for the full, current, honest list. In
short: accuracy is still modest (41.9%) and the ImageNet-pretrained-backbone
lever remains blocked in this environment; the temporal forecaster's held-out
set is small (2 storms); MC-Dropout is measurably over-confident (not yet
corrected, e.g. via temperature scaling); and there's no lightweight
deployment path (ONNX/API) yet.
