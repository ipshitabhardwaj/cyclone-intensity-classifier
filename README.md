# Cyclone Intensity Classification — SIH26070 MVP

An AI/ML system that classifies tropical cyclone intensity from multi-channel
satellite imagery (IMD's 8-category scale: LPA → Depression → Deep Depression
→ Cyclonic Storm → Severe CS → Very Severe CS → Extremely Severe CS → Super
Cyclonic Storm), estimates wind speed and central pressure, and explains each
prediction with a Grad-CAM heatmap over the storm structure that drove it.

This is the **classification slice** of the full SIH26070 pipeline (identification
+ classification + prediction). Genesis/detection and multi-day track
prediction are the next build phases — see "Next steps" below.

## Status: pipeline verified end-to-end on synthetic data

No real satellite data was available at build time (MOSDAC registration /
CyINSAT download take time to arrange), so the pipeline was built and
smoke-tested against **synthetic data that mirrors the real CyINSAT schema
exactly**. Swap in the real dataset and nothing else changes — same file
layout, same column names, same code.

A 4-epoch CPU smoke run on 24 synthetic storms already shows the loop is
learning real structure (not just running without crashing):

| Epoch | Val Accuracy | Val Wind MAE | Val Pressure MAE |
|-------|-------------|--------------|-------------------|
| 1     | 23.1%       | 44.3 km/h    | 23.5 hPa          |
| 4     | 55.1%       | 18.2 km/h    | 9.0 hPa           |

Grad-CAM on the synthetic eyewall structure correctly highlights the ring
(see `outputs/gradcam/*.png`) — the model is attending to storm structure,
not memorizing noise. With real CyINSAT data, more epochs, and a GPU, expect
substantially better numbers (published work on this exact task reports
90%+ classification accuracy and single-digit-hPa pressure MAE — see the
papers cited in the project chat history).

## Status: also trained on real INSAT-3D data (small, but real)

We found a real, already-public dataset with no approval wait — **[INSAT3D
Infrared & Raw Cyclone Imagery (2012–2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)**
by Sshubam Verma — and trained the same model on it. Be upfront about its
shape before quoting these numbers anywhere: it's **136 labeled images
total**, with **no cyclone/storm id or timestamp** (just an image and a wind
speed in knots), and **no pressure label**. That means: no storm-level split
was possible (a plain random 80/20 split was used instead — see
`kaggle_dataset.py`'s docstring), the regression head predicts wind speed
only, and both rare classes (Depression: 2 images, Super Cyclonic Storm: 1
image) are too few to evaluate meaningfully. Treat this as **"the real-data
plumbing works,"** not a benchmark result — CyINSAT (111k images, 39 storms,
proper storm ids) or full MOSDAC access is what a credible number for the
pitch deck needs.

30-epoch CPU run, no ImageNet pretraining (weight download is blocked in
this sandbox — see below), 109 train / 27 val images:

| Metric | Value |
|---|---|
| Best val accuracy (epoch 16) | 44.4% |
| Val wind speed MAE | 30.4 km/h |
| Val weighted F1 | 0.40 |

Train accuracy climbed to 93%+ while val accuracy stayed noisy in the
25–44% band — that gap **is** the 109-image dataset, not a bug; expect it to
close substantially with CyINSAT's 111k images and real ImageNet weights.
Grad-CAM outputs for 6 validation storms are in `outputs_real/gradcam/`.
Reproduce with:

```bash
python src/train_real.py --config configs/config_real.yaml
python src/evaluate_real.py --config configs/config_real.yaml --checkpoint checkpoints_real/best.pt
```

## Project structure

```
cyclone-mvp/
├── configs/config.yaml       # all paths + hyperparameters live here
├── src/
│   ├── utils.py               # IMD category thresholds, config/seed helpers
│   ├── synthetic_data.py      # generates CyINSAT-schema stand-in data
│   ├── dataset.py             # PyTorch Dataset, storm-level train/val split
│   ├── model.py                # EfficientNet-B0, N-channel input, dual head
│   ├── gradcam.py              # Grad-CAM explainability
│   ├── train.py                 # training loop (CyINSAT-schema / synthetic data)
│   ├── evaluate.py              # confusion matrix, MAE, Grad-CAM (synthetic path)
│   ├── kaggle_dataset.py        # Dataset for the real Kaggle INSAT-3D data (see below)
│   ├── train_real.py            # training loop for the real Kaggle data
│   └── evaluate_real.py         # confusion matrix, MAE, Grad-CAM (real-data path)
├── data/
│   ├── cyinsat/                  # synthetic (or real CyINSAT-schema) data lives here
│   └── kaggle_insat3d/           # the real Kaggle dataset (see "Real-data run" above)
├── checkpoints/, checkpoints_real/    # saved model weights (synthetic vs. real-data run)
├── outputs/, outputs_real/            # training logs + Grad-CAM images
└── requirements.txt
```

## Data schema (matches CyINSAT)

```
data/cyinsat/
├── details.csv        # one row per cyclone: cyclone_id, name, basin, max_wind_kmph, min_pressure_hpa, num_frames
├── parameters.csv      # one row per frame: cyclone_id, frame_idx, wind_speed_kmph, pressure_hpa, category_idx,
│                        #                     lat, lon, ir1_path, ir2_path, mir_path, wv_path
└── images/
    └── <cyclone_id>/
        ├── IR1/000.png, 001.png, ...
        ├── IR2/...
        ├── MIR/...
        └── WV/...
```

Train/val splits happen at the **storm level**, not the frame level — frames
from the same cyclone never appear in both sets, otherwise the model can
cheat by memorizing a storm's specific cloud texture instead of learning the
general wind-speed ↔ structure relationship. This matters a lot for judges
who understand ML — it's a common mistake in published student projects on
this exact task.

## Running it here (CPU, synthetic data — already done once)

```bash
pip install -r requirements.txt
python src/synthetic_data.py configs/config.yaml   # regenerate synthetic data
python src/train.py --config configs/config.yaml
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

## Scaling up beyond the 136-image Kaggle run

1. **Get more data.** Either wait on [MOSDAC](https://www.mosdac.gov.in/)
   registration approval for INSAT-3D/3DR archives directly, or track down
   the **CyINSAT** dataset (39 North Indian Ocean cyclones 2014–2022, 111k
   images, already labeled with wind/pressure/position AND cyclone ids —
   as of writing it isn't actually published on Kaggle despite the paper
   saying it would be; try emailing the corresponding author or checking
   Kaggle again later).
2. **Reshape it to match the schema above.** If the source distributes a
   different folder layout, write a small conversion script that produces
   `details.csv` / `parameters.csv` / `images/<cyclone_id>/<channel>/*.png`
   in this exact format — everything downstream (`dataset.py` onward) then
   works unchanged.
3. **Point `configs/config.yaml`'s `data.root` at the new folder.** No code
   changes needed.
4. **On Colab:** Runtime → Change runtime type → GPU. Then:
   ```bash
   !pip install -r requirements.txt
   !python src/train.py --config configs/config.yaml
   ```
   With a GPU, `pretrained: true` in the config will actually succeed in
   downloading ImageNet weights (blocked in this sandbox's network), which
   should meaningfully improve accuracy and reduce epochs needed.
5. **On Kaggle:** create a notebook, add the CyINSAT dataset as a Kaggle
   Dataset input, set accelerator to GPU T4/P100, upload this repo as a
   Kaggle Dataset or paste the `src/` files into notebook cells, then run
   the same two commands.
6. **Bump `train.epochs`** in the config to 30–50 once on a GPU — the 4-epoch
   CPU smoke test above is just a correctness check, not a real training run.

## Next steps (to complete the full SIH26070 vision)

- **Genesis/detection stage**: add a binary "cyclonic disturbance present"
  classifier or lightweight object detector running on raw MOSDAC frames
  *before* a storm is officially named, to demonstrate lead-time over IMD's
  manual process — this is the strongest differentiator for judges.
- **Track + intensity prediction**: a ConvLSTM or small transformer over a
  24–48h window of frames (this classifier's backbone can be reused as the
  per-frame feature extractor feeding into it), forecasting +24h/+48h/+72h
  track and intensity, plus a rapid-intensification early-warning flag.
- **Multi-source fusion**: concatenate scatterometer wind vectors and ERA5
  reanalysis features (SST, wind shear) with the image-derived features
  before the prediction head.
- **Dashboard**: a Streamlit app (already in `requirements.txt`) showing
  live classification, the Grad-CAM overlay, and a track map — this is the
  fastest way to turn this repo into a compelling live demo for the pitch.
- **Uncertainty quantification**: MC-dropout or an ensemble to produce a
  cone-of-uncertainty for track predictions, mirroring IMD's own bulletins.

## Notes on the synthetic data generator

`src/synthetic_data.py` renders a Gaussian "eyewall" ring whose radius,
sharpness, and (above a threshold) a visible "eye" scale with a synthetic
wind-speed lifecycle curve (ramp-up, peak, decay) per storm — it is **not**
real satellite physics, just enough structure for the pipeline to have a
genuine signal to learn, so a working training loop can be verified before
real data is in hand. Delete `data/cyinsat/` and rerun the generator any time
you want a fresh synthetic set (e.g. more storms, different `img_size`).
