# Cyclone Intensity Classifier — Project Overview

**SIH26070** · AI/ML-based system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite data · Ministry of Earth Sciences / IMD · Disaster Management theme

*Last updated: this document reflects the state of the project as of the latest retrain (checkpoint `checkpoints_combined/best.pt`, 40.4% validation accuracy).*

---

## 1. What is done

**Real data pipeline (three independent sources, unified into one training pool):**

- **Kaggle "INSAT3D Infrared & Raw Cyclone Imagery"** — 136 single-frame images, no storm ID, wind-speed label only. The original seed dataset.
- **NOAA HURSAT-B1 v06** — 680 frames across 10 real North Indian Ocean cyclones (Nargis, Phet, Phailin, Madi, Hudhud, Nilofar, Komen, Chapala, Kyant, Vardah; 2008–2016), chosen specifically because IBTrACS record counts showed Depression/Deep Depression/Cyclonic Storm were under-represented in the Kaggle set alone.
- **MOSDAC (Cyclone Amphan, 2020)** — 216 frames from a real MOSDAC data order (652 raw GeoTIFF files, TIR1 + WV bands, 30-minute cadence), matched against Amphan's official IMD best track. This is the only source with real Super Cyclonic Storm coverage — Amphan peaked at that category.

Total: **1,032 real labeled images**, spanning all 8 IMD intensity categories.

**Model:** CycloneNet — an EfficientNet-B0 backbone adapted to take a 2-channel input (IR + a second "raw" channel), with two heads: an 8-way classification head (IMD category) and a wind-speed regression head.

**Training:** `src/train_combined.py` trains on all three sources together. Validation is a **storm-held-out split** — 2 entire HURSAT-B1 storms (Phet, Nilofar) never seen in training at all, not just held-out images. This matters: a random split of 3-hourly frames from the same storm would put near-duplicate images on both sides and inflate the score without proving anything about generalization. Kaggle and Amphan images all go into training (neither has a storm ID to hold out cleanly, and Amphan is the only real Super Cyclonic Storm source, so holding it out would remove the one thing it was added to fix).

**Current best checkpoint:** 40.4% validation accuracy on 2 completely unseen storms — a harder and more honest bar than the original 44.4% baseline, which was measured on a random image split with no storm awareness.

**Explainability & trust features (already built and wired into the dashboard):**
- **Grad-CAM** (`src/gradcam.py`) — visual heatmap of what the model looked at.
- **MC-Dropout uncertainty** (`src/uncertainty.py`) — confidence estimate per prediction, flagged when low.
- **Historical precedent retrieval** (`src/historical_comparison.py`) — embeds every training image once, retrieves the closest real historical matches by cosine similarity, shown as a percentile so it's comparable across checkpoints.
- **Rapid Intensification alerting** (`src/ri_alert.py`) — real threshold-based RI detection (Kaplan & DeMaria 2003), validated against Amphan's actual best-track record, which shows the exact real-world RI episode the logic correctly flags.

**Dashboard** (`app.py`, Streamlit): 4 tabs — Assess a Storm, Historical Precedent, Early-Warning Alert, About & Data. Custom dark theme, a one-line problem statement up top, a compact operational-style alert banner, and an honest "Known limitations" section. Every number on it (accuracy, dataset size, category counts, training curve) is read live from the checkpoint and manifests — nothing is hardcoded.

---

## 2. What all has been used

**Data sources:**
| Source | What it is | Volume used |
|---|---|---|
| Kaggle INSAT3D dataset | Public Kaggle dataset, pre-processed IR/raw JPEGs | 136 images |
| NOAA HURSAT-B1 v06 | Per-storm NetCDF archives, 301×301px, ~8km resolution, 3-hourly | 680 frames / 10 storms |
| MOSDAC 3DIMG L1C_SGP | Real MOSDAC data order, GeoTIFF, TIR1+WV bands, 30-min cadence | 216 frames / 1 storm (Amphan) |
| IBTrACS v04r01 (NI basin) | Official best-track records, used to pick which HURSAT-B1 storms to download (to target under-represented categories) | reference only, not directly trained on |
| IMD best track (Amphan) | `data/besttrack/amphan_2020.csv`, used both to label MOSDAC frames and for the RI case study | 45 records |

**Libraries / stack:**
- PyTorch + torchvision (EfficientNet-B0 backbone)
- Streamlit + Plotly (dashboard)
- netCDF4 (reading HURSAT-B1 files)
- rasterio (reading MOSDAC GeoTIFFs, including reprojecting lat/lon to the file's map projection)
- PIL/Pillow, NumPy, pandas
- PyYAML for config files

**IMD 8-category intensity scale** used throughout (`src/utils.py`): Low Pressure Area, Depression, Deep Depression, Cyclonic Storm, Severe Cyclonic Storm, Very Severe Cyclonic Storm, Extremely Severe Cyclonic Storm, Super Cyclonic Storm.

---

## 3. How it is implemented

Each data source has its own preprocessing script (`src/kaggle_dataset.py`, `src/hursat_dataset.py`, `src/mosdac_dataset.py`), but all three are normalized into the **same sample format** — `img_name`, `storm`, `ir_path`, `raw_path`, `has_raw`, `knots`, `kmph`, `cat_idx` — so `train_combined.py` can just concatenate the three sample lists with zero per-source special-casing.

Key implementation details worth knowing:

- **Physical-unit consistency.** HURSAT-B1 and MOSDAC deliver real brightness temperature in Kelvin, not 0–255 pixel values. Both are converted to grayscale using the same "cold cloud tops = bright" convention that forecasters actually use for enhanced IR imagery, with clip ranges chosen from real observed data rather than textbook guesses (this was tuned after spot-checking actual output images and catching a case where the water-vapor channel came out flat/blank because the assumed brightness range was wrong for real data).
- **Multi-satellite deduplication.** HURSAT-B1 bundles multiple satellites per timestamp; the pipeline keeps only the lowest-viewing-angle (most nadir, least distorted) frame per timestamp.
- **Map projection handling.** MOSDAC's delivered GeoTIFFs are in Web Mercator (meters), not plain lat/lon degrees — the crop-around-storm-center logic reprojects the storm's lat/lon into the file's own coordinate system before cropping (an earlier version of this logic, written before real files were available, would have silently produced zero usable frames without this fix).
- **Storm-held-out validation**, as described above — this is the most important methodological choice in the whole pipeline, since it's what makes the reported accuracy trustworthy rather than inflated by leakage.
- **Everything the dashboard displays is computed live** from whatever checkpoint + config it's pointed at (`CONFIG_PATH` / `CHECKPOINT_PATH` at the top of `app.py`), so retraining and re-pointing those two lines is the entire "deploy a new model" workflow.

---

## 4. What is yet to be done / left to do

- **Accuracy is still modest (40.4%).** Levers not yet pulled: a real ImageNet-pretrained backbone (blocked in this sandbox because `download.pytorch.org` isn't reachable here — worth simply retrying training on a machine with normal internet access, since the model currently trains from random initialization, which costs real accuracy), more epochs / hyperparameter tuning, and addressing the remaining class imbalance (Low Pressure Area is still rare).
- **Pressure regression is unused.** The model has a 1-output (wind-only) regression head. Amphan's real best track *does* include pressure — wiring up a 2-output head (wind + pressure) for the sources that have both would be a real, usable improvement, not just a nice-to-have.
- **RI alert isn't yet connected to model output.** Right now it's validated against real historical best-track wind sequences, not the model's own predictions over a sequence of frames. Now that real time-series imagery exists (Amphan's 30-minute cadence data), this is buildable: feed the model's predicted wind speed over successive real frames into the same RI-detection logic.
- **This is a classifier of the current frame, not a forecaster of the future.** The problem statement's "...and **prediction** of tropical cyclone patterns" wording points at genuine forecasting — given today's image, what happens in 6/12/24 hours — which nothing in the repo does yet. See enhancement #1 below; this is likely the single highest-leverage thing left to build.
- **Repo hygiene:** no `requirements.txt`/environment file yet, no `.gitignore`, not yet a git repository at all in the working copy — needed before sharing with teammates (see section 6).

---

## 5. Other possible enhancements

Roughly in order of how much they'd differentiate the submission:

1. **Actual intensity forecasting (temporal model).** A sequence model (ConvLSTM, or a small transformer over a window of frames) that takes the last N hours of imagery and predicts intensity N hours ahead — this is what turns "classifier" into "prediction system" and would be the strongest differentiator for judges, since most hackathon submissions in this space only classify the current frame.
2. **Storm track / motion prediction** alongside intensity — where the storm is headed, not just how strong it is.
3. **Wire the pressure head** using Amphan's real pressure labels (small effort, real payoff).
4. **Connect RI detection to live model output** rather than only historical best-track data (described above).
5. **Broader basin coverage** — pretrain on a larger multi-basin HURSAT-B1 or TC PRIMED sample, then fine-tune on North Indian Ocean data, to give the backbone more to learn from before specializing.
6. **Confidence calibration check** — does the MC-Dropout confidence number actually track real accuracy (e.g., are "80% confidence" predictions right ~80% of the time)? A reliability plot would make the uncertainty feature more credible to judges who ask about it.
7. **A lightweight deployment path** — package the model as an ONNX export or a small API, so "prototype" can credibly become "pilot-ready" in the pitch.
8. **Live/near-real-time ingestion** — instead of static downloaded archives, a scheduled pull from MOSDAC's open feed would support the "operational" framing directly.

---

## 6. Sharing the codebase with your team (without pushing datasets to GitHub)

The good news: once data is *processed*, it's small. Here's what actually needs to travel with the code, and what doesn't:

| Folder | Size | Needed by teammates? |
|---|---|---|
| `src/`, `app.py`, `configs/` (all code) | a few hundred KB | **Yes — this is the actual codebase** |
| `data/kaggle_insat3d/` | ~14 MB | Yes (processed, ready to use) |
| `data/hursat_processed/` | ~11 MB | Yes (processed, ready to use) |
| `data/mosdac_processed/` | ~2.3 MB | Yes (processed, ready to use) |
| `data/besttrack/` | ~12 KB | Yes |
| `checkpoints_combined/best.pt` | ~16 MB | Yes (so they don't have to retrain from scratch) |
| `hursat_raw/` (the 10 `.tar.gz` archives) | ~490 MB | **No** — only needed once, to build `hursat_processed/`. Never needed again. |
| `mosdac_amphan/` (652 `.tif` files) | ~530 MB | **No** — same reasoning, only needed once to build `mosdac_processed/`. |

So the entire useful payload — code + everything a teammate needs to actually run the dashboard or retrain — is about **45 MB total**. The ~1 GB of raw downloads never needs to leave your machine. This directly solves the problem you're describing: nobody needs to go hunting across NOAA/MOSDAC/Kaggle, because the *already-processed, already-labeled* images are what's small enough to hand over directly.

**Recommended approach:**

1. **Initialize git in the project** (if not already) and add a `.gitignore` that excludes the raw downloads and any local venv/cache folders:
   ```
   hursat_raw/
   mosdac_amphan/
   data/cyinsat/
   __pycache__/
   *.pyc
   .venv/
   ```
2. **Commit everything else normally** — code, configs, and yes, the processed `data/` folders and the checkpoint. At ~45 MB total this is well within GitHub's limits (100 MB per-file hard limit; a repo this size is completely unremarkable). Your teammates then get everything they need with one `git clone` — no separate download step, no hunting for data, nothing to re-derive.
3. **Push it** to a GitHub repo (private, if you'd rather keep it off public view until submission) and add them as collaborators, or just share the clone URL if it's public.
4. Add a short "Setup" section to your README: `pip install -r requirements.txt` (worth generating one with `pip freeze > requirements.txt` from your working environment) and `streamlit run app.py`. That's the entire onboarding for a teammate.

If you'd rather keep the repo extremely lean and not commit any binary data at all, the alternative is: exclude `data/` and `checkpoints*/` too, zip those folders separately, and share the zip via a link (your OneDrive, Google Drive, or a GitHub Release attached to the repo — GitHub Releases accept files up to 2 GB each, well beyond what you'd need). Your README would then say "download `project-data.zip` from [link], unzip into the repo root." This keeps `git clone` fast for anyone who only wants to read the code, at the cost of one extra manual step for anyone who wants to actually run it. Given how small your processed data already is, I'd only bother with this if your team has a strict policy against committing any data files.

Either way — the raw `hursat_raw/` and `mosdac_amphan/` folders are the one thing to actually leave behind. They did their job (they built the processed data) and have no further use.
