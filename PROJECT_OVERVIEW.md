# Cyclone Intensity Classifier — Project Overview

**SIH26070** · AI/ML-based system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite data · Ministry of Earth Sciences / IMD · Disaster Management theme

*Last updated: this document reflects the state of the project after adding temporal intensity forecasting (checkpoint `checkpoints_temporal/best.pt`) on top of the classifier (checkpoint `checkpoints_combined/best.pt`, 40.4% validation accuracy).*

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

**Dashboard** (`app.py`, Streamlit): 4 tabs — Assess a Storm, Historical Precedent, Early-Warning Alert, About & Data. Custom dark theme, a one-line problem statement up top, a compact operational-style alert banner, and an honest "Known limitations" section. Every number on it (accuracy, dataset size, category counts, training curve) is read live from the checkpoint and manifests — nothing is hardcoded. (Not yet wired to the new temporal forecaster below — see section 4.)

**Temporal intensity FORECASTING — the "…and prediction" part of the problem statement (new):** `src/temporal_dataset.py`, `src/temporal_model.py`, `src/train_temporal.py`. Until now, CycloneNet only classified the *current* frame; this is a genuine forecaster: given the last 4 observed frames of a storm, it predicts wind speed at **+6h / +12h / +24h ahead**.

- **Data:** only NOAA HURSAT-B1 (10 storms, ~3-hourly) and MOSDAC/Amphan (1 storm, ~30-minute) have the real timestamps + storm identity a sequence needs — the 136-image Kaggle set has neither, so it's classifier-only. Pooling the two gives **831 real sequences across 11 storms**, built directly from each manifest's `datetime_utc` column with a per-storm adaptive time-tolerance (so 3-hourly and 30-minute cadences share one pipeline without either starving the other of valid horizon labels).
- **Architecture:** the frame-level CycloneNet backbone (already trained, `checkpoints_combined/best.pt`) is reused **frozen** as a per-frame feature extractor — there isn't remotely enough sequence data (11 storms) to also learn image features from scratch. A small GRU (550K trainable params vs. 4.5M total) runs over the 4-frame embedding sequence, and the head predicts a **delta** (km/h change from the current, last-observed wind speed) rather than an absolute value, with the current wind speed fed in explicitly. This mattered in practice: a first version predicting absolute wind speed from images alone actually lost to a trivial "predict no change" baseline at +6h/+12h, because it had no explicit reference point for "how strong is this storm right now." Reframing as a delta fixed that.
- **Validation: the same storm-held-out split as the classifier** (PHET, NILOFAR — 126 held-out sequences, 2 storms never seen in training). Every number below is measured against a **persistence baseline** (predict "no change from right now"), the honest bar for any forecasting claim:

  | Horizon | Persistence baseline (MAE) | CycloneNet forecaster (MAE) | Improvement |
  |---|---|---|---|
  | +6h  | 9.63 km/h  | 8.69 km/h  | ~10% |
  | +12h | 19.26 km/h | 14.68 km/h | ~24% |
  | +24h | 38.48 km/h | 28.62 km/h | ~26% |

  The model beats the baseline at every horizon, and the margin *grows* with horizon — exactly what a real forecasting skill signal should look like (persistence degrades fast over time; a model that's actually reading trend information from the image sequence degrades more slowly). This is a modest but genuine and honestly-validated result, not an aspirational one.

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
- **RI alert still isn't connected to the temporal model's own predictions.** It's validated against real historical best-track wind sequences (Amphan), and separately the temporal forecaster now predicts wind speed forward in time — but nothing yet feeds the forecaster's own +6h/+12h/+24h predictions into the RI-detection logic. This is now the natural next step (enhancement #1 below is DONE; this is the follow-on, previously enhancement #4).
- **Temporal forecaster isn't wired into the dashboard yet.** `src/train_temporal.py` / `checkpoints_temporal/best.pt` exist and are validated, but `app.py` has no tab that runs a live forecast — a judge can't see it in the demo yet, only in the training log.
- **Temporal forecaster's val set is small (126 sequences, 2 storms)** — the storm-held-out MAE numbers are real and honestly measured, but with only 11 storms total to draw from, a couple more held-out storms' worth of data would make the comparison to the persistence baseline more statistically solid.
- **Accuracy on the frame classifier is still modest (40.4%).** Levers not yet pulled: a real ImageNet-pretrained backbone (blocked in this sandbox because `download.pytorch.org` isn't reachable here — worth simply retrying training on a machine with normal internet access, since the model currently trains from random initialization, which costs real accuracy), more epochs / hyperparameter tuning, and addressing the remaining class imbalance (Low Pressure Area is still rare).
- **Pressure regression is unused.** The model has a 1-output (wind-only) regression head. Amphan's real best track *does* include pressure — wiring up a 2-output head (wind + pressure) for the sources that have both would be a real, usable improvement, not just a nice-to-have.
- **PPT housekeeping** (not code): fill in Team ID on the title slide before uploading.

---

## 5. Other possible enhancements

Roughly in order of how much they'd differentiate the submission:

1. ~~**Actual intensity forecasting (temporal model).**~~ **DONE** — see section 1 above (`src/train_temporal.py`). A GRU over frozen CycloneNet frame embeddings predicts +6h/+12h/+24h wind speed, beating a persistence baseline at every horizon on 2 fully held-out storms.
2. **Wire the temporal forecaster into the dashboard** — a 5th tab that runs the last 4 frames of a chosen storm through `checkpoints_temporal/best.pt` and shows the +6h/+12h/+24h forecast live, not just in a training log. Now that the model exists and validates well, this is the highest-leverage next step for the actual demo.
3. **Connect RI detection to the temporal forecaster's own predictions** rather than only historical best-track data — feed the model's own +6h/+12h/+24h sequence into the existing `src/ri_alert.py` threshold logic.
4. **Storm track / motion prediction** alongside intensity — where the storm is headed, not just how strong it is.
5. **Wire the pressure head** using Amphan's real pressure labels (small effort, real payoff).
6. **Broader basin coverage** — pretrain on a larger multi-basin HURSAT-B1 or TC PRIMED sample, then fine-tune on North Indian Ocean data, to give the backbone more to learn from before specializing.
7. **Confidence calibration check** — does the MC-Dropout confidence number actually track real accuracy (e.g., are "80% confidence" predictions right ~80% of the time)? A reliability plot would make the uncertainty feature more credible to judges who ask about it.
8. **A lightweight deployment path** — package the model as an ONNX export or a small API, so "prototype" can credibly become "pilot-ready" in the pitch.
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
