# Cyclone Intensity Classifier — Project Overview

**SIH26070** · AI/ML-based system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite data · Ministry of Earth Sciences / IMD · Disaster Management theme

*Last updated: this document reflects the state of the project after (1) wiring the live Forecast tab + Rapid Intensification check into the dashboard and (2) adding a real pressure (mb) regression head alongside wind speed, retrained and promoted as the new production checkpoint (`checkpoints_combined/best.pt`, 41.9% validation accuracy — up slightly from 40.4%, confirming the added output did not cost classification accuracy). The temporal forecaster (`checkpoints_temporal/best.pt`) was retrained on top of this new backbone and re-validated.*

---

## 1. What is done

**Real data pipeline (three independent sources, unified into one training pool):**

- **Kaggle "INSAT3D Infrared & Raw Cyclone Imagery"** — 136 single-frame images, no storm ID, wind-speed label only. The original seed dataset.
- **NOAA HURSAT-B1 v06** — 680 frames across 10 real North Indian Ocean cyclones (Nargis, Phet, Phailin, Madi, Hudhud, Nilofar, Komen, Chapala, Kyant, Vardah; 2008–2016), chosen specifically because IBTrACS record counts showed Depression/Deep Depression/Cyclonic Storm were under-represented in the Kaggle set alone.
- **MOSDAC (Cyclone Amphan, 2020)** — 216 frames from a real MOSDAC data order (652 raw GeoTIFF files, TIR1 + WV bands, 30-minute cadence), matched against Amphan's official IMD best track. This is the only source with real Super Cyclonic Storm coverage — Amphan peaked at that category.

Total: **1,032 real labeled images**, spanning all 8 IMD intensity categories.

**Model:** CycloneNet — an EfficientNet-B0 backbone adapted to take a 2-channel input (IR + a second "raw" channel), with two heads: an 8-way classification head (IMD category) and a 2-output regression head (wind speed + **pressure, mb**).

**Training:** `src/train_combined.py` trains on all three sources together. Validation is a **storm-held-out split** — 2 entire HURSAT-B1 storms (Phet, Nilofar) never seen in training at all, not just held-out images. This matters: a random split of 3-hourly frames from the same storm would put near-duplicate images on both sides and inflate the score without proving anything about generalization. Kaggle and Amphan images all go into training (neither has a storm ID to hold out cleanly, and Amphan is the only real Super Cyclonic Storm source, so holding it out would remove the one thing it was added to fix).

**Current best checkpoint:** 41.9% validation accuracy on 2 completely unseen storms (up from 40.4% before the pressure head was added — a genuine, honestly-measured improvement, not a regression traded for the new output), plus 27.2 km/h wind MAE and **10.2 mb pressure MAE** on those same held-out storms.

**Pressure regression (new):** 87% of the training pool (896/1,032 images — all of HURSAT-B1 and MOSDAC/Amphan, from real IBTrACS/best-track matching) carries a real, labeled pressure value; only the 136 Kaggle images don't. The 2nd regression output is trained with a **masked MSE loss** — Kaggle samples are simply excluded from the pressure loss/metric (mask=0), not given a fabricated label — so the pressure head is trained and validated on exclusively real data. `src/kaggle_dataset.py`'s `denormalize_pressure()` converts the normalized output back to mb; the dashboard's "Assess a Storm" tab now shows a predicted pressure alongside predicted wind speed, and the real actual pressure when the chosen image has one.

**Explainability & trust features (already built and wired into the dashboard):**
- **Grad-CAM** (`src/gradcam.py`) — visual heatmap of what the model looked at.
- **MC-Dropout uncertainty** (`src/uncertainty.py`) — confidence estimate per prediction, flagged when low.
- **Historical precedent retrieval** (`src/historical_comparison.py`) — embeds every training image once, retrieves the closest real historical matches by cosine similarity, shown as a percentile so it's comparable across checkpoints.
- **Rapid Intensification alerting** (`src/ri_alert.py`) — real threshold-based RI detection (Kaplan & DeMaria 2003), source-agnostic by design. Validated two ways: against Amphan's actual best-track record (the historical Early-Warning Alert tab), and — new — against the temporal forecaster's own live +6h/+12h/+24h predictions, in the Forecast tab. A genuine true-positive case is confirmed: an anchor on 16 May 2020 06:30 for Amphan where the model's own +24h prediction crosses the +30kt/24h threshold and correctly fires.

**Dashboard** (`app.py`, Streamlit): **5 tabs** — Assess a Storm, Historical Precedent, Early-Warning Alert, **Forecast**, About & Data. Custom dark theme, a one-line problem statement up top, a compact operational-style alert banner, and an honest "Known limitations" section. Every number on it (accuracy, dataset size, category counts, training curve, forecaster MAE) is read live from the checkpoints and manifests — nothing is hardcoded. The Forecast tab lets a judge pick any storm and time point and see the model's own +6h/+12h/+24h forecast plotted against the real historical outcome, with a clear "held out — never trained on" vs. "used in training" distinction, plus the live RI check described above.

**Temporal intensity FORECASTING — the "…and prediction" part of the problem statement (new):** `src/temporal_dataset.py`, `src/temporal_model.py`, `src/train_temporal.py`. Until now, CycloneNet only classified the *current* frame; this is a genuine forecaster: given the last 4 observed frames of a storm, it predicts wind speed at **+6h / +12h / +24h ahead**.

- **Data:** only NOAA HURSAT-B1 (10 storms, ~3-hourly) and MOSDAC/Amphan (1 storm, ~30-minute) have the real timestamps + storm identity a sequence needs — the 136-image Kaggle set has neither, so it's classifier-only. Pooling the two gives **831 real sequences across 11 storms**, built directly from each manifest's `datetime_utc` column with a per-storm adaptive time-tolerance (so 3-hourly and 30-minute cadences share one pipeline without either starving the other of valid horizon labels).
- **Architecture:** the frame-level CycloneNet backbone (already trained, `checkpoints_combined/best.pt` — now the pressure-head-equipped v2 backbone; the forecaster was retrained from scratch on top of it after the backbone changed, since its GRU had learned to read the old backbone's specific feature space) is reused **frozen** as a per-frame feature extractor — there isn't remotely enough sequence data (11 storms) to also learn image features from scratch. A small GRU (550K trainable params vs. 4.5M total) runs over the 4-frame embedding sequence, and the head predicts a **delta** (km/h change from the current, last-observed wind speed) rather than an absolute value, with the current wind speed fed in explicitly. This mattered in practice: a first version predicting absolute wind speed from images alone actually lost to a trivial "predict no change" baseline at +6h/+12h, because it had no explicit reference point for "how strong is this storm right now." Reframing as a delta fixed that.
- **Validation: the same storm-held-out split as the classifier** (PHET, NILOFAR — 126 held-out sequences, 2 storms never seen in training). Every number below is measured against a **persistence baseline** (predict "no change from right now"), the honest bar for any forecasting claim:

  | Horizon | Persistence baseline (MAE) | CycloneNet forecaster (MAE) | Improvement |
  |---|---|---|---|
  | +6h  | 9.63 km/h  | 9.05 km/h  | ~6% |
  | +12h | 19.26 km/h | 16.79 km/h | ~13% |
  | +24h | 38.48 km/h | 30.59 km/h | ~20% |

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

Each data source has its own preprocessing script (`src/kaggle_dataset.py`, `src/hursat_dataset.py`, `src/mosdac_dataset.py`), but all three are normalized into the **same sample format** — `img_name`, `storm`, `ir_path`, `raw_path`, `has_raw`, `knots`, `kmph`, `cat_idx`, `pressure_mb` (real value for HURSAT/MOSDAC, `None` for Kaggle) — so `train_combined.py` can just concatenate the three sample lists with zero per-source special-casing. The dataset's `__getitem__` pairs `pressure_mb` with a `y_reg_mask` tensor so the 136 Kaggle samples (no pressure label) are masked out of the pressure loss/metric rather than given a fabricated value — see `masked_mse()` in `src/train_combined.py`.

Key implementation details worth knowing:

- **Physical-unit consistency.** HURSAT-B1 and MOSDAC deliver real brightness temperature in Kelvin, not 0–255 pixel values. Both are converted to grayscale using the same "cold cloud tops = bright" convention that forecasters actually use for enhanced IR imagery, with clip ranges chosen from real observed data rather than textbook guesses (this was tuned after spot-checking actual output images and catching a case where the water-vapor channel came out flat/blank because the assumed brightness range was wrong for real data).
- **Multi-satellite deduplication.** HURSAT-B1 bundles multiple satellites per timestamp; the pipeline keeps only the lowest-viewing-angle (most nadir, least distorted) frame per timestamp.
- **Map projection handling.** MOSDAC's delivered GeoTIFFs are in Web Mercator (meters), not plain lat/lon degrees — the crop-around-storm-center logic reprojects the storm's lat/lon into the file's own coordinate system before cropping (an earlier version of this logic, written before real files were available, would have silently produced zero usable frames without this fix).
- **Storm-held-out validation**, as described above — this is the most important methodological choice in the whole pipeline, since it's what makes the reported accuracy trustworthy rather than inflated by leakage.
- **Everything the dashboard displays is computed live** from whatever checkpoint + config it's pointed at (`CONFIG_PATH` / `CHECKPOINT_PATH` at the top of `app.py`), so retraining and re-pointing those two lines is the entire "deploy a new model" workflow.

---

## 4. What is yet to be done / left to do

- **Accuracy is still modest (41.9%).** Levers not yet pulled: a real ImageNet-pretrained backbone (blocked in this sandbox because `download.pytorch.org` isn't reachable here — worth re-testing on a machine with normal internet access, since the model currently trains from random initialization, which costs real accuracy), more epochs / hyperparameter tuning, and addressing the remaining class imbalance (Low Pressure Area is still rare — only 4 training examples).
- **Temporal forecaster's val set is small (126 sequences, 2 storms)** — the storm-held-out MAE numbers are real and honestly measured, but with only 11 storms total to draw from, a couple more held-out storms' worth of data would make the comparison to the persistence baseline more statistically solid.
- **MC-Dropout confidence isn't calibration-checked.** The dashboard reports a confidence % per prediction, but nobody has verified whether "80% confidence" predictions are actually right ~80% of the time on held-out data — a reliability plot would make this credible rather than just plausible.
- **No lightweight deployment path yet.** Still a Streamlit dashboard + local checkpoints, not an ONNX export or small API — "prototype," not yet "pilot-ready."
- **PPT housekeeping** (not code): fill in Team ID on the title slide before uploading.

---

## 5. Other possible enhancements

Roughly in order of how much they'd differentiate the submission:

1. ~~**Actual intensity forecasting (temporal model).**~~ **DONE** — see section 1 above (`src/train_temporal.py`). A GRU over frozen CycloneNet frame embeddings predicts +6h/+12h/+24h wind speed, beating a persistence baseline at every horizon on 2 fully held-out storms.
2. ~~**Wire the temporal forecaster into the dashboard.**~~ **DONE** — the Forecast tab runs the last 4 frames of a chosen storm through `checkpoints_temporal/best.pt` live.
3. ~~**Connect RI detection to the temporal forecaster's own predictions.**~~ **DONE** — the Forecast tab's "Rapid Intensification check" feeds the model's own live +6h/+12h/+24h sequence into the existing `src/ri_alert.py` threshold logic; a genuine true-positive case is confirmed (Amphan, 16 May 06:30 anchor).
4. ~~**Wire the pressure head.**~~ **DONE** — see section 1 above. A real, masked-loss 2nd regression output (wind + pressure), retrained and promoted with no classification-accuracy regression (41.9% vs. 40.4%).
5. **Address class imbalance / accuracy.** Re-test `download.pytorch.org` reachability for a real ImageNet-pretrained backbone, and class-weight the loss to address Low Pressure Area's 4-example training set.
6. **Confidence calibration check** — does the MC-Dropout confidence number actually track real accuracy (e.g., are "80% confidence" predictions right ~80% of the time)? A reliability plot would make the uncertainty feature more credible to judges who ask about it.
7. **A lightweight deployment path** — package the model as an ONNX export or a small API, so "prototype" can credibly become "pilot-ready" in the pitch.
8. **Storm track / motion prediction** alongside intensity — where the storm is headed, not just how strong it is.
9. **Broader basin coverage** — pretrain on a larger multi-basin HURSAT-B1 or TC PRIMED sample, then fine-tune on North Indian Ocean data, to give the backbone more to learn from before specializing.
10. **Live/near-real-time ingestion** — instead of static downloaded archives, a scheduled pull from MOSDAC's open feed would support the "operational" framing directly.

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
