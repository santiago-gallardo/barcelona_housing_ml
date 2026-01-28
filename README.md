# Barcelona Housing Price Prediction (2018) — Reproducible ML Pipeline + Streamlit App

Predict **Barcelona apartment prices** from structured listing data using a reproducible training pipeline (scripts) and an interactive **Streamlit** demo.

---

## Too Long; Didn’t Read

- **Problem:** Estimate apartment **price in Barcelona** from listing attributes (area + geolocation-derived features).
- **End-to-end:** **EDA → modeling → reproducible pipeline (train/evaluate/predict) → Streamlit app** for interactive predictions.
- **Final model chosen:** **RandomForest (5 vars) + log(y)** (parsimonious feature set for maintainability).
- **Holdout test metrics:** **RMSE €82,805 | MAE €47,838 | R² 0.919**.
- **Key decision:** Chose a **simpler model** (5 features + log target) with a small performance trade-off vs the best RMSE model (**+€4,682, 6.0% RMSE**), improving interpretability and deployment/UX.

---

## 1) Problem & objective

This is an applied **machine learning project on a real-world regression problem**: predicting housing prices in Barcelona from structured listing data.

The goal is not “just training models” — it’s to demonstrate:

- **Decision-making & modeling criteria:** baselines → model comparison → final choice rationale.
- **Reproducibility:** moving from notebooks to a **scripted pipeline** that generates artifacts and reports.
- **Practical usability:** batch prediction via CLI + an interactive **Streamlit** app.

---

## 2) Project structure (mandatory)

```
barcelona_housing_ml/
├─ app/                          # Streamlit (UI)
│  └─ pages/
│     ├─ 2_Prediction.py         # Prediction page (loads model from artifacts/)
│     └─ Readme.py               # Technical README page inside the app
│
├─ artifacts/                    # Model artifacts
│  ├─ features.json              # Feature list used by the final model (contract) ✅ tracked in repo
│  └─ model.joblib               # Trained model ❌ not stored in repo (too large) → download from GitHub Releases
│
├─ data/
│  └─ raw/
│     └─ Barcelona_2018.csv      # Raw dataset (separator ';')
│
├─ notebooks/                    # EDA + narrative
│  ├─ utils/
│  ├─ 00_story_modeling.ipynb
│  └─ 01_eda.ipynb
│
├─ reports/                      # Generated outputs
│  ├─ figures/                   # Plots (png)
│  ├─ metrics.json               # Full run metadata + selection rationale
│  ├─ cv_summary.csv             # CV results on TRAIN split only
│  ├─ model_comparison.csv       # Holdout test comparison across models
│  └─ preds_full.csv             # Example batch predictions output
│
├─ scripts/                      # CLI entry points
│  ├─ train.py                   # Train + save artifacts/ and reports/
│  ├─ evaluate.py                # Print evaluation tables
│  └─ predicts.py                # Batch prediction (CSV → CSV with PRED_PRICE)
│
├─ src/                          # Reusable package code
│  ├─ app_utils/
│  └─ bcn_housing/
│
├─ pyproject.toml
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## 3) Notebooks overview 

- `01_eda.ipynb` Exploratory analysis to understand distributions, missingness, outliers, and the relationship between core drivers (e.g., constructed area, location) and price.
- **`00_story_modeling.ipynb`**
  Narrative notebook that connects: baseline → feature selection → model comparison → final choice rationale.
  The focus is on *reasoning* and *trade-offs*, not on dumping implementation details.

---

## 4) Training pipeline (scripts)

### `scripts/train.py` — reproducible training + reporting

- Loads the dataset (`data/raw/Barcelona_2018.csv`).
- Runs a **train/test split** (seed `42`, test size `0.2`).
- Trains several candidate models (baseline + interpretable + ensemble).
- Saves:
  - **Holdout comparison** → `reports/model_comparison.csv`
  - **Optional CV on TRAIN split only** → `reports/cv_summary.csv` (when `--cv`)
  - **Run metadata + selection rationale** → `reports/metrics.json`
  - **Figures** → `reports/figures/*.png`
  - **Final artifacts** → `artifacts/model.joblib` and `artifacts/features.json`

> Why CV on TRAIN only? This is a robustness check (variance across folds) without contaminating the final holdout test.

### `scripts/evaluate.py` — quick view of results

Prints the holdout comparison table, and CV summary if available.

### `scripts/predicts.py` — batch prediction (production-style)

- Loads:
  - `artifacts/model.joblib`
  - `artifacts/features.json` (feature contract)
- Reads an input CSV (default: `data/raw/Barcelona_2018.csv`).
- Writes a new CSV to `reports/` with an extra column: **`PRED_PRICE`**.

---

## 5) Streamlit App

The Streamlit app provides an interactive experience to generate price estimates:

- Select a **location in Barcelona** (lat/lon sliders + map).
- Fill feature inputs.
- Get an estimated price in **€**.

Run it from project root:

```bash
streamlit run app/Readme.py
```

The prediction page lives at: `app/pages/2_Prediction.py`.

---

## 6) Results

Dataset snapshot:

- Rows: **61,486**
- Columns: **46**
- Target range: **€37,000 → €4,866,000**

Final model (selected for **parsimony**):

- **RandomForest (5 vars) + log(y)**
- Features (**5**): `CONSTRUCTEDAREA`, `LATITUDE`, `LONGITUDE`, `DISTANCE_TO_CITY_CENTER`, `DISTANCE_TO_DIAGONAL`
- Holdout test performance:
  - **RMSE:** €82,805
  - **MAE:** €47,838
  - **R²:** 0.919

Best-by-RMSE model (for reference):

- **RandomForest (10 vars)**
- **RMSE:** €78,124 | **R²:** 0.928

Trade-off (final vs best):

- RMSE delta: **+€4,682** (**6.0%**)
- R² delta: **-0.009**

Error distribution (final model):

- Median absolute error: **€28,237**
- P90 absolute error: **€110,363**
- P95 absolute error: **€157,724**

> Note: Errors tend to be larger in the high-end tail, where fewer observations exist.

Optional visuals (if you keep `reports/figures/` in the repo):

- `reports/figures/y_true_vs_pred_final.png`
- `reports/figures/feature_importance_rf5_final.png`
- `reports/figures/rmse_evolution.png`

---

## Results Summary

| Item                                  | Value                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Final model**                 | **RandomForest (5 vars) + log(y)**                                                            |
| **Test RMSE / MAE / R²**       | **€82,805 / €47,838 / 0.919**                                                               |
| **Best RMSE model (reference)** | **RandomForest (10 vars)** — RMSE **€78,124**                                         |
| **Trade-off (1 line)**          | Simpler model chosen at **+€4,682 (6.0%) RMSE** for better maintainability + UI friendliness |

---

## Key takeaways

- **Model selection as a product decision:** I prioritized a **maintainable + interpretable** feature set with near-best holdout performance, which is more practical for a UI/demo and future deployment.
- **Clear separation of concerns:** notebooks communicate analysis and rationale; **scripts implement the reproducible pipeline** (train/evaluate/predict) that generates artifacts + reports.
- **Interpretability & maintainability matter:** the final model uses a small, well-motivated feature set that’s easier to validate, monitor, and expose in a Streamlit interface.
- **Evaluation discipline:** holdout test for final comparison + CV on TRAIN for robustness (variance across folds), avoiding leakage.

---

## 7) Tech stack

- **Python** (project tested on Windows)
- **pandas / numpy**
- **scikit-learn**
- **joblib**
- **Streamlit** (+ **pydeck** for mapping)
- **Matplotlib** for report figures

---

## 8) How to run (minimum viable)

This repo includes both:

- `requirements.txt` (runtime dependencies)
- `pyproject.toml` (so you can install the project in editable mode and run scripts **without** `PYTHONPATH`)

### Setup (create env + install deps)

### Model artifact (required for predictions)

The trained model file (`artifacts/model.joblib`) is too large to be stored in the repository.

✅ This repo includes:

- `artifacts/features.json` (tracked)

⬇️ You must download separately:

- `model.joblib` from **GitHub Releases** → **Model v1 - Barcelona Housing**
- Place it at: `artifacts/model.joblib`

Steps:

1) Open the repo **Releases** page and select **Model v1 - Barcelona Housing**
2) Download the asset: `model.joblib`
3) Move it to: `artifacts/model.joblib`

* `Release: https://github.com/santiago-gallardo/barcelona_housing_ml/releases/tag/model-v1`

> **Note:** The commands below assume you have completed the previous steps:
>
> - Environment setup (installed dependencies)
> - Downloaded the trained model and placed it at `artifacts/model.joblib`

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

#### macOS / Linux (bash/zsh)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Train (and generate reports)

```bash
python scripts/train.py --cv
```

### Evaluate (prints tables)

```bash
python scripts/evaluate.py
```

### Predict (batch)

```bash
python scripts/predicts.py --print-summary
```

### Run the Streamlit app

```bash
streamlit run app/Readme.py
```

---

## 9) What I’d do next 

These are **production-oriented extensions**, not missing pieces of this project:

- Add **uncertainty estimates** (quantile regression / conformal prediction) to communicate prediction intervals.
- Perform **hyperparameter tuning** with a clear search budget + model cards.
- Expand feature engineering: neighborhood-level signals, POIs, public transport proximity, and robust geospatial encoding.
- Add CI checks (formatting + lightweight unit tests) and a `Makefile` / task runner for one-command workflows.
