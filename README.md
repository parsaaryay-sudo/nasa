# NASA C-MAPSS (FD001) — Remaining Useful Life Prediction

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor data
(NASA C-MAPSS, subset **FD001**: 100 train engines, 100 test engines, 1 operating condition, 1 fault mode),
with an interactive **Streamlit dashboard** to explore the data and compare models.

## Project structure

```
NASA-C-MAPSS/
├── app.py                 # Streamlit dashboard
├── utils.py               # labels, windows, metrics, model loading (no Streamlit dependency)
├── requirements.txt
├── data/
│   ├── raw/               # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── processed/         # CSVs written by notebook 03 (for inspection)
├── models/                # scaler.pkl, linear_regression_model.pkl, random_forest_model.pkl,
│                          # xgb_model.pkl, dense_model.keras, gru_model.keras
└── notebooks/
    ├── 01_data_loading            (writes data/processed/*_df.csv used by 02)
    ├── 02_exploratory_data_analysis
    ├── 03_data_preprocessing      <- creates models/scaler.pkl, run before 04-08
    ├── 04_linear_regression       -> models/linear_regression_model.pkl
    ├── 05_random_forest           -> models/random_forest_model.pkl
    ├── 06_XGBRegressor            -> models/xgb_model.pkl            (needs xgboost)
    ├── 07_dense_layer             -> models/dense_model.keras        (needs tensorflow)
    └── 08_gru_layer               -> models/gru_model.keras          (needs tensorflow; 30-cycle windows)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run from the project root. To (re)build the models run the notebooks in numeric order `03 → 04 → 05 → 06 → 07 → 08` (01 and 02 are exploration only).
Every notebook finds the project root by itself, so it works whether the kernel starts in the root or in `notebooks/`.
The sidebar of the app shows ⚠️ for any model that is missing or outdated (e.g. a file trained with the old
17-feature pipeline) and skips it instead of crashing. If TensorFlow / XGBoost are not installed, those models are
skipped the same way.

## Method

* **Labels.** Train RUL = `max_cycle − cycle`. Test trajectories stop *before* failure, so the true RUL of a test row is
  `RUL_FD001[engine] + (last_cycle − cycle)` (`utils.load_raw`).
* **Capped target.** RUL is capped at 125 cycles for training (early-life RUL is not observable from the sensors).
* **Features.** 14 informative sensors, standardised (scaler fitted on training data). Constant sensors and the 3
  operating settings are dropped.
* **Models.** Linear Regression, Random Forest, XGBoost and a Dense network use one time-step; the **GRU** uses a
  window of the last 30 cycles of the same engine (`utils.make_windows`). Networks learn `RUL / 125`.
* **Validation.** Engines 1–80 train, 81–100 validation (split by *engine*, never by row). Final tree/linear models are refit
  on all 100 engines.
* **Evaluation.** Official protocol: **one prediction per test engine (last observed cycle)** versus `RUL_FD001.txt`.
  Metrics: RMSE, MAE, R², and the asymmetric **NASA/PHM08 score** (late predictions are penalised more).

Reference results on FD001 test (RMSE, lower is better), measured with this pipeline: mean baseline ≈ 52.6,
Linear Regression ≈ 21.9, Random Forest ≈ 18.0. Run notebooks 06, 07 and 08 to get XGBoost, Dense and GRU; the
dashboard's *Model comparison* tab computes all of them live.

## Dashboard

| Tab | What it does |
|---|---|
| **Dataset** | Engine lifetime distribution, sensor–RUL correlation, per-engine sensor trends |
| **Engine RUL** | Pick a test engine and a cycle; compare every model's predicted RUL with the true RUL curve |
| **Model comparison** | Official-protocol metrics for all models, bar chart, predicted-vs-true scatter, error histogram |
| **Predict your data** | Upload a C-MAPSS-style file (or enter sensor values by hand) and download predictions as CSV |

### Accepted upload formats

* 26 columns without header: `unit_id, cycle, setting_1..3, sensor_1..21` (raw C-MAPSS format), or
* a CSV with a header row containing at least the 14 sensor names in `utils.FEATURES` (`unit_id` / `cycle` optional), or
* 14 columns without header in `utils.FEATURES` order.

Rows of one engine must be in chronological order (the app sorts by `unit_id`, `cycle`).

## Ideas to improve further

Window length 20–50, LSTM / 1D-CNN variants, ensembling several seeds, hyper-parameter search with `GroupKFold`
(by engine), and the harder subsets FD002–FD004 (multiple operating conditions → needs per-condition normalisation).

## Data source

A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*,
NASA Ames Prognostics Data Repository.
