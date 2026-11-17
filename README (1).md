# NASA C-MAPSS (FD001) — Remaining Useful Life Prediction

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor data
(NASA C-MAPSS, subset **FD001**: 100 train engines, 100 test engines, 1 operating condition, 1 fault mode),
plus an interactive **Streamlit dashboard** to explore the data and compare models.

## Project structure

```
NASA-C-MAPSS/
├── app.py                 # Streamlit dashboard
├── utils.py               # data loading, scaling, metrics, model loading (no Streamlit dependency)
├── requirements.txt
├── data/
│   ├── raw/               # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── processed/         # CSVs produced by the notebooks
├── models/                # scaler.pkl, linear_regression_model.pkl, xgb_model.pkl,
│                          # dense_model.h5, gru_model.keras
└── notebooks/             # 01 loading → 02 EDA → 03 preprocessing → 04/06/07 classical ML → 08/09 neural nets
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Run the command from the project root so `data/` and `models/` are found.

## Dashboard features

| Tab | What it does |
|---|---|
| **Dataset** | Engine lifetime distribution, sensor–RUL correlation, per-engine sensor trends |
| **Engine RUL** | Pick a test engine and a cycle; compare every model's predicted RUL with the true RUL curve |
| **Model comparison** | Official-protocol evaluation (last cycle of each test engine): RMSE, MAE, R², NASA score, predicted-vs-true scatter, error histogram |
| **Predict your data** | Upload a C-MAPSS-style file (or enter sensor values by hand) and download predictions as CSV |

Models are loaded defensively: if TensorFlow or XGBoost is not installed, the corresponding model is
skipped (and the reason is shown in the sidebar) instead of crashing the app.
The sidebar also offers a **Random Forest trained with RUL capped at 125**, fitted on the fly (cached) as an
improved reference model.

### Accepted upload formats

* 26 columns without header: `unit_id, cycle, setting_1..3, sensor_1..21` (raw C-MAPSS format), or
* a CSV with a header row that contains at least the 17 feature names listed in `utils.FEATURES`
  (`unit_id` / `cycle` optional), or
* 17 columns without header already in `utils.FEATURES` order.

## Evaluation protocol (important)

The benchmark evaluates **one prediction per test engine**: the last observed cycle, compared with the
true remaining life in `RUL_FD001.txt`. Test trajectories are truncated *before* failure, so the true RUL of
earlier cycles is `RUL_FD001[engine] + (last_cycle − cycle)`. `utils.load_raw()` implements this.

Metrics reported: RMSE, MAE, R², and the asymmetric **NASA/PHM08 score** (late predictions are penalised more).

## Known limitations of the notebooks (and how to improve them)

1. **Test labels:** notebook 03 computes test RUL as `max_cycle − cycle` inside the truncated test file and never
   uses `RUL_FD001.txt`, so the test metrics printed in notebooks 04–09 are not comparable with the benchmark.
2. **Un-capped RUL (0–361):** early-life RUL is essentially unobservable from sensors. Capping the target
   (≈125 cycles) is standard practice and usually improves accuracy a lot.
3. **No temporal context:** all models see a single time step. The "GRU" notebook builds LSTM layers on sequences of
   length 1, which is equivalent to a dense network. Use sliding windows (e.g. 30 cycles) and/or rolling features.
4. **Metrics evaluated on every row** instead of the last cycle per engine; no NASA score.
5. No hyper-parameter search / cross-validation by engine; only FD001 is used.

## Data source

A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*,
NASA Ames Prognostics Data Repository.
