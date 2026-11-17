"""Core logic for the C-MAPSS (FD001) RUL dashboard.

This module has no Streamlit dependency so it can be imported and tested on its own.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"

RUL_CAP = 125  # common piece-wise linear cap used in the C-MAPSS literature

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in (1, 2, 3)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# The 17 features used in notebook 03 (constant / near-constant sensors removed).
FEATURES = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
SENSOR_FEATURES = [f for f in FEATURES if f.startswith("sensor_")]

Predictor = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- data
def data_available() -> bool:
    return all((RAW_DIR / f).exists() for f in
               ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"))


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, rul_last).

    * train["RUL"] = max_cycle - cycle (run-to-failure, so this is exact).
    * test["RUL"]  = RUL_FD001[unit] + (last_observed_cycle - cycle).
      Test trajectories are cut BEFORE failure; the official RUL file gives the
      remaining life after the last observed cycle, so it must be added back.
    """
    train = pd.read_csv(RAW_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(RAW_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

    train["RUL"] = train.groupby("unit_id")["cycle"].transform("max") - train["cycle"]

    offset = test["unit_id"].map(pd.Series(rul["RUL"].to_numpy(), index=range(1, len(rul) + 1)))
    last_cycle = test.groupby("unit_id")["cycle"].transform("max")
    test["RUL"] = offset + (last_cycle - test["cycle"])
    return train, test, rul


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Last observed row of every engine (what the official benchmark evaluates)."""
    return df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).reset_index(drop=True)


def scale(scaler, df: pd.DataFrame) -> pd.DataFrame:
    """Scale the 17 features and keep column names (sklearn/xgboost were fitted on DataFrames)."""
    return pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES, index=df.index)


# ------------------------------------------------------------------------- models
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")


def load_saved_models() -> tuple[dict[str, Predictor], dict[str, str]]:
    """Load every saved model that can be loaded. Returns (models, errors).

    A model that fails to load (e.g. TensorFlow / xgboost not installed) is reported in
    `errors` instead of crashing the app.
    """
    models: dict[str, Predictor] = {}
    errors: dict[str, str] = {}

    def sk(name: str, path: Path) -> None:
        try:
            m = joblib.load(path)
            models[name] = lambda X, m=m: np.ravel(m.predict(X))
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"

    sk("Linear Regression", MODEL_DIR / "linear_regression_model.pkl")
    sk("XGBoost", MODEL_DIR / "xgb_model.pkl")

    try:
        try:
            import keras  # Keras 3 (standalone)
        except ImportError:
            from tensorflow import keras  # type: ignore
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        errors["Dense NN"] = msg
        errors["LSTM (1 time-step)"] = msg
        return models, errors

    try:
        dense = keras.models.load_model(MODEL_DIR / "dense_model.h5", compile=False)
        models["Dense NN"] = lambda X, m=dense: m.predict(np.asarray(X, "float32"), verbose=0).ravel()
    except Exception as exc:  # noqa: BLE001
        errors["Dense NN"] = f"{type(exc).__name__}: {exc}"

    try:
        # NOTE: the file is called gru_model.keras but notebook 09 builds LSTM layers.
        seq = keras.models.load_model(MODEL_DIR / "gru_model.keras", compile=False)
        models["LSTM (1 time-step)"] = lambda X, m=seq: m.predict(
            np.asarray(X, "float32").reshape(-1, 1, len(FEATURES)), verbose=0
        ).ravel()
    except Exception as exc:  # noqa: BLE001
        errors["LSTM (1 time-step)"] = f"{type(exc).__name__}: {exc}"

    return models, errors


def train_capped_rf(train: pd.DataFrame, scaler, n_estimators: int = 200) -> Predictor:
    """Random Forest trained on ALL training engines with RUL capped at RUL_CAP.

    The original notebooks trained on un-capped RUL (0..361), which is a known cause of poor
    accuracy on C-MAPSS. This model is provided as an improved reference.
    """
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    rf.fit(scale(scaler, train), train["RUL"].clip(upper=RUL_CAP))
    return lambda X: np.ravel(rf.predict(X))


def predict(model: Predictor, X_scaled: pd.DataFrame) -> np.ndarray:
    """Predict RUL (negative values are meaningless, so they are clipped at 0)."""
    return np.clip(np.asarray(model(X_scaled), dtype=float), 0, None)


# ------------------------------------------------------------------------ metrics
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score: late predictions (pred > true) are penalised more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def evaluate_models(
    models: dict[str, Predictor],
    scaler,
    test: pd.DataFrame,
    train_mean_rul: float,
    cap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official-style evaluation: one prediction per test engine (its last observed cycle).

    Returns (metrics_table, predictions_table).
    """
    last = last_cycle_rows(test)
    Xs = scale(scaler, last)
    y_true = last["RUL"].to_numpy(float)

    preds = {"unit_id": last["unit_id"].to_numpy(), "True RUL": y_true}
    for name, model in models.items():
        preds[name] = predict(model, Xs)
    preds["Mean baseline"] = np.full(len(last), train_mean_rul)
    pred_df = pd.DataFrame(preds)

    if cap:
        pred_df["True RUL"] = pred_df["True RUL"].clip(upper=RUL_CAP)
        for c in pred_df.columns[2:]:
            pred_df[c] = pred_df[c].clip(upper=RUL_CAP)

    rows = []
    for c in pred_df.columns[2:]:
        rows.append({
            "Model": c,
            "RMSE": rmse(pred_df["True RUL"], pred_df[c]),
            "MAE": mae(pred_df["True RUL"], pred_df[c]),
            "R²": r2(pred_df["True RUL"], pred_df[c]),
            "NASA score": nasa_score(pred_df["True RUL"], pred_df[c]),
        })
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return metrics, pred_df


# ------------------------------------------------------------------------- upload
def parse_upload(raw: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV/TXT in C-MAPSS format.

    Accepted layouts:
      * header row containing the 17 feature names (unit_id / cycle optional)
      * no header, 26 columns (unit_id, cycle, 3 settings, 21 sensors)
      * no header, exactly 17 columns (already the feature set, in FEATURES order)
    Separators may be commas, semicolons, spaces or tabs.
    """
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("The file is empty.")

    has_header = any(ch.isalpha() for ch in text.splitlines()[0])
    df = pd.read_csv(
        io.StringIO(text), sep=r"[,;\s]+", engine="python", header=0 if has_header else None
    ).dropna(axis=1, how="all")

    if has_header:
        df.columns = [str(c).strip() for c in df.columns]
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
    elif df.shape[1] == len(COLUMNS):
        df.columns = COLUMNS
    elif df.shape[1] == len(FEATURES):
        df.columns = FEATURES
    else:
        raise ValueError(
            f"Expected 26 columns (raw C-MAPSS) or 17 columns (features) without a header, "
            f"but found {df.shape[1]}."
        )

    if "unit_id" not in df.columns:
        df.insert(0, "unit_id", 1)
    if "cycle" not in df.columns:
        df.insert(1, "cycle", df.groupby("unit_id").cumcount() + 1)

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("Some feature values are missing or not numeric.")
    return df
