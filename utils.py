"""Core logic for the C-MAPSS (FD001) RUL dashboard.

This module has no Streamlit dependency so it can be imported and tested on its own.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from types import SimpleNamespace
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

# The 14 informative sensors. Constant sensors (1, 5, 6, 10, 16, 18, 19) are dropped, and so are the
# 3 operating settings: FD001 has a single operating condition, so they are pure noise.
FEATURES = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
SENSOR_FEATURES = FEATURES

SEQ_WINDOW = 30  # time-steps (cycles) fed to the recurrent model
VAL_FROM_UNIT = 81  # training engines 1-80 -> train split, 81-100 -> validation split (split by ENGINE)


@dataclass
class Inputs:
    """Model inputs built from one data frame (rows aligned with the frame)."""

    tab: pd.DataFrame  # (n, 14) scaled sensors            -> Linear / RF / XGBoost / Dense
    seq: np.ndarray    # (n, SEQ_WINDOW, 14) scaled window -> GRU

    def take(self, idx) -> "Inputs":
        idx = np.asarray(idx)
        return Inputs(self.tab.iloc[idx].reset_index(drop=True), self.seq[idx])


Predictor = Callable[[Inputs], np.ndarray]


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


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by engine and cycle (windows are built assuming this order)."""
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def make_windows(scaled: np.ndarray, unit_ids: np.ndarray, window: int = SEQ_WINDOW) -> np.ndarray:
    """One window per row: the last `window` cycles of the SAME engine ending at that row.

    Early cycles are left-padded by repeating the engine's first reading, so every row gets a window
    and no information from other engines leaks in. `scaled` must be sorted by (unit_id, cycle).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    scaled = np.asarray(scaled, dtype="float32")
    out = np.empty((len(scaled), window, scaled.shape[1]), dtype="float32")
    for unit in np.unique(unit_ids):
        rows = np.where(unit_ids == unit)[0]
        a = scaled[rows]
        padded = np.vstack([np.repeat(a[:1], window - 1, axis=0), a])
        out[rows] = sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)
    return out


def build_inputs(scaler, df: pd.DataFrame, window: int = SEQ_WINDOW) -> Inputs:
    """Scale the 14 sensors and build tabular + windowed inputs. `df` must be sorted (see sort_frame)."""
    tab = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES, index=df.index)
    seq = make_windows(tab.to_numpy(), df["unit_id"].to_numpy(), window)
    return Inputs(tab.reset_index(drop=True), seq)


def load_dataset() -> SimpleNamespace:
    """Everything the model notebooks need, built consistently in one place.

    Targets: ``y_train`` is the TRAIN RUL capped at RUL_CAP; ``y_test`` is the official true RUL of each
    test engine at its last observed cycle (``last_idx`` are the matching rows of ``test``).
    """
    train, test, _ = load_raw()
    train, test = sort_frame(train), sort_frame(test)
    scaler = load_scaler()
    last_idx = test.groupby("unit_id").tail(1).index.to_numpy()
    return SimpleNamespace(
        train=train, test=test, scaler=scaler,
        X_train=build_inputs(scaler, train), X_test=build_inputs(scaler, test),
        y_train=train["RUL"].clip(upper=RUL_CAP).to_numpy(float),
        is_val=(train["unit_id"] >= VAL_FROM_UNIT).to_numpy(),
        last_idx=last_idx, y_test=test.loc[last_idx, "RUL"].to_numpy(float),
        train_mean_rul=float(train["RUL"].mean()),
    )


# ------------------------------------------------------------------------- models
def load_scaler():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise RuntimeError(
            "models/scaler.pkl was fitted on a different feature set. "
            "Re-run notebooks/03_data_preprocessing.ipynb (and the model notebooks)."
        )
    return scaler


def _stale(name: str, got) -> str:
    return (f"{name} expects {got} input features but the pipeline now uses {len(FEATURES)}. "
            "It is an old model: re-run its notebook.")


def load_saved_models() -> tuple[dict[str, Predictor], dict[str, str]]:
    """Load every saved model that can be loaded. Returns (models, errors).

    A model that fails to load (missing file, old version, TensorFlow / xgboost not installed) is
    reported in `errors` instead of crashing the app.
    """
    models: dict[str, Predictor] = {}
    errors: dict[str, str] = {}

    def sk(name: str, filename: str) -> None:
        try:
            m = joblib.load(MODEL_DIR / filename)
            n_in = getattr(m, "n_features_in_", len(FEATURES))
            if n_in != len(FEATURES):
                raise RuntimeError(_stale(name, n_in))
            models[name] = lambda x, m=m: np.ravel(m.predict(x.tab))
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"

    sk("Linear Regression", "linear_regression_model.pkl")
    sk("Random Forest", "random_forest_model.pkl")
    sk("XGBoost", "xgb_model.pkl")

    nn_names = ("Dense NN", f"GRU ({SEQ_WINDOW}-cycle window)")
    try:
        try:
            import keras  # Keras 3 (standalone)
        except ImportError:
            from tensorflow import keras  # type: ignore
    except Exception as exc:  # noqa: BLE001
        for n in nn_names:
            errors[n] = f"{type(exc).__name__}: {exc}"
        return models, errors

    def kr(name: str, filename: str, seq: bool) -> None:
        try:
            m = keras.models.load_model(MODEL_DIR / filename, compile=False)
            shape = m.input_shape
            if shape[-1] != len(FEATURES) or (len(shape) == 3) != seq:
                raise RuntimeError(_stale(name, shape))
            # Networks are trained on RUL / RUL_CAP, so scale the output back.
            if seq:
                models[name] = lambda x, m=m: m.predict(x.seq, verbose=0).ravel() * RUL_CAP
            else:
                models[name] = lambda x, m=m: m.predict(
                    x.tab.to_numpy("float32"), verbose=0).ravel() * RUL_CAP
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"

    kr(nn_names[0], "dense_model.keras", seq=False)
    kr(nn_names[1], "gru_model.keras", seq=True)
    return models, errors


def predict(model: Predictor, inputs: Inputs) -> np.ndarray:
    """Predict RUL (negative values are meaningless, so they are clipped at 0)."""
    return np.clip(np.asarray(model(inputs), dtype=float), 0, None)


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
    test = sort_frame(test)
    inputs = build_inputs(scaler, test)
    last_idx = test.groupby("unit_id").tail(1).index.to_numpy()
    last_inputs = inputs.take(last_idx)
    y_true = test.loc[last_idx, "RUL"].to_numpy(float)

    preds = {"unit_id": test.loc[last_idx, "unit_id"].to_numpy(), "True RUL": y_true}
    for name, model in models.items():
        preds[name] = predict(model, last_inputs)
    preds["Mean baseline"] = np.full(len(last_idx), train_mean_rul)
    pred_df = pd.DataFrame(preds)

    if cap:
        for c in pred_df.columns[1:]:
            pred_df[c] = pred_df[c].clip(upper=RUL_CAP)

    rows = [{
        "Model": c,
        "RMSE": rmse(pred_df["True RUL"], pred_df[c]),
        "MAE": mae(pred_df["True RUL"], pred_df[c]),
        "R²": r2(pred_df["True RUL"], pred_df[c]),
        "NASA score": nasa_score(pred_df["True RUL"], pred_df[c]),
    } for c in pred_df.columns[2:]]
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return metrics, pred_df


def score_predictions(y_true, y_pred) -> dict[str, float]:
    """All four metrics in one dict (used by the notebooks)."""
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "R2": r2(y_true, y_pred), "NASA": nasa_score(y_true, y_pred)}


# ------------------------------------------------------------------------- upload
def parse_upload(raw: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV/TXT in C-MAPSS format.

    Accepted layouts:
      * header row containing the 14 feature names (unit_id / cycle optional)
      * no header, 26 columns (unit_id, cycle, 3 settings, 21 sensors)
      * no header, exactly 14 columns (already the feature set, in FEATURES order)
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
            f"Expected 26 columns (raw C-MAPSS) or 14 columns (features) without a header, "
            f"but found {df.shape[1]}."
        )

    if "unit_id" not in df.columns:
        df.insert(0, "unit_id", 1)
    if "cycle" not in df.columns:
        df.insert(1, "cycle", df.groupby("unit_id").cumcount() + 1)

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("Some feature values are missing or not numeric.")
    return sort_frame(df)
