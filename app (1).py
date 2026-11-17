"""NASA C-MAPSS (FD001) – Remaining Useful Life (RUL) dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils as U

st.set_page_config(page_title="C-MAPSS RUL Dashboard", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner="Loading data…")
def get_data():
    return U.load_raw()


@st.cache_resource(show_spinner="Loading scaler…")
def get_scaler():
    return U.load_scaler()


@st.cache_resource(show_spinner="Loading saved models…")
def get_saved_models():
    return U.load_saved_models()


@st.cache_resource(show_spinner="Training Random Forest (RUL capped at 125)…")
def get_capped_rf():
    train, _, _ = get_data()
    return U.train_capped_rf(train, get_scaler())


# --------------------------------------------------------------------------- setup
st.title("✈️ Turbofan Engine RUL Prediction")
st.caption("NASA C-MAPSS · FD001 · Remaining Useful Life estimation")

if not U.data_available():
    st.error(
        "Raw data not found. Place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` "
        "in `data/raw/` (next to `app.py`)."
    )
    st.stop()

train, test, rul_last = get_data()
scaler = get_scaler()
saved_models, load_errors = get_saved_models()

with st.sidebar:
    st.header("Models")
    use_rf = st.toggle(
        "Add Random Forest (RUL capped at 125)",
        value=True,
        help="Trained on the fly from data/raw (a few seconds, then cached). "
             "It fixes the un-capped-RUL issue of the notebook models and is a useful reference.",
    )

models: dict[str, U.Predictor] = dict(saved_models)
if use_rf:
    models["Random Forest (cap 125)"] = get_capped_rf()

with st.sidebar:
    for name in models:
        st.write(f"✅ {name}")
    for name, err in load_errors.items():
        st.write(f"⚠️ {name}")
        st.caption(err[:200])
    st.divider()
    st.caption(
        "Saved models were trained on **un-capped** RUL (0–361) and scored row-by-row on "
        "mis-labelled test data in the notebooks. The *Model comparison* tab re-evaluates them "
        "with the official protocol."
    )

if not models:
    st.error("No model could be loaded. Check the `models/` folder and installed packages.")
    st.stop()

train_mean_rul = float(train["RUL"].mean())

tab_data, tab_engine, tab_compare, tab_predict = st.tabs(
    ["📊 Dataset", "🔧 Engine RUL", "🏁 Model comparison", "🧪 Predict your data"]
)

# ------------------------------------------------------------------------ Dataset
with tab_data:
    life = train.groupby("unit_id")["cycle"].max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training engines", train["unit_id"].nunique())
    c2.metric("Test engines", test["unit_id"].nunique())
    c3.metric("Mean life (train)", f"{life.mean():.0f} cycles")
    c4.metric("Life range", f"{life.min()} – {life.max()}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(life, nbins=25, labels={"value": "Cycles to failure"},
                           title="Distribution of engine lifetimes (train)")
        fig.update_layout(showlegend=False, yaxis_title="Engines")
        st.plotly_chart(fig)
    with right:
        corr = train[U.SENSOR_FEATURES + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
        fig = px.bar(corr, orientation="h", labels={"value": "Correlation with RUL", "index": ""},
                     title="Sensor correlation with RUL")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    st.subheader("Sensor trends")
    s1, s2 = st.columns([1, 2])
    sensor = s1.selectbox("Sensor", U.SENSOR_FEATURES, index=U.SENSOR_FEATURES.index("sensor_11"))
    units = s2.multiselect("Training engines", sorted(train["unit_id"].unique()), default=[1, 2, 3])
    x_axis = s1.radio("X axis", ["cycle", "RUL"], horizontal=True)
    sub = train[train["unit_id"].isin(units)]
    fig = px.line(sub, x=x_axis, y=sensor, color="unit_id", title=f"{sensor} vs {x_axis}")
    if x_axis == "RUL":
        fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig)

    with st.expander("Features used by the models"):
        st.write(", ".join(U.FEATURES))
        st.caption("Constant / near-constant sensors (1, 5, 6, 10, 16, 18, 19) were removed in notebook 03.")

# --------------------------------------------------------------------- Engine RUL
with tab_engine:
    c1, c2 = st.columns([1, 2])
    unit = c1.selectbox("Test engine", sorted(test["unit_id"].unique()))
    eng = test[test["unit_id"] == unit].sort_values("cycle").reset_index(drop=True)
    last_cycle = int(eng["cycle"].max())
    cycle = c2.slider("Observed up to cycle", 1, last_cycle, last_cycle)
    chosen = st.multiselect("Models", list(models), default=list(models))

    if chosen:
        Xs = U.scale(scaler, eng)
        preds = {m: U.predict(models[m], Xs) for m in chosen}
        row = eng.index[eng["cycle"] == cycle][0]
        true_now = float(eng.loc[row, "RUL"])

        cols = st.columns(len(chosen) + 1)
        cols[0].metric("True RUL (from RUL_FD001)", f"{true_now:.0f}")
        for col, m in zip(cols[1:], chosen):
            p = float(preds[m][row])
            col.metric(m, f"{p:.0f}", delta=f"{p - true_now:+.0f} vs true", delta_color="off")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eng["cycle"], y=eng["RUL"], name="True RUL",
                                 line=dict(color="black", dash="dash")))
        for m in chosen:
            fig.add_trace(go.Scatter(x=eng["cycle"], y=preds[m], name=m, mode="lines"))
        fig.add_vline(x=cycle, line_color="grey", line_dash="dot")
        fig.update_layout(title=f"Engine {unit}: predicted vs true RUL", xaxis_title="Cycle",
                          yaxis_title="RUL (cycles)", hovermode="x unified")
        st.plotly_chart(fig)

        first = float(preds[chosen[0]][row])
        msg = f"{chosen[0]} predicts ≈ {first:.0f} cycles of remaining life at cycle {cycle}."
        if first > 75:
            st.success(f"🟢 Healthy — {msg}")
        elif first > 30:
            st.warning(f"🟡 Plan maintenance — {msg}")
        else:
            st.error(f"🔴 Critical — {msg}")
        st.caption("Status thresholds (75 / 30 cycles) are illustrative only.")

        sensors = st.multiselect("Sensors to inspect", U.SENSOR_FEATURES,
                                 default=["sensor_2", "sensor_4", "sensor_11"])
        normalise = st.checkbox("Normalise sensors (z-score)", value=True)
        if sensors:
            plot_df = eng[["cycle"] + sensors].copy()
            if normalise:
                plot_df[sensors] = (plot_df[sensors] - plot_df[sensors].mean()) / plot_df[sensors].std()
            fig = px.line(plot_df, x="cycle", y=sensors, title=f"Engine {unit}: sensor readings")
            fig.add_vline(x=cycle, line_color="grey", line_dash="dot")
            st.plotly_chart(fig)
    else:
        st.info("Select at least one model.")

# ------------------------------------------------------------------ Model compare
with tab_compare:
    st.markdown(
        "Official C-MAPSS protocol: **one prediction per test engine** (its last observed cycle) "
        "compared with the true remaining life from `RUL_FD001.txt` (100 points)."
    )
    cap = st.checkbox(
        f"Cap true and predicted RUL at {U.RUL_CAP} (standard in the literature)", value=False
    )
    metrics, pred_df = U.evaluate_models(models, scaler, test, train_mean_rul, cap=cap)

    st.dataframe(
        metrics.style.format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "R²": "{:.3f}", "NASA score": "{:,.0f}"}),
        hide_index=True,
    )
    st.caption("Lower is better for RMSE / MAE / NASA score. “Mean baseline” always predicts the "
               "mean training RUL — a model that does not beat it has learned nothing useful.")

    c1, c2 = st.columns(2)
    fig = px.bar(metrics, x="Model", y="RMSE", color="Model", title="RMSE by model (lower is better)")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(fig)

    pick = c2.selectbox("Inspect model", [m for m in pred_df.columns[2:]], index=0)
    fig = px.scatter(pred_df, x="True RUL", y=pick, hover_data=["unit_id"],
                     title=f"{pick}: predicted vs true RUL (100 test engines)")
    top = float(max(pred_df["True RUL"].max(), pred_df[pick].max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect",
                             line=dict(color="red", dash="dash")))
    c2.plotly_chart(fig)

    err = pred_df[pick] - pred_df["True RUL"]
    fig = px.histogram(err, nbins=30, labels={"value": "Prediction error (pred − true)"},
                       title=f"{pick}: error distribution (positive = late prediction)")
    fig.update_layout(showlegend=False, yaxis_title="Engines")
    st.plotly_chart(fig)

# ----------------------------------------------------------------- Predict yours
with tab_predict:
    mode = st.radio("Input", ["Upload a file", "Enter sensor values manually"], horizontal=True)
    model_name = st.selectbox("Model", list(models), key="predict_model")

    if mode == "Upload a file":
        st.markdown(
            "Upload a `.txt` / `.csv` in C-MAPSS format: either **26 columns without header** "
            "(`unit_id, cycle, 3 settings, 21 sensors`) or a **header row** containing at least the "
            "17 feature names. Separators: space, tab, comma or semicolon."
        )
        file = st.file_uploader("Data file", type=["txt", "csv"])
        c1, c2 = st.columns(2)
        if c1.button("Use the built-in test file as an example"):
            file = None
            st.session_state["example"] = True
        if file is not None:
            st.session_state["example"] = False

        df_in = None
        try:
            if file is not None:
                df_in = U.parse_upload(file.getvalue())
            elif st.session_state.get("example"):
                df_in = test.drop(columns=["RUL"])
        except ValueError as exc:
            st.error(str(exc))

        if df_in is not None:
            Xs = U.scale(scaler, df_in)
            out = df_in[["unit_id", "cycle"]].copy()
            out["predicted_RUL"] = U.predict(models[model_name], Xs).round(1)
            last = U.last_cycle_rows(out)
            st.subheader("Latest prediction per engine")
            st.dataframe(last, hide_index=True)
            st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(),
                               "rul_predictions.csv", "text/csv")
            unit_pick = st.selectbox("Engine", sorted(out["unit_id"].unique()), key="up_unit")
            fig = px.line(out[out["unit_id"] == unit_pick], x="cycle", y="predicted_RUL",
                          title=f"Engine {unit_pick}: predicted RUL over time")
            st.plotly_chart(fig)
    else:
        st.markdown("Values default to the training-set mean; ranges come from the training data.")
        stats = train[U.FEATURES].agg(["min", "max", "mean"])
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(U.FEATURES):
            lo, hi, mu = (float(stats.loc[k, f]) for k in ("min", "max", "mean"))
            if hi - lo < 1e-9:
                values[f] = mu
                cols[i % 3].number_input(f, value=mu, disabled=True, format="%.4f", key=f"m_{f}")
            else:
                values[f] = cols[i % 3].slider(f, lo, hi, mu, step=(hi - lo) / 200, key=f"m_{f}")
        row = pd.DataFrame([values])
        p = float(U.predict(models[model_name], U.scale(scaler, row))[0])
        st.metric(f"{model_name} — predicted RUL", f"{p:.0f} cycles")
        st.caption("Single-snapshot prediction: the models do not use history, only the current reading.")
