import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Turbofan Engine RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

# ==========================
# LOAD MODEL FILES
# ==========================

model = joblib.load("xgboost_rul_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# ==========================
# FEATURE LIST
# ==========================

features = [
    'op_setting_1',
    'op_setting_2',
    'op_setting_3',
    'sensor_1',
    'sensor_2',
    'sensor_3',
    'sensor_4',
    'sensor_5',
    'sensor_6',
    'sensor_7',
    'sensor_8',
    'sensor_9',
    'sensor_10',
    'sensor_11',
    'sensor_12',
    'sensor_13',
    'sensor_14',
    'sensor_15',
    'sensor_16',
    'sensor_17',
    'sensor_18',
    'sensor_19',
    'sensor_20',
    'sensor_21'
]

# ==========================
# TITLE
# ==========================

st.title("✈️ Turbofan Engine Remaining Useful Life Prediction System")

st.markdown("""
Predict Remaining Useful Life (RUL) of aircraft engines using
NASA CMAPSS FD001 dataset and XGBoost Machine Learning model.
""")

# ==========================
# MODEL METRICS
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "12.49")

with col2:
    st.metric("RMSE", "17.50")

with col3:
    st.metric("R² Score", "0.82")

st.markdown("---")

# ==========================
# MODE SELECTION
# ==========================

mode = st.radio(
    "Select Prediction Mode",
    [
        "Dataset Prediction",
        "Manual Input",
        "What-If Analysis"
    ]
)

# ======================================================
# MODE 1 : DATASET PREDICTION
# ======================================================

if mode == "Dataset Prediction":

    uploaded_file = st.file_uploader(
        "Upload test_FD001.txt",
        type=["txt"]
    )

    if uploaded_file is not None:

        cols = (
            ["engine_id", "cycle"]
            + [f"op_setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )

        test_df = pd.read_csv(
            uploaded_file,
            sep=r"\s+",
            header=None
        )

        test_df = test_df.iloc[:, :26]
        test_df.columns = cols

        test_last = (
            test_df.groupby("engine_id")
            .last()
            .reset_index()
        )

        engine_ids = sorted(
            test_last["engine_id"].unique()
        )

        selected_engine = st.selectbox(
            "Select Engine ID",
            engine_ids
        )

        if st.button("Predict Engine RUL"):

            sample = pd.DataFrame([
                test_last[
                    test_last["engine_id"] ==
                    selected_engine
                ][features].iloc[0]
            ])

            selected = selector.transform(sample)
            scaled = scaler.transform(selected)

            prediction = float(
                model.predict(scaled)[0]
            )

            st.success(
                f"Predicted RUL: {prediction:.2f} cycles"
            )

            if prediction > 100:
                st.success("🟢 Healthy Engine")

            elif prediction > 50:
                st.warning("🟡 Moderate Condition")

            else:
                st.error("🔴 Maintenance Required")

            with st.expander("View Engine Data"):
                st.dataframe(sample)

# ======================================================
# MODE 2 : MANUAL INPUT
# ======================================================

elif mode == "Manual Input":

    st.subheader("Enter Sensor Values")

    values = {}

    col1, col2 = st.columns(2)

    with col1:

        values["op_setting_1"] = st.number_input(
            "Operational Setting 1",
            value=-0.0006
        )

        values["op_setting_2"] = st.number_input(
            "Operational Setting 2",
            value=0.0004
        )

        values["op_setting_3"] = st.number_input(
            "Operational Setting 3",
            value=100.0
        )

        for i in range(1, 11):
            values[f"sensor_{i}"] = st.number_input(
                f"Sensor {i}",
                value=0.0
            )

    with col2:

        for i in range(11, 22):
            values[f"sensor_{i}"] = st.number_input(
                f"Sensor {i}",
                value=0.0
            )

    if st.button("Predict Manual Input"):

        sample = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        selected = selector.transform(sample)
        scaled = scaler.transform(selected)

        prediction = float(
            model.predict(scaled)[0]
        )

        prediction = max(0, prediction)

        st.success(
            f"Predicted RUL: {prediction:.2f} cycles"
        )

        if prediction > 100:
            st.success("🟢 Healthy Engine")

        elif prediction > 50:
            st.warning("🟡 Moderate Condition")

        else:
            st.error("🔴 Maintenance Required")

# ======================================================
# MODE 3 : WHAT-IF ANALYSIS
# ======================================================

else:

    st.subheader("What-If Analysis")

    st.write(
        "Move Sensor 11 slider and observe how prediction changes."
    )

    sample_values = {
        'op_setting_1': -0.0006,
        'op_setting_2': 0.0004,
        'op_setting_3': 100.0,
        'sensor_1': 518.67,
        'sensor_2': 642.58,
        'sensor_3': 1581.22,
        'sensor_4': 1398.91,
        'sensor_5': 14.62,
        'sensor_6': 21.61,
        'sensor_7': 554.42,
        'sensor_8': 2388.08,
        'sensor_9': 9056.40,
        'sensor_10': 1.30,
        'sensor_11': 47.23,
        'sensor_12': 521.79,
        'sensor_13': 2388.06,
        'sensor_14': 8130.11,
        'sensor_15': 8.4024,
        'sensor_16': 0.03,
        'sensor_17': 393,
        'sensor_18': 2388,
        'sensor_19': 100,
        'sensor_20': 38.81,
        'sensor_21': 23.3552
    }

    sensor11 = st.slider(
        "Sensor 11",
        min_value=20.0,
        max_value=80.0,
        value=47.23
    )

    sample_values["sensor_11"] = sensor11

    sample = pd.DataFrame(
        [[sample_values[f] for f in features]],
        columns=features
    )

    selected = selector.transform(sample)
    scaled = scaler.transform(selected)

    prediction = float(
        model.predict(scaled)[0]
    )

    st.metric(
        "Predicted RUL",
        f"{prediction:.2f} cycles"
    )

    if prediction > 100:
        st.success("🟢 Healthy Engine")

    elif prediction > 50:
        st.warning("🟡 Moderate Condition")

    else:
        st.error("🔴 Maintenance Required")

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

st.markdown("---")

st.subheader("Top Feature Importance")

try:

    importance = pd.Series(
        model.feature_importances_
    )

    importance = importance.sort_values(
        ascending=False
    ).head(10)

    st.bar_chart(importance)

except:
    st.info(
        "Feature importance not available."
    )

st.subheader("Model Insights")

top_feature = importance.index[0]

st.info(
    f"Most Important Feature: {top_feature}"
)
# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
### Project Information

**Project:** Turbofan Engine Remaining Useful Life Prediction

**Dataset:** NASA CMAPSS FD001

**Algorithm:** XGBoost Regressor

**Performance:**
- MAE = 12.49
- RMSE = 17.50
- R² Score = 0.82

Developed using Python, Scikit-Learn, XGBoost and Streamlit.
""")