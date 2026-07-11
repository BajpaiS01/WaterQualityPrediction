import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import gdown

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Water Pollutants Predictor",
    page_icon="💧",
    layout="centered"
)

# ==========================
# Google Drive Model Details
# ==========================

MODEL_PATH = "pollution_model11.pkl"

MODEL_URL = (
    "https://drive.google.com/uc?id=1KonXPBNqgeSfcbWbRmkOI_M91gILRLIo"
)

# ==========================
# Download Model if Missing
# ==========================

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading machine learning model... Please wait."):
        try:
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False,
                fuzzy=True
            )
        except Exception as e:
            st.error(f"Failed to download model:\n{e}")
            st.stop()

# ==========================
# Load Model
# ==========================

try:
    model = joblib.load(MODEL_PATH)
    model_cols = joblib.load("model_columns11.pkl")
except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()

# ==========================
# UI
# ==========================

st.title("💧 Water Pollutants Predictor")

st.write(
    "Predict water pollutant levels based on the selected "
    "**Year** and **Station ID**."
)

# ==========================
# Inputs
# ==========================

year_input = st.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2100,
    value=2022
)

station_id = st.text_input(
    "Enter Station ID",
    value="1"
)

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    if station_id.strip() == "":
        st.warning("Please enter a Station ID.")
        st.stop()

    # Create dataframe
    input_df = pd.DataFrame({
        "year": [year_input],
        "id": [station_id]
    })

    # One-hot encode
    input_encoded = pd.get_dummies(
        input_df,
        columns=["id"]
    )

    # Align columns
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_cols]

    # Predict
    predicted_pollutants = model.predict(input_encoded)[0]

    pollutants = [
        "Dissolved Oxygen (O₂)",
        "Nitrate (NO₃)",
        "Nitrite (NO₂)",
        "Sulphate (SO₄)",
        "Phosphate (PO₄)",
        "Chloride (CL)"
    ]

    st.subheader(
        f"Predicted Pollutant Levels\nStation: {station_id} | Year: {year_input}"
    )

    for pollutant, value in zip(
        pollutants,
        predicted_pollutants
    ):
        st.success(f"**{pollutant}:** {value:.2f}")