# Import all the necessary libraries
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import gdown

MODEL_PATH = "pollution_model11.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1KonXPBNqgeSfcbWbRmkOI_M91gILRLIo"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading machine learning model... Please wait."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)
model_cols = joblib.load("model_columns11.pkl")

st.title("💧 Water Pollutants Predictor")
st.write("Predict water pollutant levels based on **Year** and **Station ID**.")

# User inputs
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

if st.button("Predict"):

    if not station_id.strip():
        st.warning("Please enter the Station ID.")

    else:

        # Prepare input dataframe
        input_df = pd.DataFrame({
            "year": [year_input],
            "id": [station_id]
        })

        # One-hot encoding
        input_encoded = pd.get_dummies(input_df, columns=["id"])

        # Match model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[model_cols]

        # Prediction
        predicted_pollutants = model.predict(input_encoded)[0]

        pollutants = ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]

        st.subheader(
            f"Predicted pollutant levels for Station '{station_id}' in {year_input}"
        )

        for pollutant, value in zip(pollutants, predicted_pollutants):
            st.write(f"**{pollutant}:** {value:.2f}")
