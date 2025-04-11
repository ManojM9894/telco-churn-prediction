import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

# -------------------------------
# Fallback download for artifacts
# -------------------------------
def download_if_missing(path, gdrive_url):
    if not os.path.exists(path):
        file_id = gdrive_url.split("/d/")[1].split("/")[0]
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

@st.cache_resource
def load_artifacts():
    model_path = "customer_churn_model.pkl"
    encoder_path = "encoders.pkl"
    download_if_missing(model_path, "https://drive.google.com/file/d/1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj/view?usp=sharing")
    download_if_missing(encoder_path, "https://drive.google.com/file/d/1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_/view?usp=sharing")
    model = joblib.load(model_path)["model"]
    feature_names = joblib.load(model_path)["features_names"]
    encoders = joblib.load(encoder_path)
    return model, feature_names, encoders

@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model.predict_proba, feature_names=feature_names)

# ----------- Load Artifacts ----------- #
model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model)

# ----------- Load Dataset ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# Handle invalid TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# ----------- UI ----------- #
st.title("Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown: Top 50 risky customer IDs
top_50_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", top_50_ids, key="top50dropdown")

# Get customer details
if selected_id:
    customer = df[df["customerID"] == selected_id]
    if not customer.empty:
        customer = customer.drop(columns=["customerID", "Churn"])
        gender = customer["gender"].values[0]
        st.markdown(f"**Gender**: {gender}")

        # Encoding
        encoded_input = customer.copy()
        for col in encoders:
            if col in encoded_input.columns:
                encoded_input[col] = encoders[col].transform(encoded_input[col])

        input_data = encoded_input[feature_names]

        # Predict
        prob = model.predict_proba(input_data)[0][1]
        churn_label = "Likely to Churn" if prob > 0.5 else "Unlikely to Churn"
        st.metric(label="Churn Probability", value=f"{prob * 100:.2f}%", delta=churn_label)

        # SHAP Plot
        st.subheader("SHAP Explanation (Waterfall)")
        shap_values = explainer(input_data)
        fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

