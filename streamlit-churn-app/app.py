import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

# ----------- Fallback Downloads from Google Drive ----------- #
def download_file_from_gdrive(url_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={url_id}", output, quiet=False)

download_file_from_gdrive("1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", "customer_churn_model.pkl")
download_file_from_gdrive("1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", "encoders.pkl")
download_file_from_gdrive("1pQpWm9DmeAi5Kn-SsMFbYeQMGd5vZkYy", "streamlit-churn-app/telco_churn.csv")

# ----------- Load Artifacts ----------- #
@st.cache_resource
def load_artifacts():
    model_data = joblib.load("customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("encoders.pkl")
    return model, feature_names, encoders

model, feature_names, encoders = load_artifacts()

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# ----------- SHAP Explainer ----------- #
@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

# ----------- UI ----------- #
st.title("Telco Customer Churn Predictor")
st.markdown("**Select a Customer ID to see churn prediction and explanation.**")

# Combine Top 50 + All IDs in dropdown
top_ids = top_50["customerID"].tolist()
all_ids = df["customerID"].tolist()
options = top_ids + sorted(set(all_ids) - set(top_ids))
selected_id = st.selectbox("Select Customer ID", options)

if selected_id:
    customer = df[df["customerID"] == selected_id]
    st.subheader("Prediction Details")

    gender = customer["gender"].values[0]
    st.markdown(f"**Gender:** {gender}")

    input_data = customer[feature_names].copy()

    for col in encoders:
        if col in input_data:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_label = "Likely to Churn" if prediction_proba >= 0.5 else "Not Likely to Churn"

    st.markdown("**Churn Probability**")
    st.metric(label="", value=f"{prediction_proba*100:.2f}%", delta=prediction_label)

    # SHAP Plot
    st.subheader("SHAP Explanation (Waterfall)")
    shap_values = explainer(input_data)
    fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

