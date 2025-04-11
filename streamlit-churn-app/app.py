import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

# ----------------- Helper: Download from GDrive if local missing ----------------- #
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

# ----------------- Cache: Load model and encoders ----------------- #
@st.cache_resource
def load_artifacts():
    model_path = "customer_churn_model.pkl"
    encoder_path = "encoders.pkl"

    if not os.path.exists(model_path):
        download_from_gdrive("1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", model_path)

    if not os.path.exists(encoder_path):
        download_from_gdrive("1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", encoder_path)

    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load(encoder_path)
    return model, feature_names, encoders

# ----------------- Cache: SHAP Explainer ----------------- #
@st.cache_resource
def get_shap_explainer(_model, data):
    return shap.Explainer(_model.predict_proba, data, feature_names=feature_names)

# ----------------- Load Data ----------------- #
model, feature_names, encoders = load_artifacts()

df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# ----------------- SHAP Explainer Setup ----------------- #
explainer = get_shap_explainer(model, df[feature_names])

# ----------------- Streamlit UI ----------------- #
st.title("Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown with top 50 risky customers
default_id = top_50["customerID"].iloc[0]
customer_id = st.selectbox("Select Customer ID", options=top_50["customerID"], index=0, placeholder="e.g., 4614-NUVZD")

if customer_id:
    customer_row = df[df["customerID"] == customer_id]

    if not customer_row.empty:
        st.markdown("### Prediction Details")
        gender = customer_row["gender"].values[0]
        st.markdown(f"**Gender:** {gender}")

        input_data = customer_row[feature_names]

        # Encode
        for col, encoder in encoders.items():
            if col in input_data:
                input_data[col] = encoder.transform(input_data[col])

        # Predict
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction_label = "Likely to Churn" if prediction_proba > 0.5 else "Likely to Stay"

        st.markdown("### Churn Probability")
        st.metric(label="", value=f"{prediction_proba * 100:.2f}%", delta=prediction_label)

        # SHAP Plot
        st.markdown("### SHAP Explanation (Waterfall)")
        shap_values = explainer(input_data)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.error("Customer ID not found.")

