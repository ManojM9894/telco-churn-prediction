import os
import joblib
import zipfile
import numpy as np
import pandas as pd
import shap
import streamlit as st
import gdown

st.set_page_config(page_title="Telco Churn App", layout="wide")

# ----------- Download model files from GDrive if not present ----------- #
MODEL_GDRIVE_URL = "https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj"
ENCODER_GDRIVE_URL = "https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_"

if not os.path.exists("customer_churn_model.pkl"):
    gdown.download(MODEL_GDRIVE_URL, "customer_churn_model.pkl", quiet=False)

if not os.path.exists("encoders.pkl"):
    gdown.download(ENCODER_GDRIVE_URL, "encoders.pkl", quiet=False)

# ----------- Load Model & Encoders ----------- #
@st.cache_resource
def load_artifacts():
    model_data = joblib.load("customer_churn_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# Handle parsing issue in TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# ----------- SHAP Explainer ----------- #
def get_shap_explainer(model):
    # Choose the first estimator (rf or xgb) for SHAP explanation
    return shap.Explainer(model.estimators_[0][1])

explainer = get_shap_explainer(model)

# ----------- UI ----------- #
st.title("Telco Customer Churn Predictor")
st.markdown("Enter a **Customer ID** to view their churn likelihood and SHAP explanation:")

# Combine dropdown and text input
col1, col2 = st.columns([1, 2])
with col1:
    selected_id = st.selectbox("Choose a high-risk customer (Top 50):", top_50["customerID"].tolist())
with col2:
    manual_id = st.text_input("Or enter any Customer ID:", placeholder="e.g., 1452-KIOVK")

# Decide which input to prioritize
customer_id = manual_id if manual_id else selected_id

if customer_id:
    customer_row = df[df["customerID"] == customer_id]
    if customer_row.empty:
        st.warning("Customer ID not found in the dataset.")
    else:
        # Drop ID column, keep features only
        input_data = customer_row.drop(columns=["customerID", "Churn"])

        # Apply encoders
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        # Predict churn
        proba = model.predict_proba(input_data)[0][1]
        st.metric(label="Churn Probability", value=f"{proba*100:.2f}%")

        # SHAP Waterfall Plot
        st.subheader("SHAP Explanation")
        shap_values = explainer(input_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot()

