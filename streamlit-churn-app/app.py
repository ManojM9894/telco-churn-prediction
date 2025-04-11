import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

# ----------------- Constants -----------------
MODEL_URL = "https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj"
ENCODER_URL = "https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_"

MODEL_PATH = "customer_churn_model.pkl"
ENCODER_PATH = "encoders.pkl"

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(ENCODER_PATH):
        gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)

    model_dict = joblib.load(MODEL_PATH)
    model = model_dict["model"]
    feature_names = model_dict["features_names"]
    encoders = joblib.load(ENCODER_PATH)
    return model, feature_names, encoders

@st.cache_resource
def get_shap_explainer(_model, feature_names):
    return shap.Explainer(_model.predict_proba, feature_names=feature_names)

# ----------------- Load Artifacts & Data -----------------
model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model, feature_names)

# Load main dataset
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# Clean data
df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("\U0001F4DE Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown for top 50 risky customers
customer_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", customer_ids)

if selected_id:
    customer_row = df[df["customerID"] == selected_id]
    if not customer_row.empty:
        st.subheader("Prediction Details")
        st.write(f"**Gender:** {customer_row['gender'].values[0]}")

        # Prepare features
        input_data = customer_row[feature_names].copy()
        for col, le in encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        # Predict
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction_label = "Likely to Churn" if prediction_proba > 0.5 else "Likely to Stay"

        st.markdown("### \U0001F4CA Churn Probability")
        st.metric(label="", value=f"{prediction_proba * 100:.2f}%", delta=prediction_label)

        # SHAP Waterfall
        st.markdown("### \U0001F4A1 SHAP Explanation (Waterfall)")
        input_array = input_data.values.astype(float)
        shap_values = explainer(input_array)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.error("Customer ID not found in the dataset.")

