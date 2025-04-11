import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import gdown
import os
import matplotlib.pyplot as plt

# ----------- Load Model & Encoders with Fallback ----------- #
@st.cache_resource
def load_artifacts():
    model_path = "customer_churn_model.pkl"
    encoder_path = "encoders.pkl"
    if not os.path.exists(model_path):
        gdown.download(id="1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", output=model_path, quiet=False)
    if not os.path.exists(encoder_path):
        gdown.download(id="1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", output=encoder_path, quiet=False)

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_names = model_bundle["features_names"]
    encoders = joblib.load(encoder_path)
    return model, feature_names, encoders

# ----------- SHAP Explainer with Masker ----------- #
@st.cache_resource
def get_shap_explainer(_model, background_data, feature_names):
    return shap.Explainer(_model.predict_proba, masker=background_data, feature_names=feature_names)

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model, df[feature_names].sample(100, random_state=42), feature_names)

# ----------- UI ----------- #
st.title("Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown: Top 50 risky customers
selected_id = st.selectbox("Select Customer ID", top_50["customerID"].tolist())

if selected_id:
    customer = df[df["customerID"] == selected_id].iloc[0]

    # Prediction
    input_data = pd.DataFrame([customer[feature_names]])
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    pred_proba = model.predict_proba(input_data)[0][1]
    churn_label = "Likely to Churn" if pred_proba > 0.5 else "Not Likely to Churn"

    # Display Prediction
    st.subheader("Prediction Details")
    st.markdown(f"**Gender:** {customer['gender']}")
    st.metric("Churn Probability", f"{pred_proba*100:.2f}%", churn_label)

    # SHAP Waterfall Plot
    st.subheader("SHAP Explanation (Waterfall)")
    shap_values = explainer(input_data)
    fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

