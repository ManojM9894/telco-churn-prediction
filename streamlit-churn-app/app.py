import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

# ---------- Download model if not exists ----------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("customer_churn_model.pkl"):
        gdown.download(id="1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", output="customer_churn_model.pkl", quiet=False)
    if not os.path.exists("encoders.pkl"):
        gdown.download(id="1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", output="encoders.pkl", quiet=False)

    model_data = joblib.load("customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("encoders.pkl")
    return model, feature_names, encoders

@st.cache_resource
def get_shap_explainer(_model, masker_df):
    return shap.Explainer(_model.predict_proba, masker_df)

# ---------- Load Artifacts ----------
model, feature_names, encoders = load_artifacts()

# ---------- Load Datasets ----------
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# FIX: coerce invalid TotalCharges values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# ---------- UI ----------
st.title("📞 Telco Customer Churn Predictor")
st.markdown("**Select a Customer ID to see churn prediction and explanation.**")

# Customer dropdown (Top 50)
top_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", top_ids)

# Auto-fill customer info
customer_row = df[df["customerID"] == selected_id]
if customer_row.empty:
    st.error("Customer ID not found.")
    st.stop()

# Display gender and prediction details
gender = customer_row["gender"].values[0]
st.markdown("### 📋 Prediction Details")
st.write("**Gender:**", gender)

# Preprocess input
X = customer_row.drop(columns=["customerID", "Churn"])
for col in encoders:
    if col in X.columns:
        X[col] = encoders[col].transform(X[col])

# Predict
proba = model.predict_proba(X)[0][1]
churn_label = "Likely to Churn" if proba > 0.5 else "Not Likely to Churn"

# Show churn probability
st.markdown("### 📊 Churn Probability")
st.metric(label="", value=f"{proba*100:.2f}%", delta=None)
st.success(f"🚨 {churn_label}" if proba > 0.5 else f"✅ {churn_label}")

# SHAP Explanation
st.markdown("### 💡 SHAP Explanation (Waterfall)")

# Create SHAP explainer using background (entire dataset)
explainer = get_shap_explainer(model, X)

# Generate SHAP values
shap_values = explainer(X)

# Plot SHAP waterfall
fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig)

