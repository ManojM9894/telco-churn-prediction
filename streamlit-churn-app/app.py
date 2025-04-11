import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import gdown
import os
import matplotlib.pyplot as plt

# ----------- Helper to load model & encoders ----------- #
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

# ----------- Helper to load SHAP explainer ----------- #
@st.cache_resource
def get_shap_explainer(_model, masker, feature_names):
    return shap.Explainer(_model.predict_proba, masker=masker, feature_names=feature_names)

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

model, feature_names, encoders = load_artifacts()

# ----------- Prepare SHAP masker data ----------- #
masker_data = df[feature_names].sample(100, random_state=42).copy()
for col in encoders:
    masker_data[col] = encoders[col].transform(masker_data[col])

explainer = get_shap_explainer(model, masker_data, feature_names)

# ----------- UI ----------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("### Select a Customer ID to see churn prediction and explanation.")

all_ids = df["customerID"].tolist()
top_50_ids = top_50["customerID"].tolist()

# Show dropdown
selected_id = st.selectbox("Select Customer ID", options=top_50_ids + ["Other"], index=0)

if selected_id == "Other":
    entered_id = st.text_input("🔍 Enter Customer ID manually:", placeholder="e.g., 4614-NUVZD")
    customer_id = entered_id.strip()
else:
    customer_id = selected_id

# Fetch & predict
if customer_id:
    if customer_id in df["customerID"].values:
        customer_row = df[df["customerID"] == customer_id].copy()
        gender = customer_row["gender"].values[0]
        st.markdown("### 📋 Prediction Details")
        st.write("**Gender:**", gender)

        # Encode input
        input_data = customer_row[feature_names].copy()
        for col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

        prob = model.predict_proba(input_data)[0][1]
        st.subheader("📊 Churn Probability")
        st.write(f"**{prob*100:.2f}%**")

        if prob > 0.5:
            st.markdown("**🚨 Likely to Churn**")
        else:
            st.markdown("**✅ Not Likely to Churn**")

        # SHAP Waterfall
        st.markdown("### 💡 SHAP Explanation (Waterfall)")
        shap_values = explainer(input_data)
        fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.warning("Customer ID not found.")

