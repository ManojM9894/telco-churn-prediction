import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import gdown

st.set_page_config(page_title="Telco Churn App", layout="centered")

# --- Load Artifacts (with Google Drive fallback) ---
@st.cache_resource
def load_artifacts():
    try:
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    except:
        gdown.download("https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", 
                       "streamlit-churn-app/customer_churn_model.pkl", quiet=False)
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")

    try:
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    except:
        gdown.download("https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", 
                       "streamlit-churn-app/encoders.pkl", quiet=False)
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")

    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# --- Load Dataset ---
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# --- UI Layout ---
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

all_customers = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", options=all_customers)

if selected_id:
    row = df[df["customerID"] == selected_id]
    st.subheader("📋 Prediction Details")
    st.write("**Gender:**", row["gender"].values[0])

    X = row[feature_names]
    for col in X.columns:
        if col in encoders:
            X[col] = encoders[col].transform(X[col])

    prediction_proba = model.predict_proba(X)[0][1]
    prediction_label = f"{prediction_proba*100:.2f}% 🚨 Likely to Churn" if prediction_proba > 0.5 else f"{(1 - prediction_proba)*100:.2f}% ✅ Not Likely to Churn"

    st.subheader("📊 Churn Probability")
    st.metric(label="", value=f"{prediction_proba*100:.2f}%", delta="Likely to Churn" if prediction_proba > 0.5 else "Not Likely")
    st.success(prediction_label) if prediction_proba > 0.5 else st.info(prediction_label)

    # --- SHAP Bar Chart Explanation ---
    st.subheader("💡 SHAP Explanation (Bar Chart)")
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    if isinstance(shap_values.values, list):
        values = shap_values.values[0]
    else:
        values = shap_values.values[0]

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value": values
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color="skyblue")
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Top Contributing Features")
    st.pyplot(fig)

