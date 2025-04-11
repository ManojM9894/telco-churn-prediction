import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import gdown
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Telco Churn App", layout="centered")

# ---------------- DOWNLOAD FALLBACKS ---------------- #
def download_if_missing(path, gdrive_url):
    if not os.path.exists(path):
        gdown.download(gdrive_url, path, quiet=False)

download_if_missing("streamlit-churn-app/customer_churn_model.pkl", "https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj")
download_if_missing("streamlit-churn-app/encoders.pkl", "https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_")
download_if_missing("streamlit-churn-app/telco_churn.csv", "https://drive.google.com/uc?id=1Z9QNgJHMpQxykS8trNO98DJSv7zBgpOp")

# ---------------- LOAD ARTIFACTS ---------------- #
@st.cache_resource
def load_artifacts():
    data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    model = data["model"]
    feature_names = data["features_names"]
    encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    return model, feature_names, encoders

model, feature_names, encoders = load_artifacts()

@st.cache_resource
def get_shap_explainer(_model, _data):
    return shap.Explainer(_model.predict_proba, _data)

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# ---------------- UI ---------------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

customer_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", customer_ids)

if selected_id:
    customer_data = df[df["customerID"] == selected_id]
    st.subheader("📋 Prediction Details")
    st.markdown(f"**Gender**: {customer_data['gender'].values[0]}")

    # ---------------- PREDICTION ---------------- #
    input_data = customer_data.copy()
    input_data.drop(columns=["customerID", "Churn"], inplace=True)

    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_label = "🚨 Likely to Churn" if prediction_proba > 0.5 else "✅ Likely to Stay"

    st.subheader("📊 Churn Probability")
    st.markdown(f"**{prediction_proba * 100:.2f}%**")

    if prediction_proba > 0.5:
        st.success(prediction_label)
    else:
        st.info(prediction_label)

    # ---------------- SHAP EXPLANATION ---------------- #
    st.subheader("💡 SHAP Explanation (Bar Chart)")
    explainer = get_shap_explainer(model, input_data)
    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax.set_xlabel("SHAP Value")
    ax.invert_yaxis()
    st.pyplot(fig)

