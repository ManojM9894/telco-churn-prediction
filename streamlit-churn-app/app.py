import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import requests

# --------------- Page Config --------------- #
st.set_page_config(page_title="Telco Churn App", layout="centered")

# --------------- Load Artifacts --------------- #
@st.cache_resource
def load_artifacts():
    try:
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    except FileNotFoundError:
        url = "https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj"
        r = requests.get(url, allow_redirects=True)
        os.makedirs("streamlit-churn-app", exist_ok=True)
        open("streamlit-churn-app/customer_churn_model.pkl", 'wb').write(r.content)
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")

    try:
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    except FileNotFoundError:
        url = "https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_"
        r = requests.get(url, allow_redirects=True)
        open("streamlit-churn-app/encoders.pkl", 'wb').write(r.content)
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")

    model = model_data["model"]
    feature_names = model_data["features_names"]
    return model, feature_names, encoders

@st.cache_resource
def get_shap_explainer(_model, _masker):
    masker = shap.maskers.Independent(_masker)
    return shap.Explainer(_model.predict_proba, masker)

# --------------- Load Data --------------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model, df[feature_names])

# --------------- UI --------------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

selected_id = st.selectbox("Select Customer ID", top_50["customerID"].unique())

if selected_id:
    customer_data = df[df["customerID"] == selected_id]
    gender = customer_data["gender"].values[0]
    st.subheader("📋 Prediction Details")
    st.markdown(f"**Gender:** {gender}")

    input_data = customer_data[feature_names].copy()
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_label = "Likely to Churn" if prediction_proba > 0.5 else "Not Likely to Churn"

    st.subheader("📊 Churn Probability")
    st.markdown(f"<h2 style='color:#3399ff'>{prediction_proba * 100:.2f}%</h2>", unsafe_allow_html=True)
    if prediction_proba > 0.5:
        st.success(prediction_label)
    else:
        st.info(prediction_label)

    # ---------------- SHAP ---------------- #
    st.subheader("💡 SHAP Explanation (Bar Chart)")
    shap_values = explainer(input_data)

    shap_values_1d = shap_values.values[0].flatten()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_1d[:len(feature_names)]
    }).sort_values(by="SHAP Value", key=np.abs, ascending=True)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Feature Contributions")
    st.pyplot(fig)

