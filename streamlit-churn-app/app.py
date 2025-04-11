import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import gdown

st.set_page_config(page_title="Telco Churn App", layout="centered")

# ----------- Load Model and Encoders -----------
@st.cache_resource
def load_artifacts():
    model_path = "customer_churn_model.pkl"
    encoder_path = "encoders.pkl"

    if not os.path.exists(model_path):
        gdown.download(id="1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", output=model_path, quiet=False)
    if not os.path.exists(encoder_path):
        gdown.download(id="1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", output=encoder_path, quiet=False)

    model_data = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# ----------- Load Datasets -----------
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

background_df = df[feature_names].sample(50, random_state=42).copy()
for col in background_df.columns:
    if col in encoders:
        background_df[col] = encoders[col].transform(background_df[col])

background = background_df.values  # Final fallback against SHAP error

@st.cache_resource
def get_shap_explainer(_model, _masker):
    return shap.Explainer(_model.predict, masker=_masker)

explainer = get_shap_explainer(model, background)

# ----------- Streamlit UI -----------
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

default_id = top_50["customerID"].iloc[0]
selected_id = st.selectbox("Select Customer ID", [default_id] + top_50["customerID"].tolist())

customer_row = df[df["customerID"] == selected_id]
if not customer_row.empty:
    st.subheader("📋 Prediction Details")
    st.write(f"**Gender:** {customer_row['gender'].values[0]}")

    input_data = customer_row[feature_names].copy()
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_label = "🚨 Likely to Churn" if prediction_proba > 0.5 else "✅ Not Likely to Churn"

    st.subheader("📊 Churn Probability")
    st.markdown(f"**{prediction_proba * 100:.2f}%**")
    st.success(prediction_label) if prediction_proba > 0.5 else st.info(prediction_label)

    # ----------- SHAP Explanation (Bar Chart) -----------
    st.subheader("💡 SHAP Explanation (Bar Chart)")
    shap_values = explainer(input_data.values)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

