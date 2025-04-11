import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import gdown
import matplotlib.pyplot as plt

st.set_page_config(page_title="📞 Telco Churn Predictor", layout="wide")

# ----------- Download model files ----------- #
@st.cache_resource
def load_artifacts():
    if not os.path.exists("customer_churn_model.pkl"):
        gdown.download(id="1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", output="customer_churn_model.pkl", quiet=False)
    if not os.path.exists("encoders.pkl"):
        gdown.download(id="1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", output="encoders.pkl", quiet=False)
    model_data = joblib.load("customer_churn_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model_data["model"], model_data["features_names"], encoders

# ----------- SHAP Explainer ----------- #
@st.cache_resource
def get_shap_explainer(_model):
    base_model = _model.named_estimators_['xgb']
    return shap.Explainer(base_model.predict, masker=shap.maskers.Independent(np.zeros((1, len(feature_names)))))

# ----------- Load Model & Data ----------- #
model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model)

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# ----------- UI ----------- #
st.title("📞 Telco Customer Churn Predictor")

# --- Customer ID dropdown ---
customer_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select a Top 50 Risky Customer ID", customer_ids, index=0)

if selected_id:
    customer = df[df["customerID"] == selected_id]
    if not customer.empty:
        gender = customer["gender"].values[0]
        st.success(f"**Customer ID**: {selected_id} | **Gender**: {gender}")

        # Preprocess input
        input_data = customer[feature_names].copy()
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        # Prediction
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction_label = "Will Churn" if prediction_proba > 0.5 else "Will Stay"

        st.markdown("---")
        st.subheader(f"Prediction: {prediction_label}")
        st.metric(label="Churn Probability", value=f"{prediction_proba*100:.2f} %")

        # SHAP Plot
        st.markdown("#### Why this prediction?")
        shap_values = explainer(input_data)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    else:
        st.warning("Customer ID not found in the dataset.")

