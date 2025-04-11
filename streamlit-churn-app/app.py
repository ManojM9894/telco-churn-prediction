import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import gdown
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# ----------- Auto-download model & encoder if not present ----------- #
@st.cache_resource
def load_artifacts():
    if not os.path.exists("customer_churn_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", "customer_churn_model.pkl", quiet=False)
    if not os.path.exists("encoders.pkl"):
        gdown.download("https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", "encoders.pkl", quiet=False)
    
    model_data = joblib.load("customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("encoders.pkl")
    return model, feature_names, encoders

model, feature_names, encoders = load_artifacts()

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# ----------- SHAP Explainer ----------- #
explainer = shap.Explainer(model.named_estimators_['xgb'])

# ----------- UI ----------- #
st.title("Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

top_50_ids = top_50["customerID"].tolist()
customer_id = st.selectbox("Select Customer ID", options=top_50_ids, index=0)

if customer_id:
    customer_row = df[df["customerID"] == customer_id]
    
    if not customer_row.empty:
        st.subheader("Prediction Details")
        gender = customer_row["gender"].values[0]
        st.markdown(f"**Gender**: {gender}")
        
        # Prepare input for model
        input_data = customer_row[feature_names].copy()
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])
        
        prediction_proba = model.predict_proba(input_data)[:, 1][0]
        prediction_label = "Likely to Churn" if prediction_proba > 0.5 else "Not Likely to Churn"
        
        st.metric("Churn Probability", f"{prediction_proba:.2%}", help="Probability of customer churning")
        st.success(prediction_label if prediction_proba <= 0.5 else prediction_label)

        # ----------- SHAP Explanation ----------- #
        st.subheader("💡 SHAP Explanation (Waterfall)")
        shap_values = explainer(input_data)
        fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.warning("Customer ID not found.")

