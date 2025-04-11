import os
import joblib
import zipfile
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ----------- Load Model & Encoders ----------- #
@st.cache_resource
def load_artifacts():
    model = joblib.load("customer_churn_model.pkl")["model"]
    feature_names = joblib.load("customer_churn_model.pkl")["features_names"]
    encoders = joblib.load("encoders.pkl")
    return model, feature_names, encoders

@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model.predict_proba)

model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model)

# ----------- Load Datasets ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# ----------- UI ----------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a **Customer ID** from the top 50 most at-risk list:")

customer_id = st.selectbox(
    "Select Customer ID", 
    options=top_50["customerID"].tolist(), 
    index=0,
    help="Top 50 customers most likely to churn"
)

if customer_id:
    customer = df[df["customerID"] == customer_id]

    if not customer.empty:
        X = customer[feature_names].copy()
        
        # Apply label encoders
        for col in encoders:
            if col in X.columns:
                X[col] = encoders[col].transform(X[col])

        # Predict
        churn_prob = model.predict_proba(X)[0][1]
        churn_label = "Likely to Churn" if churn_prob > 0.5 else "Unlikely to Churn"

        st.metric("Churn Probability", f"{churn_prob:.2%}", help="Based on model prediction")
        st.subheader(f"Prediction: {churn_label}")

        # SHAP Waterfall Plot
        st.subheader("SHAP Explanation")
        shap_values = explainer(X)
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot()

    else:
        st.warning("Customer ID not found in dataset.")

