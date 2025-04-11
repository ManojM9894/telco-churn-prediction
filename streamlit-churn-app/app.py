import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- Config ---------------- #
st.set_page_config(page_title="Telco Churn App", layout="centered")

# ---------------- Caching ---------------- #
@st.cache_resource
def load_artifacts():
    try:
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    except:
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj",
            "streamlit-churn-app/customer_churn_model.pkl",
            quiet=False,
        )
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    return model, feature_names, encoders

# ---------------- Load Data ---------------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# ---------------- UI ---------------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown: Top 50 customers
top_50_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", top_50_ids)

# ---------------- Model & Predict ---------------- #
model, feature_names, encoders = load_artifacts()

if selected_id:
    customer = df[df["customerID"] == selected_id]
    gender = customer["gender"].values[0]
    st.subheader("📋 Prediction Details")
    st.write(f"**Gender:** {gender}")

    # Preprocess
    X = customer[feature_names].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        le = encoders[col]
        X[col] = le.transform(X[col])

    # Predict
    prediction_proba = model.predict_proba(X)[0][1]
    prediction_label = f"{prediction_proba * 100:.2f}% 🚨 Likely to Churn" if prediction_proba > 0.5 else f"{prediction_proba * 100:.2f}% ✅ Not Likely to Churn"

    st.subheader("📊 Churn Probability")
    st.metric(label="", value=f"{prediction_proba * 100:.2f}%")
    st.success(prediction_label) if prediction_proba > 0.5 else st.info(prediction_label)

    # ---------------- SHAP TreeExplainer ---------------- #
    st.subheader("💡 SHAP Explanation (Bar Chart)")
    try:
        explainer = shap.TreeExplainer(model.estimators_[1][1])  # XGBoost inside VotingClassifier
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "SHAP Value": shap_values[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
        ax.set_xlabel("Impact on Prediction")
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.warning("Could not generate SHAP chart.")
        st.exception(e)
