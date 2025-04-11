import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Telco Churn App", layout="centered")

# ----------- Load Artifacts ----------- #
@st.cache_resource
def load_artifacts():
    try:
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    except FileNotFoundError:
        import gdown
        gdown.download("https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj",
                       "streamlit-churn-app/customer_churn_model.pkl", quiet=False)
        model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]

    if os.path.exists("streamlit-churn-app/encoders.pkl"):
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    else:
        gdown.download("https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_",
                       "streamlit-churn-app/encoders.pkl", quiet=False)
        encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    return model, feature_names, encoders

# ----------- Load Data ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
df = df[df["TotalCharges"].str.strip() != ""]  # Handle spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

model, feature_names, encoders = load_artifacts()
xgb_model = model.named_estimators_['xgb']  # Extract XGBoost model for Tree SHAP

# ----------- SHAP Explainer ----------- #
explainer = shap.TreeExplainer(xgb_model)

# ----------- Streamlit UI ----------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

selected_id = st.selectbox("Select Customer ID", top_50["customerID"].tolist())

if selected_id:
    customer = df[df["customerID"] == selected_id]
    gender = customer["gender"].values[0]
    st.markdown(f"### 📋 Prediction Details\n**Gender:** {gender}")

    # Preprocess
    X = customer[feature_names].copy()
    for col in X.select_dtypes(include='object').columns:
        if col in encoders:
            X[col] = encoders[col].transform(X[col])

    prediction_proba = model.predict_proba(X)[0][1]
    prediction_label = f"{prediction_proba * 100:.2f}%\n🚨 Likely to Churn" if prediction_proba > 0.5 else f"{prediction_proba * 100:.2f}%\n✅ Not Likely to Churn"

    st.markdown(f"### 📊 Churn Probability\n**{prediction_proba * 100:.2f}%**")
    st.success(prediction_label) if prediction_proba > 0.5 else st.info(prediction_label)

    # ----------- SHAP Chart ----------- #
    st.markdown("### 💡 SHAP Explanation (Bar Chart)")
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[1][0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax.set_xlabel("SHAP Value")
    ax.invert_yaxis()
    st.pyplot(fig)

