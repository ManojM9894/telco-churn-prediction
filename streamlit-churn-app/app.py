kimport streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import gdown
import matplotlib.pyplot as plt

# ----------- Download Fallback Files ----------- #
@st.cache_resource
def load_artifacts():
    try:
        model_data = joblib.load("customer_churn_model.pkl")
        encoders = joblib.load("encoders.pkl")
    except:
        gdown.download("https://drive.google.com/uc?id=1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj", "customer_churn_model.pkl", quiet=False)
        gdown.download("https://drive.google.com/uc?id=1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_", "encoders.pkl", quiet=False)
        model_data = joblib.load("customer_churn_model.pkl")
        encoders = joblib.load("encoders.pkl")
    return model_data["model"], model_data["features_names"], encoders

@st.cache_resource
def get_shap_explainer(_model, feature_names):
    return shap.Explainer(_model.predict_proba, shap.maskers.Independent(data=np.zeros((1, len(feature_names)))), feature_names=feature_names)

# ----------- Load Files ----------- #
model, feature_names, encoders = load_artifacts()
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")
df = df.dropna(subset=["TotalCharges"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
explainer = get_shap_explainer(model, feature_names)

# ----------- UI ----------- #
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Select a Customer ID to see churn prediction and explanation.")

# Dropdown from Top 50 or manual ID
use_top = st.checkbox("Use Top 50 Risky Customers")
if use_top:
    customer_id = st.selectbox("Select Customer ID", top_50["customerID"].values)
else:
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g., 4614-NUVZD")

if customer_id:
    customer_row = df[df["customerID"] == customer_id]
    if not customer_row.empty:
        st.subheader("📋 Prediction Details")
        st.markdown(f"**Gender:** {customer_row.iloc[0]['gender']}")
        X = customer_row[feature_names]
        for col in encoders:
            X[col] = encoders[col].transform(X[col])
        input_data = X.values

        # Predict
        prob = model.predict_proba(input_data)[0][1]
        st.subheader("📊 Churn Probability")
        st.write(f"**{prob * 100:.2f}%**")
        if prob > 0.5:
            st.warning("🚨 Likely to Churn")
        else:
            st.success("✅ Likely to Stay")

        # SHAP Explanation
        st.subheader("💡 SHAP Explanation (Waterfall)")
        shap_values = explainer(input_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.error("❌ Customer ID not found. Please try again.")

