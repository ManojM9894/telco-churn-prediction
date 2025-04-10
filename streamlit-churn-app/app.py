import os
import gdown
import joblib
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📊 Churn Predictor")
    st.markdown("Upload your customer data to predict churn.")
    st.markdown("**Steps:**\n1. Upload a CSV\n2. See predictions\n3. Explore explanations")
    st.markdown("Need help? Use the sample CSV format from the repo.")

# ----------- Download model and encoders from Google Drive ---------------- #

model_path = "model/customer_churn_model.pkl"
encoder_path = "model/encoders.pkl"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    model_url = "https://drive.google.com/uc?id=1HMXlOnLtbzdbjhDZMrDDvh5Ai9ok4NqH"
    gdown.download(model_url, model_path, quiet=False)

if not os.path.exists(encoder_path):
    encoder_url = "https://drive.google.com/uc?id=1cyvzDSL9mMyNca-ZaZQQVeSq3h0DUXld"
    gdown.download(encoder_url, encoder_path, quiet=False)

# -------------------------------------------------------------------------- #

# Load model & encoders
@st.cache_resource
def load_artifacts():
    model_data = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

st.title("Telco Customer Churn Predictor")

uploaded_file = st.file_uploader("📂 Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Encode categorical features
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Ensure feature order
    input_df = input_df[feature_names]

    # Predict
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    input_df["Churn_Prediction"] = preds
    input_df["Churn_Probability"] = probs.round(2)

    st.write("### 🧠 Prediction Results")
    st.dataframe(input_df)

    st.download_button("📥 Download Predictions", input_df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    # SHAP Explainability
    st.write("### 🔍 SHAP Explainability")
    explainer = shap.Explainer(model, input_df[feature_names])
    shap_values = explainer(input_df[feature_names])

    # SHAP Summary Plot
    st.write("#### 📊 SHAP Summary Plot")
    fig_summary, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, input_df[feature_names], plot_type="bar", show=False)
    st.pyplot(fig_summary)

    # SHAP Force Plot (selectable)
    st.write("#### 🔬 SHAP Force Plot for Selected Prediction")
    selected_row = st.number_input("Select row to explain", min_value=0, max_value=len(input_df)-1, value=0, step=1)
    shap_html = shap.plots.force(shap_values[selected_row], matplotlib=False)
    components.html(shap_html, height=300)

else:
    st.info("👆 Upload a customer CSV file to begin prediction.")

