import os
import zipfile
import joblib
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# ----------- File Setup & Debug ----------- #

zip_path = "model_bundle.zip"
model_dir = "model"
model_path = os.path.join(model_dir, "customer_churn_model.pkl")
encoder_path = os.path.join(model_dir, "encoders.pkl")

# Show current directory files (debug aid)
st.write("🗂️ Files in current directory:", os.listdir())

# Unzip model if not already extracted
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    os.makedirs(model_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(model_dir)

# ----------- Load model and encoders ----------- #

@st.cache_resource
def load_artifacts():
    model_data = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# ----------- Streamlit UI ----------- #

st.title("📊 Telco Customer Churn Predictor")

uploaded_file = st.file_uploader("📂 Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    input_df = input_df[feature_names]

    # Predictions
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    input_df["Churn_Prediction"] = preds
    input_df["Churn_Probability"] = probs.round(2)

    # KPI metric
    st.metric("✅ Total Predictions", len(input_df))

    # Churn distribution
    st.write("### 📈 Churn Distribution")
    fig_churn, ax = plt.subplots()
    sns.countplot(data=input_df, x="Churn_Prediction", palette="viridis")
    ax.set_xticklabels(['Not Churned', 'Churned'])
    st.pyplot(fig_churn)

    # Prediction results table
    st.write("### 🧠 Prediction Results")
    st.dataframe(input_df)

    # Download button
    st.download_button("📥 Download Predictions", input_df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    # SHAP Explainability
    st.write("### 🔍 SHAP Explainability")
    explainer = shap.Explainer(model, input_df[feature_names])
    shap_values = explainer(input_df[feature_names])

    st.write("#### 📊 SHAP Summary Plot")
    fig_summary, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, input_df[feature_names], plot_type="bar", show=False)
    st.pyplot(fig_summary)

    st.write("#### 🔬 SHAP Force Plot for Selected Prediction")
    selected_row = st.number_input("Select row to explain", min_value=0, max_value=len(input_df)-1, value=0)
    shap_html = shap.plots.force(shap_values[selected_row], matplotlib=False)
    components.html(shap_html, height=300)

else:
    st.info("👆 Upload a customer CSV file to begin prediction.")

