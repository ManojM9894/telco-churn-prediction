import os
import joblib
import gdown
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")

# ----------- Google Drive File IDs ----------- #
MODEL_FILE_ID = "1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj"
ENCODER_FILE_ID = "1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_"

# ----------- Download from Google Drive if not present ----------- #
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# ----------- Load Model & Encoders ----------- #
@st.cache_resource
def load_artifacts():
    st.info("Downloading model files from Google Drive...")
    if not os.path.exists("customer_churn_model.pkl"):
        download_file(MODEL_FILE_ID, "customer_churn_model.pkl")
    if not os.path.exists("encoders.pkl"):
        download_file(ENCODER_FILE_ID, "encoders.pkl")

    st.info("Loading model...")
    model_data = joblib.load("customer_churn_model.pkl")
    st.info("Model loaded.")

    st.info("Loading encoders...")
    encoders = joblib.load("encoders.pkl")
    st.info("Encoders loaded.")

    return model_data["model"], model_data["features_names"], encoders

# ----------- SHAP Explainer ----------- #
@st.cache_resource
def get_shap_explainer(model):
    return shap.Explainer(model)

model, feature_names, encoders = load_artifacts()
explainer = get_shap_explainer(model)

# ----------- Load Dataset ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", pd.NA), errors='coerce')
df = df.dropna(subset=['TotalCharges'])

st.title("Telco Customer Churn Predictor")
st.markdown("Enter a Customer ID to view their churn likelihood:")

customer_id_input = st.text_input("Customer ID", placeholder="e.g., 1452-KIOVK")

if customer_id_input:
    customer_row = df[df['customerID'] == customer_id_input]

    if customer_row.empty:
        st.error("Customer ID not found in the dataset.")
    else:
        input_dict = customer_row.iloc[0][feature_names].to_dict()

        # Encode categorical features
        for col, encoder in encoders.items():
            input_dict[col] = encoder.transform([input_dict[col]])[0]

        input_df = pd.DataFrame([input_dict])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        gender = customer_row.iloc[0]['gender']
        st.info(f"**Customer ID: {customer_id_input}**")
        st.info(f"**Gender: {gender}**")
        st.success("Likely to churn" if prediction == 1 else "Not likely to churn")
        st.metric("Churn Probability", f"{probability * 100:.2f} %")

        # ----------- SHAP Waterfall Plot ----------- #
        st.subheader("Why this prediction?")
        shap_values = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig)

