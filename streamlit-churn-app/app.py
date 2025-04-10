import os
import joblib
import gdown
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# ----------- Google Drive File IDs ----------- #
MODEL_FILE_ID = "1lKk6KmEEjwXQZjiRjTzpbFwbUcSGsdoj"
ENCODER_FILE_ID = "1_lMgMqtQ_ppqU2EOzabHl1tkvNkMJ9P_"

# ----------- Download from Google Drive if not present ----------- #
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# ----------- Load Model & Encoders ----------- #
def load_artifacts():
    if not os.path.exists("customer_churn_model.pkl"):
        download_file(MODEL_FILE_ID, "customer_churn_model.pkl")
    if not os.path.exists("encoders.pkl"):
        download_file(ENCODER_FILE_ID, "encoders.pkl")

    model_data = joblib.load("customer_churn_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# ----------- UI Starts Here ----------- #
st.title("Telco Customer Churn Predictor")

with st.form("initial_form"):
    customer_id = st.text_input("Customer ID", placeholder="e.g., C001245")
    gender = st.selectbox("Gender", ["Female", "Male"])
    next_button = st.form_submit_button("️ Next")

if next_button and customer_id and gender:
    with st.form("full_form"):
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charge = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charge = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": device_protect,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charge,
            "TotalCharges": total_charge
        }

        input_df = pd.DataFrame([input_dict])

        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.info(f"**Customer ID: {customer_id}**")
        st.success("Likely to churn" if prediction == 1 else "Not likely to churn")
        st.metric("Churn Probability", f"{probability * 100:.2f} %")

