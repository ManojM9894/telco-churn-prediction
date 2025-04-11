kimport streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

# Set Streamlit config
st.set_page_config(page_title="Telco Churn App", layout="centered")

# ----------- Load Artifacts ----------- #
@st.cache_resource
def load_artifacts():
    model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    return model, feature_names, encoders

model, feature_names, encoders = load_artifacts()

# ----------- Load Dataset ----------- #
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

# Clean TotalCharges
if "TotalCharges" in df.columns:
    df = df[df["TotalCharges"].apply(lambda x: str(x).strip()) != ""]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

# ----------- UI ----------- #
st.title("\U0001F4DE Telco Customer Churn Predictor")
st.markdown("**Select a Customer ID to see churn prediction and explanation.**")

# Dropdown of top 50 risky customers
top_50_ids = top_50["customerID"].tolist()
selected_id = st.selectbox("Select Customer ID", top_50_ids)

if selected_id:
    customer_row = df[df["customerID"] == selected_id].copy()
    gender = customer_row["gender"].values[0]
    st.markdown(f"### \U0001F4CB Prediction Details")
    st.markdown(f"**Gender**: {gender}")

    # Encode customer features for prediction
    input_data = customer_row[feature_names].copy()
    for col in input_data.select_dtypes(include=["object"]).columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # Predict
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction_label = f"{prediction_proba * 100:.2f}% \U0001F6A8 Likely to Churn" if prediction_proba > 0.5 else f"{prediction_proba * 100:.2f}% \u2705 Not Likely to Churn"

    st.markdown("### \U0001F4CA Churn Probability")
    st.metric(label="Churn Probability", value=f"{prediction_proba * 100:.2f}%")

    if prediction_proba > 0.5:
        st.success(prediction_label)
    else:
        st.info(prediction_label)

    # SHAP Explanation
    st.markdown("### \U0001F4A1 SHAP Explanation (Bar Chart)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[1][0]  # class 1 SHAP values
        }).sort_values(by="SHAP Value", key=abs, ascending=True)

        st.bar_chart(shap_df.set_index("Feature"))
    except Exception as e:
        st.error(f"SHAP explanation could not be generated: {e}")

