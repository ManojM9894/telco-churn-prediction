import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------- Page Config -----------------
st.set_page_config(page_title="Telco Churn App", layout="centered")

# ----------------- Load Artifacts -----------------
@st.cache_resource
def load_artifacts():
    model_data = joblib.load("streamlit-churn-app/customer_churn_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features_names"]
    encoders = joblib.load("streamlit-churn-app/encoders.pkl")
    return model, feature_names, encoders

model, feature_names, encoders = load_artifacts()

# ----------------- Load Dataset -----------------
df = pd.read_csv("streamlit-churn-app/telco_churn.csv")
top_50 = pd.read_csv("streamlit-churn-app/top_50_risky_customers.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# ----------------- UI Header -----------------
st.title("📞 Telco Customer Churn Predictor")
st.write("Select a Customer ID to see churn prediction and explanation.")

# ----------------- Dropdown -----------------
customer_id = st.selectbox(
    "Select Customer ID",
    options=top_50["customerID"].tolist(),
    index=0
)

# ----------------- Prediction Logic -----------------
if customer_id:
    customer_row = df[df["customerID"] == customer_id]

    if not customer_row.empty:
        gender = customer_row["gender"].values[0]
        st.markdown(f"### 📋 Prediction Details")
        st.markdown(f"**Gender:** {gender}")

        X = customer_row[feature_names]
        for col in X.select_dtypes(include="object").columns:
            le = encoders.get(col)
            if le:
                X[col] = le.transform(X[col])

        prediction_proba = model.predict_proba(X)[0][1]
        prediction_label = f"{prediction_proba * 100:.2f}% {'🚨 Likely to Churn' if prediction_proba > 0.5 else '✅ Likely to Stay'}"

        st.markdown(f"### 📊 Churn Probability")
        st.metric("Churn Probability", f"{prediction_proba * 100:.2f}%")
        if prediction_proba > 0.5:
            st.success(prediction_label)
        else:
            st.info(prediction_label)

        # ----------------- SHAP Explanation -----------------
        st.markdown("### 💡 SHAP Explanation (Bar Chart)")
        try:
            base_model = dict(model.named_estimators_)["xgb"]
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)

            shap_df = pd.DataFrame({
                "Feature": X.columns,
                "SHAP Value": shap_values[0]
            }).sort_values(by="SHAP Value", key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
            ax.set_xlabel("Impact on Prediction")
            ax.set_title("Top Feature Contributions")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP explanation could not be generated:\n\n{e}")
