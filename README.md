# Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/github/license/ManojM9894/telco-churn-prediction)
![Last Commit](https://img.shields.io/github/last-commit/ManojM9894/telco-churn-prediction)
![Repo Size](https://img.shields.io/github/repo-size/ManojM9894/telco-churn-prediction)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-churn-prediction-2ukgrvfptmc9terlkegn8i.streamlit.app)

---

## Live App

Try the churn prediction tool in your browser:  
🔗 **[Click here to open the Streamlit App](https://telco-churn-prediction-2ukgrvfptmc9terlkegn8i.streamlit.app)**

This project uses real-world telecom customer data to build a machine learning pipeline that predicts which customers are likely to churn. It is a complete end-to-end data science workflow—from data preprocessing and modeling to deployment-ready exports and interactive dashboards.

---

## Dataset Description

The **Telco Customer Churn** dataset includes demographics, account information, and service usage patterns.

| Column Name       | Description                                 |
|-------------------|---------------------------------------------|
| gender            | Male or Female                              |
| SeniorCitizen     | 1 if senior, 0 if not                       |
| tenure            | Number of months with the company           |
| PhoneService      | Whether the customer has phone service      |
| InternetService   | DSL, Fiber, or None                         |
| Contract          | Month-to-month, One year, or Two year       |
| MonthlyCharges    | Monthly bill amount                         |
| TotalCharges      | Total bill over customer lifetime           |
| Churn             | Yes or No – target variable                 |

The dataset is stored locally at: `data/telco_churn.csv`

---

## Project Structure

- `Techno_Churn_Final_Upload.ipynb` – Full pipeline with preprocessing, SMOTE, model tuning, and evaluation
- `customer_churn_model.pkl` – Saved Voting Classifier model
- `encoders.pkl` – Label encoders for deployment
- `README.md` – You’re reading it 

---

## Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost
- imbalanced-learn (SMOTE, RUS)
- Pickle (model persistence)

---

## Key Features

- Class imbalance handled using **SMOTE** and **RandomUnderSampler**
- Models trained: **Decision Tree**, **Random Forest**, **XGBoost**
- Ensemble built using a **Voting Classifier**
- Hyperparameter tuning via `GridSearchCV` + `StratifiedKFold`
- Evaluation metrics:
  - Accuracy
  - Precision / Recall
  - F1-Score
  - ROC-AUC
- Feature importance visualized and exported
- Dashboard-ready outputs (BigQuery + Looker Studio)
- Prepared for **Streamlit** deployment

---

## Best Model

| Model            | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|------------------|----------|-----------|--------|----------|---------|
| Voting Classifier| ~80%     | *XX%*     | *XX%*  | *XX%*    | *XX%*   |

*Exact values available in the final notebook section.*

---

## Dashboard & Risk Insights (Looker Studio)

This project includes a **Looker Studio dashboard** powered by **BigQuery + model predictions**, offering real-time churn visualization and business-ready reporting.

### Dashboard Pages:

- **Page 1: Churn Overview**
  - Churn by contract type, tenure, and monthly charges
- **Page 2: ML Predictions**
  - Total customers evaluated
  - Churn risk breakdown
  - Tenure vs churn probability
- **Page 3: Churn Drivers & At-Risk Customers**
  - Feature importance (Random Forest)
  - Segment heatmap (InternetService × PaymentMethod)
  - Top 50 risky customers with `customerID`

---

### Live Dashboard

- **View Interactive Dashboard**:  
  [Click here to open the Telco Churn Dashboard](https://lookerstudio.google.com/reporting/3769f0f4-c502-4488-a3a8-8a47f9d3d8a8)

- **Download Dashboard PDF**:  
  [Telco Churn Dashboard Report (PDF)](assets/Telco_Churn_Dashboard_Report.pdf)

---

## Outputs for Looker Studio

These files are exported from the ML pipeline and used in Looker Studio:

- [`feature_importance.csv`](data_outputs/feature_importance.csv) – Feature importance (Random Forest)
- [`top_50_risky_customers.csv`](data_outputs/top_50_risky_customers.csv) – Top 50 at-risk customers with `customerID`

---

## Business Impact

- Identify customers most likely to churn
- Enable proactive retention strategy
- Segment customers by risk and behavior
- Improve customer lifetime value (CLV) and reduce revenue loss

---

## Future Enhancements

- Build Streamlit UI for real-time prediction
- Integrate SHAP for explainable ML
- API deployment via Flask/FastAPI

---

## Author

**Manoj Kumar Mandava**  
[LinkedIn](https://www.linkedin.com/in/manojmandava9894)  
GitHub: [@Manoj9894](https://github.com/Manoj9894)

---

## License

MIT License
