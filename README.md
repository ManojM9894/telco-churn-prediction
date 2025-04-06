# Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/github/license/ManojM9894/telco-churn-prediction)
![Last Commit](https://img.shields.io/github/last-commit/ManojM9894/telco-churn-prediction)
![Repo Size](https://img.shields.io/github/repo-size/ManojM9894/telco-churn-prediction)

This project uses real-world telecom customer data to build a predictive model that identifies which customers are likely to churn. It's a complete end-to-end data science project from preprocessing to deployment-ready modeling.

---

## Dataset Description

This project uses the **Telco Customer Churn dataset**, which includes customer demographics, account information, and service usage.

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

The dataset is stored locally at: data/telco_churn.csv

---

## Project Structure

- `Techno_Churn_Final_Upload.ipynb` — Full notebook with preprocessing, SMOTE, hyperparameter tuning, model evaluation, and predictions
- `customer_churn_model.pkl` — Saved best model (Voting Classifier)
- `encoders.pkl` — Saved label encoders for future predictions
- `README.md` — You’re reading it :)

---

## Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE, RandomUnderSampler)
- Pickle (for model saving)

---

## Key Features

- Handles **class imbalance** using both **SMOTE** and **RandomUnderSampler**
- Trains multiple models: **Decision Tree, Random Forest, XGBoost**
- Performs **hyperparameter tuning** using `GridSearchCV` + `StratifiedKFold`
- Adds a **Voting Classifier ensemble** to combine strengths of RF and XGB
- Evaluates models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Includes **feature importance plots**
- Saves models + encoders for deployment
- Ready for **Streamlit web app** integration

---

## Best Model

| Model            | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|------------------|----------|-----------|--------|----------|---------|
| Voting Classifier| ~80%     | *XX%*     | *XX%*  | *XX%*    | *XX%*   |

_(Exact scores are shown in the notebook’s final comparison section)_

---

## Business Impact

A churn prediction model like this allows telecom companies to:
- Proactively retain at-risk customers
- Customize offers for likely churners
- Reduce revenue loss

---

## Future Enhancements

- Add Streamlit interface for real-time predictions
- Integrate SHAP for explainable AI
- Deploy via Flask or FastAPI

---

## Author

**Manoj Kumar Mandava**  
[LinkedIn](https://www.linkedin.com/in/manojmandava9894)  
GitHub: [@Manoj9894](https://github.com/Manoj9894)

---

## License

MIT License
