# Loan_Approval_Prediction
AI-based Loan Approval System using CIBIL score, salary, and loan history with ensemble ML models (XGBoost &amp; Random Forest).

# 🤖 AI-Based Loan Approval Prediction Model

This project presents a machine learning solution for predicting **loan approvals** using applicant financial data and credit history.

It leverages ensemble learning techniques and is trained on a real-world-like dataset with features such as **CIBIL Score**, **Total Income**, **Existing Loans**, and **Repayment History**.

---

## 🎯 Objective

To build a robust and accurate machine learning model that can classify whether a loan should be approved based on financial indicators.

---

## 🧠 Machine Learning Models Used

✅ **Individual Models:**
- Logistic Regression (with regularization)
- Random Forest (with GridSearchCV tuning)
- XGBoost Classifier (with hyperparameter tuning)

✅ **Final Model:**
- **Soft Voting Ensemble** using the best-performing models (Random Forest + XGBoost)

---

## 📊 Final Evaluation Metrics

| Metric         | Value   |
|----------------|---------|
| **Accuracy**   | 77.5%   |
| **F1-score**   | 0.78    |
| **ROC-AUC**    | 0.83    |
| **Recall (Class 1)** | 0.91 ✅ (minimizes false negatives)

---

## 📁 Project Files

| File                          | Description                                        |
|-------------------------------|----------------------------------------------------|
| `loan_prediction.ipynb`       | Jupyter Notebook (EDA ➝ Feature Engineering ➝ Training ➝ Evaluation) |
| `loan_approval_model.pkl`     | Trained voting classifier model (XGBoost + RF)    |
| `README.md`                   | Project summary and usage guide                   |

---

## 🔍 Features Used for Prediction

- CIBIL Score
- Total Income (`ApplicantIncome + CoapplicantIncome`)
- Loan Amount
- Loan Term
- Credit History
- Education
- Marital Status
- Employment Type (Salaried/Self-Employed)
- Dependents

---

## 🧪 Libraries & Tools

- Python (3.9+)
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib / Seaborn (for visuals)
- Jupyter Notebook

---

## 🚀 How to Use

### 1. Load the Model
```python
import joblib
model = joblib.load('loan_approval_model.pkl')



#2 Make Prediction

input_data = [[total_income, loan_amount, credit_history, ..., feature_n]]
prediction = model.predict(input_data)
print("Approved ✅" if prediction[0] == 1 else "Rejected ❌")
