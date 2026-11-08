# Heart Disease Classification — ML Assignment 2

## Problem Statement
Build a binary classification system to predict whether a patient has heart disease based on 14 clinical features. The system implements and compares 5 ML classifiers deployed as an interactive Streamlit web application.

---

## Dataset Description
- **Dataset:** Heart Disease Dataset (synthetic, based on UCI Heart Disease characteristics)
- **Instances:** 1000 patients
- **Features:** 14 clinical features (see table below)
- **Target:** Binary — `0` = No Heart Disease, `1` = Heart Disease
- **Train/Test Split:** 80% / 20% (stratified)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (0=Female, 1=Male) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg | Resting ECG result (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia type (0–3) |
| bmi | Body Mass Index |

---

## GitHub Repository Link
> **https://github.com/devanshu1010/BITS_ML_Assigment2_Heart_Disease**

Repository structure:
```
BITS_ML_Assigment2_Heart_Disease/
├── .gitignore              # Git ignore rules
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # Complete assignment documentation
├── train_models.py         # Model training script
├── test_data.csv           # Test data (200 rows)
├── heart_disease_data.csv  # Full dataset (1000 rows)
└── model/
    ├── scaler.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── model_comparison.png
    ├── confusion_matrices.png
    └── results.csv
```

---

## Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8200 | 0.9084 | 0.8560 | 0.8560 | 0.8560 | 0.6160 |
| Decision Tree | 0.8000 | 0.8921 | 0.8899 | 0.7760 | 0.8291 | 0.5989 |
| KNN | 0.8100 | 0.8643 | 0.8480 | 0.8480 | 0.8480 | 0.5947 |
| Naive Bayes | 0.8300 | 0.9202 | 0.8273 | 0.9200 | 0.8712 | 0.6309 |
| Support Vector Machine | 0.8000 | 0.9004 | 0.8244 | 0.8640 | 0.8438 | 0.5676 |

### Model Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Strong baseline with good AUC (0.9084). Assumes linear decision boundary. Provides interpretable coefficients. Good balance between precision and recall. |
| Decision Tree | Highest precision (0.8899) but lower recall. Prone to overfitting — explicitly captures non-linear patterns via decision rules. Max depth limited to 5 to prevent overfitting. |
| KNN | Instance-based learner. Performance depends on k (k=7 chosen). Slower inference on large data. Sensitive to irrelevant features. Reasonable performance overall. |
| Naive Bayes | **Best AUC (0.9202) and F1 (0.8712)**. Conditional independence assumption holds reasonably well. Fast training and inference. Highest recall (0.920) — minimizes false negatives (critical in medical diagnosis). |
| Support Vector Machine | Robust with RBF kernel. Good AUC (0.9004). Sensitive to feature scaling (addressed by StandardScaler). Competitive performance but higher training time. |
| **Overall, Winner** | **Naive Bayes** — Best F1 Score, highest AUC, and highest Recall. In heart disease prediction, minimizing false negatives is critical (missing a positive case is dangerous), making Naive Bayes the optimal choice for this dataset. |

---

## Streamlit App Features
- **Dataset upload option (CSV)** — upload test data for evaluation
- **Model selection dropdown** — choose from 5 classifiers
- **Display of evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix & classification report** per model
- **ROC curves** for all models
- **Model comparison table** with best highlighted
- **Dataset preview** with correlation heatmap

---

## How to Run Locally
```bash
git clone https://github.com/devanshu1010/BITS_ML_Assigment2_Heart_Disease.git
cd BITS_ML_Assigment2_Heart_Disease
pip install -r requirements.txt
python train_models.py           # Train and save models
streamlit run app.py             # Launch web app (or: python -m streamlit run app.py)
```

---

## Student Details
- **Name:** Devanshu Chhipani
- **ID:** 2026AJ05742
- **Course:** Machine Learning (AIMLZG565)
- **Program:** M.Tech AIML — BITS Pilani WILP
