# 🧬 Multiple Disease Prediction using Machine Learning
## 📌 Overview

This project is a machine learning application that predicts the likelihood of three diseases:

🩺 **Liver Disease** (Indian Liver Patient Dataset)

🩺 **Chronic Kidney Disease** (CKD Dataset)

🧠 **Parkinson’s Disease** (UCI Parkinson’s Dataset)

It uses scikit-learn pipelines for robust preprocessing and model training, and a Streamlit web application for user-friendly predictions.

## 🎯 Objectives

 1. Perform EDA (Exploratory Data Analysis) on each dataset

 2. Handle missing values, duplicates, and categorical encoding

 3. Build robust preprocessing pipelines (numeric + categorical handling)

 4. Train machine learning models for each disease

 5. Evaluate performance with metrics like Accuracy, F1, ROC-AUC, Precision-Recall

 6. Save models with joblib for deployment

 7. Develop an interactive Streamlit app for real-time disease prediction

## Tech Stack

 1. Python 3.9+

 2. Pandas, NumPy – data processing

 3. Matplotlib, Seaborn – visualization

 4. scikit-learn – preprocessing, ML models, pipelines

 5. imbalanced-learn – SMOTE & handling imbalance

 6. XGBoost / RandomForest / Logistic Regression – classifiers

 7. SHAP – model interpretability

 8. Joblib – model persistence

 9. Streamlit – interactive UI

## 📊 Dataset Details
 1. **Liver Disease Dataset**

 Features: Age, Gender, Bilirubin, Enzyme levels, Proteins, etc.

 Target: Dataset → mapped as 1 = Disease, 0 = No Disease

 Size: ~583 records

 2. **Kidney Disease Dataset**

 Features: Age, Blood pressure, Hemoglobin, Sugar, RBC, PCV, etc.

 Target: class → mapped as ckd = 1, notckd = 0

 Size: ~400 records

 3. **Parkinson’s Dataset**

 Features: Voice measurements (MDVP, jitter, shimmer, etc.)

 Target: status → 1 = Parkinson’s, 0 = Healthy

 Size: ~195 records

## 🔍 Workflow

 1. **Data Preprocessing**

      Removed duplicates

      Dropped unwanted columns (id, name, etc.)

      Missing values:

        Numeric → imputed with median

        Categorical → imputed with "None"

      Encoded categorical features with OneHotEncoder

      Scaled numeric features with StandardScaler

2. **Model Training**

      RandomForest, Logistic Regression, and XGBoost tested

      Cross-validation using StratifiedKFold

      Hyperparameter tuning with RandomizedSearchCV

      Best model saved using joblib

 3. **Evaluation Metrics**

       Accuracy

       Precision, Recall, F1-score

       ROC-AUC & PR-AUC curves

       Confusion Matrix

 4. **Model Interpretability**

       Feature importance (RandomForest/XGBoost)

       SHAP values for local & global explanations

 5. **Deployment**

       Built with Streamlit

       User inputs patient details → Model predicts disease probability

## ✅ Results

**Liver Disease Model**: ROC-AUC ≈ 0.82

**Kidney Disease Model**: ROC-AUC ≈ 0.96

**Parkinson’s Model**: ROC-AUC ≈ 0.89
(values will depend on your tuning and preprocessing)

## 🌟 Future Improvements

Add more datasets (e.g., Heart Disease, Diabetes)

Use Deep Learning models for improved performance

Deploy on cloud (Heroku, Streamlit Cloud, AWS)

Add Explainable AI dashboards with SHAP/ELI5

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis. Always consult healthcare professionals for medical advice.
