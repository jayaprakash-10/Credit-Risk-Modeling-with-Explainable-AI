# Credit Risk Modeling with Explainable AI

## 🚀 Project Overview
This repository contains code to build, evaluate, and explain a binary classification model that predicts whether a loan applicant will default. It uses both an interpretable **Logistic Regression** model and a high-performance **XGBoost** model, with **SHAP** and **LIME** for global and local explainability.

## 💾 Dataset
We use the Lending Club “accepted” loan data (2007–Q4 2018) from Kaggle:
- **Link:** https://www.kaggle.com/datasets/wordsforthewise/lending-club  
- Download `accepted_2007_to_2018Q4.csv.gz` and place it in the project root (or update the path in the notebook).

> **Note:** The “rejected” file contains unfunded applications and has no `loan_status` column, so it’s not used for default prediction.

## 🛠️ Tech Stack
- **Python 3.7+**  
- Data wrangling: `pandas`, `numpy`  
- Modeling: `scikit-learn`, `xgboost`  
- Interpretability: `shap`, `lime`  
- UI (optional): `streamlit` or `ipywidgets`  
- Environment management: `venv` or `conda`

## 🔑 Key Steps
1. **Data Loading & Cleaning**  
   - Read the compressed CSV.  
   - Define target: `default = 1` if `loan_status` ∈ {“Charged Off”, “Default”}.  
   - Select and parse features (`int_rate`, `term`, `emp_length`, etc.), drop rows with missing critical fields.

2. **Feature Engineering & Preprocessing**  
   - Numeric features → median imputation + standard scaling  
   - Categorical features → constant-value imputation + one-hot encoding  

3. **Model Training**  
   - Split into train/test (80/20).  
   - Pipeline 1: Logistic Regression  
   - Pipeline 2: XGBoost Classifier

4. **Evaluation**  
   - Accuracy, ROC AUC, Precision/Recall/F1 (classification report)  

5. **Explainability**  
   - **SHAP (Global):** `TreeExplainer` on XGBoost → summary & dependence plots  
   - **LIME (Local):** `LimeTabularExplainer` on encoded data → per-instance explanation  

6. **Optional UI**  
   - **Streamlit** app: interactive sliders + SHAP force plots  
   - **ipywidgets** dashboard: inline Colab widgets + SHAP waterfall plot

## 📁 Repository Structure
