# Sprint Project 2 – Home Credit Default Risk

## 📌 Project Overview

This project focuses on predicting **Credit Default Risk**, i.e., whether a customer applying for a home loan will be able to repay their debt or not.  
The work is based on the **Home Credit Default Risk** Kaggle competition and uses real-world financial data provided by **Home Credit Group**, a leading financial institution specializing in home credit.

The project applies **data analysis techniques** and **machine learning concepts** to build an end-to-end pipeline for preprocessing, modeling, hyperparameter tuning, and prediction.

---

## 🎯 Business Goals

- Understand how **Credit Default Risk** can be formulated as a **Machine Learning classification problem**.
- Build a model to predict whether a client will **repay or default** on a loan.
- Expand domain knowledge in **finance and credit risk assessment**.

---

## 🧠 What You Will Learn

- Feature engineering and feature selection for tabular data.
- Handling large datasets with thousands of rows and many features.
- Building and evaluating machine learning models for binary classification.
- Interpreting model results and evaluation metrics.
- Using **AUC-ROC** as a key performance metric.
- Hyperparameter tuning using **Cross Validation**.
- Creating end-to-end **ML pipelines**.
- Achieving competitive performance using **model ensembles**.

---

## 📊 Dataset Description

- Data source: **Home Credit Group**
- Each row represents **one loan application**
- Features include:
  - Client demographic information
  - Financial and credit history data
  - Loan-related attributes
- Target variable:
  - Binary label indicating whether the client **defaulted (1)** or **repaid (0)** the loan

⚠️ **Note:** The training dataset must be removed from the `dataset/` folder before submission.

---

## 🛠️ Technical Objectives

- Perform **Exploratory Data Analysis (EDA)**
- Apply **data preprocessing techniques**, including:
  - Feature selection
  - Outlier detection and handling
  - Missing value imputation
  - Normalization / standardization
  - Categorical feature encoding
- Train and compare machine learning models:
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - LightGBM (optional)
- Perform **hyperparameter search** using Cross Validation
- Evaluate models using **AUC-ROC**
- Analyze and interpret model results

---

## 🧩 Project Workflow

1. Obtain and load the data  
2. Perform Exploratory Data Analysis (EDA)  
3. Data preprocessing:
   - Feature selection
   - Outlier handling
   - Imputation, normalization, and encoding
4. Train a **baseline Logistic Regression model**
5. Train a **Random Forest Classifier**
6. Perform **Randomized Hyperparameter Search with Cross Validation**
7. Analyze and compare model performance
8. *(Optional)* Train LightGBM models and use pipelines
9. *(Optional)* Build custom models and engineered features

---

## 🧪 Models Used

- **Logistic Regression** (baseline model)
- **Random Forest Classifier**
- **LightGBM** *(optional / advanced)*

---

## 📐 Evaluation Metric

- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**  
  Used to evaluate the performance of binary classification models, especially useful for imbalanced datasets.

---

## 💻 Working Environment

Recommended setup:

- Python **3.9**
- Virtual environment
- VS Code (or similar IDE)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
