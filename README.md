# 🛡️ Premium Insurance with Lifestyle Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict personalized annual health and life insurance premiums based on individual demographics, financial status, physical habits, stress levels, and pre-existing medical conditions.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [How Risk Scoring Works](#-how-risk-scoring-works)
  - [1. Lifestyle Risk Score](#1-lifestyle-risk-score)
  - [2. Total Medical Risk Score](#2-total-medical-risk-score)
- [Project Architecture](#-project-architecture)
- [Model & Preprocessing Pipeline](#-model--preprocessing-pipeline)
- [Model Evaluation & Benchmark Metrics](#-model-evaluation--benchmark-metrics)
  - [Model Benchmark Comparison](#model-benchmark-comparison)
  - [Best Model In-Depth Performance](#best-model-in-depth-performance)
  - [Subgroup Analysis by Policy Tier](#subgroup-analysis-by-policy-tier)
- [Installation & Setup](#-installation--setup)
- [Running the Streamlit Application](#-running-the-streamlit-application)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 📖 Overview

Traditional insurance premium estimation often relies solely on basic demographic parameters such as age and gender. This project enhances premium estimation by incorporating modern lifestyle metrics:
- **Physical Activity & Stress Levels**
- **Body Mass Index (BMI) & Smoking Habits**
- **Specific Pre-existing Medical Conditions**
- **Income & Family Dependants**
- **Policy Tier Selection (Bronze, Silver, Gold)**

A tuned **XGBoost Regressor** trained on comprehensive customer data provides accurate, data-driven premium estimations in real-time through an intuitive **Streamlit** user interface.

---

## ✨ Key Features

- **Intuitive Web UI**: Built with pure Streamlit, featuring a clean 3-column input layout for demographics, financial/policy details, and lifestyle metrics.
- **Custom Risk Scoring Engine**: Translates lifestyle habits and diagnosed medical conditions into actionable numerical risk scores.
- **Dynamic Encodings**: Handles one-hot dummy transformations matching reference categories from training.
- **Instant Output**: Displays both annual premium and monthly equivalent costs, along with an interactive risk and profile summary.
- **Serialized Artifacts**: Includes pre-trained `best_model.joblib` and fitted `scaler.joblib` for zero-setup inference.

---

## 🧮 How Risk Scoring Works

### 1. Lifestyle Risk Score
Derived from a combination of regular physical activity and daily stress level:

$$\text{Lifestyle Risk Score} = \text{Score}_{\text{Physical Activity}} + \text{Score}_{\text{Stress Level}}$$

| Level | Physical Activity Score | Stress Level Score |
| :--- | :---: | :---: |
| **Low** | 4 | 0 |
| **Medium** | 1 | 1 |
| **High** | 0 | 4 |

### 2. Total Medical Risk Score
Calculated by summing the risk weights associated with diagnosed conditions:

| Pre-existing Condition | Risk Weight |
| :--- | :---: |
| **Heart Disease** | 8 |
| **Diabetes** | 6 |
| **High Blood Pressure** | 6 |
| **Thyroid** | 5 |
| **No Disease / None** | 0 |

---

## 📁 Project Architecture

```
premium-insurance-with-lifestyle-prediction/
│
├── artifacts/
│   ├── best_model.joblib          # Trained XGBoost Regressor model
│   └── scaler.joblib              # Fitted StandardScaler (18 features)
│
├── main.py                        # Streamlit web application interface
├── prediction_helper.py           # Feature preprocessing & inference pipeline
├── premiums.ipynb                 # EDA, feature engineering, and model training notebook
├── premiums_with_life_style.xlsx  # Dataset used for training and validation
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore file for cache and environments
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

---

## ⚙️ Model & Preprocessing Pipeline

1. **Data Cleaning & Filtering**:
   - Outlier removal for extreme ages ($> 100$), dependants ($> 5$), and extreme incomes ($> 100$ Lakhs).
2. **Feature Engineering**:
   - Creation of `life_style_risk_score` and `total_risk_score`.
   - Ordinal mapping of `insurance_plan` (Bronze: 1, Silver: 2, Gold: 3).
3. **Multicollinearity Elimination**:
   - Variance Inflation Factor (VIF) analysis performed; redundant `income_level` dropped in favor of continuous `income_lakhs`.
4. **Encoding & Scaling**:
   - Categorical nominal variables transformed using one-hot encoding (`drop_first=True`).
   - Normalization applied using `StandardScaler` across all 18 features.
5. **Model Optimization**:
   - Model selection evaluated across Linear Regression, Ridge, Lasso, Random Forest, and XGBoost.
   - Hyperparameter tuning performed via `RandomizedSearchCV` on `XGBRegressor`.

---

## 📊 Model Evaluation & Benchmark Metrics

The models were evaluated on an independent 30% test holdout (`test_size=0.3`, `random_state=42`, $N_{\text{test}} = 2,958$) within [premiums.ipynb](premiums.ipynb).

### Model Benchmark Comparison

| Model | Train R² | Test R² | Test MSE | Test RMSE | Test MAE | Test MAPE (%) | Extreme Errors (>10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.95834 | 0.95788 | 3,271,494.45 | ₹1,808.73 | ₹1,390.46 | 8.74% | 943 (31.88%) |
| **Lasso Regression** | 0.95834 | 0.95788 | 3,271,552.98 | ₹1,808.74 | ₹1,390.36 | 8.74% | 947 (32.01%) |
| **Ridge Regression** | 0.95834 | 0.95788 | 3,271,588.35 | ₹1,808.75 | ₹1,390.48 | 8.74% | 944 (31.91%) |
| **Random Forest** | 0.99867 | 0.99038 | 747,138.13 | ₹864.37 | ₹676.74 | 4.34% | 274 (9.26%) |
| **XGBoost (Baseline)** | 0.99795 | 0.99388 | 475,123.53 | ₹689.29 | ₹569.93 | 3.76% | 180 (6.09%) |
| **Tuned XGBoost (Best Model)** | **0.99659** | **0.99499** | **389,342.56** | **₹623.97** | **₹526.45** | **3.55%** | **151 (5.10%)** |

### Best Model In-Depth Performance

The serialized production model (`artifacts/best_model.joblib`) was tuned using `RandomizedSearchCV` on `XGBRegressor`:

- **High Explained Variance**: Test R² of **0.9950**, capturing 99.5% of the variance in annual premium amounts.
- **Low Absolute Error**: Mean Absolute Error (MAE) of **₹526.45** and Root Mean Squared Error (RMSE) of **₹623.97** against a test target standard deviation of ₹8,814.30 (normalized RMSE ~7.08%).
- **Minimal Residual Bias**: Mean residual ($y - \hat{y}$) of **-₹11.46** with a median absolute percentage error of **2.64%**.
- **Error Distribution**: 90% of test predictions fall within an 8.04% error margin, and 95% fall within 10.05%.

### Subgroup Analysis by Policy Tier

Evaluating error tolerance across insurance plan tiers:

| Plan Tier | Test Count | Mean Premium | MAE | MAPE | Extreme Errors (>10%) | Extreme Error % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bronze** | 1,119 | ₹11,454.84 | ₹469.30 | 4.67% | 135 | 12.06% |
| **Silver** | 1,086 | ₹20,134.37 | ₹548.17 | 2.87% | 16 | 1.47% |
| **Gold** | 753 | ₹30,107.79 | ₹579.94 | 1.99% | 0 | 0.00% |

> [!NOTE]
> **Key Finding**: Over **89% (135 out of 151)** of all extreme errors (>10% deviation) occur on the **Bronze** plan. Because Bronze policies have lower base premiums (₹3,500–₹12,000), small absolute differences (~₹600–₹900) lead to higher relative percentage errors, whereas higher-tier policies (Gold) maintain virtually zero high-margin errors.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yogeshwaran-cse/premium-insurance-with-lifestyle-prediction.git
cd premium-insurance-with-lifestyle-prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit Application

Start the local development server:

```bash
streamlit run main.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 🛠️ Technologies Used

- **Language**: Python 3.10+
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Model Serialization**: [joblib](https://joblib.readthedocs.io/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
