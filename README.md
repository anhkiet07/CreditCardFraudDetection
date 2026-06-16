# Credit Card Fraud Detection using Machine Learning (Logistic Regression, XGBoost & Random Forest)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Data Science](https://img.shields.io/badge/Fields-Machine%20Learning%20%7C%20Fraud%20Detection-success)
![Libraries](https://img.shields.io/badge/Libraries-XGBoost%20%7C%20Scikit--Learn-orange)

## 📌 Project Overview
This project focuses on building a highly robust and accurate machine learning infrastructure to detect fraudulent credit card transactions. Credit card fraud detection presents a classic **highly imbalanced dataset** challenge, where fraudulent activities comprise only a minute fraction (approx. 0.17%) of the entire transactional dataset. 

To tackle this real-world imbalance effectively, this project implements, evaluates, and benchmarks three distinct algorithmic architectures:
1. **Logistic Regression** (Optimized with custom probability decision thresholds)
2. **XGBoost Classifier** (High-performance gradient boosting framework)
3. **Random Forest Classifier** (Ensemble bagging framework with tuned threshold execution)

A comprehensive theoretical and empirical research paper is also included in the repository as a complete thesis guide (`CreditCardFraudDetection.pdf`).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning & Modeling:** `scikit-learn`, `xgboost`
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

---

## 📂 Project Structure

The repository is structurally organized into dedicated directories for documentation and executable source code:

```text
├── Report/
│   └── CreditCardFraudDetection.pdf       # Full academic research paper and project thesis (in Vietnamese)
├── Scripts/
│   ├── logistic.ipynb                     # Jupyter Notebook for the Optimized Logistic Regression pipeline
│   ├── xgboost.ipynb                      # Jupyter Notebook for the High-Performance XGBoost pipeline
│   └── randomforest.ipynb                 # Jupyter Notebook for the Tuned Random Forest ensemble pipeline
└── README.md                              # Comprehensive project documentation
