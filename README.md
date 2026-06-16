# Credit Card Fraud Detection using Machine Learning (Logistic Regression & XGBoost)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Data Science](https://img.shields.io/badge/Fields-Machine%20Learning%20%7C%20Fraud%20Detection-success)
![Libraries](https://img.shields.io/badge/Libraries-XGBoost%20%7C%20Scikit--Learn-orange)

## 📌 Project Overview
This project focuses on building a highly accurate machine learning system to detect fraudulent credit card transactions. Credit card fraud detection is a classic example of an **extremely imbalanced dataset** problem, where fraudulent transactions account for only a tiny fraction (approx. 0.17%) of the entire dataset. 

To solve this challenge, the project implements and compares two distinct approaches:
1. **Logistic Regression** (Optimized with custom decision thresholds)
2. **XGBoost Classifier** (Advanced gradient boosting framework)

A comprehensive theoretical and empirical research paper is also included in the repository as a complete thesis guide (`CreditCardFraudDetection.pdf`).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning & Modeling:** `scikit-learn`, `xgboost`
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

---

## 📂 Project Structure

The project directory is structured as follows to ensure easy replication and clean modular deployment:

```text
├── CreditCardFraudDetection.pdf       # Full academic research paper and project thesis (in Vietnamese)
├── LinearRegression.ipynb             # Auxiliary regression analysis for baseline evaluations
├── logistic.ipynb                     # Jupyter Notebook for the Optimized Logistic Regression pipeline
├── xgboost.ipynb                      # Jupyter Notebook for the High-Performance XGBoost pipeline
└── README.md                          # Comprehensive project documentation
