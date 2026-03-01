# 🏦 Credit Risk Prediction & Loan Approval Classifier

## 📌 Project Overview
Financial institutions face significant risks when approving loans. The goal of this project is to minimize financial loss by building a robust machine learning classification model that accurately predicts the likelihood of a loan applicant defaulting. 

This repository contains an end-to-end data science pipeline that processes historical financial data, handles inconsistencies, and trains a highly accurate predictive model.

## ⚙️ The Data Science Pipeline

### 1. Data Preprocessing & Cleaning
Real-world financial data is often messy. The first step was ensuring the data was logically sound:
* **Outlier Removal:** Identified and removed logical impossibilities in the dataset (e.g., applicants with an age > 100 years). This step also organically resolved correlated errors in the employment experience column, establishing a clean baseline.
* **Handling Missing Values:** Ensured the dataset was free of nulls before feeding it into the model.

### 2. Feature Engineering & Encoding
Machine learning algorithms require numerical input, so categorical variables were carefully transformed to preserve their inherent business logic:
* **Ordinal Encoding:** Education levels (`person_education`) were mapped sequentially (e.g., High School to Master's) to preserve the hierarchical weight of a candidate's educational background.
* **One-Hot Encoding:** Applied `pd.get_dummies` to nominal categories such as `loan_intent` (Venture, Medical, Personal, etc.) and `person_home_ownership` (Rent, Own, Mortgage) to prevent the model from falsely assuming mathematical relationships between distinct categories.
* **Binary Mapping:** Converted binary features (`gender`, `default` history) into standard 0/1 formats.

### 3. Predictive Modeling
* Built a classification model to identify high-risk applicants.
* Utilized Scikit-Learn to split the data into training and testing sets to ensure the model could generalize to unseen future applicants.
* Leveraged hyperparameter tuning and cross-validation techniques to optimize the model's decision boundaries.

### 4. Model Evaluation & Metrics
The model was evaluated using strict classification metrics, prioritizing the model's ability to distinguish between classes:
* **ROC-AUC Score:** Evaluated the model's ability to rank applicants by risk.
* **Accuracy & Classification Report:** Measured overall correctness, precision, and recall.

## 📊 Results & Business Impact
The model demonstrated exceptional predictive power, making it a viable tool for automating safe lending decisions:
* **ROC-AUC Score:** `0.974` (Outstanding class separation)
* **Test Accuracy:** `~93%` 

This high ROC-AUC score means the model is highly capable of flagging potential defaulters without incorrectly penalizing safe, reliable borrowers.

## 🛠️ Technologies Used
* **Python** (Core programming language)
* **Pandas & NumPy** (Data manipulation and linear algebra)
* **Matplotlib & Seaborn** (Exploratory Data Analysis and statistical visualizations)
* **Scikit-Learn** (Machine learning modeling, preprocessing, and evaluation)
