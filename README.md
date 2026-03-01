🏦 Loan Approval Prediction & Credit Risk Analysis
📌 Project Overview

Financial institutions must carefully evaluate loan applications to minimize default risk while maximizing approval rates for reliable borrowers.

This project focuses on building a machine learning classification model that predicts whether a loan application should be approved (1) or rejected (0) based on financial and demographic features.

The repository demonstrates a complete end-to-end data science pipeline, from raw data preprocessing to building a high-performing predictive model.

⚙️ The Data Science Pipeline
1. Data Preprocessing & Cleaning

Real-world financial data often contains inconsistencies and unrealistic values. Ensuring data quality was the first critical step:

Outlier Removal: Removed illogical entries such as applicants with age greater than 100 years to maintain data integrity.

Data Validation: Checked distributions and ensured all values fall within realistic financial ranges.

Missing Values Handling: Cleaned or removed missing entries to prevent bias during model training.

2. Feature Engineering & Encoding

Machine learning models require numerical input, so categorical features were transformed while preserving their real-world meaning:

Ordinal Encoding: Applied to ordered features such as education levels to reflect hierarchy (e.g., High School < Bachelor < Master < Doctorate).

One-Hot Encoding: Used for nominal features like loan purpose and home ownership to avoid introducing false relationships.

Binary Encoding: Converted yes/no features (e.g., previous defaults) into 0/1 format for model compatibility.

3. Exploratory Data Analysis (EDA)

EDA was conducted to understand patterns and relationships between features and loan approval:

Compared distributions of numerical features across approved and rejected loans

Used boxplots to detect outliers and group differences

Analyzed correlations to identify important predictors

Investigated how income, credit score, and loan ratios impact approval

4. Predictive Modeling

Multiple machine learning models were implemented to evaluate performance:

Logistic Regression: Used as a baseline linear model

XGBoost Classifier: Captures complex non-linear relationships and feature interactions

The dataset was split into training and testing sets to ensure proper generalization.

5. Model Evaluation & Metrics

The models were evaluated using classification metrics to measure their effectiveness:

Accuracy: Measures overall prediction correctness

Classification Report: Includes precision, recall, and F1-score

Confusion Matrix: Shows distribution of correct and incorrect predictions

Special attention was given to avoid overfitting and ensure reliable performance on unseen data.

📊 Results & Business Impact

The models achieved strong predictive performance, making them suitable for real-world financial applications:

Logistic Regression Accuracy: ~85%

XGBoost Accuracy: ~93%

🔍 Interpretation

XGBoost significantly outperforms the baseline model

Small gap between training and testing accuracy indicates good generalization

The model effectively identifies high-risk applicants while minimizing false approvals

This performance demonstrates the model’s ability to support automated loan decision systems and reduce financial risk.

🧠 Key Insights

Applicants with higher income and credit scores are more likely to be approved

Loan-to-income ratio is a critical risk indicator

Previous loan defaults strongly increase rejection probability

Financial behavior plays a larger role than demographic features

🛠️ Technologies Used

Python – Core programming language

Pandas & NumPy – Data manipulation and analysis

Matplotlib & Seaborn – Data visualization

Scikit-Learn – Machine learning models and preprocessing

XGBoost – Advanced gradient boosting algorithm
