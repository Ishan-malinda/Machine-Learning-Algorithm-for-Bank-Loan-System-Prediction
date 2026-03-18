# Bank Loan Approval & Max Amount Prediction

## 📌 Project Overview
This machine learning project aims to assist bankers and financial risk analysts in assessing loan applications. The project is divided into two main case studies:
1. **Classification (Case Study A):** Predicting the loan approval status (Approved/Rejected) to minimize the risk of future defaulted payments. The primary business goal is to maximize the correct identification of "Rejected" applications.
2. **Regression (Case Study B):** Estimating the maximum loan amount a client can be offered.

## 📊 Dataset
The dataset contains financial and demographic records of loan applicants. Key features include:
- **Demographics:** Age, Employment Length, Home Ownership
- **Financials:** Income, Loan Amount, Loan Intent, Loan Interest Rate, Loan-to-Income Ratio (LTI)
- **Credit History:** Payment Default on File, Credit History Length
- **Targets:** `loan_approval_status` (Classification) and `max_allowed_loan` (Regression)

## 📁 Project Structure / Workflow
The project workflow is divided into three sequential Jupyter Notebooks:

### 1. Data Understanding and Preprocessing (`notebook1_cw.ipynb`)
- **Exploratory Data Analysis (EDA):** Analyzed variable distributions, types, and checked for data quality issues.
- **Data Cleaning:** Addressed missing values (e.g., mean imputation for continuous variables like age and interest rate, listwise deletion for payment default).
- **Outlier Removal:** Filtered out erroneous and impossible records (e.g., biologically impossible ages like 123, and negative maximum loan amounts).
- **Output:** Generated clean datasets (`loan_classification_prepared.csv` and `loan_regression_prepared.csv`) for modeling.

### 2. Classification Modelling & Hyperparameter Tuning (`notebook2_cw.ipynb`)
- **Objective:** Predict `loan_approval_status`. 
- **Methodology:** Implemented an 80/20 Train-Test split with class stratification (`stratify=y`) to handle the highly imbalanced dataset (~85.8% Rejected, 14.2% Approved).
- **Models Built:** Logistic Regression, Naïve Bayes, and K-Nearest Neighbours (KNN).
- **Hyperparameter Tuning:** Applied `GridSearchCV` with 5-fold cross-validation, optimizing for Recall.
- **Key Results:** **K-Nearest Neighbours (KNN)** emerged as the best model. After tuning (`n_neighbors=3`, `metric=euclidean`, `weights=distance`), the model effectively satisfied the business criteria of catching risky loans, achieving:
  - **Recall (Class 0 / Rejected):** 0.97
  - **Precision (Class 0 / Rejected):** 0.93
  - **F1-Score (Class 0 / Rejected):** 0.95

### 3. Regression Modelling (`notebook3_cw.ipynb`)
- **Objective:** Estimate the `max_allowed_loan` for clients.
- **Models Built:** Decision Tree Regressors (DT-1 and DT-2).
- **Methodology:** Evaluated tree depths and parameters to accurately estimate the maximum loan offer limit for approved clients.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Logistic Regression, Naïve Bayes, KNN, Decision Trees, GridSearchCV)
- **Data Visualization:** Plotly Express, Matplotlib

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
