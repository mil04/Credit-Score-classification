# Machine Learning Project: Predicting Default

## Overview
This project focuses on predicting default using machine learning techniques. The dataset, sourced from synthetic data generation, comprises various financial and personal attributes of individuals, including income, savings, debts, and expenditures. Our objective was to build a robust model that accurately predicts whether a borrower would default on their loan.

## [Dataset Description](https://www.kaggle.com/datasets/conorsully1/credit-score)
The dataset contains 87 columns and 1000 observations. Key features include:
- **DEFAULT**: Binary variable indicating default (1) or non-default (0).
- **CREDIT_SCORE**: Creditworthiness score of individuals.
- **CUST_ID**: Unique identifier for each individual.
- **INCOME, SAVINGS, DEBT**: Financial attributes.
- **Expenditure Categories**: Such as clothing, education, entertainment, fines, gambling, groceries, health, housing, taxes, travel, and utilities.

## Data Preprocessing
1. **Missing Values**: No missing values were found in the dataset.
2. **Imbalanced Data**: The target variable (DEFAULT) was imbalanced, with more instances of non-defaults than defaults.
3. **Categorical Variables**: 
   - Converted `CAT_GAMBLING` to ordinal encoding ('No' -> 1, 'Low' -> 2, 'High' -> 3).
   - Removed `CUST_ID` as it does not provide predictive value.
4. **Outliers and Transformations**: Identified and replaced outliers in columns like SAVINGS, DEBT, and various expenditures. Applied transformations as necessary but found most distributions were normal or exponential.
5. **Feature Engineering**: Created new features by calculating ratios of different expenditures over time periods (6 months vs. 12 months).

## Exploratory Data Analysis (EDA)
- **Target Variable Distribution**: Identified imbalance in the DEFAULT variable distribution.
- **Feature Correlation**: Removed highly correlated features (correlation > 95%).
- **Important Features**: Identified `R_DEBT_INCOME` as a significant predictor for DEFAULT.

## Model Training
Several machine learning models were trained and evaluated using cross-validation. Our primary evaluation metric was Accuracy, with secondary importance given to Recall (particularly for the DEFAULT=1 class).

1. **Random Forest**: Applied PCA for dimensionality reduction, used StandardScaler, and optimized hyperparameters using RandomizedSearchCV.
2. **Support Vector Classification (SVC)**: Utilized PipelineBasic for preprocessing, SelectKBest for feature selection, and GridSearchCV for hyperparameter tuning.
3. **Logistic Regression**: Employed StandardScaler, Recursive Feature Elimination for feature selection, and GridSearchCV for optimization.
4. **XGBoost Classifier**: Integrated PCA for feature selection and optimized using BayesSearchCV.
5. **Gradient Boosting Classifier**: Followed a similar approach as XGBoost without normalization.
6. **Stacking**: Combined predictions from XGBoost, Logistic Regression, SVC, and Gradient Boosting Classifier using Logistic Regression as the meta-classifier.

## Results
The models were evaluated based on their cross-validation accuracy and recall scores. The stacking model demonstrated improved overall performance by leveraging the strengths of individual base models.

## Conclusion
The project successfully developed a predictive model for borrower default, with the stacking model providing the best performance. Future work could explore additional feature engineering and advanced ensemble methods to further enhance prediction accuracy.

## Collaborators
- **Pahasian Milanna**
- **Badzeika Hleb**
- **Bokhan Katsiaryna**

Department of Mathematics and Information Sciences, Warsaw University of Technology
