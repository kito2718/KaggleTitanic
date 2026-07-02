---
title: "Kaggle Titanic: Improving Survival Prediction with Random Forest Age Imputation"
published: false
description: "How predicting missing age values using RandomForestRegressor improved our CV score to 0.8519 and Kaggle Public Score to 0.78947."
tags: "machinelearning, kaggle, python, datascience"
canonical_url: "https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation"
---

# Abstract
- Changed age imputation from median values by title to predictive imputation using `RandomForestRegressor`.
- The best 5-Fold CV (Cross-Validation) score improved from 0.8507 to 0.8519 (Logistic Regression).
- Kaggle Public Score increased from 0.78708 to 0.78947.
- Validation code is committed to GitHub: [titanic_eda_20260702_2031_age_imputation.ipynb](https://github.com/kito2718/KaggleTitanic/blob/main/notebooks/titanic_eda_20260702_2031_age_imputation_.ipynb).

# Overview
In the Kaggle Titanic: Machine Learning from Disaster competition, passenger Age is a critical factor for predicting survival.
Previously, we filled missing values with the median age of each passenger title (Mr, Miss, Mrs, Master, Rare). This time, we tried a more advanced approach: predicting the missing ages using a machine learning model (RandomForestRegressor) based on other features (Pclass, Sex, SibSp, Parch, Fare, Embarked, Deck).

# Implementation
Here is the preprocessing code for the imputation. We trained `RandomForestRegressor` on passengers with known ages and predicted the missing values.

```python
from sklearn.ensemble import RandomForestRegressor

# Features used for predicting Age
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'FamilySize', 'IsAlone', 'Age']
df_age_prep = df_all[age_features].copy()

# One-Hot Encoding for categorical features
cat_cols_for_age = ['Sex', 'Embarked', 'Title', 'Deck']
df_age_encoded = pd.get_dummies(df_age_prep, columns=cat_cols_for_age, drop_first=True)

# Split into known and unknown age datasets
train_age = df_age_encoded[df_age_encoded['Age'].notnull()]
test_age = df_age_encoded[df_age_encoded['Age'].isnull()]

X_train_age = train_age.drop(columns=['Age'])
y_train_age = train_age['Age']
X_test_age = test_age.drop(columns=['Age'])

# Train regressor and predict missing age
age_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
age_regressor.fit(X_train_age, y_train_age)
predicted_ages = age_regressor.predict(X_test_age)

# Impute missing values in the original dataframe
df_all.loc[df_all['Age'].isnull(), 'Age'] = predicted_ages
```

# Validation Results
Comparison of 5-Fold CV (Cross-Validation) accuracy across different models:

| Model | Before (Median by Title) | After (Random Forest Imputation) | Difference |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.8507 +/- 0.0104 | **0.8519 +/- 0.0115** | **+0.0012** |
| Random Forest | 0.8204 +/- 0.0193 | 0.8249 +/- 0.0348 | +0.0045 |
| XGBoost | 0.8215 +/- 0.0241 | 0.8226 +/- 0.0244 | +0.0011 |
| LightGBM | 0.8496 +/- 0.0211 | 0.8485 +/- 0.0147 | -0.0011 |

Logistic Regression achieved our personal best 5-Fold CV accuracy of **0.8519**.
We also observed accuracy improvements in tree-based models like Random Forest and XGBoost.

# Kaggle Submission Score
We predicted the test dataset using the updated Logistic Regression model and submitted it to Kaggle.
Our Public Score successfully improved from **0.78708 to 0.78947**!
It is encouraging to see that the local CV score improvement translated directly to the Kaggle Public Score.

# Summary & Next Steps
By estimating passenger age from other relevant features instead of using simple median values, the model could learn a more realistic passenger representation.
For our next attempt, we will target hyperparameter tuning (using Optuna) and model ensembling to achieve further improvements.

Hope this helps!
