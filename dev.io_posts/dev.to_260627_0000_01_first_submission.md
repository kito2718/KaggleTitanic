---
title: "Kaggle Titanic: My First Submission (EDA, Feature Engineering, and Model Evaluation)"
published: true
description: "My first submission journey for the Kaggle Titanic competition. Covers EDA, simple feature engineering, and model comparison (Logistic Regression, Random Forest, XGBoost, and LightGBM)."
tags: python, kaggle, machinelearning, dataanalysis
canonical_url: https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission
---

[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle Practice 3: Feature Engineering for Cabin](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)

https://www.kaggle.com/c/titanic

[Available on GitHub](https://github.com/kito2718/KaggleTitanic)

# Abstract
- Participated in the Kaggle Titanic competition and performed Exploratory Data Analysis (EDA).
- Performed basic feature engineering such as extracting title prefix (Title) and family size (FamilySize).
- Compared four modeling approaches (Logistic Regression, Random Forest, XGBoost, and LightGBM) using 5-fold cross-validation..
- LightGBM yielded the best CV score, resulting in a public leaderboard score of 0.77272.

# Introduction
This is a continuation of my Kaggle Titanic journey.
In this post, we'll start by visualizing and analyzing variables likely related to survival (such as gender and passenger class) and handling missing values as part of the exploratory data analysis (EDA) process.
After that, we'll extract titles from passenger names, build new features like family size, and compare model performance using 5-Fold Cross-Validation (CV).

# Exploratory Data Analysis (EDA)
The Titanic dataset contains 891 training rows and 418 test rows. Missing values are one of the first challenges in the Titanic dataset:
- **Cabin**: 77.1% (mostly missing)
- **Age**: 19.9% (about 20% missing)
- **Embarked**: 0.2% (only 2 missing values)

## Visualization
Let's plot basic variables against survival rate.

### 1. Survival Rate by Gender and Passenger Class
Looking at survival rates by gender and passenger class (Pclass) reveals strong insights. Female passengers had a significantly higher survival rate, and passengers in 1st class (Pclass = 1) had the highest chance of survival.
![Survival Rate by Gender and Passenger Class](https://static.zenn.studio/user-upload/0d803e9b5315-20260628.png)

### 2. Age Distribution
Here is the histogram of age distribution, split by survival status. Younger passengers, especially infants and small children, tended to have higher survival rates. and we can observe specific age groups with higher mortality.
![Age Distribution Histogram](https://static.zenn.studio/user-upload/c95d3e0068ea-20260628.png)

### 3. Correlation Map
Let's examine the correlation heatmap between numerical features. As expected, Pclass and Fare exhibit a strong negative correlation because higher-class tickets were generally more expensive.
![Correlation Heatmap](https://static.zenn.studio/user-upload/cd4aebddf5ca-20260628.png)

Here is the formula for calculating correlation ($r$):
$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

Simplifying this using covariance and standard deviation:
$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
Where $\text{Cov}(X, Y)$ is the covariance between $X$ and $Y$ (representing how they move together), and $\sigma_X, \sigma_Y$ are their respective standard deviations (representing the spread of the data).

# Preprocessing and Feature Engineering
Based on our EDA findings, I wrote a `feature_engineering` function to prepare the data:
1. **Extract Title**: Extract honorifics (Mr, Miss, Mrs, Master, etc.) from `Name` using regular expressions, mapping rare titles to `Rare`.
2. **Impute Age**: Instead of using the global median, impute missing age values using the median age of each specific title group. This approach produces more realistic age estimates than using a single global median. (e.g., child-level ages for 'Master' and adult-level ages for 'Mr/Mrs').
3. **FamilySize and IsAlone**: Sum sibling/spouse count (`SibSp`) and parent/child count (`Parch`) plus one (for the passenger themselves) to create `FamilySize`. If `FamilySize` is 1, set the `IsAlone` flag to 1.
4. **Additional imputations**: Fill missing `Fare` with the median and `Embarked` with the mode.
5. **Categorical Encoding**: Encode `Sex`, `Embarked`, and `Title` using `LabelEncoder`.

Here is the code:
```python
def feature_engineering(df):
    df = df.copy()
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')
    
    # Impute missing Age with median of each Title group
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Family-related features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Impute Fare and Embarked
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Label Encoding for categorical variables
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title']:
        df[col] = le.fit_transform(df[col])
    return df
```

After preprocessing, we selected 10 features:
`['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']`.

# Model Evaluation and Results
We compared four popular machine learning models using Stratified 5-Fold Cross-Validation (CV).

Here is the evaluation code:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM':           lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, max_depth=3, num_leaves=7, learning_rate=0.05)
}
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f'{name:<25}: {scores.mean():.4f} +/- {scores.std():.4f}')
```

The average cross-validation scores are shown below:
| Model | 5-Fold CV Score (Accuracy) |
| :--- | :--- |
| Logistic Regression | 0.8014 +/- 0.0133 |
| Random Forest | 0.8227 +/- 0.0077 |
| XGBoost | 0.8181 +/- 0.0220 |
| LightGBM | 0.8350 +/- 0.0178 |

We plotted the CV score distributions using boxplots:
![Model Comparison](https://static.zenn.studio/user-upload/04350e7741f2-20260628.png)

LightGBM achieved the best CV score of **0.8350**.
Random Forest also performed well, achieving a score of 0.8227.

## Feature Importance
Let's visualize the feature importances of our best LightGBM model:
![Feature Importance](https://static.zenn.studio/user-upload/e3613eb19157-20260628.png)

The model relied heavily on Sex, Title, Fare, and Age, which aligns well with the patterns observed during EDA.

# First Kaggle Submission
Using our best LightGBM model, we generated predictions for the test dataset and created `submission.csv`:
```python
preds = best_model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
submission.to_csv('../submissions/submission.csv', index=False)
```

Leaderboard Score: **0.77272**.
This became my personal best score at the time.
It is common for public leaderboard scores to be slightly lower than local CV scores due to differences between the training folds and the hidden test set.

# Summary
We walked through the EDA, feature engineering, and model evaluation workflow. Evaluating multiple models proved to be very effective.
In the next article, we'll tackle the heavily missing Cabin feature and see whether it can help us push the score even higher.

Japanese version:
[Kaggle Practice 1: Setting up Kaggle Titanic Environment on a Local PC](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle Practice 2: First Submission](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle Practice 3: Feature Engineering for Cabin](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
