---
title: "Kaggle Titanic Practice 6: Elevating Survival Predictions with Group Features & CatBoost Tuning"
published: false
description: "Learn how we achieved our highest CV score of 0.8563 using advanced passenger group features, CatBoost, and automated hyperparameter tuning with Optuna."
tags: kaggle, machinelearning, catboost, optuna
canonical_url: https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning
---

https://www.kaggle.com/c/titanic

← [Kaggle Practice 14: Game AI & Reinforcement Learning](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle Practice 2 (xxxx)](https://zenn.dev/rg687076/articles/xxxx) →

[GitHub Repository](https://github.com/kito2718/KaggleTitanic)

[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://dev.to/kito2718/kaggle-titanic-my-first-submission-eda-feature-engineering-and-model-evaluation-896)
[Kaggle Practice 3: Feature Engineering for Cabin](https://dev.to/kito2718/kaggle-titanic-cabin-feature-engineering-is-it-really-effective-44nc)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://dev.to/kito2718/kaggle-titanic-improving-survival-prediction-with-random-forest-age-imputation-5b3l)
[Kaggle Practice 5: Feature Engineering (Fare Log-Transformation and Age Stage Binning)](https://dev.to/kito2718/kaggle-titanic-improving-score-with-fare-log-transformation-and-age-stage-binning-2d5e)

# Abstract
* Created advanced passenger group statistics (Group_Size, Group_Female_Child_Ratio, etc.) based on last names and ticket numbers.
* Introduced CatBoost Classifier to handle categorical values and optimized hyperparameters automatically using Optuna.
* Reached our highest 5-Fold Cross-Validation (CV) accuracy of **0.8563**, though the Kaggle Public Score remained at **0.79665**.

# Overview
This post covers feature engineering refinement, model diversification using CatBoost, and hyperparameter tuning with Optuna. Although the Public Score didn't increase from our previous high, the CV score significantly improved.

## 1. Advanced Group Features
During the Titanic evacuation, passengers tended to act in groups (families or travel companions). To capture this behavior, we identified passengers sharing the same ticket number or last name and fare as a group.
We engineered the following four group features:

1. **Group Size (`Group_Size`)**: The actual number of companions sharing the ticket or last name and fare. Unlike the traditional `Family_Size` (which only counts biological/legal relatives), this captures friends, couples, and staff traveling together.
2. **Group Female & Child Ratio (`Group_Female_Child_Ratio`)**: The ratio of priority rescue candidates (females or children under 16) within the group.
3. **Group Mean Age (`Group_Mean_Age`)**: The average age of the group.
4. **Group Fare Median Difference (`Group_Fare_Median_Diff`)**: The difference between the passenger's fare and the median fare of their class (Pclass). This acts as a proxy for cabin location and quality.

### Visualization & Analysis of Group Features
Let's look at the relationship between these group features and survival rate.

#### ① Group Size vs Survival Rate (`Group_Size`)
![Group Size vs Survival Rate](https://raw.githubusercontent.com/kito2718/zenn_articles/main/articles/images/group_size_survival.png)


* **X-axis**: Number of people in the group. **Y-axis**: Survival rate.
* **Analysis**: Mid-sized groups of 2-4 people show high survival rates (50-70%). In contrast, solo travelers and large families (5+ people) show lower survival rates. This indicates that having a moderate number of companions to cooperate with during the crisis increased the chance of survival.

#### ② Group Female & Child Ratio Distribution (`Group_Female_Child_Ratio`)
![Group Female & Child Ratio Distribution](https://raw.githubusercontent.com/kito2718/zenn_articles/main/articles/images/group_female_child_ratio_survival.png)

* **X-axis**: Female and child ratio within the group (0.0 to 1.0). **Y-axis**: Probability density.
* **Analysis**:
  * **Left Side (Near 0.0)**: Groups with no women or children (adult males only) have a dense cluster of deceased passengers (red area). This represents the lowest priority for rescue.
  * **Right Side (Near 1.0)**: Groups consisting entirely of women and children have a massive peak of surviving passengers (blue area), indicating they were evacuated first.
  * **Takeaway**: Even if a passenger is an adult male (who typically has a very low survival rate), if he belongs to a group with many women and children, his chance of survival increases because he was more likely guided to a lifeboat alongside his group.

#### ③ Group Mean Age Distribution (`Group_Mean_Age`)
![Group Mean Age Distribution](https://raw.githubusercontent.com/kito2718/zenn_articles/main/articles/images/group_mean_age_survival.png)

* **X-axis**: Average age of the group. **Y-axis**: Probability density.
* **Analysis**: The surviving passengers (blue area) show peaks around younger averages (families with children) and mature age groups (35-40, likely family heads). The overall age profile of a group plays a significant role in their mobility and evacuation priority.

#### ④ Group Fare Median Difference (`Group_Fare_Median_Diff`)
![Group Fare Median Difference](https://raw.githubusercontent.com/kito2718/zenn_articles/main/articles/images/group_fare_median_diff_survival.png)

* **X-axis**: Survival status (0 = Deceased, 1 = Survived). **Y-axis**: Difference from Pclass median fare.
* **Analysis**: Surviving passengers tended to belong to groups that paid a higher fare relative to their class median. This suggests their cabins were situated in more accessible locations (closer to the deck or evacuation routes).

### Implementation Code
Below is the feature engineering code implemented in our pipeline:

```python
# --- Advanced Group Features Creation ---
# Define groups (Group_Id) by ticket number, or last name and fare for solo tickets
df_all['Ticket_Group_Size'] = df_all.groupby('Ticket')['PassengerId'].transform('count')
df_all['Group_Id'] = df_all['Ticket']
mask = df_all['Ticket_Group_Size'] == 1
df_all.loc[mask, 'Group_Id'] = df_all.loc[mask, 'Last_Name'] + '_' + df_all.loc[mask, 'Fare'].astype(str)

# 1. Group Size
df_all['Group_Size'] = df_all.groupby('Group_Id')['PassengerId'].transform('count')

# 2. Female and Child Ratio within Group
df_all['Is_Female_or_Child'] = ((df_all['Sex'] == 'female') | (df_all['Age'] < 16)).astype(int)
df_all['Group_Female_Child_Ratio'] = df_all.groupby('Group_Id')['Is_Female_or_Child'].transform('mean')

# 3. Mean Age of Group
df_all['Group_Mean_Age'] = df_all.groupby('Group_Id')['Age'].transform('mean')

# 4. Difference from Pclass median fare
pclass_fare_median = df_all.groupby('Pclass')['Fare'].transform('median')
df_all['Group_Fare_Median_Diff'] = df_all['Fare'] - pclass_fare_median

# Drop temporary columns
df_all = df_all.drop(columns=['Ticket_Group_Size', 'Is_Female_or_Child'])
```

## 2. CatBoost & Optuna Tuning
The Titanic dataset contains many categorical variables (`Sex`, `Embarked`, `Title`, `Deck`). To handle these effectively, we introduced **CatBoost** (CatBoostClassifier), which features robust target statistics encoding natively.

We also performed hyperparameter optimization using the `optuna` library across LightGBM, XGBoost, and CatBoost.

### Tuning Implementation Code
Here is our Optuna search script, evaluating each trial with 5-fold cross-validation accuracy over 30 trials:

```python
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
from catboost import CatBoostClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- LightGBM Tuning ---
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'verbosity': -1,
        'random_state': 42
    }
    model = lgb.LGBMClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
print(f"Best LightGBM CV Score: {study_lgb.best_value:.4f}")

# --- CatBoost Tuning ---
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)
print(f"Best CatBoost CV Score: {study_cat.best_value:.4f}")
```

The optimized parameter scores were:
* **LightGBM**: CV `0.8530` (improved)
* **XGBoost**: CV `0.8530` (significantly improved)
* **CatBoost**: CV **0.8563** (New Highest CV Score)
  * Best params: `iterations=186, depth=4, learning_rate=0.068, l2_leaf_reg=5.63`

## 3. Results & Submission

| Model | Baseline CV (Age Imputation) | Feature Addition CV | Hyperparameter Optimized CV | Kaggle Public Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | **0.8519** | 0.8485 | - | - |
| LightGBM | 0.8485 | 0.8518 | 0.8530 | 0.79425 (Step 1) |
| XGBoost | 0.8226 | 0.8272 | 0.8530 | - |
| **CatBoost** | - | - | **0.8563** | **0.79665** (Highest Score Tie) |

We also tested stacking ensembles (Meta: Ridge Classifier, CV: 0.8485), but due to the small dataset size, the meta-model suffered from overfitting. The standalone tuned CatBoost classifier provided the best results.

# Summary
Integrating advanced group features with CatBoost and Optuna yielded a new local CV record of **0.8563** and tied our highest Kaggle Public Score of **0.79665**.

The full code adjustments and experimental log can be reviewed in [ToBeContinued.md](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/ToBeContinued.md).
We will continue searching for further improvements.

Japanese version:
[Kaggle Titanic Practice 1: Setting up Kaggle Titanic Environment on Local PC](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle Titanic Practice 2: First Submission](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle Titanic Practice 3: Feature Engineering with Cabin](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle Titanic Practice 4: Feature Engineering (Age Imputation using Random Forest)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle Titanic Practice 5: Feature Engineering (Non-linear Transforms & Binning of Numerical Features)](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)
[Kaggle Titanic Practice 6: Feature Engineering (Advanced Group Features), CatBoost & Optuna Tuning](https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning)
