---
title: "Kaggle Titanic: Improving Score with Fare Log-Transformation and Age Stage Binning"
published: false
description: "How log-transforming Fare and stage binning Age improved our LightGBM model, achieving a new Kaggle Public Score of 0.79665."
tags: "machinelearning, kaggle, python, datascience"
canonical_url: "https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning"
---

# Abstract

- Applied log-transformation to passenger Fare to mitigate skewness and stage binning to Age for life stage classification.
- While the linear model (Logistic Regression) accuracy decreased due to representation changes, tree-based models (Random Forest, XGBoost, LightGBM) showed improved CV accuracy under Pattern C (excluding original continuous Fare and Age).
- Submitted predictions from the improved LightGBM model (Pattern C) and achieved a new best Kaggle Public Score of **0.79665** (previously 0.78947).

---

## Introduction

So far, we achieved a 5-Fold CV of 0.8519 and a Kaggle Public Score of 0.78947 by introducing ML-based age imputation. In this iteration, we focused on scaling numerical features and handling non-linear relationships to push our score further.

---

## Preprocessing & Features

### 1. Log-Transformation of Fare
The ticket `Fare` in Titanic has a highly right-skewed distribution because a small number of wealthy passengers paid extremely high fares compared to the majority of third-class passengers.
To mitigate this skewness, we applied `log1p(Fare)`. This transforms the skewed distribution into a normal-like bell curve, which stabilizes the training of linear models.

![Fare Log-Transformation](https://raw.githubusercontent.com/kito2718/KaggleTitanic/main/dev.io_posts/images/fare_log_transform.png)

Linear models like Logistic Regression generally perform better and stabilize training when the input numerical features follow a normal distribution.

### 2. Age Binning (Life Stages)
Using continuous values of `Age` limits linear models to capturing simple monotonic relationships (e.g., higher age equals higher survival probability). However, actual survival rates vary non-linearly across age groups (higher for children/infants, lower for young adults).

![Age Stage Binning](https://raw.githubusercontent.com/kito2718/KaggleTitanic/main/dev.io_posts/images/age_binning.png)

To capture this non-linear relationship, we split passengers into 5 life stages (bins) and dummy-encoded them:
- Infant (0-5 years old)
- Child (6-15 years old)
- Youth (16-30 years old)
- Middle-aged (31-55 years old)
- Senior (56+ years old)

This classification allows models to capture specific group boundaries (e.g., protecting children during evacuation) more effectively.

---

## Evaluation Patterns for Multicollinearity

Since the log-transformed `Log_Fare` and binned `Age_Bin` dummy variables are highly correlated with their original continuous counterparts, keeping both might cause instability due to multicollinearity. Thus, we evaluated 4 different feature configurations using 5-Fold Stratified Cross-Validation:

- **Pattern A**: Keep original `Fare` and `Age`, add `Log_Fare` and `Age_Bin` dummy variables.
- **Pattern B**: Exclude `Fare` (replace with `Log_Fare`), keep `Age`, add `Age_Bin` dummy variables.
- **Pattern C**: Exclude `Fare` (replace with `Log_Fare`), exclude `Age` (use only `Age_Bin` dummy variables).
- **Pattern D**: Keep `Fare`, exclude `Age` (use only `Age_Bin` dummy variables).

---

## Validation Results (5-Fold CV Accuracy)

| Model | Baseline (RF Age Imputed) | Pattern A | Pattern B | Pattern C | Pattern D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8519** | 0.8474 | 0.8474 | 0.8474 | 0.8440 |
| **Random Forest** | 0.8249 | 0.8238 | 0.8170 | **0.8339 (+0.0090)** | 0.8271 |
| **XGBoost** | 0.8226 | 0.8159 | 0.8170 | **0.8283 (+0.0057)** | 0.8272 |
| **LightGBM** | 0.8485 | 0.8474 | 0.8474 | **0.8496 (+0.0011)** | 0.8496 |

### Discussion
- **Logistic Regression**: Converting Age to dummy variables resulted in a loss of granular numerical information, which led to a slightly lower CV score compared to the baseline (0.8519).
- **Tree-Based Models**: In **Pattern C** (excluding continuous Fare and Age), all tree models showed significant accuracy improvements. Removing redundant continuous variables likely prevented trees from splitting too deeply, acting as a form of regularization.

---

## Kaggle Submission Result

Although the overall best CV score did not exceed the baseline Logistic Regression, the LightGBM model trained on **Pattern C** features achieved a better generalization performance.

- **Kaggle Public Score**: **0.79665** (Improved from the previous best of 0.78947!)

The results demonstrate that while linear models struggled with the binned representations, the tree-based models benefited significantly, translating to a better Public Score.

The corresponding code has been committed to GitHub: [titanic_eda_20260703_2025_fare_log_and_age_binning.ipynb](https://github.com/kito2718/KaggleTitanic/blob/main/notebooks/titanic_eda_20260703_2025_fare_log_and_age_binning.ipynb).

---

## Conclusion & Next Steps

Log-transformation and binning proved to be a highly effective combination for boosting tree-based models.
For our next attempt, we will target hyperparameter tuning (using Optuna) and model ensembling (stacking/blending) to push the score even further.
