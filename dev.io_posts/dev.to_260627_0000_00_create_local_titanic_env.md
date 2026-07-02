---
title: "Setting Up Kaggle Titanic Environment on a Local PC"
published: true
description: "A step-by-step guide to setting up a local Python environment for the Kaggle Titanic competition, including API configuration and directory structure."
tags: python, kaggle, machinelearning, tutorial
canonical_url: https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env
---

[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle Practice 3: Feature Engineering for Cabin](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)

https://www.kaggle.com/c/titanic

[Available on GitHub](https://github.com/kito2718/KaggleTitanic)

# Abstract
- Set up a local development environment for the Kaggle Titanic competition.

# Introduction
Kaggle Notebooks are convenient, but local development offers faster iteration, better IDE support, and easier version control with Git.
I've been participating in the Kaggle Titanic competition and trying to improve my score. However, logging into Kaggle every time I wanted to submit an experiment quickly became tedious.

# Environment Summary
Surprisingly, all I really needed was Python.

| Item | Value |
|------|------|
| Python | 3.14.6 |
| Virtual Env | `.venv` (venv) |
| Project Path | `51_googleantigravity\1st_` |
| pip | 26.1.x |

---

# Pitfalls/Things to Keep in Mind
I ran into a small issue during setup, so I'll document it here.

> **Crucial Point:**
> - **Avoid using non-ASCII characters (such as Japanese characters) in the virtual environment path.**
> C extension libraries (DLLs) such as `scipy` and `scikit-learn` failed to load properly when the path contained non-ASCII characters.

# 1. Step-by-Step Guide
## 1.1. Installing Python

Download and install from the [Official Python Release Page](https://www.python.org/downloads/release/python-3146/).
![](https://static.zenn.studio/user-upload/1d1950b8d6ca-20260627.png =500x)
*Scroll down to the bottom of the page to find the installer.*

Also, add Python to your PATH:
```batch
setx PATH "%PATH%;<Python_Installation_Folder>"
```

## 1.2. Creating the Project Folder
Create it anywhere you like on your PC.
```batch
mkdir 51_googleantigravity\1st_
```

## 1.3. Creating the Virtual Environment
Create a Python virtual environment:
```batch
# Move to the project folder
cd 51_googleantigravity\1st_
python.exe -m venv .venv
```
Structure after creation:
```tree
1st_/
└── .venv/
    ├── Scripts/
    │   ├── activate
    │   ├── python.exe
    │   └── pip.exe
    └── Lib/
```

## 1.4. Installing Packages
Install various packages required for the Python code:
```batch
python.exe -m pip install --upgrade pip pandas numpy scikit-learn matplotlib seaborn jupyter kaggle xgboost lightgbm
```

Major packages installed:

| Package | Version | Purpose |
|-----------|-----------|------|
| pandas | 3.0.3 | Data manipulation |
| numpy | 2.5.0 | Numerical calculations |
| scikit-learn | 1.9.0 | Machine learning |
| matplotlib | 3.11.0 | Data visualization |
| seaborn | 0.13.2 | Statistical data visualization |
| xgboost | 3.3.0 | Gradient boosting |
| lightgbm | 4.6.0 | Gradient boosting |
| jupyter / notebook | 7.6.0 | Jupyter Notebook environment |
| kaggle | 2.2.3 | Kaggle CLI |

## 1.5. Directory Structure
Create directories for datasets, notebooks, and outputs:
```batch
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir submissions
```
Organized folder structure:
```plaintext
1st_/
├── .venv/
├── data/
│   ├── raw/          ← Kaggle raw data
│   └── processed/    ← Visualizations and preprocessed data
├── notebooks/        ← Notebook files
└── submissions/      ← Submission CSV files
```

## 1.6. Creating requirements.txt
```batch
python.exe -m pip freeze > requirements.txt
```

## 1.7. Creating the Notebook
Create `notebooks/titanic_eda.ipynb` with the following structure:

| Cell | Content |
|------|------|
| 1 | Import libraries |
| 2 | Load data (train.csv / test.csv) |
| 3 | Exploratory Data Analysis (missing values, survival rate, age distribution, correlation heatmap) |
| 4 | Feature Engineering (extracting titles, imputing age, family size, etc.) |
| 5 | Model training |
| 6 | Generate submission file (output to submissions/submission.csv) |


## 1.8. Launching Jupyter Notebook
Start the notebook server:
```batch
# Move to project directory
cd 51_googleantigravity\1st_
# Activate virtual environment
.venv\Scripts\activate
# Start notebook
jupyter notebook notebooks\titanic_eda.ipynb
```
Your browser should open automatically. If not, copy the URL displayed in the terminal and paste it into your browser.

## 1.9. Preparing Kaggle CLI
1. Go to [Kaggle Settings](https://www.kaggle.com/settings) -> "API Tokens" tab.
2. Click "Create Legacy API Key" under the Legacy API Credentials section.
3. Place the downloaded `kaggle.json` file at `C:\Users\<your_username>\.kaggle\kaggle.json`.
4. Run the following:

```batch
# Activate virtual environment
.venv\Scripts\activate
# Download Titanic data using Kaggle API
kaggle competitions download -c titanic -p data\raw
# Extract ZIP (using standard Windows tar command)
tar -xf data\raw\titanic.zip -C data\raw
```

Alternatively, you can download files manually:
> Download the following 3 files from https://www.kaggle.com/c/titanic/data and place them in `data\raw\`:
> - `train.csv`
> - `test.csv`
> - `gender_submission.csv`

## 1.10. Submitting to Kaggle
```batch
kaggle competitions submit -c titanic -f submissions\submission.csv -m "Your comment"
```
That's it! You now have a fully functional local environment for experimenting with Kaggle Titanic and submitting results directly from your machine.

Japanese version:
[Kaggle Practice 1: Setting up Kaggle Titanic Environment on a Local PC](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle Practice 2: First Submission](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle Practice 3: Feature Engineering for Cabin](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
