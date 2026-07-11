# Titanic Survival Prediction

An end-to-end machine learning pipeline predicting Titanic passenger survival, built with Python, Pandas, NumPy, Matplotlib, and scikit-learn.

## Overview

This project walks through the complete ML workflow on a real, messy dataset:

1. Loading raw data
2. Exploratory data analysis
3. Handling missing values
4. Encoding categorical variables
5. Feature/target split
6. Train/test split
7. Training a Logistic Regression classifier
8. Evaluating with accuracy, confusion matrix, and classification report
9. Interpreting feature importance

## Dataset

The classic [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — 891 passengers, with features including class, sex, age, fare, and family size.

## Results

- **Accuracy:** ~81.5%
- **Key predictors:** `Sex` and `Pclass` were the strongest drivers of survival, consistent with the historically documented "women and children first" evacuation protocol and the survival advantage of wealthier, first-class passengers.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook titanic_survival_prediction.ipynb
```

## Project Structure

```
.
├── titanic_survival_prediction.ipynb   # Main notebook
├── requirements.txt                     # Dependencies
└── README.md
```

## Possible Next Steps

- Compare against other models (Random Forest, Gradient Boosting)
- Feature engineering (e.g. extracting titles from passenger names)
- Cross-validation for more robust evaluation
