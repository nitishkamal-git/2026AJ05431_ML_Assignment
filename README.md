# ML_Assignment
# Bits ID: 2026AJ05431
# Diabetes Readmission Prediction Using Machine Learning

## 1. Project Overview

This project implements and compares five machine learning classification algorithms to predict hospital readmission outcomes for diabetic patients.

The objective is to evaluate different classification techniques and identify their performance using multiple evaluation metrics.

The project also includes an interactive Streamlit application that allows users to upload a test dataset, select a trained machine learning model, and view its evaluation results.

## Problem Statement

Hospital readmission increases treatment costs and can indicate gaps in patient care. The objective of this project is to build and compare machine learning classifiers that predict a diabetic patient's hospital readmission category from clinical and administrative encounter data.

The prediction is a multiclass classification problem with the following outcomes:

- `<30`: readmitted within 30 days
- `>30`: readmitted after 30 days
- `NO`: not readmitted

The model should predict the correct outcome from patient encounter features. Its performance is evaluated on a held-out patient-level test dataset without using the training records for assessment.

## 2. Dataset Description

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Dataset characteristics:

- Total records: 101,766
- Original dataset columns: 50
- Input features used: 44
- Categorical features: 36
- Numerical features: 8
- Target variable: `readmitted`

The original data contains encounter-level records from 130 US hospitals over the period 1999–2008. Each row represents a hospital encounter, while `patient_nbr` identifies the patient. Patient-level splitting prevents different encounters belonging to the same patient from appearing in both training and testing data.

The target is `readmitted`, and identifiers plus columns removed during preprocessing are excluded from model predictions.

### Target Classes

| Class | Description |
|---|---|
| NO | No readmission |
| >30 | Readmission after 30 days |
| <30 | Readmission within 30 days |

## 3. Data Preprocessing

The following data-cleaning and preprocessing activities were performed:

- Removed `encounter_id` and `patient_nbr` from model input features.
- Preserved patient identifiers separately for group-based data splitting.
- Removed the `weight` column because of its high percentage of missing values.
- Removed constant columns: `examide` and `citoglipton`.
- Replaced special unknown or unavailable values with `Unknown`.
- Retained useful categorical and numerical features.
- Used median imputation for missing numerical values.
- Used constant-value imputation for missing categorical values.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features for most classifiers.
- Applied MinMaxScaler for the Naive Bayes Classifier - Multinomial variant.

Missing-value imputation and feature transformations were implemented within Scikit-learn pipelines to avoid data leakage.

## 4. Train/Test Split

The dataset was divided into training and testing sets using StratifiedGroupKFold.

| Dataset | Records |
|---|---:|
| Training | 81,187 |
| Testing | 20,579 |
| Total | 101,766 |

A patient-level group split was used to prevent encounters belonging to the same patient from appearing in both training and testing datasets.

The patient overlap between the two datasets was zero.

## 5. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Support Vector Machine

Each classifier was trained using its corresponding preprocessing pipeline.

## 6. Evaluation Metrics

The models were evaluated using six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

Precision, Recall and F1-score use macro averaging to give equal importance to all three target classes.

AUC uses the multiclass One-vs-Rest approach with macro averaging.

For Support Vector Machine, softmax-normalized decision scores were used for AUC calculation because LinearSVC does not directly provide probability estimates. These scores are not calibrated probabilities.

## 7. Model Performance

The following results were obtained on the held-out test dataset.

| Model | Accuracy | Macro AUC (OVR) | Macro Precision | Macro Recall | Macro F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5474 | 0.6450 | 0.4900 | 0.3554 | 0.2809 | 0.0988 |
| Decision Tree Classifier | 0.4627 | 0.5357 | 0.3740 | 0.3745 | 0.3737 | 0.0817 |
| K-Nearest Neighbor Classifier | 0.5024 | 0.5725 | 0.3962 | 0.3910 | 0.3915 | 0.1138 |
| Naive Bayes Classifier - Multinomial | 0.5424 | 0.6172 | 0.4334 | 0.3918 | 0.3784 | 0.1397 |
| Support Vector Machine | 0.5447 | 0.6422 | 0.5077 | 0.3455 | 0.2622 | 0.0736 |

### Performance Observations

- Logistic Regression achieved the highest accuracy (0.5474) and highest macro One-vs-Rest AUC (0.6450).
- Support Vector Machine had the highest macro precision (0.5077), but its macro recall and F1-score were low because it strongly favored the `NO` class.
- Naive Bayes Classifier - Multinomial achieved the highest macro recall (0.3918), while K-Nearest Neighbor Classifier achieved the highest macro F1-score (0.3915).
- Naive Bayes Classifier - Multinomial also had the highest MCC among the five models (0.1397).
- Decision Tree Classifier had the weakest overall ranking, with accuracy 0.4627 and macro AUC 0.5357.
- All models struggled to predict the minority `<30` and `>30` categories compared with the majority `NO` category.

### Overall Winning Model

**Logistic Regression is selected as the overall winning model.** It achieved the highest accuracy (0.5474) and highest macro One-vs-Rest AUC (0.6450) on the held-out test dataset. It performed strongly overall despite class imbalance, while K-Nearest Neighbor achieved the highest macro F1-score and Naive Bayes achieved the highest macro recall. Future improvements should prioritize recall and F1-score for the underrepresented `<30` class.