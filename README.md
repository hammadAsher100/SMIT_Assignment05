# 🧠 Stroke Prediction System using Artificial Neural Network (ANN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green.svg)](https://scikit-learn.org/)

## 📋 Overview

This project implements a **Stroke Prediction System** using an Artificial Neural Network (ANN) to predict the likelihood of a patient having a stroke based on health and demographic factors.

**Dataset:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 samples, 12 features)

## 🎯 Objective

Develop a binary classification model that predicts stroke risk with comprehensive data preprocessing, ANN training, and multi-metric evaluation.

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Float | Patient age in years |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `avg_glucose_level` | Float | Average blood glucose level |
| `bmi` | Float | Body Mass Index (201 missing values) |
| `gender` | Categorical | Male, Female, Other |
| `ever_married` | Categorical | Yes, No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban, Rural |
| `smoking_status` | Categorical | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | **Target** | 0 = No Stroke, 1 = Stroke (5% positive) |

## 🛠️ Technologies Used

```python
- Python 3.8+
- TensorFlow 2.x / Keras
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Joblib
