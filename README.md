# 🏙️ Machine Learning-Based Prediction of Household Energy Consumption for Smart Cities

This project uses machine learning to predict household appliance energy consumption based on indoor sensor data and outdoor weather conditions. Designed for smart city applications, it supports energy optimization, sustainability, and intelligent resource management.

## 📦 Dataset

- **Source**: UCI Machine Learning Repository – [Appliances Energy Prediction](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
- **Target Variable**: `appliances` (energy consumption in Wh)
- **Features**:
  - Indoor temperature and humidity (`T1–T9`, `RH_1–RH_9`)
  - Outdoor weather (`T_out`, `RH_out`, `Windspeed`, `Visibility`, `Press_mm_hg`, `Tdewpoint`)
  - Time-based patterns (`date`)
  - Random variables (`rv1`, `rv2`)

## 📊 Exploratory Data Analysis

- Visualized feature distributions and correlations
- Identified key environmental drivers of energy consumption
- Handled missing values and scaled features for modeling

## 🤖 Machine Learning Models

Two algorithms were implemented and compared:

### 1. Random Forest Regressor
- Captures non-linear relationships
- Provides feature importance
- Robust to overfitting

### 2. Linear Regression
- Simple and interpretable
- Serves as a baseline model

## 📈 Evaluation Metrics

- **Regression**: RMSE, MAE, R²
- **Classification (if discretized)**: Confusion Matrix

## 🧪 Feature Selection

- **Random Forest**: Gini importance
- **Linear Regression**: Correlation and p-values

## 🚀 Deployment

- **App**: Built with Streamlit (`app.py`)
- **Model**: Pre-trained Random Forest (`rf_model_compressed.pkl`)
- **Scaler**: StandardScaler object (`scaler.pkl`)
- **Dependencies**: Listed in `requirements.txt`

## ✅ Outcome

Random Forest was selected for deployment due to its superior performance and robustness. The model enables smart homes to predict and manage energy usage more efficiently.

---
