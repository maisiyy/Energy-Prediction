# 🏙️ Machine Learning-Based Prediction of Household Energy Consumption for Smart Cities

This project applies machine learning techniques to classify household energy consumption as **High** or **Low**, supporting smarter and more sustainable energy management in smart cities. Using environmental and indoor sensor data from the UCI Appliances Energy Prediction dataset, the system predicts energy usage and identifies key influencing factors. The final model is deployed as an interactive web application accessible across devices.

## 📘 Project Overview

Energy efficiency is a major concern in modern smart cities. As urban populations grow, optimizing energy consumption becomes essential to reduce environmental impact and operational costs. Machine learning enables data-driven insights that help predict energy usage and support intelligent energy management.

This project uses the **Appliances Energy Prediction** dataset to classify energy consumption into **High** or **Low**, based on 28 environmental and indoor features such as temperature, humidity, windspeed, visibility, and atmospheric pressure.

---

## 📦 Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Records:** 19,735  
- **Features:** 28  
- **Target Variable:** `appliances` (converted into binary: High/Low using median threshold)

### Key Feature Categories
- **Indoor Conditions:**  
  Temperatures (T1–T9), Humidity levels (RH_1–RH_9)
- **Outdoor Weather:**  
  `T_out`, `RH_out`, `Windspeed`, `Visibility`, `Press_mm_hg`, `Tdewpoint`
- **Other Variables:**  
  `lights`, `rv1`, `rv2`
- **Timestamp:**  
  `date` (removed during preprocessing)

✅ **No missing values** were found in the dataset, simplifying preprocessing.

---

## 📊 Exploratory Data Analysis (EDA)

EDA revealed:

- Strong correlations between indoor temperature/humidity and appliance energy usage  
- Time-based patterns influencing consumption  
- Outdoor weather conditions affecting indoor climate and energy demand  
- Random variables (`rv1`, `rv2`) showing no meaningful real-world relevance

Visualizations included correlation heatmaps, distribution plots, and feature importance graphs.

---

## 🧹 Data Preprocessing

Steps included:

- Dropping irrelevant columns (`date`)
- Converting continuous target into binary classes:
  - **High** = values above median  
  - **Low** = values below or equal to median
- Splitting dataset into **80% training** and **20% testing**
- Scaling features using **StandardScaler**
- Saving processed dataset as `energydata_updated.csv`

---

## 🤖 Machine Learning Models

Two classification models were developed and compared:

### 1. **Logistic Regression**
- Linear model for binary classification  
- Interpretable coefficients  
- Efficient and simple  
- Works best when relationships are linear  
- Feature selection: **Recursive Feature Elimination (RFE)**

### 2. **Random Forest Classifier**
- Ensemble of decision trees  
- Captures non-linear relationships  
- Robust to noise and overfitting  
- Provides built-in **feature importance**  
- Feature selection: **Top features based on importance scores**

---

## ✅ Model Performance Evaluation

Confusion matrices and metrics were computed for both models.

### **Logistic Regression (Top 10 Features)**

| Metric | Score |
|--------|--------|
| Accuracy | 0.7454 |
| Precision | 0.7365 |
| Recall | 0.6973 |
| F1 Score | 0.7163 |

---

### **Random Forest Classifier (Top 10 Features)**

| Metric | Score |
|--------|--------|
| Accuracy | 0.8946 |
| Precision | 0.8926 |
| Recall | 0.8769 |
| F1 Score | 0.8847 |

✅ **Random Forest significantly outperformed Logistic Regression** in all metrics.

---

## 🏆 Model Selection

Based on accuracy, precision, recall, and F1-score, the **Random Forest Classifier** was selected for deployment. It demonstrated:

- Higher predictive accuracy  
- Better handling of non-linear relationships  
- Lower false positives and false negatives  
- Stronger generalization performance  

---

## 🚀 Deployment

The final model was deployed using **Streamlit Cloud**.

### Deployment Components
- `rf_model_compressed.pkl` — compressed Random Forest model  
- `scaler.pkl` — StandardScaler object  
- `app.py` — Streamlit application  
- `requirements.txt` — dependencies  
- `energydata_updated.csv` — processed dataset  

### Features of the Web App
- User-friendly interface  
- Accepts environmental inputs  
- Predicts **High** or **Low** energy consumption  
- Accessible on:
  - ✅ Mobile phones  
  - ✅ Laptops  
  - ✅ Desktop PCs  

---

## 📁 Repository Structure

