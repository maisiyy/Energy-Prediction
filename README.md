🏙️ Machine Learning-Based Prediction of Household Energy Consumption for Smart Cities
This project leverages machine learning to predict household appliance energy usage using environmental and time-based variables. Built for smart city applications, it aims to optimize energy efficiency, reduce waste, and support sustainable urban living.

🔍 Project Overview
Goal: Predict household appliance energy consumption using sensor and weather data.

Dataset: UCI Appliances Energy Prediction — includes indoor temperature/humidity readings, outdoor weather conditions, and timestamped energy usage.

Target Variable: appliances (energy consumption in Wh)

📊 Key Features
Indoor climate data: T1–T9, RH_1–RH_9

Outdoor weather: T_out, RH_out, Windspeed, Visibility, Press_mm_hg, Tdewpoint

Time-based patterns: date

Random variables: rv1, rv2 (for simulation purposes)

🧠 Machine Learning Workflow
Exploratory Data Analysis: Visualize trends, correlations, and feature distributions.

Preprocessing: Handle missing values, scale features, and engineer time-based variables.

Modeling: Compare two algorithms:

Random Forest Regressor: Robust to non-linear relationships and feature importance analysis.

Linear Regression: Interpretable baseline model for comparison.

Evaluation: Metrics include RMSE, MAE, and R². Confusion matrix used for classification-based evaluation (if discretized).

Feature Selection:

RF: Based on Gini importance

LR: Based on correlation and p-values

🚀 Deployment
Streamlit app (app.py) for interactive prediction

Pre-trained model (rf_model_compressed.pkl) and scaler (scaler.pkl) included

Requirements listed in requirements.txt

✅ Outcomes
Identified key environmental drivers of energy consumption

Demonstrated model performance and interpretability

Proposed Random Forest as the preferred model for deployment due to higher accuracy and robustness
