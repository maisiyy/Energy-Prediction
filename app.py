import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model and scaler using joblib
try:
    rf_model = joblib.load("rf_model_compressed.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Define the top 10 features used for prediction
top_rf_features = ['RH_8', 'lights', 'RH_out', 'T2', 'RH_9', 'RH_6', 'RH_5', 'RH_1', 'T8', 'Press_mm_hg']

# Streamlit app interface
st.title("💡 Energy Consumption Prediction App")
st.write("Enter values for the following features to predict whether the energy consumption is **High or Low**:")

# Create input fields for each feature
user_input = []
for feature in top_rf_features:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert input to numpy array and reshape
        input_array = np.array(user_input).reshape(1, -1)

        # Scale the input
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = rf_model.predict(input_scaled)[0]

        # Display result
        if prediction == 1:
            st.success("🔺 Predicted Energy Consumption: **High**")
        else:
            st.success("🟢 Predicted Energy Consumption: **Low**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
