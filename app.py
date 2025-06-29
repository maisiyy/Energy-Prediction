import streamlit as st
import pickle
import joblib
import numpy as np

# Load the trained Random Forest model using joblib
rf_model = joblib.load("rf_model_compressed.pkl")

# Load the scaler using pickle
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the top 10 features used for prediction with descriptions
top_rf_features = {
    'RH_8': 'Humidity in the teenager room 2 (%)',
    'lights': 'Energy consumption of light fixtures (Wh)',
    'RH_out': 'Humidity outside the building (%)',
    'T2': 'Temperature in the living room (°C)',
    'RH_9': 'Humidity in the parents\' room (%)',
    'RH_6': 'Humidity outside the building (north side) (%)',
    'RH_5': 'Humidity in the bathroom (%)',
    'RH_1': 'Humidity in the kitchen area (%)',
    'T8': 'Temperature in the teenager room 2 (°C)',
    'Press_mm_hg': 'Atmospheric pressure (mm Hg)'
}

# Streamlit app interface
st.title("🏠 Smart Home Energy Consumption Predictor")
st.write("Enter values for the following features to predict whether the energy consumption is High or Low:")

# Create input fields for each feature
user_input = []
for feature, description in top_rf_features.items():
    value = st.number_input(f"{feature}", value=0.0, help=description)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    # Convert input to numpy array and reshape
    input_array = np.array(user_input).reshape(1, -1)

    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = rf_model.predict(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.success("Predicted Energy Consumption: High 🔥")
    else:
        st.success("Predicted Energy Consumption: Low ❄️")

# Explain influencing factors
st.subheader("📈 Key Influencing Factors")
st.markdown("""
This prediction is based on the following top 10 features that influence energy consumption in a smart home:

- **RH_8**: Humidity in the teenager room 2 (%)
- **lights**: Energy consumption of light fixtures (Wh)
- **RH_out**: Humidity outside the building (%)
- **T2**: Temperature in the living room (°C)
- **RH_9**: Humidity in the parents' room (%)
- **RH_6**: Humidity outside the building (north side) (%)
- **RH_5**: Humidity in the bathroom (%)
- **RH_1**: Humidity in the kitchen area (%)
- **T8**: Temperature in the teenager room 2 (°C)
- **Press_mm_hg**: Atmospheric pressure (mm Hg)

These features were selected based on their statistical importance in predicting whether the household's energy consumption is high or low.
""")
# Footer
st.markdown("---")
st.caption("""
*Note: Predictions are based on median energy consumption thresholds.*  
*Model trained on the UCI Appliances Energy Prediction Dataset.*
""")
