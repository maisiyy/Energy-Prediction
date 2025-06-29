import streamlit as st
import pickle
import joblib
import numpy as np

# Load the trained Random Forest model using joblib
rf_model = joblib.load("rf_model_compressed.pkl")

# Load the scaler using pickle
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the top 10 features used for prediction with user-friendly labels
top_rf_features = {
    'RH_8': 'Humidity in Children Room (%)',
    'lights': 'Lighting Energy Consumption (Wh)',
    'RH_out': 'Outdoor Humidity (%)',
    'T2': 'Living Room Temperature (°C)',
    'RH_9': 'Parents\' Room Humidity (%)',
    'RH_6': 'North Side Outdoor Humidity (%)',
    'RH_5': 'Bathroom Humidity (%)',
    'RH_1': 'Kitchen Humidity (%)',
    'T8': 'Children Room Temperature (°C)',
    'Press_mm_hg': 'Atmospheric Pressure (mm Hg)'
}

# Streamlit app interface
st.title("🏠 Smart Home Energy Consumption Predictor")
st.write("Enter values for the following features to predict whether the energy consumption is High or Low:")

# Create two columns for input fields
col1, col2 = st.columns(2)

# Split features into two groups (5 each)
features_list = list(top_rf_features.items())
left_features = features_list[:5]
right_features = features_list[5:]

user_input = []

# Left column inputs
with col1:
    for feature, label in left_features:
        value = st.number_input(f"{label}", value=0.0, key=f"left_{feature}")
        user_input.append(value)

# Right column inputs
with col2:
    for feature, label in right_features:
        value = st.number_input(f"{label}", value=0.0, key=f"right_{feature}")
        user_input.append(value)

# Predict button (centered)
st.write("")  # Add some space
if st.button("Predict", type="primary"):
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

- **Humidity in Children Room (RH_8)**: Humidity in the teenager room 
- **Lighting Energy Consumption (lights)**: Energy used by light fixtures
- **Outdoor Humidity (RH_out)**: Humidity outside the building
- **Living Room Temperature (T2)**: Temperature in the living room
- **Parents' Room Humidity (RH_9)**: Humidity in the parents' room
- **North Side Outdoor Humidity (RH_6)**: Humidity outside the building (north side)
- **Bathroom Humidity (RH_5)**: Humidity in the bathroom
- **Kitchen Humidity (RH_1)**: Humidity in the kitchen area
- **Children Room Temperature (T8)**: Temperature in the teenager room 
- **Atmospheric Pressure (Press_mm_hg)**: Pressure from the building

These features were selected based on their statistical importance in predicting whether the household's energy consumption is high or low.
""")

# Footer
st.markdown("---")
st.caption("""
*Note: Predictions are based on median energy consumption thresholds.*  
*Model trained on the UCI Appliances Energy Prediction Dataset.*
""")
