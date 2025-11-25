import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and features
model = joblib.load("electricity_demand_model.pkl")
features = joblib.load("model_features.pkl")

st.title("⚡ Electricity Demand Forecasting App")
st.write("Enter the input values to predict electricity demand in kW.")

# -----------------------
# User input fields
# -----------------------
year = st.number_input("Year", min_value=2000, max_value=2035, value=2000)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
hour_of_day = st.number_input("Hour of Day (0–23)", min_value=0, max_value=23, value=0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
solar_generation = st.number_input("Solar Generation (kW)", min_value=0.0, max_value=5000.0, value=0.0)
is_holiday = st.selectbox("Is Holiday?", [0, 1])
is_weekend = st.selectbox("Is Weekend?", [0, 1])
day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)

# -----------------------
# Prepare input DataFrame
# -----------------------
input_data = pd.DataFrame({
    'day_of_week': [day_of_week],
    'hour_of_day': [hour_of_day],
    'is_weekend': [is_weekend],
    'temperature': [temperature],
    'is_holiday': [is_holiday],
    'solar_generation': [solar_generation],
    'year': [year],
    'month': [month],
    'day': [day],
    'is_month_start': [1 if day == 1 else 0],
    'is_month_end': [1 if day == pd.Period(f"{year}-{month}").days_in_month else 0]
})

# Ensure correct feature order
input_data = input_data[features]

# -----------------------
# Predict button
# -----------------------
if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"⚡ Predicted Electricity Demand: {prediction:.2f} kW")
