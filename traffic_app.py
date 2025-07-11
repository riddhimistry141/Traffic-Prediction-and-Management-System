import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and label encoders
model = joblib.load('traffic_model.pkl')
le_location = joblib.load('le_location.pkl')
le_road_condition = joblib.load('le_road_condition.pkl')
le_weather = joblib.load('le_weather.pkl')
le_day = joblib.load('le_day.pkl')
le_incident = joblib.load('le_incident.pkl')

# Load dataset
df = pd.read_csv("traffic_prediction_dataset_large (1000).csv")

# Convert Timestamp to datetime (adjusted to correct date format)
df['Timestamp'] = pd.to_datetime('2025-07-11 ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
df['Hour'] = df['Timestamp'].dt.hour

# Streamlit app
st.title("Traffic Prediction and Management System")

# Display dataset
st.header("Traffic Dataset")
st.write(df)

# Display visualizations
st.header("Exploratory Data Analysis")
st.subheader("Traffic Volume by Hour")
st.image("traffic_volume_by_hour.png")

st.subheader("Average Speed by Weather")
st.image("speed_by_weather.png")

st.subheader("Traffic Volume by Road Condition")
st.image("traffic_by_road_condition.png")

st.subheader("Traffic Volume by Incident Reported")
st.image("traffic_by_incident.png")

# User input for prediction
st.header("Predict Traffic Volume")
hour = st.slider("Select Hour of Day", 0, 23, 8)
location = st.selectbox("Select Location", le_location.classes_)
road_condition = st.selectbox("Select Road Condition", le_road_condition.classes_)
weather = st.selectbox("Select Weather", le_weather.classes_)
day_of_week = st.selectbox("Select Day of Week", le_day.classes_)
incident_reported = st.selectbox("Incident Reported?", le_incident.classes_)

# Encode inputs
input_data = pd.DataFrame({
    'Hour': [hour],
    'Location': [le_location.transform([location])[0]],
    'Road_Condition': [le_road_condition.transform([road_condition])[0]],
    'Weather': [le_weather.transform([weather])[0]],
    'Day_of_Week': [le_day.transform([day_of_week])[0]],
    'Incident_Reported': [le_incident.transform([incident_reported])[0]]
})

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Traffic Volume: {int(prediction)} vehicles/hour")

# Download dataset
st.header("Download Dataset")
with open("traffic_prediction_dataset_large (1000).csv", "rb") as file:
    st.download_button(
        label="Download traffic_prediction_dataset_large (1000).csv",
        data=file,
        file_name="traffic_prediction_dataset_large (1000).csv",
        mime="text/csv"
    )
