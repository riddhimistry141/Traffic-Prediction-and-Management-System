import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set base directory to AI Project folder on Desktop
base_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "AI Project")

# Load model and label encoders
model = joblib.load(os.path.join(base_dir, 'traffic_model.pkl'))
le_location = joblib.load(os.path.join(base_dir, 'le_location.pkl'))
le_road_condition = joblib.load(os.path.join(base_dir, 'le_road_condition.pkl'))
le_weather = joblib.load(os.path.join(base_dir, 'le_weather.pkl'))
le_day = joblib.load(os.path.join(base_dir, 'le_day.pkl'))
le_incident = joblib.load(os.path.join(base_dir, 'le_incident.pkl'))

# Load dataset
df = pd.read_csv(os.path.join(base_dir, "traffic_prediction_dataset_large (1000).csv"))

# Convert Timestamp to datetime (adjust format based on your data, e.g., from previous fix)
df['Timestamp'] = pd.to_datetime('2025-07-06 ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
df['Hour'] = df['Timestamp'].dt.hour

# Streamlit app
st.title("Traffic Prediction and Management System")

# Display dataset
st.header("Traffic Dataset")
st.write(df)

# Display visualizations
st.header("Exploratory Data Analysis")
st.subheader("Traffic Volume by Hour")
st.image(os.path.join(base_dir, "traffic_volume_by_hour.png"))

st.subheader("Average Speed by Weather")
st.image(os.path.join(base_dir, "speed_by_weather.png"))

st.subheader("Traffic Volume by Road Condition")
st.image(os.path.join(base_dir, "traffic_by_road_condition.png"))

st.subheader("Traffic Volume by Incident Reported")
st.image(os.path.join(base_dir, "traffic_by_incident.png"))

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
with open(os.path.join(base_dir, "traffic_prediction_dataset_large (1000).csv"), "rb") as file:
    st.download_button(
        label="Download traffic_prediction_dataset_large (1000).csv",
        data=file,
        file_name="traffic_prediction_dataset_large (1000).csv",
        mime="text/csv"
    )