import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set base directory to AI Project folder under OneDrive Desktop
base_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "AI Project")

# Load the dataset
df = pd.read_csv(os.path.join(base_dir, "traffic_prediction_dataset_large (1000).csv"))

# Convert Timestamp to datetime with time-only format, using current date as base
df['Timestamp'] = pd.to_datetime('2025-07-06 ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

df['Hour'] = df['Timestamp'].dt.hour

# Encode categorical variables
le_location = LabelEncoder()
le_road_condition = LabelEncoder()
le_weather = LabelEncoder()
le_day = LabelEncoder()
le_incident = LabelEncoder()

df['Location'] = le_location.fit_transform(df['Location'])
df['Road_Condition'] = le_road_condition.fit_transform(df['Road_Condition'])
df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Day_of_Week'] = le_day.fit_transform(df['Day_of_Week'])
df['Incident_Reported'] = le_incident.fit_transform(df['Incident_Reported'])

# Save label encoders
joblib.dump(le_location, os.path.join(base_dir, 'le_location.pkl'))
joblib.dump(le_road_condition, os.path.join(base_dir, 'le_road_condition.pkl'))
joblib.dump(le_weather, os.path.join(base_dir, 'le_weather.pkl'))
joblib.dump(le_day, os.path.join(base_dir, 'le_day.pkl'))
joblib.dump(le_incident, os.path.join(base_dir, 'le_incident.pkl'))

# Features and target
X = df[['Hour', 'Location', 'Road_Condition', 'Weather', 'Day_of_Week', 'Incident_Reported']]
y = df['Traffic_Volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R^2 Score: {train_score:.2f}")
print(f"Testing R^2 Score: {test_score:.2f}")

# Save the model
joblib.dump(model, os.path.join(base_dir, 'traffic_model.pkl'))
print("Model and label encoders saved.")