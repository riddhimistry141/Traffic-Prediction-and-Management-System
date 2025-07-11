import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set base directory to project folder under Desktop
base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "project")

# Load the dataset
df = pd.read_csv(os.path.join(base_dir, "traffic_prediction_dataset_large (1000).csv"))

# Convert Timestamp to datetime with specified format (adjust format as needed)
df['Timestamp'] = pd.to_datetime('2025-07-11 ' + df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

df['Hour'] = df['Timestamp'].dt.hour

# Exploratory Data Analysis
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Traffic Volume by Hour
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Traffic_Volume', data=df, marker='o')
plt.title('Traffic Volume by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Traffic Volume (vehicles/hour)')
plt.savefig(os.path.join(base_dir, 'traffic_volume_by_hour.png'))
plt.close()

# Average Speed by Weather
plt.figure(figsize=(10, 6))
sns.boxplot(x='Weather', y='Average_Speed_kmph', data=df)
plt.title('Average Speed by Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Average Speed (kmph)')
plt.savefig(os.path.join(base_dir, 'speed_by_weather.png'))
plt.close()

# Traffic Volume by Road Condition
plt.figure(figsize=(10, 6))
sns.barplot(x='Road_Condition', y='Traffic_Volume', data=df)
plt.title('Traffic Volume by Road Condition')
plt.xlabel('Road Condition')
plt.ylabel('Traffic Volume (vehicles/hour)')
plt.savefig(os.path.join(base_dir, 'traffic_by_road_condition.png'))
plt.close()

# Incident Impact on Traffic Volume
plt.figure(figsize=(10, 6))
sns.boxplot(x='Incident_Reported', y='Traffic_Volume', data=df)
plt.title('Traffic Volume by Incident Reported')
plt.xlabel('Incident Reported')
plt.ylabel('Traffic Volume (vehicles/hour)')
plt.savefig(os.path.join(base_dir, 'traffic_by_incident.png'))
plt.close()

print("EDA complete. Visualizations saved as PNG files.")