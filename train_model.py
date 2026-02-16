import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Convert smoking_status to numbers
smoking_map = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 3
}

df['smoking_status'] = df['smoking_status'].map(smoking_map)

# Select features
X = df[['age', 'hypertension', 'heart_disease',
        'avg_glucose_level', 'bmi', 'smoking_status']]

y = df['stroke']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
