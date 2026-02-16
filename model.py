import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset (for demo)
data = {
    "age": [25, 45, 30, 50, 23, 60],
    "heart_rate": [72, 95, 80, 110, 70, 120],
    "blood_pressure": [120, 140, 130, 150, 110, 160],
    "stroke": [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["age", "heart_rate", "blood_pressure"]]
y = df["stroke"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

