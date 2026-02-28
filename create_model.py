import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a dummy dataset
X = np.array([
    [25, 0, 0, 85, 22, 0],
    [70, 1, 1, 245, 32, 2],
    [50, 0, 1, 180, 28, 1],
    [40, 1, 0, 150, 30, 0]
])
y = np.array([0, 1, 1, 0])  # 0 = Low Risk, 1 = High Risk

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dummy model created and saved as stroke_model.pkl")