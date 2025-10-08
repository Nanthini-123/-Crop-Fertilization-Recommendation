import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace with actual dataset path)
X_train = np.array([[45, 43, 56, 20, 32, 34, 202]])  # Example, replace with actual dataset

# Train a new StandardScaler
crop_scaler = StandardScaler()
crop_scaler.fit(X_train)  # Fit scaler on actual training data

# Save the scaler correctly
with open("models/crop_scaler.pkl", "wb") as f:
    pickle.dump(crop_scaler, f)

print("✅ Crop scaler saved successfully!")
