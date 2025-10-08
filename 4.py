import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define dataset paths
CROP_DATASET = "/Users/nanthinik/Desktop/crop and fertilization recommendation system/dataset/Crop_recommendation.csv"
FERTILIZER_DATASET = "/Users/nanthinik/Desktop/crop and fertilization recommendation system/dataset/Fertilizer Prediction.csv"

# Load datasets
crop_df = pd.read_csv(CROP_DATASET)
fertilizer_df = pd.read_csv(FERTILIZER_DATASET)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Select only the required features
X = data[['N', 'P', 'K', 'temp', 'hum', 'ph', 'rain']]
y = data['Crop Type']  # Target column

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(crop_model, 'models/crop_model.pkl')
joblib.dump(scaler, 'models/crop_scaler.pkl')

print("✅ Crop Recommendation Model Saved Successfully")
# ======= Train Crop Recommendation Model =======
X_crop = crop_df.drop(columns=["label"])  # Drop target column
y_crop = crop_df["label"]

# Split the data
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Normalize numerical features
crop_scaler = StandardScaler()
X_crop_train = crop_scaler.fit_transform(X_crop_train)
X_crop_test = crop_scaler.transform(X_crop_test)

# Train RandomForest model for crop recommendation
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop_train, y_crop_train)

# Save the crop recommendation model and scaler
with open("crop_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

with open("crop_scaler.pkl", "wb") as f:
    pickle.dump(crop_scaler, f)

print("✅ Crop Recommendation Model Saved Successfully!")

# ======= Train Fertilizer Prediction Model =======
# Trim whitespace from column names
fertilizer_df.columns = fertilizer_df.columns.str.strip()

# Ensure correct target column
TARGET_COLUMN = "Fertilizer Name"
if TARGET_COLUMN not in fertilizer_df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in dataset!")

# Separate features and target
X_fert = fertilizer_df.drop(columns=[TARGET_COLUMN])
y_fert = fertilizer_df[TARGET_COLUMN]

# Identify categorical columns
categorical_cols = ["Soil Type", "Crop Type"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(X_fert[categorical_cols])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Merge encoded features with numerical data
fertilizer_data = pd.concat([encoded_df, X_fert.drop(columns=categorical_cols)], axis=1)

# Split the data
X_fert_train, X_fert_test, y_fert_train, y_fert_test = train_test_split(fertilizer_data, y_fert, test_size=0.2, random_state=42)

# Normalize numerical features
fert_scaler = StandardScaler()
X_fert_train = fert_scaler.fit_transform(X_fert_train)
X_fert_test = fert_scaler.transform(X_fert_test)

# Train RandomForest model for fertilizer prediction
fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fert_train, y_fert_train)

# Save the fertilizer model, scaler, and encoder
with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(fertilizer_model, f)

with open("fertilizer_scaler.pkl", "wb") as f:
    pickle.dump(fert_scaler, f)

with open("fertilizer_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Fertilizer Prediction Model Saved Successfully!")
