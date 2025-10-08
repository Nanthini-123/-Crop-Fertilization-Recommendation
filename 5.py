import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define dataset paths
CROP_DATASET = "dataset/Crop_recommendation.csv"
FERTILIZER_DATASET = "dataset/Fertilizer Prediction.csv"

# ======= Train Crop Recommendation Model =======
print("🚀 Loading Crop Dataset...")
crop_df = pd.read_csv(CROP_DATASET)

# Ensure column consistency
CROP_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COLUMN = 'label'  

if TARGET_COLUMN not in crop_df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in dataset!")

# Extract features and target
X_crop = crop_df[CROP_FEATURES]
y_crop = crop_df[TARGET_COLUMN]

# Split the data
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Normalize numerical features
crop_scaler = StandardScaler()
X_crop_train = crop_scaler.fit_transform(X_crop_train)
X_crop_test = crop_scaler.transform(X_crop_test)

# Train RandomForest model for crop recommendation
print("🌱 Training Crop Model...")
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop_train, y_crop_train)

# Save the crop recommendation model and scaler
joblib.dump(crop_model, "models/crop_model.pkl")
joblib.dump(crop_scaler, "models/crop_scaler.pkl")

print("✅ Crop Recommendation Model Saved Successfully!")

# ======= Train Fertilizer Prediction Model =======
print("\n🚜 Loading Fertilizer Dataset...")
fertilizer_df = pd.read_csv(FERTILIZER_DATASET)

# Trim whitespace from column names
fertilizer_df.columns = fertilizer_df.columns.str.strip()

# Ensure correct target column
TARGET_COLUMN = "Fertilizer Name"
if TARGET_COLUMN not in fertilizer_df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in dataset!")

# Extract features and target
X_fert = fertilizer_df.drop(columns=[TARGET_COLUMN])
y_fert = fertilizer_df[TARGET_COLUMN]

# Identify categorical columns
CATEGORICAL_FEATURES = ["Soil Type", "Crop Type"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(X_fert[CATEGORICAL_FEATURES])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))

# Merge encoded features with numerical data
X_fert = pd.concat([encoded_df, X_fert.drop(columns=CATEGORICAL_FEATURES)], axis=1)

# Split the data
X_fert_train, X_fert_test, y_fert_train, y_fert_test = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)

# Normalize numerical features
fert_scaler = StandardScaler()
X_fert_train = fert_scaler.fit_transform(X_fert_train)
X_fert_test = fert_scaler.transform(X_fert_test)

# Train RandomForest model for fertilizer prediction
print("🧪 Training Fertilizer Model...")
fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fert_train, y_fert_train)

# Save the fertilizer model, scaler, and encoder
joblib.dump(fertilizer_model, "models/fertilizer_model.pkl")
joblib.dump(fert_scaler, "models/fertilizer_scaler.pkl")
joblib.dump(encoder, "models/fertilizer_encoder.pkl")

print("✅ Fertilizer Prediction Model Saved Successfully!")
