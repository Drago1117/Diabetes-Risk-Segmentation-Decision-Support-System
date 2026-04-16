# Import required libraries
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Paths
# Load dataset into DataFrame
data_path = 'data/Diabetes_and_Lifestyle_Dataset_.csv'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(df.columns.tolist())

# Risk Classification (Person 3)
# Features that may cause data leakage (contain target-related info)
leakage_feats = ['diabetes_stage', 'diabetes_risk_score', 'glucose_fasting', 'glucose_postprandial', 'hba1c', 'insulin_level']

# Remove leakage features from input features
X_class = df.drop(leakage_feats, axis=1)

# Convert categorical variables into numeric (one-hot encoding)
X_class = pd.get_dummies(X_class, drop_first=True)

# Target variable (what we want to predict)
y_class = df['diabetes_stage']

# Split data into training and testing sets 
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Encode target labels into numeric values
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_c)
y_test_enc = le.transform(y_test_c)

# Train models
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_c, y_train_c)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_c, y_train_c)

xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_c, y_train_enc)

# Save best (XGB)
joblib.dump(xgb, 'models/xgb_classifier.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(X_class.columns.tolist(), 'models/class_features.pkl')
print("Classification models saved.")

# Segmentation (Person 4)
df_clean = df.dropna()
X_seg = df_clean.drop('diabetes_stage', axis=1)
X_seg = pd.get_dummies(X_seg, drop_first=True).astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_seg)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X_seg.columns.tolist(), 'models/seg_features.pkl')
print("Segmentation models saved.")

print("All models trained and saved!")

