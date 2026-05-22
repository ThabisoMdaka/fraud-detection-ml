import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("   FRAUD DETECTION — Data Preprocessing Pipeline")
print("=" * 60)

# ===============================
# 1. LOAD RAW DATA
# ===============================
print("\n--- Loading Raw Data ---")
df = pd.read_csv('data/raw/creditcard.csv')
print(f"Raw dataset shape: {df.shape}")

# ===============================
# 2. FEATURE SCALING
# ===============================
print("\n--- Scaling Features ---")

# Amount and Time are the only non-PCA features
# They need scaling to match the V1-V28 PCA scale
scaler_amount = StandardScaler()
scaler_time   = StandardScaler()

df['Amount_Scaled'] = scaler_amount.fit_transform(
    df[['Amount']]
)
df['Time_Scaled'] = scaler_time.fit_transform(
    df[['Time']]
)

# Drop original unscaled columns
df.drop(columns=['Amount', 'Time'], inplace=True)

print("Amount scaled: mean=0, std=1")
print("Time scaled:   mean=0, std=1")
print(f"Dataset shape after scaling: {df.shape}")

# ===============================
# 3. FEATURE / TARGET SPLIT
# ===============================
X = df.drop(columns=['Class'])
y = df['Class']

print(f"\nFeatures (X): {X.shape}")
print(f"Target   (y): {y.shape}")
print(f"Fraud rate:   {y.mean()*100:.3f}%")

# ===============================
# 4. STRATIFIED TRAIN/TEST SPLIT
# ===============================
print("\n--- Splitting Data ---")
print("Strategy: Stratified split — preserves fraud ratio")
print("Split:    80% Training | 20% Testing")
print("CRITICAL: Split BEFORE any oversampling to prevent")
print("          data leakage into the test set")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set:  {len(X_train):,} samples")
print(f"  Fraud:       {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"  Legitimate:  {(y_train==0).sum():,}")
print(f"\nTest set:      {len(X_test):,} samples")
print(f"  Fraud:       {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
print(f"  Legitimate:  {(y_test==0).sum():,}")

# ===============================
# 5. SAVE PROCESSED DATA
# ===============================
print("\n--- Saving Processed Data ---")

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv',  index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv',  index=False)

# Save scalers for use in prediction pipeline
joblib.dump(scaler_amount, 'models/scaler_amount.pkl')
joblib.dump(scaler_time,   'models/scaler_time.pkl')

feature_names = list(X.columns)
joblib.dump(feature_names, 'models/feature_names.pkl')

print("Saved: data/processed/X_train.csv")
print("Saved: data/processed/X_test.csv")
print("Saved: data/processed/y_train.csv")
print("Saved: data/processed/y_test.csv")
print("Saved: models/scaler_amount.pkl")
print("Saved: models/scaler_time.pkl")
print("Saved: models/feature_names.pkl")

print("\n" + "="*60)
print("   PREPROCESSING COMPLETE")
print("="*60)
print(f"  Total features: {X.shape[1]}")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples:     {len(X_test):,}")
print("="*60)