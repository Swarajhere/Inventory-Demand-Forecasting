# forecasting.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# ---------------------------
# Load dataset
# ---------------------------
print("Loading dataset...")
df = pd.read_csv("data/train.csv")

# ---------------------------
# Feature Engineering
# ---------------------------
print("Preprocessing dataset...")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Extract time-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# Create lag feature (previous day sales per store-item)
df = df.sort_values(["store", "item", "date"])
df["lag_1"] = df.groupby(["store", "item"])["sales"].shift(1)
df["lag_7"] = df.groupby(["store", "item"])["sales"].shift(7)

# Fill missing lag values with median
df["lag_1"].fillna(df["lag_1"].median(), inplace=True)
df["lag_7"].fillna(df["lag_7"].median(), inplace=True)

# Drop date column
df = df.drop(columns=["date"])

# ---------------------------
# Train-test split
# ---------------------------
X = df.drop(columns=["sales"])
y = df["sales"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# Train Models
# ---------------------------
results = {}

# 1. Random Forest
print("\nTraining Random Forest...")
start = time.time()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
results["RandomForest"] = rf_rmse
print(f"Random Forest RMSE: {rf_rmse:.4f} (time: {time.time() - start:.2f}s)")

# 2. XGBoost
print("\nTraining XGBoost...")
start = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
results["XGBoost"] = xgb_rmse
print(f"XGBoost RMSE: {xgb_rmse:.4f} (time: {time.time() - start:.2f}s)")

# 3. LightGBM
print("\nTraining LightGBM...")
start = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
results["LightGBM"] = lgb_rmse
print(f"LightGBM RMSE: {lgb_rmse:.4f} (time: {time.time() - start:.2f}s)")

# ---------------------------
# Final Results
# ---------------------------
print("\nModel Performance (RMSE):")
for model, score in results.items():
    print(f"{model}: {score:.4f}")