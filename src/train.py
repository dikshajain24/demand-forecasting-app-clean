# src/train.py
"""
Train a demand forecasting model using LightGBM
- Drops 'store' from features (kept only as ID)
- Encodes categorical columns
- Splits into train/validation
- Saves model + feature list + metrics
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def train():
    # Load features
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df = df.sort_values("date")

    # Target and features
    target = "sales"
    drop_cols = ["date", "sales", "store"]   # 🚨 exclude store from features
    features = [c for c in df.columns if c not in drop_cols]

    # Encode categorical columns to numeric codes
    cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print("Encoding categorical columns:", cat_cols)
        for col in cat_cols:
            df[col] = df[col].astype("category").cat.codes

    # Split train/validation
    last_date = df["date"].max()
    valid_start = last_date - pd.Timedelta(days=90)
    train_df = df[df["date"] < valid_start]
    valid_df = df[df["date"] >= valid_start]

    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]

    print(f"Training rows: {X_train.shape[0]}, Validation rows: {X_valid.shape[0]}")
    print(f"Features used: {len(features)}")

    # Model
    model = LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    rmse = np.sqrt(mse)
    mape_val = mape(y_valid.values, preds)

    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAPE: {mape_val:.2f}%")

    # Save model
    model_file = MODEL_DIR / "lgb_model.pkl"
    joblib.dump(model, model_file)
    print(f"✅ Model saved to {model_file}")

    # Save feature list
    features_file = MODEL_DIR / "features_used.txt"
    with open(features_file, "w") as f:
        for feat in features:
            f.write(f"{feat}\n")
    print(f"✅ Feature list saved to {features_file}")

    # Save metrics
    metrics_file = MODEL_DIR / "metrics.json"
    metrics = {"Validation_RMSE": float(rmse), "Validation_MAPE": float(mape_val)}
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train()
