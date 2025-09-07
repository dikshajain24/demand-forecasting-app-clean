# src/predict.py
"""
Predict next N days per store using the saved LightGBM model
- Loads models/lgb_model.pkl and models/features_used.txt
- Loads last snapshot of features from data/processed/features.parquet
- Ensures prediction DataFrame matches training feature set
- Iteratively predicts next N days (default=7)
- Saves results to models/next_Nday_preds.csv
"""
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path

PROCESSED = Path("data/processed")
MODEL_FILE = Path("models/lgb_model.pkl")
FEATURES_FILE = Path("models/features_used.txt")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_object_cols(df, feature_cols):
    obj_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("category").cat.codes
    return df

def predict_next_n_days(n_days=7):
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model file not found. Run train.py first.")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError("Feature list file not found. Run train.py again.")

    # Load model + features
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, "r") as f:
        trained_features = [line.strip() for line in f.readlines()]

    # Load dataset
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df = df.sort_values(["store", "date"])

    # Last row per store
    last = df.groupby("store").tail(1).set_index("store")

    # Exclude 'store' from features (identifier only)
    trained_features = [f for f in trained_features if f != "store"]

    # Encode objects
    last = encode_object_cols(last, trained_features)

    # Ensure all features match training
    for col in trained_features:
        if col not in last.columns:
            last[col] = 0
    X = last[trained_features]

    # Predict iteratively
    preds_matrix = np.zeros((len(last), n_days))
    work = last.copy()

    for d in range(n_days):
        day_pred = model.predict(work[trained_features])
        preds_matrix[:, d] = day_pred

        # Simple lag updates for iterative forecasting
        if "sales_lag_1" in work.columns:
            work["sales_lag_1"] = day_pred
        if "rolling_mean_7" in work.columns:
            work["rolling_mean_7"] = (work["rolling_mean_7"] * 6 + day_pred) / 7
        if "rolling_mean_30" in work.columns:
            work["rolling_mean_30"] = (work["rolling_mean_30"] * 29 + day_pred) / 30

    # Build result
    out = pd.DataFrame(index=last.index.astype(str))
    for i in range(n_days):
        out[f"day_{i+1}_pred"] = preds_matrix[:, i]
    out.reset_index(inplace=True)
    out.rename(columns={"index": "store"}, inplace=True)

    out_path = OUT_DIR / f"next_{n_days}day_preds.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Number of days to predict (default=7)")
    args = parser.parse_args()
    predict_next_n_days(n_days=args.days)
