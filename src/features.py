# src/features.py
"""
Feature engineering for Rossmann data:
- Adds calendar features (dow, month, etc.)
- Adds lag features (1, 7, 30 days)
- Adds rolling averages (7, 30 days)
- Saves to data/processed/features.parquet
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")

def add_features():
    # Load cleaned parquet file
    df = pd.read_parquet(PROCESSED / "daily_sales.parquet")

    # Sort by store and date
    df = df.sort_values(["store", "date"])

    # --- Calendar features ---
    df["dow"] = df["date"].dt.dayofweek   # 0=Monday
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # --- Lag features (previous day, 7 days ago, 30 days ago) ---
    df["sales_lag_1"] = df.groupby("store")["sales"].shift(1)
    df["sales_lag_7"] = df.groupby("store")["sales"].shift(7)
    df["sales_lag_30"] = df.groupby("store")["sales"].shift(30)

    # --- Rolling means ---
    df["rolling_mean_7"] = df.groupby("store")["sales"].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df.groupby("store")["sales"].shift(1).rolling(30).mean()

    # Fill NA values with 0 for simplicity
    df = df.fillna(0)

    # Save output
    out_path = PROCESSED / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved feature dataset to {out_path}")


if __name__ == "__main__":
    add_features()
