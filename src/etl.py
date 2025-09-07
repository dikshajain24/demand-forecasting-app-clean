# src/etl.py
"""
Robust ETL for Rossmann dataset: reads data/raw/train.csv and data/raw/store.csv,
normalizes columns, coerces problematic columns to string so parquet write succeeds.
"""
import pandas as pd
from pathlib import Path
import sys

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path, parse_dates=None):
    # use low_memory=False to avoid dtype guessing in chunks
    print(f"Reading {path} with low_memory=False ...")
    return pd.read_csv(path, low_memory=False, parse_dates=parse_dates)


def prepare_rossmann():
    train_path = RAW / "train.csv"
    store_path = RAW / "store.csv"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Put train.csv into data/raw/ and try again.")
        sys.exit(1)
    if not store_path.exists():
        print(f"ERROR: {store_path} not found. Put store.csv into data/raw/ and try again.")
        sys.exit(1)

    # --- read files ---
    train = safe_read_csv(train_path, parse_dates=["Date"])
    store = safe_read_csv(store_path, parse_dates=None)

    # normalize column names to lowercase
    train.columns = train.columns.str.lower()
    store.columns = store.columns.str.lower()

    # show a quick dtype summary for debugging
    print("\n--- train dtypes (sample) ---")
    print(train.dtypes.head(12))
    if "stateholiday" in train.columns:
        print("\nUnique values in stateholiday (sample 50):")
        print(pd.Series(train["stateholiday"].dropna().unique()[:50]))

    # Keep only open stores if 'open' exists
    if "open" in train.columns:
        train = train[(train["open"].isna()) | (train["open"] == 1)]

    # Merge store metadata if possible
    if "store" in store.columns and "store" in train.columns:
        train = train.merge(store, on="store", how="left")

    # Ensure date column exists and is datetime
    if "date" not in train.columns:
        # attempt to find any date-like column
        for c in train.columns:
            if "date" in c.lower():
                train.rename(columns={c: "date"}, inplace=True)
                break
    train["date"] = pd.to_datetime(train["date"], errors="coerce")

    # sort data
    if "store" in train.columns:
        train = train.sort_values(["store", "date"])
    else:
        train = train.sort_values("date")

    # --- Fix known problematic column(s) explicitly ---
    # stateholiday often has mixed types in Rossmann; coerce to string
    if "stateholiday" in train.columns:
        print("Coercing 'stateholiday' to string (to avoid mixed dtypes)...")
        train["stateholiday"] = train["stateholiday"].astype(str)

    # --- General fallback: convert any object-dtype columns to string ---
    obj_cols = train.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print("Converting object columns to string for parquet compatibility:", obj_cols)
        for c in obj_cols:
            train[c] = train[c].astype(str)

    # Remove any unnamed columns (leftover from CSV exports)
    unnamed = [c for c in train.columns if c.lower().startswith("unnamed")]
    if unnamed:
        print("Dropping unnamed columns:", unnamed)
        train = train.drop(columns=unnamed)

    # final dtype check
    print("\n--- final dtypes (sample) ---")
    print(train.dtypes.head(12))

    # Save to parquet (pyarrow must be installed)
    out_path = PROCESSED / "daily_sales.parquet"
    print(f"\nWriting parquet to {out_path} ...")
    train.to_parquet(out_path, index=False)
    print(f"✅ Saved processed dataset to {out_path}")


if __name__ == "__main__":
    prepare_rossmann()



