"""
preprocessing.py
----------------
Standalone preprocessing utilities imported by the Jupyter notebook
and usable independently from the command line.
"""

import re
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_review_text(text: str) -> str:
    """Apply regex-based cleaning to a review string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                   # non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()                 # extra whitespace
    return text


# ── Data loading & initial cleaning ──────────────────────────────────────────
def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    original_rows = len(df)

    # ① Drop duplicates
    df = df.drop_duplicates()
    print(f"  Duplicates removed : {original_rows - len(df)}")

    # ② Handle missing values
    df = df.copy()
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["review"] = df["review"].fillna("no review provided")

    # ③ Type conversion
    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["rating"]       = df["rating"].astype(float)
    df["price"]        = df["price"].astype(float)
    df["quantity"]     = df["quantity"].astype(int)
    df["total_amount"] = df["total_amount"].astype(float)

    # ④ Clean review text
    df["cleaned_review"] = df["review"].apply(clean_review_text)

    # ⑤ Derived time features
    df["year"]  = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["dow"]   = df["timestamp"].dt.dayofweek   # 0=Mon … 6=Sun

    print(f"  Rows after cleaning: {len(df):,}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """
    Returns (X, y, feature_names, encoders_dict, scaler).
    All transformers are fitted here so the GUI / inference code can reuse them.
    """
    df = df.copy()

    # Label-encode low-cardinality categoricals
    le_gender  = LabelEncoder()
    le_loc     = LabelEncoder()
    le_loyalty = LabelEncoder()
    le_cat     = LabelEncoder()
    le_pay     = LabelEncoder()

    df["gender_enc"]  = le_gender.fit_transform(df["gender"])
    df["location_enc"]= le_loc.fit_transform(df["location"])
    df["loyalty_enc"] = le_loyalty.fit_transform(df["loyalty_tier"])
    df["category_enc"]= le_cat.fit_transform(df["category"])
    df["payment_enc"] = le_pay.fit_transform(df["payment_method"])

    feature_cols = [
        "age","gender_enc","location_enc","loyalty_enc",
        "category_enc","payment_enc",
        "price","quantity","total_amount","rating",
        "year","month","dow",
    ]

    from sklearn.impute import SimpleImputer
    X_raw = df[feature_cols].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    y = df["will_purchase_again"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    encoders = {
        "gender": le_gender, "location": le_loc,
        "loyalty": le_loyalty, "category": le_cat, "payment": le_pay,
    }

    return X_scaled, y, feature_cols, encoders, scaler


# ── Customer aggregation for clustering ──────────────────────────────────────
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per customer for K-Means clustering."""
    agg = df.groupby("customer_id").agg(
        total_spend    = ("total_amount","sum"),
        num_orders     = ("transaction_id","count"),
        avg_order_val  = ("total_amount","mean"),
        avg_rating     = ("rating","mean"),
        unique_cats    = ("category","nunique"),
        recency_days   = ("timestamp", lambda x: (df["timestamp"].max() - x.max()).days),
    ).reset_index()
    return agg


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ecommerce_data.csv"
    print(f"Loading {path} …")
    clean_df = load_and_clean(path)
    out = path.replace(".csv","_cleaned.csv")
    clean_df.to_csv(out, index=False)
    print(f"Saved → {out}")
