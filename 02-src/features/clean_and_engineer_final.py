#!/usr/bin/env python3
# clean_and_engineer.py
# Usage:
# C:\Users\shamb\Desktop\AutoWorth\src\data\combined\__pycache__\all_cars_combined.csv


import argparse
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================
# ----------------- HELPER FUNCTIONS -------------------
# ======================================================

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension: {ext}")

def write_any(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df.to_parquet(path, index=False)
    elif ext in [".csv", ".txt"]:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")

def safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def iqr_clip(series: pd.Series, whisker: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series
    lo = q1 - whisker * iqr
    hi = q3 + whisker * iqr
    return series.clip(lower=lo, upper=hi)

# ======================================================
# ------------------- CLEANING -------------------------
# ======================================================

def drop_other_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["fuelType", "transmission"]:
        if safe_col(df, col):
            mask_other = df[col].astype(str).str.strip().str.lower().eq("other")
            df = df.loc[~mask_other].copy()
    return df

def filter_year(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    if safe_col(df, "year"):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.loc[df["year"] >= min_year].copy()
        df["year"] = df["year"].astype(int)
    return df

def clip_outliers_iqr(df: pd.DataFrame, target: str, exclude_cols=("year",)) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col == target or col in exclude_cols:
            continue
        df[col] = iqr_clip(df[col])
    return df

# ======================================================
# ---------------- FEATURE ENGINEERING -----------------
# ======================================================

def collapse_rare_categories(s: pd.Series, min_count: int = 50) -> pd.Series:
    vc = s.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    return s.map(lambda x: "OTHER" if x in rare else x)

def engineer_features(df: pd.DataFrame, min_cat_count: int = 50) -> pd.DataFrame:
    df = df.copy()

    # 1) Car age & flags
    if safe_col(df, "year"):
        current_year = datetime.now().year
        df["car_age"] = (current_year - pd.to_numeric(df["year"], errors="coerce")).clip(lower=0)
        df["is_post_2019"] = (pd.to_numeric(df["year"], errors="coerce") >= 2019).astype(int)

    # 2) Usage intensity
    mileage_col_candidates = ["mileage", "odometer", "km", "kilometers"]
    mil_col = next((c for c in mileage_col_candidates if safe_col(df, c)), None)
    if mil_col and safe_col(df, "year"):
        age = (datetime.now().year - pd.to_numeric(df["year"], errors="coerce")).replace(0, 1)
        df["mileage_per_year"] = (pd.to_numeric(df[mil_col], errors="coerce") / age).replace([np.inf, -np.inf], np.nan)

    # 3) Transmission flags
    if safe_col(df, "transmission"):
        t = df["transmission"].astype(str).str.lower().str.strip()
        df["gearbox_auto"] = t.isin(["automatic", "semi-auto", "semi automatic", "semiauto"]).astype(int)

    # 4) Engine power buckets
    power_col = None
    for c in ["engineSize", "engine_size", "power", "bhp", "ps", "kw"]:
        if safe_col(df, c):
            power_col = c
            break
    if power_col:
        p = pd.to_numeric(df[power_col], errors="coerce")
        df["power_bucket"] = pd.qcut(p, q=5, duplicates="drop").astype(str)

    # 5) Car segment
    explicit_seg = None
    for c in ["car_segment", "segment", "bodyType", "body_type", "body"]:
        if safe_col(df, c):
            explicit_seg = c
            break
    if explicit_seg:
        df["car_segment"] = df[explicit_seg].astype(str).str.lower().str.strip()

    if not safe_col(df, "car_segment"):
        model_col = None
        for c in ["model", "level_1", "derivative", "name", "trim"]:
            if safe_col(df, c):
                model_col = c
                break

        if model_col:
            m = df[model_col].astype(str).str.lower().str.strip()

            SEG_MAP = {
                "focus": "hatchback_compact",
                "fiesta": "hatchback_compact",
                "fabia": "hatchback_compact",
                "i10": "hatchback_compact",
                "i30": "hatchback_compact",
                "a3": "hatchback_compact",
                "1 series": "hatchback_compact",
                "2 series": "hatchback_compact",
                "a4": "sedan",
                "c class": "sedan_executive",
                "e class": "sedan_executive",
                "3 series": "sedan_executive",
                "5 series": "sedan_executive",
                "q3": "suv_crossover",
                "kuga": "suv_crossover",
                "tucson": "suv_crossover",
                "tiguan": "suv_crossover",
                "t-roc": "suv_crossover",
                "octavia": "suv_crossover",
            }

            def seg_from_model(x: str) -> str:
                for key, val in SEG_MAP.items():
                    if key in x:
                        return val
                if any(k in x for k in ["q", "x", "tiguan", "t-roc", "kuga", "tucson", "rav", "sportage", "captur"]):
                    return "suv_crossover"
                if any(k in x for k in ["class", "series", "passat", "mondeo"]):
                    return "sedan_executive"
                if any(k in x for k in ["fiesta", "focus", "fabia", "clio", "polo", "a3", "i10", "i20", "i30"]):
                    return "hatchback_compact"
                return "other"

            df["car_segment"] = m.map(seg_from_model)

    # 6) Normalize categoricals and collapse rares
    for cat in ["brand", "model", "car_segment", "transmission", "fuelType"]:
        if safe_col(df, cat):
            df[cat] = df[cat].astype(str).str.lower().str.strip()
            df[cat] = collapse_rare_categories(df[cat], min_count=min_cat_count)

    return df

# ======================================================
# ------------------- PREPROCESSING --------------------
# ======================================================

def build_preprocessor(df: pd.DataFrame, target: str):
    feature_df = df.drop(columns=[target]) if target in df.columns else df
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_cols, categorical_cols

# ======================================================
# ---------------------- MAIN --------------------------
# ======================================================

def main():
    ap = argparse.ArgumentParser(description="Clean, engineer, and preprocess dataset.")
    ap.add_argument("--input", required=True, help="Path to raw input file (.csv or .parquet)")
    ap.add_argument("--output-x", required=True, help="Path to write processed dataset (.csv or .parquet)")
    ap.add_argument("--save-preprocessor", required=True, help="Path to persist fitted preprocessor (joblib)")
    ap.add_argument("--target", required=True, help="Target column name (e.g., price)")
    ap.add_argument("--min-year", type=int, default=2019, help="Filter rows with year < min-year")
    ap.add_argument("--iqr-whisker", type=float, default=1.5, help="Whisker multiplier for IQR clipping")
    args = ap.parse_args()

    # 1. Load
    df = read_any(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns: {list(df.columns)}")

    # 2. Cleaning
    df = drop_other_categories(df)
    df = filter_year(df, args.min_year)
    df = clip_outliers_iqr(df, target=args.target)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Preprocessing (encoding + scaling)
    pre, num_cols, cat_cols = build_preprocessor(df, target=args.target)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    pre.fit(X)

    Xt = pre.transform(X)
    feat_names = pre.get_feature_names_out()
    X_proc = pd.DataFrame(Xt, columns=feat_names, index=X.index)
    X_proc[args.target] = y.values

    # 5. Save processed data + preprocessor
    write_any(X_proc, args.output_x)
    os.makedirs(os.path.dirname(args.save_preprocessor), exist_ok=True)
    joblib.dump({"preprocessor": pre, "numeric_cols": num_cols, "categorical_cols": cat_cols}, args.save_preprocessor)

    # 6. Report
    print("=== CLEAN & ENGINEER SUMMARY ===")
    print(f"Input file: {args.input}")
    print(f"Rows after cleaning: {len(df):,}")
    if "fuelType" in df.columns:
        print("Fuel type counts:", df["fuelType"].value_counts().to_dict())
    if "transmission" in df.columns:
        print("Transmission counts:", df["transmission"].value_counts().to_dict())
    if "year" in df.columns:
        print(f"Year range: {int(df['year'].min())}–{int(df['year'].max())}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
    print(f"Processed dataset saved to: {args.output_x}")
    print(f"Preprocessor saved to: {args.save_preprocessor}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
