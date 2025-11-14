from pathlib import Path
import pandas as pd
import glob

# Define folders
RAW_DIR = Path("04-data/raw")
COMBINED_DIR = Path("04-data/combined")
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

# Expected columns 
EXPECTED_COLS = [
    "model", "year", "price", "transmission",
    "mileage", "fuelType", "tax", "mpg", "engineSize"
]

def combine_raw():
    all_files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not all_files:
        raise FileNotFoundError("No CSV files found in data/raw/")

    dfs = []
    for file in all_files:
        brand_name = Path(file).stem.lower()  # e.g. "vw.csv" -> "vw"

        # Fix known mislabels (file name ≠ brand)
        brand_map = {
            "focus": "ford",
            "cclass": "mercedes",
            "merc": "mercedes",
            "hyundi": "hyundai"
        }
        brand_name = brand_map.get(brand_name, brand_name)

        try:
            df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns]

            # Keep only expected columns
            df = df[[c for c in EXPECTED_COLS if c in df.columns]]

            # Add brand column (overwrite existing if present)
            df["brand"] = brand_name

            dfs.append(df)
            print(f"[OK] {brand_name}: {len(df)} rows")
        except Exception as e:
            print(f"[ERROR] {brand_name}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns so brand comes first
    ordered_cols = ["brand"] + [c for c in EXPECTED_COLS if c in combined_df.columns]
    combined_df = combined_df[ordered_cols]

    # Convert numeric fields safely
    for col in ["year", "price", "mileage", "tax", "mpg", "engineSize"]:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # Drop clearly invalid entries
    combined_df.dropna(subset=["price", "year", "mileage"], inplace=True)

    # Save the result
    out_path = COMBINED_DIR / "all_cars_combined.csv"
    combined_df.to_csv(out_path, index=False)

    print(f"\n✅ Combined dataset saved to: {out_path}")
    print(f"Total rows: {len(combined_df)} | Columns: {list(combined_df.columns)}")

if __name__ == "__main__":
    combine_raw()
