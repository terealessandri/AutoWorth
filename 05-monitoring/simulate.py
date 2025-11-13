"""
Simulate real-time requests to the running FastAPI model and log predictions.

Usage:
    python simulate.py
This will:
- Load real NYC Taxi data
- Send random samples to the /predict endpoint
- Collect predictions + ground truth durations
- Append them to data/predictions.csv
"""

import time
import json
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
API_URL = "http://localhost:9696/predict"
LOG_PATH = Path("data/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data(n_rows=100, year=2023, month=1):
    """Load a sample of NYC taxi data for simulation."""
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    print(f"📥 Loading data from {url}")
    df = pd.read_parquet(url)

    # Compute trip duration in minutes
    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Filter for reasonable values
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    # Sample subset and reset index
    df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    print(f"✓ Loaded {len(df)} rows for simulation")
    return df


def simulate_requests(df: pd.DataFrame, sleep_s=0.05):
    """Send each row to the prediction API and log the results."""
    rows = []

    for i, row in df.iterrows():
        payload = {
            "PULocationID": int(row["PULocationID"]),
            "DOLocationID": int(row["DOLocationID"]),
            "trip_distance": float(row["trip_distance"]),
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            pred = resp.json()["duration"]

            rows.append(
                {
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "PU_DO": f"{payload['PULocationID']}_{payload['DOLocationID']}",
                    "trip_distance": payload["trip_distance"],
                    "prediction": pred,
                    "duration": row["duration"],  # ground truth
                }
            )

        except Exception as e:
            print(f"⚠️  Request failed: {e}")

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i + 1}/{len(df)}")
        time.sleep(sleep_s)

    return pd.DataFrame(rows)


def main():
    print("\n🚕 Starting simulation...\n")
    df = load_data(n_rows=100)
    out = simulate_requests(df)

    if out.empty:
        print("❌ No predictions recorded. Make sure app.py is running.")
        return

    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(LOG_PATH, index=False)
    print(f"✅ Wrote {len(out)} total rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
