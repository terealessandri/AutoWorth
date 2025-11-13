"""
Generate an Evidently HTML report comparing past vs recent predictions.

Usage:
    python monitor.py
This will:
- Load the logged predictions from data/predictions.csv
- Split them into reference (older) vs current (newer) data
- Generate a data + performance drift report
"""

import pandas as pd
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset


LOG_PATH = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")


def main():
    print("\n📊 Starting monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError("❌ No logged predictions found. Run simulate.py first!")

    df = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    df = df.dropna(subset=["prediction", "duration"])
    print(f"✓ Loaded {len(df)} logged predictions")

    # Sort by timestamp and split into reference (older) vs current (recent)
    df = df.sort_values("ts")
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    # Configure mapping for Evidently
    column_mapping = ColumnMapping(
        target="duration",
        prediction="prediction",
        numerical_features=["trip_distance"],
        categorical_features=["PU_DO"],
    )

    # Build report
    print("\n🧮 Generating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # Convert Path to str for Windows compatibility
    report.save_html(str(REPORT_PATH))
    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open it in your browser to explore drift metrics.\n")


if __name__ == "__main__":
    main()
