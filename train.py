# train.py
import os, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("AutoWorth-Training")

DATA_PATH = "04-data/processed/cars_clean.csv"
TARGET = "price"
ARTIFACT_DIR = "05-artifacts/models"
TOL_EUR = 500

from pathlib import Path
Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def pct_within_tol(y_true, y_pred, tol=TOL_EUR):
    return float(np.mean(np.abs(y_true - y_pred) <= tol) * 100.0)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_pred = np.full_like(y_val, fill_value=float(np.median(y_tr)), dtype=float)
baseline = {
    "name": "Baseline_Median",
    "r2": r2_score(y_val, baseline_pred),
    "mae": mean_absolute_error(y_val, baseline_pred),
    "rmse": rmse(y_val, baseline_pred),
    "mape": mape(y_val, baseline_pred),
    "pct_within_€500": pct_within_tol(y_val, baseline_pred),
}

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=400, n_jobs=-1, random_state=42
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )

# ... keep all your imports and helper functions ...

results = []
mlflow.set_experiment("AutoWorth-Training")

with mlflow.start_run(run_name="Baseline_Median"):
    mlflow.log_param("model", "Baseline_Median")
    mlflow.log_metric("r2", baseline["r2"])
    mlflow.log_metric("mae", baseline["mae"])
    mlflow.log_metric("rmse", baseline["rmse"])
    mlflow.log_metric("mape_pct", baseline["mape"])
    mlflow.log_metric("pct_within_eur_500", baseline["pct_within_€500"])
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    mlflow.log_artifact(os.path.join(ARTIFACT_DIR, "baseline_metrics.json"))



results = []
mlflow.set_experiment("AutoWorth-Training")

with mlflow.start_run(run_name="Baseline_Median"):
    mlflow.log_param("model", "Baseline_Median")
    mlflow.log_metric("r2", baseline["r2"])
    mlflow.log_metric("mae", baseline["mae"])
    mlflow.log_metric("rmse", baseline["rmse"])
    mlflow.log_metric("mape_pct", baseline["mape"])
    mlflow.log_metric("pct_within_eur_500", baseline["pct_within_€500"])
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    mlflow.log_artifact(os.path.join(ARTIFACT_DIR, "baseline_metrics.json"))

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Handle XGBoost separately
        if name.lower().startswith("xgboost") or "xgb" in name.lower():
            X_fit, X_val_in = X_tr.to_numpy(), X_val.to_numpy()
        else:
            X_fit, X_val_in = X_tr, X_val

        t0 = time.time()
        model.fit(X_fit, y_tr)
        train_time_s = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_val_in)
        infer_time_s = time.time() - t0

        # Compute metrics
        r2     = r2_score(y_val, preds)
        mae    = mean_absolute_error(y_val, preds)
        _rmse  = rmse(y_val, preds)
        _mape  = mape(y_val, preds)
        within = pct_within_tol(y_val, preds, TOL_EUR)

        # Log to MLflow (metrics only)
        mlflow.log_params({
            "model": name,
            "target": TARGET,
            "tolerance_eur": TOL_EUR
        })
        mlflow.log_metrics({
            "r2": r2,
            "mae": mae,
            "rmse": _rmse,
            "mape_pct": _mape,
            "pct_within_eur_500": within,
            "train_time_s": train_time_s,
            "infer_batch_time_s": infer_time_s
        })

        # Store in results for selection
        results.append({
            "name": name,
            "mae": mae,
            "r2": r2,
            "rmse": _rmse,
            "mape": _mape,
            "pct_within_eur_500": within,
            "estimator": model
        })


# Choose the lowest MAE
best = sorted(results, key=lambda d: d["mae"])[0]

# Save ONLY the best model locally
Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
best_path = os.path.join(ARTIFACT_DIR, f"{best['name']}_model.joblib")
joblib.dump(best["estimator"], best_path)

# Write summary
summary = {
    "baseline": baseline,
    "candidates": [
        {k: v for k, v in d.items() if k != "estimator"} for d in results
    ],
    "best": {
        "name": best["name"],
        "mae": best["mae"],
        "rmse": best["rmse"],
        "r2": best["r2"],
        "mape": best["mape"],
        "pct_within_eur_500": best["pct_within_eur_500"],
        "path": best_path
    }
}

with open(os.path.join(ARTIFACT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Print clear summary
print("\n================== TRAINING SUMMARY ==================")
for r in results:
    print(f"{r['name']:<20} MAE={r['mae']:.2f} | RMSE={r['rmse']:.2f} | "
          f"R2={r['r2']:.3f} | MAPE={r['mape']:.2f}% | "
          f"Within±€500={r['pct_within_eur_500']:.1f}%")

print("------------------------------------------------------")
print(f"BEST MODEL: {best['name']}")
print(f"Saved to:   {best_path}")
print("======================================================\n")