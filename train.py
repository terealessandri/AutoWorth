# train.py
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- MLflow helper (no-op si USE_MLFLOW=0 o si no está instalado) ---
# Permite importar utils desde 02-src (namespace package)
sys.path.append("02-src")
from utils.safe_mlflow import (  # noqa: E402
    start_run, set_tracking_uri, set_experiment,
    log_params, log_metrics, log_artifact, log_model
)

# --- XGBoost opcional ---
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ===================== Config =====================
DATA_PATH = "04-data/processed/cars_clean.csv"
TARGET = "price"
ARTIFACT_DIR = "05-artifacts/models"
TOL_EUR = 500

# Si activas MLflow: export USE_MLFLOW=1
# Puedes cambiar la URI por HTTP (server) o 'file:./mlruns'
USE_MLFLOW = os.getenv("USE_MLFLOW", "0").lower() in ("1", "true")

if USE_MLFLOW:
    set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    set_experiment(os.getenv("MLFLOW_EXPERIMENT", "AutoWorth-Training"))

Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# ===================== Métricas auxiliares =====================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def pct_within_tol(y_true, y_pred, tol=TOL_EUR):
    return float(np.mean(np.abs(y_true - y_pred) <= tol) * 100.0)

# ===================== Carga de datos =====================
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== Baseline =====================
baseline_pred = np.full_like(y_val, fill_value=float(np.median(y_tr)), dtype=float)
baseline = {
    "name": "Baseline_Median",
    "r2": r2_score(y_val, baseline_pred),
    "mae": mean_absolute_error(y_val, baseline_pred),
    "rmse": rmse(y_val, baseline_pred),
    "mape": mape(y_val, baseline_pred),
    "pct_within_€500": pct_within_tol(y_val, baseline_pred),
}

# Logueo baseline (no-op si MLflow desactivado)
with start_run(run_name="Baseline_Median"):
    log_params({"model": "Baseline_Median"})
    log_metrics({
        "r2": baseline["r2"],
        "mae": baseline["mae"],
        "rmse": baseline["rmse"],
        "mape_pct": baseline["mape"],
        "pct_within_eur_500": baseline["pct_within_€500"],
    })
    # Guardar y (si aplica) loggear artifact con métricas baseline
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    baseline_path = os.path.join(ARTIFACT_DIR, "baseline_metrics.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    log_artifact(baseline_path)

# ===================== Modelos candidatos =====================
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

results = []

for name, model in models.items():
    with start_run(run_name=name):
        # Para XGB usamos arrays numpy
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

        # Métricas
        r2     = r2_score(y_val, preds)
        mae    = mean_absolute_error(y_val, preds)
        _rmse  = rmse(y_val, preds)
        _mape  = mape(y_val, preds)
        within = pct_within_tol(y_val, preds, TOL_EUR)

        # Logueo (no-op si MLflow desactivado)
        log_params({
            "model": name,
            "target": TARGET,
            "tolerance_eur": TOL_EUR
        })
        log_metrics({
            "r2": r2,
            "mae": mae,
            "rmse": _rmse,
            "mape_pct": _mape,
            "pct_within_eur_500": within,
            "train_time_s": train_time_s,
            "infer_batch_time_s": infer_time_s
        })

        results.append({
            "name": name,
            "mae": mae,
            "r2": r2,
            "rmse": _rmse,
            "mape": _mape,
            "pct_within_eur_500": within,
            "estimator": model
        })

# ===================== Selección y guardado =====================
best = sorted(results, key=lambda d: d["mae"])[0]

Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
best_path = os.path.join(ARTIFACT_DIR, f"{best['name']}_model.joblib")
joblib.dump(best["estimator"], best_path)

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

summary_path = os.path.join(ARTIFACT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

# (Opcional) registrar artifacts si MLflow activo
log_artifact(summary_path)

# ===================== Print resumen =====================
print("\n================== TRAINING SUMMARY ==================")
for r in results:
    print(f"{r['name']:<20} MAE={r['mae']:.2f} | RMSE={r['rmse']:.2f} | "
          f"R2={r['r2']:.3f} | MAPE={r['mape']:.2f}% | "
          f"Within±€500={r['pct_within_eur_500']:.1f}%")
print("------------------------------------------------------")
print(f"BEST MODEL: {best['name']}")
print(f"Saved to:   {best_path}")
print("======================================================\n")
