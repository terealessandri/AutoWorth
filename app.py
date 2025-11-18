# ==========================================================
# app.py — AutoWorth API (FastAPI)
# Estructura asumida:
#  - 05-artifacts/models/summary.json   → contiene "best.path"
#  - 05-artifacts/models/*.joblib       → modelos serializados
#  - (opcional) 05-artifacts/models/feature_columns.json
# Cómo lanzar:  uvicorn app:app --reload
# ==========================================================
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------
# Config
# ---------------------------
APP_VERSION = "1.0.0"

SUMMARY_PATH = Path("05-artifacts/models/summary.json")
FEATURE_COLS_PATH = Path("05-artifacts/models/feature_columns.json")  # opcional
RUN_ID_PATH = Path("run_id.txt")  # opcional, visible en /meta

# Estado global (cargado en startup)
model = None
feature_cols: Optional[List[str]] = None
RUN_ID: str = "unknown"
MODEL_ID: str = "unknown"
MODEL_PATH: Optional[Path] = None


def _model_id(obj) -> str:
    """Retorna un hash corto para trazar la versión del modelo cargado."""
    try:
        return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()[:8]
    except Exception:
        return "unknown"


def _resolve_model_path() -> Path:
    """
    Lee summary.json y obtiene la ruta del mejor modelo (best.path).
    Si no existe, intenta un fallback razonable.
    """
    if SUMMARY_PATH.exists():
        try:
            data = json.loads(SUMMARY_PATH.read_text())
            best_path = data["best"]["path"]
            return Path(best_path)
        except Exception as e:
            print(f"[startup] WARNING: no pude leer summary.json: {e}")

    # Fallback: intenta algunos nombres comunes dentro de 05-artifacts/models
    candidates = [
        "05-artifacts/models/RandomForest_model.joblib",
        "05-artifacts/models/LinearRegression_model.joblib",
        "05-artifacts/models/XGBoost_model.joblib",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p

    # Último recurso (probablemente fallará al cargar en startup)
    return Path("05-artifacts/models/model.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y columnas una sola vez al iniciar la app."""
    global model, feature_cols, RUN_ID, MODEL_ID, MODEL_PATH

    # 1) Run id visible (opcional)
    if RUN_ID_PATH.exists():
        RUN_ID = RUN_ID_PATH.read_text().strip()

    # 2) Columnas esperadas (opcional)
    if FEATURE_COLS_PATH.exists():
        try:
            feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
            if not isinstance(feature_cols, list):
                feature_cols = None
        except Exception as e:
            print(f"[startup] WARNING: no pude leer feature_columns.json: {e}")
            feature_cols = None

    # 3) Resolver y cargar el mejor modelo
    try:
        MODEL_PATH = _resolve_model_path()
        model = joblib.load(MODEL_PATH)
        MODEL_ID = _model_id(model)
        print(f"[startup] Loaded model: {MODEL_PATH} (id={MODEL_ID})")
    except Exception as e:
        print(f"[startup] ERROR loading model from {MODEL_PATH}: {e}")
        model = None

    yield  # ----------------- app running -----------------


# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(
    title="AutoWorth API",
    description="Predict used-car prices using the trained model (AutoWorth).",
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS abierto para pruebas locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Schemas
# ---------------------------
class PredictIn(BaseModel):
    brand: str
    model: str
    year: int = Field(..., ge=1950, le=2100)
    mileage: int = Field(..., ge=0)
    fuel: Literal["Petrol", "Diesel", "Hybrid", "Electric"]
    transmission: Literal["Manual", "Automatic", "Semi-Auto"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand": "ford",
                "model": "Fiesta",
                "year": 2019,
                "mileage": 42000,
                "fuel": "Petrol",
                "transmission": "Manual",
            }
        }
    }


class PredictOut(BaseModel):
    price: float
    currency: Literal["EUR"] = "EUR"
    latency_ms: int
    model_id: str
    model_version: str


class BatchIn(BaseModel):
    items: List[PredictIn] = Field(..., min_items=1, max_length=1000)


class BatchOut(BaseModel):
    prices: List[float]
    currency: Literal["EUR"] = "EUR"
    latency_ms: int
    model_id: str
    model_version: str


class ErrorOut(BaseModel):
    detail: str


# ---------------------------
# Helpers
# ---------------------------
def _to_frame(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte la lista de items a DataFrame y, si tenemos feature_cols,
    asegura el orden/ausencia de columnas (rellena faltantes con None).
    """
    df = pd.DataFrame(items)
    if feature_cols:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = None
        df = df[feature_cols]
    return df


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Si el modelo espera numpy (algunos pipelines con XGBoost), to_numpy:
    try:
        return model.predict(df)
    except Exception:
        return model.predict(df.to_numpy())


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/", tags=["meta"])
def root():
    return {"name": "AutoWorth API", "version": APP_VERSION}


@app.get("/health", tags=["meta"])
def health():
    ok = model is not None
    return {"status": "ok" if ok else "error", "model_loaded": ok}


@app.get("/meta", tags=["meta"])
def meta():
    return {
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
        "model_id": MODEL_ID,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "feature_columns": feature_cols or "unknown",
    }


@app.post("/predict", response_model=PredictOut, responses={503: {"model": ErrorOut}}, tags=["infer"])
def predict_one(body: PredictIn):
    t0 = time.time()
    row = body.dict()
    df = _to_frame([row])
    pred = float(_predict_df(df)[0])
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "price": pred,
        "currency": "EUR",
        "latency_ms": latency_ms,
        "model_id": MODEL_ID,
        "model_version": RUN_ID,
    }


@app.post("/predict_batch", response_model=BatchOut, responses={503: {"model": ErrorOut}}, tags=["infer"])
def predict_batch(body: BatchIn):
    t0 = time.time()
    df = _to_frame([item.dict() for item in body.items])
    preds = _predict_df(df).astype(float).tolist()
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "prices": preds,
        "currency": "EUR",
        "latency_ms": latency_ms,
        "model_id": MODEL_ID,
        "model_version": RUN_ID,
    }
