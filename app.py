from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="AutoWorth API", version="1.0")

# ---------- Load model ----------
MODEL_PATH = "05-artifacts/models/XGBoost_model.joblib"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
else:
    print(f"[WARNING] Model not found at {MODEL_PATH}")

# ---------- Define schemas ----------
class PredictIn(BaseModel):
    year: int
    mileage: float
    engine_size: float
    brand: Optional[str] = None
    fuel: Optional[str] = None
    model_name: Optional[str] = None
    transmission: Optional[str] = None

class PredictBatchIn(BaseModel):
    items: List[PredictIn]

# ---------- Helpers ----------
EXPECTED = list(getattr(model, "feature_names_in_", [])) or ["year", "mileage", "engine_size"]

def to_frame_one(p: PredictIn) -> pd.DataFrame:
    row = {"year": p.year, "mileage": p.mileage, "engine_size": p.engine_size}
    df = pd.DataFrame([row]).reindex(columns=EXPECTED, fill_value=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise HTTPException(status_code=422, detail="Invalid numeric value in payload")
    return df

def to_frame_batch(b: PredictBatchIn) -> pd.DataFrame:
    rows = [{"year": it.year, "mileage": it.mileage, "engine_size": it.engine_size} for it in b.items]
    df = pd.DataFrame(rows).reindex(columns=EXPECTED, fill_value=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise HTTPException(status_code=422, detail="Invalid numeric value in batch payload")
    return df

# ---------- Endpoints ----------
@app.get("/health", tags=["meta"])
def health():
    ok = model is not None
    return {
        "status": "ok" if ok else "error",
        "model_loaded": ok,
        "expected_features": EXPECTED,
    }

@app.post("/predict", tags=["inference"])
def predict(inp: PredictIn):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = to_frame_one(inp)
    y = model.predict(df)[0]
    return {"prediction": float(y)}

@app.post("/predict_batch", tags=["inference"])
def predict_batch(inp: PredictBatchIn):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = to_frame_batch(inp)
    preds = model.predict(df).tolist()
    return {"predictions": [float(v) for v in preds]}
