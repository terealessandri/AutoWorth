"""FastAPI service for NYC Taxi Duration Prediction.

Model is baked into Docker image at build time:
    /app/models/<run_id>/model
    /app/run_id.txt
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Any

import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Global state
RUN_ID: Optional[str] = None
model: Optional[Any] = None


# Pydantic models
class RideRequest(BaseModel):
    PULocationID: int = Field(..., ge=1, description="Pickup Location ID")
    DOLocationID: int = Field(..., ge=1, description="Dropoff Location ID")
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles")

    class Config:
        json_schema_extra = {
            "example": {
                "PULocationID": 138,
                "DOLocationID": 236,
                "trip_distance": 2.5,
            }
        }


class PredictionResponse(BaseModel):
    duration: float
    model_version: str


# Lifespan: load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    # 1. Read run_id.txt
    run_id_path = Path("run_id.txt")
    if run_id_path.exists():
        RUN_ID = run_id_path.read_text().strip()
        print(f"[startup] Found run_id: {RUN_ID}")
    else:
        print("[startup] run_id.txt not found – health will report 'unknown'.")
        RUN_ID = None

    # 2. Load model from standard deployment path
    model_dir = Path("models/model")
    if model_dir.exists():
        try:
            model = mlflow.sklearn.load_model(str(model_dir))
            print(f"[startup] Model loaded from {model_dir}")
        except Exception as e:
            print(f"[startup] Failed to load model: {e}")
            model = None
    else:
        print(f"[startup] Model directory not found: {model_dir}")
        model = None

    yield


app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Predict taxi trip duration (minutes).",
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the NYC Taxi Duration prediction API"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "run_id": RUN_ID or "unknown",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(ride: RideRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    feature_dict = {
        "PU_DO": f"{ride.PULocationID}_{ride.DOLocationID}",
        "trip_distance": ride.trip_distance,
    }
    pred = model.predict([feature_dict])[0]
    return PredictionResponse(
        duration=float(pred),
        model_version=RUN_ID or "unknown",
    )


# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
