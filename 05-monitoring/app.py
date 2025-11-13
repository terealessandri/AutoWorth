"""FastAPI service for NYC Taxi Duration Prediction.

Loads a single sklearn Pipeline artifact (DictVectorizer + XGBRegressor) from MLflow
using the run id in run_id.txt and exposes /predict.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc  # load unified pipeline
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RUN_ID: Optional[str] = None
model = None  # sklearn Pipeline (pyfunc)

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------
class RideRequest(BaseModel):
    """Input payload for a duration prediction."""
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


# ---------------------------------------------------------------------------
# Lifespan: load artifacts once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    with open("run_id.txt", "r") as f:
        RUN_ID = f.read().strip()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/model")
    print("[startup] Loaded Pipeline artifact 'model'.")
    yield
    # (No teardown needed)


app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Predict taxi trip duration (minutes).",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the NYC Taxi Duration prediction API"}


@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(ride: RideRequest):
    feature_dict = {
        "PU_DO": f"{ride.PULocationID}_{ride.DOLocationID}",
        "trip_distance": ride.trip_distance,
    }
    pred = model.predict([feature_dict])[0]
    return PredictionResponse(duration=float(pred), model_version=RUN_ID or "unknown")


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)

