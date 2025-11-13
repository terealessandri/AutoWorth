"""Train NYC Taxi Duration Prediction model and log to MLflow.

Simplified version:
1. Load January 2023 data
2. Build dict features (PU_DO, trip_distance)
3. Fit a sklearn Pipeline: (DictVectorizer -> XGBRegressor)
4. Log params, metrics (RMSE, MAE, R2) and ONE artifact: 'model'
5. Persist run_id.txt for serving.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow
import mlflow.xgboost  # ensure XGBoost flavor registered
import mlflow.sklearn


MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "nyc-taxi-duration"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"


def load_data(limit: int = 100_000) -> pd.DataFrame:
    """Load taxi parquet file and compute trip duration (minutes)."""
    print("📥 Loading raw data ...")
    df = pd.read_parquet(DATA_URL)

    # Compute duration
    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Basic filtering
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    df = df.head(limit).copy()
    print(f"✓ Loaded {len(df):,} rows after filtering")
    return df


def prepare_features(df: pd.DataFrame):
    """Create list-of-dicts features and numeric target.

    Output X (List[Dict[str, Any]]), y (np.ndarray).
    """
    print("🔧 Preparing features ...")
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    feature_dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    target = df["duration"].values
    return feature_dicts, target


class DictVectorizerWrapper(BaseEstimator, TransformerMixin):
    """Thin sklearn-compatible wrapper around DictVectorizer.

    Accepts list-of-dicts and produces sparse feature matrix.
    """
    def __init__(self):
        self.dv = DictVectorizer(sparse=True)

    def fit(self, X, y=None):  # X: List[Dict]
        self.dv.fit(X)
        return self

    def transform(self, X):
        return self.dv.transform(X)


def train_and_log(X_train_dicts, y_train, X_val_dicts, y_val):
    """Train XGBoost model, evaluate, and log everything to MLflow."""
    print("🚀 Training model ...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    pipeline = Pipeline([
        ("vectorizer", DictVectorizerWrapper()),
        ("regressor", xgb.XGBRegressor(**params)),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train_dicts, y_train)

        # Evaluate on validation set (list-of-dicts directly)
        y_pred = pipeline.predict(X_val_dicts)
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"✓ RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.3f}")

        # Log parameters & metrics
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("val_rows", len(y_val))

        # Single artifact: the sklearn Pipeline (includes DictVectorizer + XGBRegressor)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"💾 Saved run_id.txt (run: {run_id})")
        print("🖥  View MLflow UI: http://localhost:5000")
        print("📦 Logged single Pipeline artifact 'model'")
        return run_id


def main():
    print("\n=== NYC Taxi Duration Training ===\n")
    df = load_data()
    feature_dicts, target = prepare_features(df)

    X_train_dicts, X_val_dicts, y_train, y_val = train_test_split(
        feature_dicts, target, test_size=0.2, random_state=42
    )

    run_id = train_and_log(X_train_dicts, y_train, X_val_dicts, y_val)
    print("\n✅ Training complete. Next: python app.py\n")
    return run_id


if __name__ == "__main__":
    main()
