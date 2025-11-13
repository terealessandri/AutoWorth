"""Train NYC Taxi Duration model and package for deployment."""

from __future__ import annotations

from pathlib import Path
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2023-01.parquet"
)
DEPLOYMENT_MODEL_PATH = Path("models/model")


def load_data(limit: int = 100_000) -> pd.DataFrame:
    """Load and filter NYC taxi data."""
    print("Loading data...")
    df = pd.read_parquet(DATA_URL)
    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df.head(limit).copy()
    print(f"Loaded {len(df):,} rows")
    return df


def prepare_features(df: pd.DataFrame):
    """Create feature dicts and target."""
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    features = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    target = df["duration"].values
    return features, target


def train_and_log(X_train, y_train, X_val, y_val):
    """Train model, log to MLflow, and copy artifact for deployment."""
    print("Training model...")
    mlflow.set_experiment("nyc-taxi-duration")

    pipeline = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    max_depth=8,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipeline, "model")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

    # Save the trained pipeline directly to the deployment path
    print("Creating deployment-ready model...")
    if DEPLOYMENT_MODEL_PATH.exists():
        print(f"Removing existing model at {DEPLOYMENT_MODEL_PATH}")
        shutil.rmtree(DEPLOYMENT_MODEL_PATH)

    mlflow.sklearn.save_model(pipeline, str(DEPLOYMENT_MODEL_PATH))
    print(f"Model saved to standard deployment path: {DEPLOYMENT_MODEL_PATH}")

    # Save run_id for the artifact
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    return run_id


def main() -> None:
    """Main training pipeline."""
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_and_log(X_train, y_train, X_val, y_val)
    print("Training complete.")


if __name__ == "__main__":
    main()
