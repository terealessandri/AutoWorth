# 02-src/utils/safe_mlflow.py
import os
from contextlib import contextmanager

USE_MLFLOW = os.getenv("USE_MLFLOW", "0") in ("1", "true", "True")

try:
    import mlflow  # noqa: F401
    _mlflow_available = True
except Exception:
    _mlflow_available = False

_enabled = USE_MLFLOW and _mlflow_available

@contextmanager
def start_run(**kwargs):
    if _enabled:
        import mlflow
        with mlflow.start_run(**kwargs) as run:
            yield run
    else:
        # no-op context manager
        yield None

def set_tracking_uri(uri: str):
    if _enabled:
        import mlflow
        mlflow.set_tracking_uri(uri)

def set_experiment(name: str):
    if _enabled:
        import mlflow
        mlflow.set_experiment(name)

def log_params(d: dict):
    if _enabled:
        import mlflow
        mlflow.log_params(d)

def log_metrics(d: dict, step: int | None = None):
    if _enabled:
        import mlflow
        mlflow.log_metrics(d, step=step)

def log_artifact(path: str):
    if _enabled:
        import mlflow
        mlflow.log_artifact(path)

def log_model(model, artifact_path: str):
    if _enabled:
        import mlflow
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)