AutoWorth — Documentation

Predicting used-car prices with a production-grade ML system. This repo captures what we built, how the system works, and the decisions we made across data, modeling, serving, and operations.

------------------------------------------------------------
1) Overview
------------------------------------------------------------
AutoWorth is an end-to-end MLOps project that turns raw vehicle listings into price estimates via a reproducible pipeline and an online inference API. The system follows the whole loop (data → model → deploy → monitor)

Key outcomes:
- Single, versioned model artifact promoted to serving.
- Deterministic preprocessing consistent between train and inference.
- Experiment lineage captured (params, metrics, artifacts, run IDs).
- Portable service boundary with a stable REST contract.

------------------------------------------------------------
2) High-level architecture
------------------------------------------------------------
        Data sources                Training & Tracking               Packaging & Serving               Clients / Tests
  ----------------------------------------------------------------------------------------------
  • Listings (tabular CSV)    • Feature pipeline (identical)      • FastAPI app (HTTP /predict)     • Swagger UI (/docs)
                              • Model selection                    • Run ID surfaced via /health     • pytest integration tests
                              • MLflow runs & artifacts           • Docker image as deploy unit      • cURL / Postman
                                  (Reproducible, traced)               (Portable, reproducible)          (External, black-box)

------------------------------------------------------------
3) Data & features
------------------------------------------------------------
Input domain:
Used-car listings with standard attributes (brand, model/trim, year, mileage, fuel, transmission, engine specs, location, etc.). After exploratory analysis, we removed unhelpful “other” buckets (fuel/transmission) and consolidated rare categories to avoid leakage from spurious frequency noise. Outliers were handled via IQR clipping for most numerics, with a domain rule for year based on market context. Missingness was addressed with simple, fast imputers consistent at train and serve time.

Feature engineering:
- Categorical variables → one-hot with UNKNOWN bucket to avoid injecting fake order and to handle unseen categories safely.
- Numeric variables → winsorized/clipped as per data profile; optional scaling where model-class benefits.
- Optional derived signals (e.g., mileage bands, age of vehicle, fuel x city interactions) were evaluated but only retained when they moved the validation metric meaningfully.

------------------------------------------------------------
4) Modeling & experiment tracking
------------------------------------------------------------
Task: Supervised regression (price)
Baselines: Linear models and tree ensembles to establish a usefulness floor.
Candidates: Regularized linear, Random Forest, Gradient Boosting/XGBoost.
Metrics: MAE and RMSE as primary; R² as auxiliary for interpretability.

Experiment management (MLflow):
- Every run logs: parameters, data hash/split seed, metrics, feature list, and the trained model.
- The chosen run’s Run ID is persisted (run_id.txt) and the model artifact exported (models/model.joblib) for serving.
- The split between backend store (metadata) and artifact store follows MLflow best practices.

------------------------------------------------------------
5) Service design (FastAPI)
------------------------------------------------------------
Endpoints:
- GET /health → returns {"status": "ok", "run_id": "<mlflow-run-id>"} for provenance checks.
- POST /predict → accepts feature JSON, applies the same preprocessing as training, returns {"run_id": "...", "prediction": <float>}.

Contract:
- Stable schema for inputs/outputs; breaking changes require new versioned routes.
- Swagger UI exposed at /docs for easy testing.

Why HTTP online prediction:
User-facing scenario → synchronous, per-request inference. Batch unsuitable; streaming unnecessary.

------------------------------------------------------------
6) Packaging & runtime
------------------------------------------------------------
Container:
The app is packaged as a Docker image to ensure reproducibility and portability. It includes code, environment, and model artifact.

Image contents:
- app.py + preprocessing + model.joblib
- Minimal base image, non-root user, explicit port exposure
- Health endpoint for orchestration readiness

------------------------------------------------------------
7) Ops notes & future work
------------------------------------------------------------
- Monitoring: expand beyond /health; add drift and performance tracking.
- Retraining: event/schedule-based with validation gates.
- Traffic: introduce canary, A/B, or shadow rollouts.
- Explainability: lightweight SHAP or feature-importance for transparency.
- Security: input validation, rate limiting, no secrets in image.

------------------------------------------------------------
8) Repository layout
------------------------------------------------------------
AutoWorth/
├─ data/
├─ notebooks/
├─ src/
│  ├─ train.py
│  ├─ app.py
│  ├─ test_api.py
│  └─ utils/
├─ models/
│  ├─ model.joblib
│  └─ feature_columns.json
├─ run_id.txt
├─ requirements.txt
├─ Dockerfile
└─ README.md



