# 04 – Model Deployment

Deploy the NYC Taxi trip duration model as a REST API using FastAPI + MLflow.

---
## What You Will Do
1. Start MLFlow
2. Train the model (logs single Pipeline artifact + creates `run_id.txt`)
3. Start the API service (`python app.py`)
4. Explore manually via Swagger (`/predict` in browser)
5. Test the API automatically (`pytest -q test_api.py`)

---
## Folder Structure
```
04-deployment/
├── train.py            # Train & log single Pipeline artifact (creates run_id.txt)
├── app.py              # FastAPI service (loads Pipeline from MLflow)
├── test_api.py         # Pytest to exercise /health and /predict endpoints
├── run_id.txt          # Generated MLflow run ID (after training)
└── README.md           # This guide
```

---
## 1. Start MLflow tracking server (New VSCode terminal)
Activate Environment
```powershell
.venv\Scripts\activate
```
You should see `(.venv)` at the start of the prompt.

Go to the folder:
```powershell
cd 04-deployment
```

And run:

```powershell
mlflow server --host 127.0.0.1 --port 5000
```
Open http://localhost:5000 (leave running).

---
## 2. Train the Model (New VSCode terminal)
```powershell
.venv\Scripts\activate
```
You should see `(.venv)` at the start of the prompt.

```powershell
cd 04-deployment
```

And run:

```powershell
python train.py
```
Example output:
```
=== NYC Taxi Duration Training ===
📥 Loading raw data ...
✓ Loaded 100,000 rows after filtering
🔧 Preparing features ...
🚀 Training model ...
✓ RMSE: 5.15  MAE: 3.47  R2: 0.747
✓ Saved run_id.txt (run: <RUN_ID>)
→ View MLflow UI: http://localhost:5000
✅ Training complete. Next: python app.py
```
This writes `run_id.txt` (the MLflow run id) and logs ONE artifact: `model` (a sklearn Pipeline that includes the `DictVectorizer` and `XGBRegressor`).

---
## 3. Start the API
```powershell
python app.py
```
Sample output:
```
🔄 Loading artifacts from MLflow...
Tracking URI: http://localhost:5000
Run ID: <RUN_ID>
✅ Artifacts loaded
INFO:     Uvicorn running on http://0.0.0.0:9696 (Press CTRL+C to quit)
```
Keep this terminal open. The service loads the model.

---

## 4. Test via Browser
Open http://localhost:9696/docs
1. Click `POST /predict` → Try it out
2. Use:
```json
{
	"DOLocationID": 236,
	"PULocationID": 138,
	"trip_distance": 2.5
}
```
3. Execute → Response shows `duration` and `model_version`.

Sample output:
```
Response body
Download
{
  "duration": 12.679303169250488,
  "model_version": "7ea8c7189b044775adc4a05219bf695a"
}
```
---

## 5. Automated API Test
Run endpoint tests (requires API running):

Activate Environment
```powershell
.venv\Scripts\activate
```
You should see `(.venv)` at the start of the prompt.

Go to the folder:
```powershell
cd 04-deployment
```

And run:

```powershell
pytest -q test_api.py
```
Expect `2 passed`.



---
---

---
## API Endpoints
| Endpoint | Purpose |
|----------|---------|
| /        | Welcome/info |
| /health  | Service + model status |
| /predict | Make a prediction |
| /docs    | Swagger UI |

---
## Troubleshooting
| Issue | Fix |
|-------|-----|
| "RUN_ID not found" | Run `python train.py` first |
| MLflow connection error | Start server (step 2) |
| Module not found | Activate env: `.venv\Scripts\activate` |
| API not responding | Ensure Uvicorn running from `app.py` |

---
## Behind the Scenes
- `train.py`: download & filter January 2023 data → build features (`PU_DO`, `trip_distance`) → train a Pipeline (`DictVectorizer` + `XGBRegressor`) → log params & metrics and ONE artifact `model` → write `run_id.txt`.
- `app.py`: on startup reads `run_id.txt` → loads the single Pipeline artifact → `/predict` builds a feature dict and calls `model.predict`.
- Feature schema for prediction: `PULocationID` (int), `DOLocationID` (int), `trip_distance` (float). Internally combined into `PU_DO` then vectorized.
- MLflow UI: inspect runs & artifacts at http://localhost:5000.
- Environment variable (optional): set `MLFLOW_TRACKING_URI` if using a non-default server address.


---
## Quick Commands
| Action | Command |
|--------|---------|
| Activate env | `.venv\Scripts\activate` |
| Start MLflow | `mlflow server --host 127.0.0.1 --port 5000` |
| Train model | `python train.py` |
| Run API | `python app.py` |
| Test API | Open `http://localhost:9696/docs` |
| API pytest | `pytest -q 04-deployment/test_api.py` |

