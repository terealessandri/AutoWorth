# 🚀 05 – Monitoring: From Training to Drift Detection

This module demonstrates a **complete MLOps loop**: train → serve → simulate production traffic → monitor drift.

You'll run four key scripts in sequence to experience how real-world ML systems are monitored in production.

---

Copy folder 04-deployment and create 05-monitoring. Add `simulate.py` and `monitor.py` from blackboard. 

## ✅ Folder Structure
```
05-monitoring/
├── app.py          # FastAPI service (loads model from MLflow)
├── train.py        # Model training & MLflow logging
├── simulate.py     # NEW: Simulates live predictions (creates predictions.csv)
├── monitor.py      # NEW: Generates drift report from predictions.csv
├── test_api.py     # Basic health + predict tests
├── run_id.txt      # Written by train.py (points to MLflow run)
└── README.md       # This guide
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
cd 05-monitoring
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
cd 05-monitoring
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

## 4. (Optional) Test via Browser
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
cd 05-monitoring
```

And run:

```powershell
pytest -q test_api.py
```
Expect `2 passed`.



---

## 6. Simulate Live Predictions

In a **new terminal**:

```powershell
.venv\Scripts\activate
cd 05-monitoring
python simulate.py
```

Output:

```
🚕 Starting simulation...

📥 Loading data from ...yellow_tripdata_2023-01.parquet
✓ Loaded 100 rows for simulation
   Progress: 20/100
   Progress: 40/100
   Progress: 60/100
   Progress: 80/100
   Progress: 100/100
✅ Wrote 100 total rows to data/predictions.csv
```

✅ This script:

* Calls your `/predict` endpoint ~100 times with real taxi trip data
* Logs predictions + ground truth into `data/predictions.csv`
* Simulates production traffic (inference logs)

**Important:** The API (step 3) must be running, or you'll see connection errors.

---

## 7. Generate Monitoring Report

In the same terminal:

```powershell
python monitor.py
```

Output:

```
📊 Starting monitoring report...

✓ Loaded 100 logged predictions
Reference: 50  |  Current: 50

🧮 Generating Evidently drift report...
✅ Report saved: monitoring_report.html
Open it in your browser to explore drift metrics.
```

✅ This script:

* Loads all historical predictions from `data/predictions.csv`
* Splits them into **reference (older)** vs **current (newer)** data
* Uses **Evidently** to detect:
  * Input data drift (`PU_DO` location pairs / `trip_distance`)
  * Performance drift (predicted vs actual duration)
* Saves a visual dashboard: `monitoring_report.html`

---

## 8. View Results

1. **Open `monitoring_report.html` in your browser**
   → You'll see interactive data-drift and regression-performance dashboards.

2. **Open [http://localhost:5000](http://localhost:5000)**
   → Inspect your MLflow run, parameters, metrics, and logged model artifact.

---



## 🧠 What Each Script Does

| Stage         | Script        | Concept                                               |
| ------------- | ------------- | ----------------------------------------------------- |
| **Train**     | `train.py`    | MLflow experiment tracking & model versioning         |
| **Serve**     | `app.py`      | Model deployment via FastAPI                          |
| **Simulate**  | `simulate.py` | Live inference logging (simulated production traffic) |
| **Monitor**   | `monitor.py`  | Drift detection & model performance monitoring        |
| **MLflow UI** | —             | Centralized tracking & reproducibility                |

---

## 🔍 Quick Checklist Before Each Step

| Step | Must Be True | How to Check |
|------|--------------|--------------|
| 2 (Train) | Virtual env active | Prompt shows `(.venv)` |
| 3 (API) | `run_id.txt` exists | `Get-Content run_id.txt` shows run ID |
| 4 (Simulate) | API running | Visit `http://localhost:9696/health` returns JSON |
| 5 (Monitor) | `predictions.csv` exists | `Test-Path .\data\predictions.csv` returns True |

---

## ⚡ Quick Reference Commands

| Action | Command |
|--------|---------|
| Activate env | `.venv\Scripts\activate` |
| Start MLflow | `mlflow server --host 127.0.0.1 --port 5000` |
| Train model | `python train.py` |
| Run API | `python app.py` |
| Test API (manual) | Open `http://localhost:9696/docs` |
| Test API (pytest) | `pytest -q test_api.py` |
| Simulate traffic | `python simulate.py` |
| Generate report | `python monitor.py` |
| View drift report | Open `monitoring_report.html` |

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` in `monitor.py` | No predictions logged yet | Run `simulate.py` first to create `data/predictions.csv` |
| `❌ No predictions recorded` in `simulate.py` | API not running | Ensure `python app.py` is active in another terminal |
| `requests.exceptions.ConnectionError` | API unreachable | Check API is on port 9696; verify with `/health` endpoint |
| `mlflow.pyfunc.load_model` fails | MLflow server not running or wrong run_id | Start MLflow server; re-run `train.py` |
| `ModuleNotFoundError: evidently` | Package not installed | `pip install evidently` |
| Port already in use (9696 or 5000) | Another process using port | Change port in command or stop conflicting process |

---

## 🎓 Conceptual Summary

✅ **End result:**
You've built a *complete, monitored ML system*:

* **Trained** → **Logged** → **Served** → **Queried** → **Monitored**

This is a full **"MLOps loop"** in four scripts — perfect for experiencing deployment + monitoring in one coherent flow.

---

## API Endpoints Reference

| Endpoint | Purpose |
|----------|---------|
| `/`      | Welcome message |
| `/health` | Service + model status |
| `/predict` | Make a duration prediction |
| `/docs` | Interactive Swagger UI |

---

## Behind the Scenes

- **`train.py`**: Downloads January 2023 data → filters & engineers features (`PU_DO`, `trip_distance`) → trains sklearn Pipeline (DictVectorizer + XGBRegressor) → logs to MLflow → writes `run_id.txt`

- **`app.py`**: On startup reads `run_id.txt` → loads Pipeline artifact from MLflow → `/predict` builds feature dict and returns prediction

- **`simulate.py`**: Loads real taxi data → sends batch requests to `/predict` → collects predictions + ground truth → appends to CSV log

- **`monitor.py`**: Reads prediction log → splits into reference/current windows → generates Evidently HTML report showing data & performance drift

- **MLflow UI**: Inspect all runs, parameters, metrics, and artifacts at http://localhost:5000

---

## 📚 Additional Notes

- Feature schema for prediction: `PULocationID` (int), `DOLocationID` (int), `trip_distance` (float)
- Internally combined into `PU_DO` categorical feature then vectorized
- Ground truth `duration` computed from pickup/dropoff timestamps
- Evidently compares distributions between reference and current data windows
- Optional: Set `MLFLOW_TRACKING_URI` environment variable if using non-default server

