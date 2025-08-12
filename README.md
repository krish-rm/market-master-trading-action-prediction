## 📈 Market Master — Trading Action Prediction MLOps Pipeline

A production-quality MLOps pipeline for multi-class trading action prediction — built with scikit-learn, MLflow, FastAPI, and pytest.

This project demonstrates the full local ML lifecycle: from training and experiment tracking to serving a prediction API and running tests.

---

## 📋 Problem & Solution
Modern markets produce continuous OHLCV data, making multi-indicator decisions hard to execute consistently. We scope the task to a single asset and predict the next trading action among five classes using a small, practical set of indicators (returns, rolling stats, RSI-like momentum, moving averages, ATR, Bollinger Band width).

Our pipeline trains a baseline classifier (RandomForest), logs metrics/artifacts to MLflow, and serves predictions via FastAPI with class probabilities for risk-aware decision thresholds.

### Solution overview
- **Small, focused feature set**: log-returns, rolling mean/volatility, momentum (RSI-like), moving averages, ATR, and Bollinger Band width.
- **Baseline model first**: scikit-learn classifier (RandomForest to start; LogisticRegression as a simple alternative) that is fast and easy to iterate on.
- **Local experiment tracking**: MLflow to log metrics, parameters, and the model artifact.
- **Local serving**: FastAPI endpoint for real-time predictions from the trained artifact.
- **Tests**: unit tests for features and an API smoke test to ensure the core path works.

### 💼 Business impact (single-asset local prototype)
- **Consistency in decisions**: Map indicator signals to clear actions (strong_sell/sell/hold/buy/strong_buy), reducing emotional noise.
- **Operational efficiency**: Automates multi-indicator synthesis, reducing manual analysis time.
- **Risk awareness**: Multi-class probabilities support confidence-based thresholds and risk controls.
- **Path to scale**: Validated locally, the same structure extends to model registry, monitoring, and cloud later.

## 🚀 Features
- **Training script** that: loads CSV, engineers practical indicators, trains a scikit-learn classifier for 5 actions, logs metrics and artifacts to local MLflow.
- **Prediction API (FastAPI)** that loads the saved model artifact (or via MLflow URI) and returns one of {strong_sell, sell, hold, buy, strong_buy} with class probabilities.
- **Tests (pytest)** for features and API smoke tests.
- **Reproducible runs** with a tiny sample dataset in `data/raw/` and local MLflow in `./mlruns`.

---

## 🛠️ Stack

| Layer            | Tool(s)                                                                              |
|------------------|--------------------------------------------------------------------------------------|
| 🧠 Model         | scikit-learn (RandomForest baseline), LabelEncoder for action labels                 |
| ⚙️ Features      | pandas, NumPy (OHLCV-derived: returns, rolling stats, RSI-like, MA, ATR, BB width)  |
| 📊 Monitoring    | Evidently (planned)                                                                  |
| ⚙️ Orchestration | Prefect 2 (planned)                                                                  |
| 📦 Tracking      | MLflow (local `./mlruns`, SQLite backend by default)                                 |
| 🌐 API           | FastAPI + Uvicorn                                                                    |
| 🛢️ Storage       | Local files (CSV, `artifacts/`); PostgreSQL via Docker Compose (optional, later)     |
| 🧪 Testing       | pytest                                                                               |

Items marked “planned” will be added after the core local loop is stable and verified.

---

## 🐳 Quickstart (Local)

1) Clone and enter the repo
```bash
git clone <your-repo-url>
cd market-master-trading-prediction-system
```

2) Create virtual environment
- Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```
- macOS/Linux (bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

3) Install deps (we will add `requirements.txt` next)
```bash
pip install -r requirements.txt
```

4) (Optional) Start MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
# Open http://localhost:5000 in your browser
```

5) Train the model (script coming in `src/train.py`)
```bash
python src/train.py
```
This will: read `data/raw/sample_ohlcv.csv`, create features, train a 5-class model, and log params/metrics/artifacts to MLflow. The final model is also saved to `artifacts/model/`.

6) Serve the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for Swagger UI
```

7) Sample prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"open": 101.2, "high": 102.1, "low": 100.5, "close": 101.7, "volume": 123456}'
```
Response will be a JSON with `action` in {"strong_sell","sell","hold","buy","strong_buy"} and per-class probabilities.

8) Run tests
```bash
pytest -q
```

---

## ⚙️ Project Structure
```bash
.
├── README.md
├── requirements.txt                 # to be added
├── data/
│   └── raw/
│       └── sample_ohlcv.csv         # tiny sample dataset (to be added)
├── mlruns/                          # local MLflow artifacts (created at run)
├── src/
│   ├── data.py                      # load/split data
│   ├── features.py                  # indicators (returns, rolling stats, RSI, MA, ATR, BB width)
│   ├── train.py                     # train multi-class model + log to MLflow + persist artifact
│   ├── infer.py                     # CLI/batch inference
│   └── api.py                       # FastAPI serving (loads latest model)
└── tests/
    ├── test_features.py             # unit tests for feature funcs
    └── test_api.py                  # API smoke test
```

---

## 📊 Models & Metrics

- **Target**: 5-class action label: {strong_sell, sell, hold, buy, strong_buy}.
- **Baseline model**: `RandomForestClassifier` (robust default for tabular features).
- **Alternative**: `LogisticRegression` (simple/explainable baseline for comparison).
- **Features**: log-returns, rolling mean/volatility, momentum (RSI-like), moving averages (short/long), ATR, Bollinger Band width, and simple price range ratios.
- **Metrics logged (MLflow)**: overall accuracy, macro F1, per-class precision/recall/F1, confusion matrix artifact, and PR/ROC where applicable.

Example outcomes from the earlier implementation (for reference, will be recomputed locally on this dataset):
- Accuracy ~ 0.69
- Macro F1 ~ 0.72
- Inference latency < 1s per sample

We will treat these as directional targets only and rely on the new local dataset and runs for authoritative metrics.

---

## ✅ Evaluation Readiness (per `project-requirement.txt`)
 The project follows the rubric in `project-requirement.txt` and is built to be graded easily.

- **Problem description**: Clearly defined, narrow trading objective (5-class trading actions for one asset) with rationale and constraints.
- **Cloud**: Local-only for now (0–2 points). A minimal, optional Dockerization and later registry can enable easy promotion to cloud when we’re ready.
- **Experiment tracking and model registry**: MLflow tracking enabled from day one. Registry to be added after confirmation (current: tracking; target: tracking + registry).
- **Workflow orchestration**: Initially manual CLI steps for clarity. A simple workflow (e.g., Prefect) can be added to run train → evaluate → persist as one flow after local path is verified.
- **Model deployment**: Local FastAPI service with `/predict` and `/health` and a repeatable model load path.
- **Model monitoring**: Basic local metrics via MLflow; we plan to add a minimal Evidently drift report generated on a held-out slice.
- **Reproducibility**: Clear instructions, pinned `requirements.txt`, tiny sample dataset included, deterministic seeds, and tests.
- **Best practices**: Unit tests, integration (API) test, formatter/linter, optional Makefile, and later a minimal CI on push.

We will mark each criterion as “Implemented” or “Planned” in the repository once we add the corresponding files.

---

## 🔧 Configuration (Local)
Environment variables are optional. Reasonable defaults will be embedded for local use.
- `MLFLOW_TRACKING_URI`: default `file:./mlruns`
- `MODEL_URI`: default to latest saved artifact in `artifacts/model/`

---

## 🚆 Execution Flow (Local)
1) Data: read `data/raw/sample_ohlcv.csv` → drop NAs → train/test split.
2) Features: compute small set of indicators (no over-engineering).
3) Train: scikit-learn baseline (e.g., LogisticRegression or RandomForest) → log params/metrics/artifacts to MLflow.
4) Serve: load saved model (local path or `models:/name@alias` when we enable registry) → expose `/health` and `/predict`.
5) Test: unit tests for features, an API smoke test, and a tiny integration check in CI later.

---

## 📌 Acceptance Criteria
- Fresh clone → set up → train → serve API → make a prediction → tests pass, all on a local machine, without editing code.
- MLflow UI shows at least one run with 5-class metrics and a stored model artifact.
- No unused services, no broken instructions.

---

## 🗺️ Roadmap (post-confirmation)
- Add a tiny `Makefile`/`tasks.py` for convenience (optional on Windows).
- Add MLflow Model Registry and `@production` alias for API loading.
- Add basic monitoring (Evidently) and a drift report generated locally.
- Swap the sample CSV for a small real OHLCV snapshot with licensing-safe source and attribution.
- Add Docker for local reproducibility (compose for MLflow UI).
- Only after all the above are stable, consider a minimal cloud target.


