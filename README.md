## 📈 Market Master — Trading Action Prediction MLOps Pipeline

A production-quality MLOps pipeline for multi-class trading action prediction — built with scikit-learn, MLflow, FastAPI, and pytest.

This project demonstrates the full local ML lifecycle: from training and experiment tracking to serving a prediction API and running tests. We now plan to scale from a single-symbol 5‑minute prototype to a real fintech scenario using 1‑hour OHLCV and a component‑weighted index signal (e.g., NASDAQ‑100 → /NQ).

---

## 📋 Problem & Solution
Modern markets produce continuous OHLCV data, making multi-indicator decisions hard to execute consistently.
- Phase A (done): single-symbol 5‑class action prediction from recent OHLCV.
- Phase B (next): component‑weighted index signal using 1‑hour OHLCV for index constituents (e.g., NASDAQ‑100) and a pooled champion classifier.

Our pipeline trains classifiers, logs to MLflow, serves a champion via FastAPI, and will aggregate per‑component predictions into an index futures signal.

### Operational framing (Phase A, example: NVDA, 5‑minute bars)
- **Scope**: Single asset, intraday 5‑minute OHLCV candles.
- **Horizon & cadence**: Predict the next bar’s action (t+1) after each completed candle; ~78 predictions per full trading day.
- **History window**: Use the latest W bars (e.g., 20–60) to compute rolling features; first W bars are warm‑up (no signal).
- **Labeling**: Forward return r_next mapped to 5 classes via thresholds t1 < t2 (strong_sell, sell, hold, buy, strong_buy) with no look‑ahead.
- **API outputs per bar**: `action`, per‑class `probabilities`, `confidence` (max prob), `timestamp_next`, and `model_version` metadata.
- **Example decision policy**: If `confidence ≥ 0.6`, act per class mapping; otherwise `hold`. Pair with risk controls (position limits, SL/TP, flat by close).
- **Batch mode**: Provide a 10‑day CSV → receive a table of timestamps, actions, and probabilities for offline evaluation/backtesting.

### Operational framing (Phase B, 1‑hour constituents and index aggregation)
- Scope: Multiple symbols (index constituents, e.g., NASDAQ‑100/QQQ), intraday 1‑hour OHLCV candles.
- Horizon & cadence: Predict the next hour’s action (t+1) per symbol after each completed hourly candle; aggregate to an index signal once per hour.
- History window: Use the latest W bars (e.g., 12–20 hours) for features; first W bars are warm‑up (no signal).
- Labeling: Map next‑bar return to 5 classes via thresholds (t1 < t2). For 1‑hour bars, prefer dynamic thresholds (quantile- or volatility-based) validated via CV.
- Per‑symbol outputs per hour: action, per‑class probabilities, confidence (max prob), timestamp_next.
- Index aggregation: Convert actions to scores {strong_buy:+2, buy:+1, hold:0, sell:−1, strong_sell:−2}; compute Weighted Sentiment Score (WSS) = Σ(score_i × weight_i).
- Index decision policy: If WSS ≥ +0.5 → BUY futures (/NQ). If WSS ≤ −0.5 → SELL futures. Otherwise HOLD. Tune thresholds by backtest.
- Batch mode: Provide 21‑day 1‑hour CSVs for all constituents → receive per‑symbol tables, WSS time series, and final hourly signals for offline evaluation/backtesting.

### Solution overview
- Phase A (done): focused features; 5‑class labels; RF baseline with candidate comparison (RF/ET/GB/HGBT/MLP/SVC/LogReg); MLflow + FastAPI; tests.
- Phase B (next): 1‑hour OHLCV for index constituents; per‑symbol predictions + weighted aggregation (WSS) to emit /NQ signal; pooled training with symbol feature; champion–challenger.

### Phase B (in progress)
- Switching to 1‑hour OHLCV and pooled training across index constituents (e.g., NASDAQ‑100).
- Aggregating per‑symbol actions to WSS and emitting /NQ signals; logging per‑symbol tables and WSS to MLflow.

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

Example outcomes from earlier prototypes are directional targets only. Actual results depend on dataset, interval (5‑min vs 1‑hour), and seeds.

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
- Add MLflow Model Registry and `@Production` alias for serving.
- Add basic monitoring (Evidently) and a drift report generated locally.
- Add Prefect flow for fetch → train_compare → gate → promote → predict_index.
- Add Docker for local reproducibility (compose for MLflow UI + API).
- Only after all the above are stable, consider a minimal cloud target.


