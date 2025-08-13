## 📈 Market Master — Component‑Weighted Index Signal (1‑Hour) MLOps Pipeline

A production‑quality local MLOps pipeline that predicts 5‑class trading actions per index constituent and aggregates them into a component‑weighted index signal (e.g., NASDAQ‑100 → /NQ). Built with scikit‑learn, MLflow (with Model Registry), FastAPI, and pytest.

---

## 📋 Problem & Solution
Modern markets produce continuous OHLCV data, making multi‑indicator decisions hard to execute consistently. We frame the problem at the index level using constituent signals:

### Operational framing (1‑hour constituents and index aggregation)
- Scope: Multiple symbols (index constituents, e.g., NASDAQ‑100/QQQ), intraday 1‑hour OHLCV candles.
- Horizon & cadence: Predict the next hour’s action (t+1) per symbol after each completed hourly candle; aggregate to an index signal once per hour.
- History window: Use the latest W bars (e.g., 12–20 hours) for features; first W bars are warm‑up (no signal).
- Labeling: Map next‑bar return to 5 classes via thresholds (t1 < t2). For 1‑hour bars, prefer dynamic thresholds (quantile- or volatility-based) validated via CV.
- Per‑symbol outputs per hour: action, per‑class probabilities, confidence (max prob), timestamp_next.
- Index aggregation: Convert actions to scores {strong_buy:+2, buy:+1, hold:0, sell:−1, strong_sell:−2}; compute Weighted Sentiment Score (WSS) = Σ(score_i × weight_i).
- Index decision policy: If WSS ≥ +0.5 → BUY futures (/NQ). If WSS ≤ −0.5 → SELL futures. Otherwise HOLD. Tune thresholds by backtest.
- Batch mode: Provide 21‑day 1‑hour CSVs for all constituents → receive per‑symbol tables, WSS time series, and final hourly signals for offline evaluation/backtesting.
 - MVP scope: To move fast and cover the majority of index weight, we start with the top‑10 QQQ constituents (normalized weights). This can be expanded to the full NASDAQ‑100 once the pipeline is validated.

### Solution overview (implemented)
- 1‑hour OHLCV for index constituents (top‑10 QQQ in MVP).
- Pooled training across constituents with a symbol feature; multiple candidates compared; champion selected.
- Per‑symbol predictions aggregated to WSS to emit an /NQ signal; logged to MLflow.
- Champion served via FastAPI, loading from MLflow Model Registry alias `@Production`.

### 💼 Business impact
- **Consistency in decisions**: Clear actions (strong_sell/sell/hold/buy/strong_buy) per constituent reduce emotional noise.
- **Operational efficiency**: Automates multi‑indicator synthesis; hourly index‑level signal for faster decisioning.
- **Risk awareness**: Per‑class probabilities enable confidence‑based thresholds and risk controls.
- **Path to scale**: Structure extends to full NASDAQ‑100, registry‑based promotion, and cloud later.

## 🚀 Features
- Pooled training and model comparison (RF/ET/GB/HGBT/MLP/SVC/LogReg) with MLflow nested runs; champion persisted.
- MLflow Model Registry integration with aliases (Staging/Production); API serves `models:/market-master-component-classifier@Production`.
- Per‑symbol predictions and WSS aggregation for index signal; logs artifacts/metrics to MLflow.
- Monitoring with Evidently (drift and classification quality); gating criteria and promotion utility.
- FastAPI with `/health`, `GET /predict/component`, `GET /signal/index`, and `POST /predict`.

---

## 🛠️ Stack

| Layer            | Tool(s)                                                                              |
|------------------|--------------------------------------------------------------------------------------|
| 🧠 Model         | scikit‑learn candidates (RF/ET/GB/HGBT/MLP/SVC/LogReg), LabelEncoder                 |
| ⚙️ Features      | pandas, NumPy (returns, rolling stats, RSI‑like, MA, ATR, BB width)                  |
| 📊 Monitoring    | Evidently (implemented)                                                               |
| ⚙️ Orchestration | Runner script (`src/run_pipeline.py`) + Windows Task Scheduler; Prefect 2 (optional) |
| 📦 Tracking      | MLflow (SQLite) + Model Registry with aliases                                         |
| 🌐 API           | FastAPI + Uvicorn                                                                    |
| 🛢️ Storage       | Local files (CSV, `artifacts/`, `mlruns/`)                                           |
| 🧪 Testing       | pytest                                                                               |

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

3) Install deps
```bash
pip install -r requirements.txt
```

4) Start MLflow UI (optional but recommended)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
# Open http://localhost:5000 in your browser
```

5) Run the end‑to‑end pipeline (fetch → train/compare → predict index → monitor → gate → promote)
```bash
python -m src.run_pipeline --interval 1h --days 30 --max-symbols 10
```

6) Serve the API (loads Production model from MLflow registry with local fallback)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for Swagger UI
```

7) Sample predictions
- Single component (latest history from CSV):
```bash
curl "http://localhost:8000/predict/component?symbol=NVDA"
```
- Index signal (WSS aggregation):
```bash
curl "http://localhost:8000/signal/index?universe=qqq"
```
- POST /predict with bars (JSON body):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "symbol": "NVDA",
        "interval": "1h",
        "bars": [
          {"open": 101.2, "high": 102.1, "low": 100.5, "close": 101.7, "volume": 123456}
        ]
      }'
```
Response includes `action` in {"strong_sell","sell","hold","buy","strong_buy"} and per‑class probabilities.
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
├── requirements.txt
├── data/
│   ├── components/                  # per‑symbol 1h OHLCV CSVs
│   └── weights/qqq_weights.csv      # normalized QQQ weights (top‑10 in MVP)
├── mlruns/                          # MLflow artifacts (created at run)
├── src/
│   ├── features.py                  # indicators (returns, rolling stats, RSI, MA, ATR, BB width)
│   ├── data.py                      # dataset utilities and metadata persistence
│   ├── fetch_symbol.py              # robust yfinance fetcher with market‑hours filter
│   ├── fetch_components.py          # orchestrates multi‑symbol fetch
│   ├── fetch_weights_qqq.py         # fetch/normalize QQQ weights (with fallback)
│   ├── model_candidates.py          # candidate estimators for comparison
│   ├── train_pooled_compare.py      # pooled training, MLflow logging, registry registration
│   ├── predict_index.py             # per‑symbol predictions → WSS → index signal
│   ├── monitor_drift.py             # Evidently drift + classification quality reports
│   ├── gate_and_report.py           # promotion gating and summary
│   ├── registry.py                  # promote/rollback aliases (Staging/Production)
│   ├── run_pipeline.py              # end‑to‑end runner (local orchestration)
│   ├── hourly_predict.py            # hourly scheduler entrypoint (Windows Task Scheduler)
│   └── api.py                       # FastAPI serving (registry‑first model load)
└── tests/
    ├── test_features.py             # unit tests for feature funcs
    └── test_api.py                  # API smoke test
```

---

## 📊 Models & Metrics

- **Target**: 5‑class action: {strong_sell, sell, hold, buy, strong_buy}.
- **Candidates**: RF/ET/GB/HGBT/MLP/SVC/LogReg trained on pooled dataset (one‑hot `symbol`).
- **Features**: log‑returns, rolling mean/volatility, RSI‑like momentum, MAs (short/long), ATR, Bollinger Band width, range ratios.
- **Selection**: macro F1 on time‑based split; accuracy as tiebreaker.
- **Logging (MLflow)**: params, metrics, per‑candidate artifacts; champion registered as `market-master-component-classifier` with alias `Staging` → gated promotion to `Production`.

---

## ✅ Evaluation Readiness
 The project is structured for easy grading and local execution.

- **Problem description**: Clearly defined 1‑hour constituent signals aggregated to an index decision.
- **Cloud**: Local‑only. Docker/compose optional later.
- **Experiment tracking & registry**: MLflow tracking implemented; Model Registry with aliases implemented.
- **Workflow orchestration**: Local runner (`src/run_pipeline.py`) and Windows Task Scheduler for hourly runs; Prefect flow included as optional (version‑pin pending).
- **Model deployment**: FastAPI with `/health`, `/predict/component`, `/signal/index`, and `/predict`; loads registry `@Production` model with artifact fallback.
- **Monitoring**: Evidently drift and classification quality reports saved to `data/monitoring/` and logged to MLflow.
- **Reproducibility**: Pinned requirements, deterministic seeds, and tests.
- **Best practices**: Unit tests, API smoke test; gating and rollback utilities.

---

## 🔧 Configuration (Local)
Environment variables are optional. Defaults are embedded for local use.
- `MLFLOW_TRACKING_URI`: `sqlite:///mlflow.db`
- `MODEL_URI`: `models:/market-master-component-classifier@Production` (API tries this first; falls back to `artifacts/model/model.pkl`)

---

## 🚆 Execution Flow (Local)
1) Fetch: QQQ weights → 1‑hour OHLCV per constituent to `data/components/`.
2) Features/labels: rolling indicators and dynamic thresholds; persist metadata.
3) Train/compare: candidates on pooled dataset; select champion; log to MLflow and register as Staging.
4) Predict index: per‑symbol actions → WSS → /NQ signal; log table/WSS to MLflow.
5) Monitor: Evidently drift and classification quality reports; log to MLflow.
6) Gate: evaluate drift and F1 thresholds; if pass → promote Staging → Production.
7) Serve: API loads Production model via registry.

---

## 📌 Acceptance Criteria
- Fresh clone → set up → run pipeline → serve API → call endpoints → tests pass; no code edits required.
- MLflow UI shows runs for candidates, artifacts, WSS logs; Model Registry has aliases Staging/Production.
- API `/health` shows registry model; `/signal/index` returns WSS and signal.

---

## 🗺️ Roadmap
- Expand to full NASDAQ‑100 constituents and improve symbol metadata.
- Add probability calibration (e.g., Platt scaling) for selected models.
- Tighten gates (delta F1 vs champion, latency SLO) and auto‑rollback on alerts.
- Prefect pinning and UI deployment once dependency versions are aligned.
- Docker Compose for API + MLflow UI; minimal CI to run tests on push.


