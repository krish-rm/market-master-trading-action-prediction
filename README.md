## 📈 Market Master — Component-Weighted Hourly Index Prediction MLOps Pipeline

<div align="center">
  <img src="docs/concept_diagram.png" alt="Concept Diagram" width="500" />
</div>

A complete end-to-end machine learning pipeline that predicts the next-hour trading action for each major stock in an index, then combines these into a single weighted signal for the index itself (e.g., NASDAQ-100 → /NQ). The system uses historical price data, technical indicators, and a trained model to give clear buy, sell, or hold signals for index futures trading.

---

## Problem
Index futures like NASDAQ-100 (/NQ) are a popular way to trade the overall direction of the market, especially for tech-heavy portfolios. The index is made up of many constituent stocks — companies like Apple, Microsoft, Amazon, and Nvidia — each assigned a specific weight based on its market value. Larger companies influence the index more. Traders watch the index because it acts as a single benchmark for multiple companies, but its movement is really the result of all constituents combined.

Manually tracking dozens of stocks, calculating indicators, and then deciding the next move for the index is challenging. Each stock may give different signals, and without a structured way to combine them, decisions can be slow, inconsistent, and prone to emotional bias.

## Solution
Our system automates this process using machine learning and a clear aggregation method. Every hour, we collect OHLCV (Open, High, Low, Close, Volume) data for the top-weighted NASDAQ-100 stocks. For each stock, we compute technical features and indicators, then predict the next hour’s price movement using a trained model. Each prediction is mapped to a score — from strong buy to strong sell — and weighted according to the stock’s influence in the index.

These weighted scores are combined into a Weighted Sentiment Score (WSS), representing the predicted direction of the index for the next hour. If the WSS exceeds a buy threshold, we issue a “buy” signal for index futures; if it falls below a sell threshold, we issue a “sell” signal; otherwise, we hold.

This approach ensures consistent, data-driven decisions, removes human guesswork, and allows traders to act confidently on the likely direction of the index in the coming hour — ideal for trading index derivatives.

### Solution overview 
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
| 📊 Monitoring    | Evidently                                                                            |
| ⚙️ Orchestration | Prefect 2 (task-based workflow orchestration) |
| 📦 Tracking      | MLflow (SQLite) + Model Registry with aliases                                         |
| 🌐 API           | FastAPI + Uvicorn                                                                    |
| 🛢️ Storage       | Local files (CSV, `artifacts/`, `mlruns/`)                                           |
| 🧪 Testing       | pytest                                                                               |
| 🐳 Containerization | Docker + Docker Compose (API + MLflow UI)                                           |
| 🔄 CI/CD         | GitHub Actions (tests, linting, pipeline smoke test)                                 |

---

## 📊 Models & Metrics

- **Target**: 5‑class action: {strong_sell, sell, hold, buy, strong_buy}.
- **Candidates**: RF/ET/GB/HGBT/MLP/SVC/LogReg trained on pooled dataset (one‑hot `symbol`).
- **Features**: log‑returns, rolling mean/volatility, RSI‑like momentum, MAs (short/long), ATR, Bollinger Band width, range ratios.
- **Selection**: macro F1 on time‑based split; accuracy as tiebreaker.
- **Logging (MLflow)**: params, metrics, per‑candidate artifacts; champion registered as `market-master-component-classifier` with alias `Staging` → gated promotion to `Production`.


---

## 🚆 Execution Flow (Local)
1) **Fetch**: QQQ weights → 1‑hour OHLCV per constituent to `data/components/`.
2) **Features/labels**: rolling indicators and dynamic thresholds; persist metadata.
3) **Train/compare**: candidates on pooled dataset; select champion; log to MLflow and register as Staging.
4) **Predict index**: per‑symbol actions → WSS → /NQ signal; log table/WSS to MLflow.
5) **Monitor**: Evidently drift and classification quality reports; log to MLflow.
6) **Gate**: evaluate drift and F1 thresholds; if pass → promote Staging → Production.
7) **Serve**: API loads Production model via registry.

**Orchestration**: All steps are managed by Prefect tasks with proper logging, error handling, and conditional deployment.


---


## 🐳 Quickstart (Local)

### Option A: Makefile Commands 

1) Clone and enter the repo
```bash
git clone <your-repo-url>
cd market-master-trading-action-prediction-system
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

3) Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes both production and development dependencies (black, flake8, isort, mypy, pre-commit) for code quality tools.

4) Run the orchestrated pipeline
```bash
# Run complete orchestrated pipeline (recommended)
make prefect-flow

# Alternative: Run non-orchestrated pipeline
make pipeline
```

5) Start monitoring and serving
```bash
# Start MLflow UI for experiment tracking (open in new terminal)
make mlflow-ui

# Start model serving API (open in new terminal)
make model-serving

# Start Prefect server (open in new terminal)
make prefect-start
```

6) Access services
- API: http://localhost:8001/docs (Swagger UI)
- MLflow UI: http://localhost:5000
- Prefect dashboard: http://localhost:4200

**💡 Tip**: See the [Available Commands](#️-available-commands) section below for all available options.

### Option B: Docker Setup

1) Clone and enter the repo
```bash
git clone <your-repo-url>
cd market-master-trading-prediction-system
```

2) Run the pipeline and start services
```bash
# Run the full pipeline first
make prefect-flow

# Start services with Docker Compose
docker-compose up -d
```

3) Access services
- API: http://localhost:8000/docs (Swagger UI)
- MLflow UI: http://localhost:5000

**💡 Tip**: See the [Available Commands](#️-available-commands) section below for all available options.


## 🛠️ Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `make prefect-flow` | Run complete orchestrated pipeline | **Main command** - Start here |
| `make pipeline` | Run full pipeline (non-orchestrated) | Alternative to prefect-flow |
| `make smoke-test` | Quick pipeline + API test | Testing the system |
| `make clean` | Remove all artifacts and data | Fresh start |
| `make install` | Install dependencies | First time setup |
| `make install-dev` | Install + pre-commit hooks | Development setup |
| `make mlflow-ui` | Start MLflow experiment tracking | Monitor experiments |
| `make model-serving` | Start model serving API | Serve predictions |
| `make model-serving-test` | Test model serving endpoints | Validate API |
| `make prefect-start` | Start Prefect server | Advanced orchestration |
| `make prefect-worker` | Start Prefect worker | Advanced orchestration |
| `make prefect-deploy` | Deploy scheduled flows | Advanced orchestration |
| `make test` | Run all tests | Quality assurance |
| `make test-unit` | Run unit tests only | Component testing |
| `make test-integration` | Run integration tests only | System testing |
| `make lint` | Run code linting | Code quality |
| `make format` | Format code with black | Code formatting |
| `make type-check` | Run type checking | Code quality |
| `make docker-build` | Build Docker image | Containerization |
| `make docker-run` | Start Docker services | Container deployment |
| `make promote-staging` | Promote model to staging | Model management |
| `make rollback-production` | Rollback production model | Model management |


### **Model Serving API Endpoints**
When using `make model-serving`, the following endpoints are available:
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model-info` - Model information
- `POST /reload-model` - Reload model from registry

---

7) Sample predictions
- Single component (latest history from CSV):
```bash
curl "http://localhost:8000/predict/component?symbol=NVDA"
```
- Index signal (WSS aggregation):
```bash
curl "http://localhost:8000/signal/index?universe=qqq"
```
- POST /predict with custom bars (requires 21+ bars for warm-up):
```bash
# PowerShell (Windows)
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{
  "symbol": "NVDA",
  "interval": "1h", 
  "bars": [
    {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 100000},
    {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 120000},
    {"open": 101.5, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 150000},
    {"open": 102.5, "high": 104.0, "low": 102.0, "close": 103.5, "volume": 180000},
    {"open": 103.5, "high": 105.0, "low": 103.0, "close": 104.5, "volume": 200000},
    {"open": 104.5, "high": 106.0, "low": 104.0, "close": 105.5, "volume": 220000},
    {"open": 105.5, "high": 107.0, "low": 105.0, "close": 106.5, "volume": 250000},
    {"open": 106.5, "high": 108.0, "low": 106.0, "close": 107.5, "volume": 280000},
    {"open": 107.5, "high": 109.0, "low": 107.0, "close": 108.5, "volume": 300000},
    {"open": 108.5, "high": 110.0, "low": 108.0, "close": 109.5, "volume": 320000},
    {"open": 109.5, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 350000},
    {"open": 110.5, "high": 112.0, "low": 110.0, "close": 111.5, "volume": 380000},
    {"open": 111.5, "high": 113.0, "low": 111.0, "close": 112.5, "volume": 400000},
    {"open": 112.5, "high": 114.0, "low": 112.0, "close": 113.5, "volume": 420000},
    {"open": 113.5, "high": 115.0, "low": 113.0, "close": 114.5, "volume": 450000},
    {"open": 114.5, "high": 116.0, "low": 114.0, "close": 115.5, "volume": 480000},
    {"open": 115.5, "high": 117.0, "low": 115.0, "close": 116.5, "volume": 500000},
    {"open": 116.5, "high": 118.0, "low": 116.0, "close": 117.5, "volume": 520000},
    {"open": 117.5, "high": 119.0, "low": 117.0, "close": 118.5, "volume": 550000},
    {"open": 118.5, "high": 120.0, "low": 118.0, "close": 119.5, "volume": 580000},
    {"open": 119.5, "high": 121.0, "low": 119.0, "close": 120.5, "volume": 600000},
    {"open": 120.5, "high": 122.0, "low": 120.0, "close": 121.5, "volume": 620000},
    {"open": 121.5, "high": 123.0, "low": 121.0, "close": 122.5, "volume": 650000},
    {"open": 122.5, "high": 124.0, "low": 122.0, "close": 123.5, "volume": 680000},
    {"open": 123.5, "high": 125.0, "low": 123.0, "close": 124.5, "volume": 700000}
  ]
}'

# Bash/Linux/macOS
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NVDA",
    "interval": "1h",
    "bars": [
      {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 100000},
      {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 120000},
      {"open": 101.5, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 150000},
      {"open": 102.5, "high": 104.0, "low": 102.0, "close": 103.5, "volume": 180000},
      {"open": 103.5, "high": 105.0, "low": 103.0, "close": 104.5, "volume": 200000},
      {"open": 104.5, "high": 106.0, "low": 104.0, "close": 105.5, "volume": 220000},
      {"open": 105.5, "high": 107.0, "low": 105.0, "close": 106.5, "volume": 250000},
      {"open": 106.5, "high": 108.0, "low": 106.0, "close": 107.5, "volume": 280000},
      {"open": 107.5, "high": 109.0, "low": 107.0, "close": 108.5, "volume": 300000},
      {"open": 108.5, "high": 110.0, "low": 108.0, "close": 109.5, "volume": 320000},
      {"open": 109.5, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 350000},
      {"open": 110.5, "high": 112.0, "low": 110.0, "close": 111.5, "volume": 380000},
      {"open": 111.5, "high": 113.0, "low": 111.0, "close": 112.5, "volume": 400000},
      {"open": 112.5, "high": 114.0, "low": 112.0, "close": 113.5, "volume": 420000},
      {"open": 113.5, "high": 115.0, "low": 113.0, "close": 114.5, "volume": 450000},
      {"open": 114.5, "high": 116.0, "low": 114.0, "close": 115.5, "volume": 480000},
      {"open": 115.5, "high": 117.0, "low": 115.0, "close": 116.5, "volume": 500000},
      {"open": 116.5, "high": 118.0, "low": 116.0, "close": 117.5, "volume": 520000},
      {"open": 117.5, "high": 119.0, "low": 117.0, "close": 118.5, "volume": 550000},
      {"open": 118.5, "high": 120.0, "low": 118.0, "close": 119.5, "volume": 580000},
      {"open": 119.5, "high": 121.0, "low": 119.0, "close": 120.5, "volume": 600000},
      {"open": 120.5, "high": 122.0, "low": 120.0, "close": 121.5, "volume": 620000},
      {"open": 121.5, "high": 123.0, "low": 121.0, "close": 122.5, "volume": 650000},
      {"open": 122.5, "high": 124.0, "low": 122.0, "close": 123.5, "volume": 680000},
      {"open": 123.5, "high": 125.0, "low": 123.0, "close": 124.5, "volume": 700000}
    ]
  }'
```

**Note**: The POST endpoint requires at least 21 bars of historical data due to rolling window calculations (20-period warm-up + 1 current bar). For daily use, prefer the GET endpoints which use stored CSV data.
Response includes `action` in {"strong_sell","sell","hold","buy","strong_buy"} and per‑class probabilities.

8) Run tests
```bash
# Using Makefile (recommended)
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only
make test-verbose      # Run tests with verbose output
make test-coverage     # Run tests with coverage report

# Direct pytest commands
pytest -q              # Run all tests
pytest tests/unit/ -q  # Run unit tests only
pytest tests/integration/ -q  # Run integration tests only
pytest tests/integration/ -q
```



---

## ⚙️ Project Structure
```bash
.
├── README.md
├── requirements.txt
├── Makefile                         # Convenience commands for development
├── Dockerfile                       # Containerized API service
├── docker-compose.yml               # Local environment (API + MLflow UI)
├── .github/workflows/ci.yml         # GitHub Actions CI/CD
├── data/
│   ├── components/                  # per‑symbol 1h OHLCV CSVs
│   ├── weights/qqq_weights.csv      # normalized QQQ weights (top‑10 in MVP)
│   └── monitoring/                  # Evidently drift and quality reports
├── mlruns/                          # MLflow artifacts (created at run)
├── artifacts/
│   ├── model/                       # Champion model and metadata
│   ├── models/                      # All candidate models
│   ├── index/                       # Index signal outputs
│   └── backtest/                    # Batch backtest results
├── src/
│   ├── features.py                  # indicators (returns, rolling stats, RSI, MA, ATR, BB width)
│   ├── data.py                      # dataset utilities and metadata persistence
│   ├── fetch_symbol.py              # robust yfinance fetcher with market‑hours filter
│   ├── fetch_components.py          # orchestrates multi‑symbol fetch (with sanity checks)
│   ├── fetch_weights_qqq.py         # fetch/normalize QQQ weights (with fallback)
│   ├── model_candidates.py          # candidate estimators for comparison
│   ├── train_pooled_compare.py      # pooled training, MLflow logging, registry registration
│   ├── predict_index.py             # per‑symbol predictions → WSS → index signal (with batch mode)
│   ├── monitor_drift.py             # Evidently drift + classification quality reports
│   ├── gate_and_report.py           # promotion gating and summary
│   ├── registry.py                  # promote/rollback aliases (Staging/Production)
│   ├── run_pipeline.py              # end‑to‑end runner (local orchestration)
│   ├── hourly_predict.py            # hourly scheduler entrypoint (Windows Task Scheduler)
│   └── api.py                       # FastAPI serving (registry‑first model load)
├── flows/
│   └── enhanced_orchestration.py    # Prefect flow definition
└── tests/
    ├── unit/                        # Unit tests
    │   ├── test_features.py         # Feature engineering tests
    │   ├── test_pipeline.py         # Pipeline component tests
    │   └── test_mlops.py            # MLOps functionality tests
    ├── integration/                 # Integration tests
    │   └── test_api.py              # API endpoint tests
    └── conftest.py                  # Shared test fixtures
```



---

## ✅ Evaluation Readiness
  The project is structured for easy grading and local execution.

- **Problem description**: Clearly defined 1‑hour constituent signals aggregated to an index decision.
- **Cloud**: Local‑only with Docker containerization for reproducibility.
- **Experiment tracking & registry**: MLflow tracking implemented; Model Registry with aliases implemented.
- **Workflow orchestration**: Prefect 2 task-based orchestration with proper logging, error handling, and conditional deployment.
- **Model deployment**: FastAPI with `/health`, `/predict/component`, `/signal/index`, and `/predict`; loads registry `@Production` model with artifact fallback.
- **Monitoring**: Evidently drift and classification quality reports saved to `data/monitoring/` and logged to MLflow.
- **Reproducibility**: Pinned requirements, deterministic seeds, tests, and Docker containerization.
- **Best practices**: 
  - ✅ **Unit tests**: Comprehensive test suite covering features, pipeline, and MLOps components
  - ✅ **Integration tests**: API endpoint testing with FastAPI TestClient
  - ✅ **Linter and formatter**: Black (code formatting), isort (import sorting), flake8 (linting), mypy (type checking)
  - ✅ **Makefile**: Comprehensive development and deployment commands
  - ✅ **Pre-commit hooks**: Automated code quality checks before commits
  - ✅ **CI/CD pipeline**: GitHub Actions with automated testing, linting, and pipeline smoke tests

---

## 🔧 Configuration (Local)
Environment variables are optional. Defaults are embedded for local use.
- `MLFLOW_TRACKING_URI`: `sqlite:///mlflow.db`
- `MODEL_URI`: `models:/market-master-component-classifier@Production` (API tries this first; falls back to `artifacts/model/model.pkl`)


---

## 🗺️ Roadmap
- Expand to full NASDAQ‑100 constituents and improve symbol metadata.
- Add probability calibration (e.g., Platt scaling) for selected models.
- Tighten gates (delta F1 vs champion, latency SLO) and auto‑rollback on alerts.
- Prefect UI deployment and advanced scheduling features.
- Cloud deployment (AWS/GCP) with managed MLflow and monitoring services.
- Real-time data streaming integration for live trading signals.


