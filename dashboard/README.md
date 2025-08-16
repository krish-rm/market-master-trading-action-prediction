# 📊 Dashboard

This directory contains the interactive Streamlit dashboard for the Market Master Trading System.

## 🎯 Streamlit App

### `streamlit_app.py`
Interactive web dashboard that provides real-time analysis of QQQ constituents with ML-powered trading predictions.

**Features:**
- 📈 Real-time stock prices for QQQ constituents
- 🤖 ML-powered individual stock predictions
- 📊 Weighted sentiment score calculation
- 🎯 Futures trading signals (/NQ)
- 📊 Technical analysis and risk metrics
- 🔄 Auto-refresh every 30 seconds

### 🚀 Usage

```bash
# From project root
make streamlit-dashboard

# Or directly
streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 🌐 Access
- **URL**: http://localhost:8501
- **Port**: 8501 (configurable)

### 📋 Requirements
- Streamlit 1.32.0+
- Plotly 5.19.0+
- yfinance 0.2.65+
- All other dependencies from main requirements.txt

### 🎨 Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Data**: Live stock prices and predictions
- **Interactive Charts**: Candlestick charts, volume analysis
- **Auto-refresh**: Updates every 30 seconds
- **Professional UI**: Custom CSS styling

### 🔧 Configuration
The dashboard can be configured by modifying:
- Refresh intervals
- Number of stocks displayed
- Chart timeframes
- Risk thresholds

### 📱 Dashboard Sections
1. **Live Prices**: Real-time stock price cards
2. **Predictions**: ML model predictions for each stock
3. **Analysis**: Technical analysis and charts
4. **Trading Signals**: Futures trading recommendations
5. **Risk Metrics**: Volatility and risk assessment
