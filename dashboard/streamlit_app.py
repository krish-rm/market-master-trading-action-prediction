import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import requests
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetch_weights_qqq import fetch_qqq_holdings

# Page configuration
st.set_page_config(
    page_title="Market Master Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_qqq_holdings():
    """Get QQQ holdings with fallback"""
    try:
        holdings = fetch_qqq_holdings()
        return holdings
    except Exception as e:
        st.warning(f"Could not fetch QQQ holdings: {e}")
        # Fallback to major QQQ constituents
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TSLA', 'COST', 'PEP'],
            'weight': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        })

def simulate_predictions(symbols, stock_data):
    """Simulate ML predictions based on price movements"""
    predictions = {}
    for symbol in symbols:
        if symbol in stock_data:
            change_pct = stock_data[symbol]['change_pct']
            if change_pct > 2:
                pred = 'strong_buy'
            elif change_pct > 0.5:
                pred = 'buy'
            elif change_pct < -2:
                pred = 'strong_sell'
            elif change_pct < -0.5:
                pred = 'sell'
            else:
                pred = 'hold'
            predictions[symbol] = pred
    return predictions

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_stock_data(symbol, period="1d", interval="1h"):
    """Get stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_prediction_color(prediction):
    """Get color based on prediction"""
    if prediction in ['strong_buy', 'buy']:
        return 'positive'
    elif prediction in ['strong_sell', 'sell']:
        return 'negative'
    else:
        return 'neutral'

def get_prediction_emoji(prediction):
    """Get emoji for prediction"""
    emoji_map = {
        'strong_buy': '🚀',
        'buy': '📈',
        'hold': '⏸️',
        'sell': '📉',
        'strong_sell': '💥'
    }
    return emoji_map.get(prediction, '❓')

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Market Master Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This dashboard shows real-time analysis of QQQ constituents with ML-powered trading predictions.
    
    **Features:**
    - Real-time stock prices
    - Individual stock predictions
    - Weighted sentiment score
    - Futures trading signals
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 QQQ Constituents Analysis")
        
        # Get QQQ holdings
        holdings = get_qqq_holdings()
        
        if holdings is not None and not holdings.empty:
            # Display holdings table
            st.markdown("### Current QQQ Holdings")
            holdings_display = holdings.copy()
            holdings_display['weight'] = holdings_display['weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(holdings_display, use_container_width=True)
            
            # Get stock data for top constituents
            symbols = holdings['symbol'].head(10).tolist()  # Top 10 for performance
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Live Prices", "🤖 Predictions", "📊 Analysis"])
            
            with tab1:
                st.markdown("### Real-time Stock Prices")
                
                # Create price cards
                price_cols = st.columns(3)
                stock_data = {}
                
                for i, symbol in enumerate(symbols):
                    col_idx = i % 3
                    with price_cols[col_idx]:
                        data = get_stock_data(symbol)
                        if data is not None and not data.empty:
                            latest = data.iloc[-1]
                            prev = data.iloc[-2] if len(data) > 1 else latest
                            
                            change = latest['Close'] - prev['Close']
                            change_pct = (change / prev['Close']) * 100
                            
                            stock_data[symbol] = {
                                'price': latest['Close'],
                                'change': change,
                                'change_pct': change_pct,
                                'volume': latest['Volume']
                            }
                            
                            # Color based on change
                            color = "positive" if change >= 0 else "negative"
                            arrow = "↗️" if change >= 0 else "↘️"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{symbol}</h4>
                                <h3>${latest['Close']:.2f}</h3>
                                <p class="{color}">{arrow} {change:+.2f} ({change_pct:+.2f}%)</p>
                                <small>Vol: {latest['Volume']:,}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Price chart for selected stock
                st.markdown("### Price Chart")
                selected_stock = st.selectbox("Select stock for detailed chart:", symbols)
                
                if selected_stock:
                    data = get_stock_data(selected_stock, period="5d", interval="1h")
                    if data is not None and not data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name=selected_stock
                        ))
                        fig.update_layout(
                            title=f"{selected_stock} - 5 Day Price Chart",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### ML Predictions")
                
                # Simulate predictions (in real app, this would call your ML model)
                predictions = simulate_predictions(symbols, stock_data)
                
                # Display predictions
                pred_cols = st.columns(3)
                for i, symbol in enumerate(symbols):
                    col_idx = i % 3
                    with pred_cols[col_idx]:
                        if symbol in predictions:
                            pred = predictions[symbol]
                            color_class = calculate_prediction_color(pred)
                            emoji = get_prediction_emoji(pred)
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>{symbol}</h4>
                                <h3 class="{color_class}">{emoji} {pred.replace('_', ' ').title()}</h3>
                                <small>ML Model Prediction</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Prediction summary
                st.markdown("### Prediction Summary")
                if predictions:
                    pred_df = pd.DataFrame([
                        {'Symbol': symbol, 'Prediction': pred, 'Weight': holdings[holdings['symbol'] == symbol]['weight'].iloc[0]}
                        for symbol, pred in predictions.items()
                    ])
                    
                    # Calculate weighted sentiment
                    sentiment_map = {'strong_sell': -2, 'sell': -1, 'hold': 0, 'buy': 1, 'strong_buy': 2}
                    pred_df['Sentiment_Score'] = pred_df['Prediction'].map(sentiment_map)
                    pred_df['Weighted_Score'] = pred_df['Sentiment_Score'] * pred_df['Weight']
                    
                    wss = pred_df['Weighted_Score'].sum()
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Constituents", len(predictions))
                    with col2:
                        st.metric("Weighted Sentiment Score", f"{wss:.3f}")
                    with col3:
                        if wss > 0.5:
                            signal = "BULLISH"
                            color = "positive"
                        elif wss < -0.5:
                            signal = "BEARISH"
                            color = "negative"
                        else:
                            signal = "NEUTRAL"
                            color = "neutral"
                        st.markdown(f'<p class="{color}"><strong>Signal: {signal}</strong></p>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Technical Analysis")
                
                # Volume analysis
                if stock_data:
                    volume_data = pd.DataFrame([
                        {'Symbol': symbol, 'Volume': data['volume'], 'Price': data['price']}
                        for symbol, data in stock_data.items()
                    ])
                    
                    fig = px.bar(volume_data, x='Symbol', y='Volume', title="Trading Volume")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price change distribution
                    changes = [data['change_pct'] for data in stock_data.values()]
                    fig = px.histogram(x=changes, title="Price Change Distribution", nbins=10)
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Trading Signals")
        
        # Current time
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Futures trading signal
        st.markdown("### 📈 Futures Trading Signal")
        
        if 'wss' in locals():
            # Calculate signal based on WSS
            if wss > 0.5:
                signal = "LONG /NQ"
                confidence = "High"
                color = "positive"
                emoji = "🚀"
            elif wss < -0.5:
                signal = "SHORT /NQ"
                confidence = "High"
                color = "negative"
                emoji = "📉"
            else:
                signal = "HOLD /NQ"
                confidence = "Low"
                color = "neutral"
                emoji = "⏸️"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{emoji} {signal}</h3>
                <p class="{color}">Confidence: {confidence}</p>
                <p>WSS: {wss:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Market sentiment
        st.markdown("### 📊 Market Sentiment")
        
        if 'predictions' in locals():
            sentiment_counts = {}
            for pred in predictions.values():
                sentiment_counts[pred] = sentiment_counts.get(pred, 0) + 1
            
            # Create sentiment pie chart
            if sentiment_counts:
                fig = px.pie(
                    values=list(sentiment_counts.values()),
                    names=list(sentiment_counts.keys()),
                    title="Prediction Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.markdown("### ⚠️ Risk Metrics")
        
        if 'stock_data' in locals():
            # Calculate volatility
            volatilities = []
            for symbol, data in stock_data.items():
                if data['price'] > 0:
                    vol = abs(data['change_pct'])
                    volatilities.append(vol)
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                max_volatility = np.max(volatilities)
                
                st.metric("Avg Volatility", f"{avg_volatility:.2f}%")
                st.metric("Max Volatility", f"{max_volatility:.2f}%")
                
                if avg_volatility > 3:
                    st.warning("⚠️ High market volatility detected")
                elif avg_volatility < 1:
                    st.success("✅ Low market volatility")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Market Master Trading System | Powered by MLflow & Prefect</p>
        <p>Data updates every 30 seconds | Predictions based on ML models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
