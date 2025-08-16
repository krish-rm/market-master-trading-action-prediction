"""
MLflow Model Serving for Market Master Trading Prediction System
Provides REST API endpoints for model inference with health checks and monitoring.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime, timezone
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Market Master Model Serving API",
    description="MLflow-based model serving for trading prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata
MODEL = None
META = None
MODEL_VERSION = None
LAST_LOAD_TIME = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[float] = Field(..., description="Feature vector for prediction")
    symbol: Optional[str] = Field(None, description="Stock symbol (optional)")
    timestamp: Optional[str] = Field(None, description="Prediction timestamp")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str = Field(..., description="Predicted action (buy/sell/hold/strong_buy/strong_sell)")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    last_load_time: Optional[str] = Field(None, description="Last model load time")
    uptime: Optional[str] = Field(None, description="Service uptime")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features_list: List[List[float]] = Field(..., description="List of feature vectors")
    symbols: Optional[List[str]] = Field(None, description="List of stock symbols")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


def load_model_from_registry(model_name: str = "market-master-component-classifier", alias: str = "Production") -> None:
    """Load model from MLflow registry."""
    global MODEL, META, MODEL_VERSION, LAST_LOAD_TIME
    
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Load model using alias
        model_uri = f"models:/{model_name}@{alias}"
        MODEL = mlflow.sklearn.load_model(model_uri)
        
        # Load metadata
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, alias)
        MODEL_VERSION = str(model_version.version)
        
        # Load metadata from artifacts
        artifacts_dir = Path("artifacts/model")
        meta_path = artifacts_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                META = json.load(f)
        else:
            META = {"feature_columns": [], "model_info": "Metadata not found"}
        
        LAST_LOAD_TIME = datetime.now(timezone.utc).isoformat()
        logger.info(f"Model loaded successfully: {model_name}@{alias} (version {MODEL_VERSION})")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def predict_single(features: List[float], symbol: Optional[str] = None) -> Dict[str, Any]:
    """Make a single prediction."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features], columns=META.get("feature_columns", []))
        
        # Make prediction
        prediction = MODEL.predict(feature_df)[0]
        probabilities = MODEL.predict_proba(feature_df)[0]
        
        # Map class indices to labels
        class_labels = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
        prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model_from_registry()
        logger.info("Model serving API started successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = "N/A"  # Could implement actual uptime tracking
    
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        last_load_time=LAST_LOAD_TIME,
        uptime=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    result = predict_single(request.features, request.symbol)
    
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=MODEL_VERSION or "unknown",
        timestamp=result["timestamp"]
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        feature_df = pd.DataFrame(request.features_list, columns=META.get("feature_columns", []))
        
        # Make batch predictions
        preds = MODEL.predict(feature_df)
        probas = MODEL.predict_proba(feature_df)
        
        for i, (pred, prob) in enumerate(zip(preds, probas)):
            class_labels = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
            prob_dict = {label: float(p) for label, p in zip(class_labels, prob)}
            
            prediction_result = {
                "prediction": pred,
                "confidence": float(max(prob)),
                "probabilities": prob_dict,
                "symbol": request.symbols[i] if request.symbols and i < len(request.symbols) else None,
                "index": i
            }
            predictions.append(prediction_result)
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_version=MODEL_VERSION or "unknown",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload model from registry."""
    try:
        background_tasks.add_task(load_model_from_registry)
        return {"message": "Model reload initiated", "status": "reloading"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": MODEL_VERSION,
        "last_load_time": LAST_LOAD_TIME,
        "feature_columns": META.get("feature_columns", []),
        "model_type": type(MODEL).__name__,
        "metadata": META
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Market Master Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "reload_model": "/reload-model",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


def start_server(host: str = "0.0.0.0", port: int = 8001, reload: bool = False):
    """Start the model serving server."""
    uvicorn.run(
        "src.model_serving:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
