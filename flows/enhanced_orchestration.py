import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient

from prefect import flow, task
from prefect.server.schemas.schedules import CronSchedule
from prefect.blocks.system import Secret
from prefect import get_run_logger

from src.fetch_weights_qqq import fetch_qqq_holdings
from src.fetch_symbol import fetch_symbol, save_csv
from src.train_pooled_compare import main as train_pooled_main
from src.predict_index import main as predict_index_main
from src.monitor_drift import main as monitor_drift_main
from src.gate_and_report import main as gate_and_report_main


@task
def task_fetch_weights(output_path: str = "data/weights/qqq_weights.csv") -> str:
    """Fetch QQQ weights with error handling and logging."""
    logger = get_run_logger()
    try:
        df = fetch_qqq_holdings()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Successfully fetched weights for {len(df)} symbols")
        return str(out)
    except Exception as e:
        logger.error(f"Failed to fetch weights: {e}")
        raise


@task
def task_fetch_components(
    weights_path: str, 
    interval: str = "1h", 
    days: int = 30, 
    max_symbols: Optional[int] = None
) -> Dict[str, Any]:
    """Fetch component data with progress tracking."""
    import pandas as pd
    logger = get_run_logger()
    
    w = pd.read_csv(weights_path)
    w["symbol"] = w["symbol"].astype(str).str.upper()
    symbols = w["symbol"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]
    
    saved = 0
    failed_symbols = []
    
    for sym in symbols:
        try:
            logger.info(f"Fetching data for {sym}...")
            df = fetch_symbol(sym, interval=interval, days=days)
            tail_n = 820 if interval == "5m" else 220
            df = df.tail(tail_n)
            save_csv(df, Path(f"data/components/{sym}_{interval}.csv"))
            saved += 1
            logger.info(f"Successfully saved {sym}")
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")
            failed_symbols.append(sym)
            continue
    
    logger.info(f"Fetch completed: {saved}/{len(symbols)} symbols saved")
    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols}")
    
    return {
        "saved": saved,
        "total": len(symbols),
        "failed": failed_symbols
    }


@task
def task_train_pooled() -> Dict[str, Any]:
    """Train pooled model with performance tracking."""
    logger = get_run_logger()
    try:
        train_pooled_main()
        logger.info("Model training completed successfully")
        
        # Get latest model performance
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        versions = client.search_model_versions("name='market-master-component-classifier'")
        if versions:
            latest = max(versions, key=lambda v: v.version)
            return {
                "model_version": latest.version,
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


@task
def task_monitor_and_gate() -> Dict[str, Any]:
    """Monitor model performance and run gating logic."""
    logger = get_run_logger()
    try:
        # Run monitoring
        monitor_drift_main()
        logger.info("Model monitoring completed")
        
        # Run gating
        gate_and_report_main()
        logger.info("Gating logic completed")
        
        # Check if gate passed
        import json
        gate_path = Path("artifacts/index/gate_summary.json")
        if gate_path.exists():
            with open(gate_path) as f:
                gate_result = json.load(f)
            logger.info(f"Gate result: {gate_result}")
            return gate_result
        else:
            logger.warning("Gate summary not found")
            return {"passed": False, "reason": "No gate summary"}
            
    except Exception as e:
        logger.error(f"Monitoring/gating failed: {e}")
        raise


@task
def task_predict_index() -> Dict[str, Any]:
    """Generate index predictions."""
    logger = get_run_logger()
    try:
        predict_index_main()
        logger.info("Index prediction completed")
        
        # Read prediction results
        import json
        pred_path = Path("artifacts/index/wss_summary.json")
        if pred_path.exists():
            with open(pred_path) as f:
                pred_result = json.load(f)
            logger.info(f"Prediction result: {pred_result}")
            return pred_result
        else:
            logger.warning("Prediction summary not found")
            return {"status": "completed", "wss": None}
            
    except Exception as e:
        logger.error(f"Index prediction failed: {e}")
        raise


@task
def task_deploy_model_if_gate_passed(gate_result: Dict[str, Any]) -> bool:
    """Deploy model to production if gate passed."""
    logger = get_run_logger()
    
    if gate_result.get("passed", False):
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            client = MlflowClient()
            name = "market-master-component-classifier"
            
            # Get staging version
            staging_version_obj = client.get_model_version_by_alias(name, "Staging")
            staging_version = int(staging_version_obj.version)
            
            # Promote to production
            client.set_registered_model_alias(name, "Production", staging_version)
            logger.info(f"Model version {staging_version} promoted to Production")
            return True
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    else:
        logger.info("Gate failed - model not deployed")
        return False


@flow(name="market-master-enhanced-flow")
def enhanced_index_flow(
    interval: str = "1h",
    days: int = 30,
    max_symbols: Optional[int] = None
) -> Dict[str, Any]:
    """Enhanced orchestrated flow with monitoring and conditional deployment."""
    logger = get_run_logger()
    logger.info("Starting enhanced market master flow")
    
    # Step 1: Fetch data
    weights = task_fetch_weights()
    fetch_result = task_fetch_components(weights, interval, days, max_symbols)
    
    # Step 2: Train model
    train_result = task_train_pooled()
    
    # Step 3: Generate predictions
    pred_result = task_predict_index()
    
    # Step 4: Monitor and gate
    gate_result = task_monitor_and_gate()
    
    # Step 5: Conditional deployment
    deployed = task_deploy_model_if_gate_passed(gate_result)
    
    # Return comprehensive results
    return {
        "fetch_result": fetch_result,
        "train_result": train_result,
        "prediction_result": pred_result,
        "gate_result": gate_result,
        "deployed": deployed,
        "status": "completed"
    }


def create_deployments() -> None:
    """Create deployments using the new Prefect 3.x API."""
    print("Creating deployments using Prefect 3.x API...")
    
    # Note: In Prefect 3.x, deployments are created differently
    # The old Deployment.build_from_flow() method is no longer available
    # Instead, you can use flow.serve() or create deployments via the UI/CLI
    
    print("To create deployments, use one of these methods:")
    print("1. Use the Prefect UI: prefect server start, then create deployments via web interface")
    print("2. Use flow.serve() for local development")
    print("3. Use 'prefect deploy' CLI command")
    
    # Example of how to serve the flow locally
    print("\nTo serve this flow locally, run:")
    print("prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n 'enhanced-index-signal-hourly'")
    print("prefect deployment apply enhanced_index_flow-deployment.yaml")


if __name__ == "__main__":
    # Run the flow directly (deployments can be created separately via Prefect UI)
    print("Running enhanced market master flow...")
    result = enhanced_index_flow()
    
    # Print detailed results including model version
    print("\n" + "="*60)
    print("🎯 ENHANCED MARKET MASTER FLOW COMPLETED")
    print("="*60)
    
    # Fetch results
    fetch_result = result.get("fetch_result", {})
    print(f"📊 Data Fetch: {fetch_result.get('saved', 0)}/{fetch_result.get('total', 0)} symbols processed")
    
    # Training results
    train_result = result.get("train_result", {})
    model_version = train_result.get("model_version", "N/A")
    print(f"🤖 Model Training: Version {model_version} created and registered")
    
    # Prediction results
    pred_result = result.get("prediction_result", {})
    wss = pred_result.get("wss", "N/A")
    print(f"📈 Index Prediction: WSS = {wss}")
    
    # Gate results
    gate_result = result.get("gate_result", {})
    passed = gate_result.get("passed", False)
    drift_features = gate_result.get("drift_features", 0)
    f1_score = gate_result.get("macro_f1", 0)
    print(f"🔍 Quality Gate: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"   - Drift Features: {drift_features}")
    print(f"   - Macro F1 Score: {f1_score:.3f}")
    
    # Deployment results
    deployed = result.get("deployed", False)
    if deployed:
        print(f"🚀 Model Deployment: Version {model_version} promoted to Production")
        print(f"   - Production Model: models:/market-master-component-classifier@Production")
    else:
        print("🚫 Model Deployment: Gate failed - model not promoted")
    
    print("="*60)
    print("\nTo create scheduled deployments, use Prefect UI or CLI:")
    print("1. Start Prefect server: make prefect-start")
    print("2. Configure API URL: make prefect-setup")
    print("3. Open http://127.0.0.1:4200")
    print("4. Create deployments from the UI")
    print("\nOr use the new deployment commands:")
    print("prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n 'enhanced-index-signal-hourly'")
    print("prefect deployment apply enhanced_index_flow-deployment.yaml")
