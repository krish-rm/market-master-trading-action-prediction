from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from src.fetch_symbol import fetch_symbol, save_csv
from src.fetch_weights_qqq import fetch_qqq_holdings
from src.train_pooled_compare import main as train_pooled_main


def step_weights(output_path: str = "data/weights/qqq_weights.csv") -> None:
    df = fetch_qqq_holdings()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[weights] saved {len(df)} rows -> {out}")


def step_fetch_components(
    weights_path: str, interval: str, days: int, max_symbols: int | None
) -> None:
    import pandas as pd

    w = pd.read_csv(weights_path)
    w["symbol"] = w["symbol"].astype(str).str.upper()
    symbols = w["symbol"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]

    saved = 0
    failed_symbols = []

    for sym in symbols:
        try:
            print(f"Fetching data for {sym}...")
            df = fetch_symbol(sym, interval=interval, days=days)
            tail_n = 820 if interval == "5m" else 220
            df = df.tail(tail_n)
            save_csv(df, Path(f"data/components/{sym}_{interval}.csv"))
            saved += 1
            print(f"Successfully saved {sym}")
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
            failed_symbols.append(sym)
            continue

    print(f"[fetch] saved {saved}/{len(symbols)} symbols to data/components")
    if failed_symbols:
        print(f"Failed symbols: {failed_symbols}")

    # If no symbols were successfully fetched, raise an error
    if saved == 0:
        raise RuntimeError(
            "No symbols were successfully fetched. "
            "Check network connectivity and API availability."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end local pipeline")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "5m"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--max-symbols", type=int, default=10)
    args = parser.parse_args()

    step_weights()
    step_fetch_components(
        "data/weights/qqq_weights.csv", args.interval, args.days, args.max_symbols
    )
    print("[train] pooled compare + select best")
    train_pooled_main()
    print("[signal] index WSS + decision")
    subprocess.run(
        [
            "python",
            "-m",
            "src.predict_index",
            "--weights",
            "data/weights/qqq_weights.csv",
            "--interval",
            args.interval,
        ],
        check=True,
    )
    print("[monitor] drift + classification quality")
    subprocess.run(["python", "-m", "src.monitor_drift"], check=True)
    print("[gate] promotion decision")
    subprocess.run(["python", "-m", "src.gate_and_report"], check=True)
    # If gate passed, promote Staging to Production in MLflow Model Registry
    # (best-effort)
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Use environment variable or default to local SQLite for Option A
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        name = "market-master-component-classifier"
        # Get version currently tagged Staging
        try:
            staging_version_obj = client.get_model_version_by_alias(name, "Staging")
            staging_version = int(staging_version_obj.version)
            client.set_registered_model_alias(name, "Production", staging_version)
            print(f"[registry] Promoted version {staging_version} to Production")
        except Exception as e:
            print(f"[registry] No Staging version found to promote: {e}")
    except Exception as e:
        print(f"[registry] Skipped promotion: {e}")


if __name__ == "__main__":
    main()
