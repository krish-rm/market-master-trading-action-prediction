from pathlib import Path
from typing import Optional

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from src.fetch_weights_qqq import fetch_qqq_holdings
from src.fetch_symbol import fetch_symbol, save_csv
from src.train_pooled_compare import main as train_pooled_main
from src.predict_index import main as predict_index_main


@task
def task_fetch_weights(output_path: str = "data/weights/qqq_weights.csv") -> str:
    df = fetch_qqq_holdings()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


@task
def task_fetch_components(weights_path: str, interval: str = "1h", days: int = 30, max_symbols: Optional[int] = None) -> int:
    import pandas as pd

    w = pd.read_csv(weights_path)
    w["symbol"] = w["symbol"].astype(str).str.upper()
    symbols = w["symbol"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]
    saved = 0
    for sym in symbols:
        df = fetch_symbol(sym, interval=interval, days=days)
        tail_n = 820 if interval == "5m" else 220
        df = df.tail(tail_n)
        save_csv(df, Path(f"data/components/{sym}_{interval}.csv"))
        saved += 1
    return saved


@task
def task_train_pooled() -> None:
    train_pooled_main()


@task
def task_predict_index() -> None:
    predict_index_main()


DEFAULT_INTERVAL = "1h"
DEFAULT_DAYS = 30
DEFAULT_MAX_SYMBOLS = 10


@flow(name="market-master-index-flow")
def index_flow() -> None:
    weights = task_fetch_weights()
    _ = task_fetch_components(weights, DEFAULT_INTERVAL, DEFAULT_DAYS, DEFAULT_MAX_SYMBOLS)
    task_train_pooled()
    task_predict_index()


def build_hourly_deployment() -> None:
    # Hourly schedule: fetch → predict (light run) or full flow if desired
    dep = Deployment.build_from_flow(
        flow=index_flow,
        name="index-signal-hourly",
        schedule=CronSchedule(cron="0 * * * *", timezone="UTC"),
    )
    dep.apply()


if __name__ == "__main__":
    # Default: create deployment for scheduler; running the flow still works directly
    try:
        build_hourly_deployment()
    except Exception:
        pass
    index_flow()


