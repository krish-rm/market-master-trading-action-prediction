from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .fetch_symbol import fetch_symbol, save_csv


def fetch_one(symbol: str, interval: str, days: int) -> Tuple[str, bool, str]:
    try:
        df = fetch_symbol(symbol=symbol, interval=interval, days=days)
        out = Path(f"data/components/{symbol}_{interval}.csv")
        save_csv(df, out)
        return symbol, True, str(out)
    except Exception as e:
        return symbol, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 1h OHLCV for index constituents")
    parser.add_argument("--weights", type=str, default="data/weights/qqq_weights.csv", help="CSV with columns: symbol,weight")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "5m"], help="Bar interval to fetch")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in calendar days")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")
    wdf = pd.read_csv(weights_path)
    if "symbol" not in wdf.columns:
        raise SystemExit("Weights CSV must contain 'symbol' column")
    symbols: List[str] = list(wdf["symbol"].astype(str).str.upper().unique())

    results: List[Tuple[str, bool, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(fetch_one, sym, args.interval, args.days): sym for sym in symbols}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    print(f"Fetched {len(ok)} / {len(results)} symbols")
    if ok:
        print("Saved:")
        for sym, _, out in ok:
            print(f"  {sym}: {out}")
    if fail:
        print("Failed:")
        for sym, _, err in fail:
            print(f"  {sym}: {err}")


if __name__ == "__main__":
    main()


