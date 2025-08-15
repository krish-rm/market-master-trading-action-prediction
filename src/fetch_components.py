from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

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


def sanity_checks(symbols: List[str], interval: str) -> Dict[str, Dict]:
    """Perform sanity checks on fetched data"""
    checks = {}

    for symbol in symbols:
        csv_path = Path(f"data/components/{symbol}_{interval}.csv")
        if not csv_path.exists():
            checks[symbol] = {"status": "missing_file", "details": "CSV file not found"}
            continue

        try:
            df = pd.read_csv(csv_path)

            # Check for required columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                checks[symbol] = {
                    "status": "missing_columns",
                    "details": f"Missing: {missing_cols}",
                }
                continue

            # Coverage check
            total_expected = len(df)
            non_null_count = df[required_cols].notna().all(axis=1).sum()
            coverage_pct = (
                (non_null_count / total_expected) * 100
                if total_expected > 0 else 0
            )

            # Missing bars check (assuming hourly data should have ~6.5 hours per
            # trading day)
            expected_hours_per_day = 6.5
            expected_total = int(expected_hours_per_day * 30)  # 30 days
            missing_bars = max(0, expected_total - total_expected)

            # Duplicate timestamps check
            duplicates = df["timestamp"].duplicated().sum()

            # Session bounds check (basic market hours validation)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            outside_market_hours = ((df["hour"] < 9) | (df["hour"] > 16)).sum()

            checks[symbol] = {
                "status": "ok",
                "coverage_pct": round(coverage_pct, 2),
                "total_bars": total_expected,
                "missing_bars": missing_bars,
                "duplicate_timestamps": duplicates,
                "outside_market_hours": outside_market_hours,
            }

        except Exception as e:
            checks[symbol] = {"status": "error", "details": str(e)}

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch 1h OHLCV for index constituents"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="data/weights/qqq_weights.csv",
        help="CSV with columns: symbol,weight",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1h", "5m"],
        help="Bar interval to fetch",
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Lookback window in calendar days"
    )
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
        futures = {
            ex.submit(fetch_one, sym, args.interval, args.days): sym
            for sym in symbols
        }
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

    # Run sanity checks on successfully fetched data
    if ok:
        print("\n=== SANITY CHECKS ===")
        successful_symbols = [r[0] for r in ok]
        checks = sanity_checks(successful_symbols, args.interval)

        for symbol, check in checks.items():
            print(f"\n{symbol}:")
            if check["status"] == "ok":
                print(f"  ✓ Coverage: {check['coverage_pct']}%")
                print(f"  ✓ Total bars: {check['total_bars']}")
                print(f"  ✓ Missing bars: {check['missing_bars']}")
                print(f"  ✓ Duplicate timestamps: {check['duplicate_timestamps']}")
                print(f"  ✓ Outside market hours: {check['outside_market_hours']}")
            else:
                print(f"  ✗ {check['status']}: {check['details']}")


if __name__ == "__main__":
    main()
