from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf


def _normalize_df(df: pd.DataFrame, tz_ny: pytz.BaseTzInfo) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        chosen_level = None
        for lvl in range(df.columns.nlevels):
            vals = [str(v).lower() for v in df.columns.get_level_values(lvl)]
            if any(
                k in vals
                for k in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "adj_close",
                    "volume",
                ]
            ):
                chosen_level = lvl
                break
        if chosen_level is not None:
            df.columns = df.columns.get_level_values(chosen_level)
        else:
            df.columns = df.columns.get_level_values(-1)

    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns from provider: {missing}. Got: {list(df.columns)}"
        )

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(tz_ny)
        else:
            df.index = df.index.tz_convert(tz_ny)

    try:
        df = df.reset_index(names="timestamp")
    except TypeError:
        df = df.reset_index()
        if df.columns[0] not in df.columns[1:]:
            df = df.rename(columns={df.columns[0]: "timestamp"})

    ts = pd.to_datetime(df["timestamp"])
    # NYSE hours 9:30–16:00 ET
    mask_open = (ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 30))
    mask_close = (ts.dt.hour < 16) | ((ts.dt.hour == 16) & (ts.dt.minute == 0))
    df = df[mask_open & mask_close]

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols].dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_symbol(symbol: str, interval: str, days: int) -> pd.DataFrame:
    tz_ny = pytz.timezone("America/New_York")
    period_days = max(days, 14)

    # Try multiple methods with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Method 1: yf.download
            df = yf.download(
                symbol,
                interval=interval,
                period=f"{period_days}d",
                auto_adjust=True,
                prepost=False,
                progress=False,
                threads=False,
                timeout=30,
            )
            if df is not None and not df.empty:
                return _normalize_df(df, tz_ny)
        except Exception as e:
            print(f"Attempt {attempt + 1} - yf.download failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    # Method 2: yf.Ticker
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(symbol)
            df = t.history(
                period=f"{period_days}d",
                interval=interval,
                actions=False,
                prepost=False,
            )
            if df is not None and not df.empty:
                return _normalize_df(df, tz_ny)
        except Exception as e:
            print(f"Attempt {attempt + 1} - yf.Ticker failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    # If all methods fail, raise error with more details
    raise RuntimeError(
        f"No data returned from yfinance for {symbol} {interval} "
        f"after {max_retries} attempts with both methods."
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV for a symbol")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "5m"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    df = fetch_symbol(args.symbol, args.interval, args.days)
    tail_n = 820 if args.interval == "5m" else 220
    df = df.tail(tail_n).copy()
    out = args.output or (f"data/components/{args.symbol}_{args.interval}.csv")
    save_csv(df, Path(out))
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
