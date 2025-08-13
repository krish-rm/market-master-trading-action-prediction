import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf


def _normalize_df(df: pd.DataFrame, tz_ny: pytz.BaseTzInfo) -> pd.DataFrame:
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the level that contains OHLCV field names
        chosen_level = None
        for lvl in range(df.columns.nlevels):
            vals = [str(v).lower() for v in df.columns.get_level_values(lvl)]
            if any(k in vals for k in ["open", "high", "low", "close", "adj close", "adj_close", "volume"]):
                chosen_level = lvl
                break
        if chosen_level is not None:
            df.columns = df.columns.get_level_values(chosen_level)
        else:
            # Fallback: take the last level
            df.columns = df.columns.get_level_values(-1)

    # Standardize column names
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Ensure required OHLCV columns exist
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from provider: {missing}. Got: {list(df.columns)}")

    # Ensure timezone-aware timestamp in New York time
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(tz_ny)
        else:
            df.index = df.index.tz_convert(tz_ny)

    # Reset index to a single 'timestamp' column
    try:
        df = df.reset_index(names="timestamp")
    except TypeError:
        # Older pandas fallback
        df = df.reset_index()
        if df.columns[0] not in df.columns[1:]:
            df = df.rename(columns={df.columns[0]: "timestamp"})

    # Regular market hours 9:30-16:00 ET
    ts = pd.to_datetime(df["timestamp"])  # ensure datetime
    mask_open = (ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 30))
    mask_close = (ts.dt.hour < 16) | ((ts.dt.hour == 16) & (ts.dt.minute == 0))
    df = df[mask_open & mask_close]

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols].dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_nvda_5min(days: int) -> pd.DataFrame:
    tz_ny = pytz.timezone("America/New_York")
    # Try period-based first (more reliable for intraday)
    period_days = max(days, 14)
    try:
        df = yf.download(
            "NVDA",
            interval="5m",
            period=f"{period_days}d",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        # Fallback to Ticker.history
        try:
            t = yf.Ticker("NVDA")
            df = t.history(period=f"{period_days}d", interval="5m", actions=False, prepost=False)
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance for NVDA 5m (both methods).")
    return _normalize_df(df, tz_ny)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NVDA 5-minute OHLCV sample.")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in calendar days (default: 14)")
    parser.add_argument("--output", type=str, default="data/raw/sample_ohlcv.csv", help="Output CSV path")
    args = parser.parse_args()

    df = fetch_nvda_5min(days=args.days)
    # Reduce to roughly last 10 trading days (~78 bars/day * 10 ≈ 780). Keep most recent subset.
    df = df.tail(820).copy()
    save_csv(df, Path(args.output))
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()


