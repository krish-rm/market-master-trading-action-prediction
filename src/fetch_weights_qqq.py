from __future__ import annotations

import argparse
import time
from pathlib import Path


import pandas as pd
import requests


def fetch_qqq_holdings() -> pd.DataFrame:
    # Fetch holdings table from Slickcharts with headers to avoid 403
    url = "https://www.slickcharts.com/qqq"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Try multiple times with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                break
            else:
                print(f"Attempt {attempt + 1}: HTTP {resp.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Attempt {attempt + 1}: Network error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue
    else:
        print("All attempts failed, using fallback data")
        # Fallback: minimal top holdings (approximate), normalized
        data = {
            "symbol": [
                "AAPL",
                "MSFT",
                "NVDA",
                "AMZN",
                "GOOGL",
                "META",
                "AVGO",
                "TSLA",
                "COST",
                "PEP",
            ],
            "weight": [
                0.12,
                0.11,
                0.10,
                0.08,
                0.07,
                0.05,
                0.05,
                0.04,
                0.03,
                0.03,
            ],
        }
        df_fb = pd.DataFrame(data)
        df_fb["weight"] = df_fb["weight"] / df_fb["weight"].sum()
        return df_fb

    try:
        tables = pd.read_html(resp.content)
        if not tables:
            raise RuntimeError("No tables found on Slickcharts QQQ page")
        # Usually the first table contains the holdings
        df = tables[0]
        # Expected columns: #, Company, Symbol, Weight, Price, Chg, % Chg
        if "Symbol" not in df.columns or "Weight" not in df.columns:
            raise RuntimeError(f"Unexpected columns: {df.columns}")
        df = df[["Symbol", "Weight"]].copy()
        # Convert Weight like '12.34%' to float fraction 0..1
        df["weight"] = (
            df["Weight"].astype(str).str.replace("%", "", regex=False).astype(float)
            / 100.0
        )
        df = df.drop(columns=["Weight"]).rename(columns={"Symbol": "symbol"})
        # Normalize to sum to 1 in case parsing artifacts
        total = df["weight"].sum()
        if total > 0:
            df["weight"] = df["weight"] / total
        return df
    except Exception as e:
        print(f"Error parsing HTML tables: {e}, using fallback data")
        # Fallback: minimal top holdings (approximate), normalized
        data = {
            "symbol": [
                "AAPL",
                "MSFT",
                "NVDA",
                "AMZN",
                "GOOGL",
                "META",
                "AVGO",
                "TSLA",
                "COST",
                "PEP",
            ],
            "weight": [
                0.12,
                0.11,
                0.10,
                0.08,
                0.07,
                0.05,
                0.05,
                0.04,
                0.03,
                0.03,
            ],
        }
        df_fb = pd.DataFrame(data)
        df_fb["weight"] = df_fb["weight"] / df_fb["weight"].sum()
        return df_fb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch QQQ holdings weights from Slickcharts"
    )
    parser.add_argument("--output", type=str, default="data/weights/qqq_weights.csv")
    args = parser.parse_args()

    df = fetch_qqq_holdings()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} symbols with weights to {out}")


if __name__ == "__main__":
    main()
