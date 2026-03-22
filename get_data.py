"""
This script download data via `yfinance` in the format we need.
"""

import yfinance as yf
import pandas as pd

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
# This file contains the member of SPX as of Mar 14 2026
file_path = SCRIPT_DIR / "data" / "SPX as of Mar 14 20261.xlsx"


def main():
    df = pd.read_excel(file_path)
    # Convert BBG ticker format to Yahoo! Finance style
    df["y_ticker"] = df["Ticker"].str.split(" ").str[0]
    df["y_ticker"] = df["y_ticker"].str.replace("/", "-")   # ticker like BRK/B

    tickers = [t for t in df["y_ticker"]]
    print(f"There are {len(tickers)} tickers")

    raw_data = yf.download(
        tickers=tickers, 
        period="1y", 
        interval="1d", 
        group_by='ticker', 
        auto_adjust=True, 
        threads=True
    )

    """
    Raw data queried above looks like this:
    Column 0: Tickers
    Column 1: Open High Low Close Volume
    Index   : Time series, daily
    """
    if raw_data is not None and not raw_data.empty:
        panel_data = raw_data.stack(level=0)
    else:
        raise ValueError("Empty return from yfinance.")
    
    panel_data = panel_data.reset_index()
    panel_data.columns.name = None

    panel_data.to_parquet(SCRIPT_DIR / "data" / "spx_1y_ohlcv.parquet")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise RuntimeError(f"Error: {e}")