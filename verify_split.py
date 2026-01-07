
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
from data.price_fetcher import PriceFetcher

def verify():
    pf = PriceFetcher()
    # Fetch AAPL spanning 2020 split (Aug 2020)
    prices = pf.fetch("AAPL", start_date="2020-08-20", end_date="2020-09-05")
    
    # 2020-08-28 (Pre) -> 2020-08-31 (Post)
    print(prices[['open', 'close']].loc["2020-08-25":"2020-09-02"])
    
    # Check Return
    p_pre = prices.loc["2020-08-28", "open"]
    p_post = prices.loc["2020-08-31", "open"]
    
    ret = (p_post - p_pre) / p_pre
    print(f"\nOpen(Aug 31) / Open(Aug 28) - 1 = {ret:.2%}")
    
    if ret < -0.5:
        print("CRITICAL: DATA IS NOT SPLIT ADJUSTED!")
    else:
        print("Data is Split Adjusted.")

if __name__ == "__main__":
    verify()
