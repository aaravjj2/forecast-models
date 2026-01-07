
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from config import CONFIG
from strategies.strategy_wrapper import StrategyWrapper

def debug():
    pf = PriceFetcher()
    prices = pf.fetch("AAPL", start_date=CONFIG.START_DATE, end_date=CONFIG.END_DATE)
    
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
    prices = prices.sort_index()
    
    # RSI 2
    delta = prices['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Logic: < 10 Buy, > 90 Short. Else Flat? 
    # Or Hold until > 50?
    # Simple: < 10 -> 1.0. > 90 -> -1.0. Else 0.
    signals = pd.Series(0.0, index=prices.index)
    signals[rsi < 10] = 1.0
    signals[rsi > 90] = -1.0
    
    strat = StrategyWrapper()
    res = strat.backtest_conditional(prices, signals, cost_bps=10.0)
    
    print("--- RSI(2) MR Backtest ---")
    print(f"Total Return: {res['total_return']:.2%}")
    print(f"Sharpe: {res['sharpe']:.2f}")
    print(f"Trades: {res['n_trades']}")
    print(f"Turnover: {res['turnover_count']}")

if __name__ == "__main__":
    debug()
