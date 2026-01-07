
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
    
    # Ensure DatetimeIndex
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
    prices = prices.sort_index()
    
    # Filter for 2017+
    start_idx = prices.index.searchsorted(pd.Timestamp("2017-01-01"))
    prices = prices.iloc[start_idx:]
    
    # Calculate simple Buy & Hold
    buy_hold_ret = (prices['close'][-1] / prices['close'][0]) - 1
    print(f"Buy & Hold Return: {buy_hold_ret:.2%}")
    
    # Calculate Momentum (Close > MA50)
    prices['ma_50'] = prices['close'].rolling(50).mean()
    mom_signals = np.where(prices['close'] > prices['ma_50'], 1.0, -1.0)
    signals = pd.Series(mom_signals, index=prices.index)
    
    strat = StrategyWrapper()
    res = strat.backtest_conditional(prices, signals, cost_bps=10.0)
    
    print("--- Momentum Backtest ---")
    print(f"Total Return: {res['total_return']:.2%}")
    print(f"Sharpe: {res['sharpe']:.2f}")
    print(f"Trades (Days Active): {res['n_trades']}")
    print(f"Turnover: {res['turnover_count']}")
    
    # Check Turnover Frequency
    # turnover_count = sum(|diff|)
    # max possible ~ 2 * len
    print(f"Turnover Ratio: {res['turnover_count'] / len(prices):.2f}")
    
    # Check if Next Day Returns are correct
    px_check = prices[['open', 'close']].copy()
    px_check['next_open'] = px_check['open'].shift(-1)
    px_check['next_2_open'] = px_check['open'].shift(-2)
    px_check['calc_ret'] = (px_check['next_2_open'] - px_check['next_open']) / px_check['next_open']
    
    # Strategy calc
    # next_day_ret (in strategy) = (Open[t+2] - Open[t+1]) / Open[t+1]
    # aligned with Signal[t].
    
    # Print first few rows of Signal vs Return
    print("\nSample Data:")
    df_debug = pd.DataFrame({
        'Close': prices['close'],
        'MA50': prices['ma_50'],
        'Signal': signals,
        'NextRet': px_check['calc_ret']
    }).dropna().head(20)
    print(df_debug)

if __name__ == "__main__":
    debug()
