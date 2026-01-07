
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.backtest.walkforward_backtest import WalkForwardBacktest
from src.ensemble.meta_ensemble import MetaEnsemble

class MockEnsemble:
    def __init__(self):
        pass

class TestBacktestPnL(unittest.TestCase):
    def test_pnl_logic(self):
        # Setup dummy data
        dates = pd.date_range('2023-01-01', periods=5)
        
        # Predictions: Buy, Buy, Sell, Sell, Abstain
        predictions = pd.DataFrame({
            'signal': [1, 1, -1, -1, 0],
            'confidence': [0.9] * 5
        }, index=dates)
        
        # Actual Returns (Open-to-Open target)
        # 1. +10%
        # 2. -5%
        # 3. -10% (Short should profit)
        # 4. +5% (Short should lose)
        # 5. +1% (Abstain)
        actuals = pd.DataFrame({
            'actual_return': [0.10, -0.05, -0.10, 0.05, 0.01],
            'actual_direction': [1, -1, -1, 1, 1]
        }, index=dates)
        
        # Prices (dummy, shouldn't be used for PnL anymore)
        prices = pd.DataFrame({'close': [100]*5}, index=dates)
        
        # Init backtest with 0 commission for clarity
        bt = WalkForwardBacktest(MockEnsemble(), initial_capital=100.0, commission_rate=0.0)
        
        results = bt._simulate_pnl(predictions, actuals, prices)
        
        # Expected Logic:
        # T=0: Signal 1. Ret 0.10. Cap = 100 * 1.1 = 110. Pos=1.
        # T=1: Signal 1. Ret -0.05. Cap = 110 * 0.95 = 104.5. Pos=1.
        # T=2: Signal -1. Ret -0.10. Short implies profit on negative ret. 
        #      Strategy Return = -1 * -0.10 = +0.10.
        #      Cap = 104.5 * 1.1 = 114.95. Pos=-1.
        # T=3: Signal -1. Ret 0.05. Short implies loss on positive ret.
        #      Strategy Return = -1 * 0.05 = -0.05.
        #      Cap = 114.95 * 0.95 = 109.2025. Pos=-1.
        # T=4: Signal 0. Ret 0.01. No trade.
        #      Cap = 109.2025.
        
        final_cap = results['final_capital']
        print(f"Final Capital: {final_cap}")
        
        self.assertAlmostEqual(final_cap, 109.2025, places=4)
        print("PnL Logic Verified!")

if __name__ == '__main__':
    unittest.main()
