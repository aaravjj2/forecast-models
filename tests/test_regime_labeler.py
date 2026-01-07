import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime, InfoRegime

class TestRegimeLabeler(unittest.TestCase):
    def setUp(self):
        # Create synthetic price data for testing
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200) # Increased length
        # Generate random walk for baseline volatility
        returns = np.random.normal(0, 0.01, size=len(dates)) # 1% daily vol baseline
        price_path = 100 * (1 + returns).cumprod()
        
        self.prices = pd.DataFrame({
            'open': price_path,
            'high': price_path * 1.01,
            'low': price_path * 0.99,
            'close': price_path,
            'volume': 1000000
        }, index=dates)
        
    def test_crash_label(self):
        # Simulate a crash: massive drop in 5 days
        # At index 10, future 5 days drop 20%
        # Sync OHLC for the crash day
        crash_price = 80.0
        self.prices.loc[self.prices.index[15], 'close'] = crash_price
        self.prices.loc[self.prices.index[15], 'low'] = crash_price * 0.99
        self.prices.loc[self.prices.index[15], 'high'] = crash_price * 1.01
        self.prices.loc[self.prices.index[15], 'open'] = crash_price
        
        # This means simple returns between T=10 and T=15 will be negative
        # Wait, pct_change(5).shift(-5) at T=10 compares Close_15 / Close_10 - 1
        
        labeler = RegimeLabeler(forecast_horizon=5)
        labeled = labeler.label_regimes(self.prices)
        
        # Check index 10 (which sees the crash at 15)
        # 80/100 - 1 = -0.20
        regime = labeled.iloc[10]['regime']
        self.assertEqual(regime, MarketRegime.CRASH)
        
    def test_trend_bull_label(self):
        # Smooth ascent relative to whatever price is at index 20
        start_price = self.prices.iloc[20]['close']
        
        # Grow 1% per day for 5 days -> ~5% total return (> 3%)
        # This is STRICTLY a Trend. 
        # With new logic: TREND_BULL was merged into TRENDING (1).
        
        for i in range(1, 6):
            target_price = start_price * (1.01 ** i)
            self.prices.iloc[20+i, self.prices.columns.get_loc('close')] = target_price
            self.prices.iloc[20+i, self.prices.columns.get_loc('open')] = target_price 
            self.prices.iloc[20+i, self.prices.columns.get_loc('high')] = target_price
            self.prices.iloc[20+i, self.prices.columns.get_loc('low')] = target_price
            
        labeler = RegimeLabeler(forecast_horizon=5)
        labeled = labeler.label_regimes(self.prices)
        
        regime = labeled.iloc[20]['regime']
        self.assertEqual(regime, MarketRegime.TRENDING)

    def test_trend_quality(self):
        """Test Robust (Clean) vs Noisy (Choppy) trend detection via ADX"""
        start_price = 100
        
        # 1. Create a CLEAN trend (Robust)
        # Linear growth -> Strong ADX
        # Need enough history for ADX(14) to warm up. 
        # T=100.
        for i in range(100):
            target_price = start_price * (1.01 ** i)
            self.prices.iloc[i, self.prices.columns.get_loc('close')] = target_price
            self.prices.iloc[i, self.prices.columns.get_loc('high')] = target_price * 1.005
            self.prices.iloc[i, self.prices.columns.get_loc('low')] = target_price * 0.995
            self.prices.iloc[i, self.prices.columns.get_loc('open')] = target_price
            
        labeler = RegimeLabeler(forecast_horizon=5)
        labeled = labeler.label_regimes(self.prices)
        
        # Check Robust Trend (at T=50, should have high Future ADX)
        quality_robust = labeled.iloc[50]['trend_quality']
        self.assertEqual(quality_robust, TrendQuality.ROBUST, 
                         f"Index 50 should be ROBUST (High ADX). Got {labeled.iloc[50]['trend_quality_name']} (ADX={labeled.iloc[50]['future_adx']:.2f})")
                         
        # 2. Create a NOISY drift (Noisy)
        # Random walk with mean reversion -> Low ADX
        # T=100 to T=200
        for i in range(100, 200):
            prev = self.prices.iloc[i-1]['close']
            target_price = prev + np.random.normal(0, 1.0) # Chop
            self.prices.iloc[i, self.prices.columns.get_loc('close')] = target_price
            self.prices.iloc[i, self.prices.columns.get_loc('high')] = target_price + 2
            self.prices.iloc[i, self.prices.columns.get_loc('low')] = target_price - 2
            
        labeled = labeler.label_regimes(self.prices)
        
        # Check Noisy Trend (at T=150)
        # ADX should be low.
        quality_noisy = labeled.iloc[150]['trend_quality']
        # Might be Noisy or Neutral depending on threshold.
        # But definitely not Robust.
        self.assertNotEqual(quality_noisy, TrendQuality.ROBUST,
                        f"Index 150 should NOT be ROBUST. Got {labeled.iloc[150]['trend_quality_name']} (ADX={labeled.iloc[150]['future_adx']:.2f})")

    def test_liquidity_stress(self):
        """Test Liquidity Regime labeling (High Amihud = Stressed)"""
        # Create a period of High Illiquidity (High Return / Low Volume)
        # T=150 to T=160
        start_price = 100
        for i in range(150, 160):
             # Huge move (High Abs Return)
             self.prices.iloc[i, self.prices.columns.get_loc('close')] = start_price * (1.1 ** (i-150))
             # Tiny Volume
             self.prices.iloc[i, self.prices.columns.get_loc('volume')] = 100 
             
        labeler = RegimeLabeler(forecast_horizon=5)
        labeled = labeler.label_regimes(self.prices)
        
        # At T=145, looking ahead 5 days -> High Future Amihud
        regime = labeled.iloc[145]['liquidity_regime']
        self.assertEqual(regime, LiquidityRegime.STRESSED,
                        f"Index 145 should be STRESSED. Got {labeled.iloc[145]['liquidity_name']}")

    def test_info_drift(self):
        """Test Info Regime (Drifting). Future Return follows Past 5d Return"""
        # Create a persistent momentum (Drift)
        # T=0 to T=30: Constant 1% UP
        start_price = 100
        for i in range(30):
             self.prices.iloc[i, self.prices.columns.get_loc('close')] = start_price * (1.01 ** i)
             
        labeler = RegimeLabeler(forecast_horizon=5)
        labeled = labeler.label_regimes(self.prices)
        
        # At T=10
        # Past Return (5d) is Positive.
        # Future Return (5d) is Positive.
        # Divergence? No, Alignment.
        # RegimeLabeler logic: sign(Future) == sign(Past) -> DRIFTING
        
        regime = labeled.iloc[10]['info_regime']
        self.assertEqual(regime, InfoRegime.DRIFTING,
                        f"Index 10 should be DRIFTING. Got {labeled.iloc[10]['info_name']}")

if __name__ == '__main__':
    unittest.main()
