import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_builder import FeatureBuilder

class TestStructuralFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100)
        returns = np.random.normal(0, 0.01, size=len(dates))
        price_path = 100 * (1 + returns).cumprod()
        volume = np.random.randint(1000, 5000, size=len(dates))
        
        self.prices = pd.DataFrame({
            'open': price_path,
            'high': price_path * 1.01,
            'low': price_path * 0.99,
            'close': price_path,
            'volume': volume
        }, index=dates)
        
        # Synthetic news
        self.news = pd.DataFrame({
            'date': dates,
            'headline': ['Good news bull strong'] * len(dates) # Always sentiment positive
        })
        
    def test_structural_features_exist(self):
        builder = FeatureBuilder()
        features = builder.build_all_features(self.prices, news=self.news)
        
        # B1: Vol Term Structure
        self.assertIn('vol_term_structure', features.columns)
        self.assertIn('intraday_range_pct', features.columns)
        self.assertIn('overnight_gap_pct', features.columns)
        
        # B2: Liquidity
        self.assertIn('amihud_illiquidity', features.columns)
        self.assertIn('volume_ratio_10', features.columns)
        
        # B3: News Divergence
        self.assertIn('news_price_divergence', features.columns)
        
    def test_divergence_logic(self):
        # Test specific divergence scenario
        # T-1: Good News, Bad Return -> Divergence should be High Positive
        # We need to control the data precisely.
        
        dates = pd.date_range(start='2023-01-01', periods=10)
        prices_df = pd.DataFrame({
            'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 100
        }, index=dates)
        
        # Make Day T-1 (Index 4) have -5% return
        prices_df.iloc[4, prices_df.columns.get_loc('close')] = 95 # Return T-1 = 95/100 - 1 = -0.05
        # Make Day T (Index 5) have flat return
        prices_df.iloc[5, prices_df.columns.get_loc('close')] = 95 
        
        # Make News at T-1 (Index 4) be Positive
        # News df needs date column
        news_df = pd.DataFrame({
            'date': dates[4], # Date T-1
            'headline': ['Positive Growth'] # Score > 0
        }, index=[0])
        
        builder = FeatureBuilder()
        # Mock sentiment scoring to return +1.0 for our headline
        builder._simple_sentiment = lambda x: 1.0
        
        features = builder.build_all_features(prices_df, news=news_df)
        
        # Check Divergence at T (Index 5)
        # Should compare News(T-1) with Return(T-1)
        # News(T-1) = 1.0
        # Return(T-1) from Index 4 to 5? No, Return(T-1) is at Index 4 (Close 4 / Close 3)
        # OR Return(T-1) is stored at Index 5 as 'return_1d.shift(1)'?
        # In `build_price_features`, `return_1d` at Index 4 is Close[4]/Close[3].
        # In `build_news_features`, we access `return_1d.shift(1)` at Index 5. 
        # `return_1d` at Index 5 is Close[5]/Close[4].
        # `return_1d` shift(1) at Index 5 is Close[4]/Close[3].
        # Wait, if Close[4]=95, Close[3]=100. return_1d[4] = -0.05.
        # At Index 5 (Day T), we access shift(1) -> return_1d[4] -> -0.05.
        # So we compare News(T-1) [aligned to T via shift] vs Return(T-1).
        
        # Check Index 5
        divergence = features.iloc[5]['news_price_divergence']
        # Sentiment = 1.0
        # Return = -0.05.
        # Vol? Vol might be 0 or small if flat before?
        # If volatile is computed 20d, and history is flat, vol is 0?
        # safe_divide might return 0 if vol is 0.
        # Then divergence = 1.0 - 0 = 1.0.
        # If vol is non-zero, z-score is negative huge. 1.0 - (-huge) = huge positive.
        # Either way, divergence should be >= 1.0 (Positive Sentiment, Negative Return).
        
        self.assertGreater(divergence, 0.5)

if __name__ == '__main__':
    unittest.main()
