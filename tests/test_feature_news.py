
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.features.feature_builder import FeatureBuilder

class TestFeatureNews(unittest.TestCase):
    def setUp(self):
        # 3 Days of prices
        self.prices = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'close': [100, 101, 102],
            'open': [100, 101, 102],
            'high': [100, 101, 102],
            'low': [100, 101, 102],
            'volume': [100, 100, 100]
        }).set_index('date')
        
        # News with specific times
        self.news = pd.DataFrame({
            'date': pd.to_datetime([
                '2023-01-01 10:00:00', # During market
                '2023-01-01 18:00:00', # After market (Leakage if used at 16:00?)
                '2023-01-02 09:00:00'
            ]),
            'headline': ['Good news', 'Late news', 'Morning news'],
            'summary': ['Good', 'Late', 'Morning']
        })
        
        self.builder = FeatureBuilder()

    def test_news_alignment(self):
        # Current implementation likely fails to match timestamps
        df = self.builder.build_news_features(self.prices, self.news)
        
        # Check if news_count_1d is > 0
        print("News Count 1d:", df['news_count_1d'].tolist())
        
        # If logical bug exists (timestamp mismatch), these will be 0
        
        # Check if shift is needed for safety.
        # If I see news count > 0 at 2023-01-01, it means it aggregated.
        # But wait, feature_builder reindexes.
        
if __name__ == '__main__':
    unittest.main()
