
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.features.feature_builder import FeatureBuilder

class TestFeatureTargets(unittest.TestCase):
    def setUp(self):
        # Create a deterministic mock dataframe
        # Dates: 2023-01-01 to 2023-01-10 (10 days)
        dates = pd.date_range(start='2023-01-01', periods=10)
        
        # Open price increases by 10% each day: 100, 110, 121, ...
        # Close price is Open + 5
        self.prices = pd.DataFrame({
            'date': dates,
            'open': [100.0 * (1.1 ** i) for i in range(10)],
            'high': [100.0 * (1.1 ** i) + 10 for i in range(10)],
            'low': [100.0 * (1.1 ** i) - 5 for i in range(10)],
            'close': [100.0 * (1.1 ** i) + 5 for i in range(10)],
            'volume': [1000 for _ in range(10)]
        })
        self.prices = self.prices.set_index('date')
        self.builder = FeatureBuilder()

    def test_target_logic(self):
        """
        Verify target logic.
        
        Old Logic (Incorrect): Close-to-Close
        Target_t = (Close_{t+1} - Close_t) / Close_t
        
        New Logic (Correct): Open-to-Open
        Signal at Close_t -> Buy Open_{t+1}, Sell Open_{t+2}
        Target_t = (Open_{t+2} - Open_{t+1}) / Open_{t+1}
        """
        df = self.builder.build_all_features(self.prices)
        
        # We need to manually calculate what we EXPECT
        # Let's look at the first row (index 0, t=0)
        # Open_0 = 100, Close_0 = 105
        # Open_1 = 110, Close_1 = 115
        # Open_2 = 121, Close_2 = 126
        
        # If we successfully implemented T+1 execution (Open-to-Open):
        # For decision at t=0 (Close_0):
        # We enter at Open_1 (110)
        # We exit at Open_2 (121)
        # Return = (121 - 110) / 110 = 11 / 110 = 0.10 (10%)
        
        expected_return_t0 = (self.prices['open'].iloc[2] - self.prices['open'].iloc[1]) / self.prices['open'].iloc[1]
        
        # Check first valid target
        # Note: Depending on implementation, the dataframe might be truncated at the end
        # But t=0 should be valid
        
        actual_return_t0 = df['target_return_1d'].iloc[0]
        
        print(f"Index 0 Date: {df.index[0]}")
        print(f"Open T+1: {self.prices['open'].iloc[1]}")
        print(f"Open T+2: {self.prices['open'].iloc[2]}")
        print(f"Expected Target (Open-to-Open): {expected_return_t0}")
        print(f"Actual Target in DF: {actual_return_t0}")

        # Assert almost equal
        self.assertAlmostEqual(actual_return_t0, expected_return_t0, places=4, 
                               msg=f"Expected {expected_return_t0} but got {actual_return_t0}. Logic likely incorrect.")

if __name__ == '__main__':
    unittest.main()
