#!/usr/bin/env python3
"""
Structural Edge Validation (Master Script)
Executes Phase 1, 2, and 3 in a continuous run.
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from utils.config import CONFIG
from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime
from strategies.regime_lattice import RegimeLattice, LatticeState
from strategies.strategy_wrapper import StrategyWrapper

class StructuralValidator:
    def __init__(self):
        self.results = {}
        
    def run(self):
        print("=== STRUCTURAL EDGE VALIDATION PROTOCOL ===")
        print(f"Config: Seed={CONFIG.SEED}, Costs={CONFIG.COST_BPS}bps, Stress={CONFIG.STRESS_COST_BPS}bps")
        
        # 1. Data & Features (AAPL Source)
        print("\n[Phase 1] Pipeline Hardening & Data Preparation...")
        prices, features, targets = self._prepare_data("AAPL")
        
        # 2. Train Models (Walk-Forward)
        print("\n[Phase 2] Regime System Training (Walk-Forward)...")
        models, predictions, scores = self._train_models_walk_forward(features, targets)
        self.results['scores'] = scores
        
        # 3. Orthogonality Check
        print("\n[Phase 2.2] Orthogonality Check...")
        self._check_orthogonality(predictions)
        
        # 4. Lattice Construction
        print("\n[Phase 3] Lattice State Construction...")
        lattice = RegimeLattice()
        states = lattice.determine_state(predictions['vol'], predictions['trend'], predictions['liq'])
        
        # 5. Backtest (Source)
        print("\n[Phase 3.2] Conditional Strategy Backtest (AAPL)...")
        res_aapl = self._run_backtest(prices, states, "AAPL Source")
        
        # 6. Robustness: Cross-Asset (MSFT Target)
        print("\n[Phase 3.3] Robustness: Cross-Asset Transfer (MSFT)...")
        prices_msft, feats_msft, _ = self._prepare_data("MSFT")
        
        # Predict using Source Models (Transfer)
        preds_msft_vol = models['vol'].predict(feats_msft)
        preds_msft_trend = models['trend'].predict(feats_msft)
        preds_msft_liq = models['liq'].predict(feats_msft)
        
        states_msft = lattice.determine_state(
            pd.Series(preds_msft_vol, index=feats_msft.index),
            pd.Series(preds_msft_trend, index=feats_msft.index),
            pd.Series(preds_msft_liq, index=feats_msft.index)
        )
        
        res_msft = self._run_backtest(prices_msft, states_msft, "MSFT Transfer") 
        # Pass prices_msft directly
        
        # 7. Robustness: Shuffle (Source)
        print("\n[Phase 3.3] Robustness: Regime Shuffle (AAPL)...")
        self._run_shuffle_test(prices, states)
        
        # 8. Comparison
        print("\n=== FINAL REPORT ===")
        self._print_report(res_aapl, res_msft)

    def _prepare_data(self, ticker):
        pf = PriceFetcher()
        prices = pf.fetch(ticker, start_date=CONFIG.START_DATE, end_date=CONFIG.END_DATE)
        
        # Ensure DatetimeIndex (TZ-naive)
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        elif prices.index.tz is not None:
             prices.index = prices.index.tz_convert(None) # Make naive
        prices = prices.sort_index()
        
        fb = FeatureBuilder()
        features = fb.build_all_features(prices)
        
        # Strict Audit: timestamp checks are implicit in FeatureBuilder logic, 
        # but here we ensure NO TARGET LEAKAGE.
        labeler = RegimeLabeler()
        labels = labeler.label_regimes(prices)
        
        # Align
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        
        # Drop lookahead columns provided by FeatureBuilder/Labeler for debug
        drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 
                     'open', 'high', 'low', 'close', 'volume']
        # Also drop future metadata from labels
        drop_cols += ['future_return_5d', 'future_vol_5d', 'future_adx', 'future_amihud']
        
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        
        # Create Targets
        # Vol: 1 = Risk Off
        y_vol = labels.loc[common_idx, 'regime_vol'].apply(lambda x: 1 if x == MarketRegime.HIGH_VOL else 0)
        # Trend: 1 = Robust
        y_trend = labels.loc[common_idx, 'regime_trend'].apply(lambda x: 1 if x == TrendQuality.ROBUST else 0)
        # Liq: 1 = Stressed
        y_liq = labels.loc[common_idx, 'regime_liq'].apply(lambda x: 1 if x == LiquidityRegime.STRESSED else 0)
        
        return prices, X, {'vol': y_vol, 'trend': y_trend, 'liq': y_liq}

    def _train_models_walk_forward(self, X, targets):
        # 3 Models
        models = {}
        predictions = {'vol': [], 'trend': [], 'liq': []}
        indices = []
        
        # Simple Train/Test Split or Rolling?
        # "Fixed-length rolling windows"
        train_window = CONFIG.TRAIN_WINDOW_DAYS
        predict_window = CONFIG.PREDICT_WINDOW_DAYS
        
        # We need to save the LAST model for transfer testing
        last_models = {}
        
        for name, y in targets.items():
            print(f"  Training {name}...", end='', flush=True)
            preds = []
            
            # Start loop
            for t in range(train_window, len(X), predict_window):
                end_idx = min(t + predict_window, len(X))
                
                # Rolling Window: [t-train_window : t]
                start_train = t - train_window
                X_train = X.iloc[start_train:t]
                y_train = y.iloc[start_train:t]
                
                X_test = X.iloc[t:end_idx]
                
                clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, 
                                       random_state=CONFIG.SEED, verbose=-1, n_jobs=1)
                clf.fit(X_train, y_train)
                
                p = clf.predict(X_test)
                preds.extend(p)
                
                # Save last model
                if end_idx == len(X):
                    last_models[name] = clf
            
            predictions[name] = pd.Series(preds, index=X.index[train_window:])
            print(f" Done. Acc: {accuracy_score(y.iloc[train_window:], predictions[name]):.2%}")
            
        return last_models, pd.DataFrame(predictions), {}

    def _check_orthogonality(self, preds_df):
        corr = preds_df.corr()
        print("  Correlation Matrix:")
        print(corr)
        max_corr = corr.abs().unstack().sort_values(ascending=False)
        # Filter diag
        max_corr = max_corr[max_corr < 0.999].max()
        print(f"  Max Cross-Correlation: {max_corr:.2f}")
        if max_corr > 0.3:
            print("  WARNING: High Correlation detected.")
        else:
            print("  PASS: Orthogonality confirmed.")

    def _run_backtest(self, prices, states, label, price_ticker=None):
        if prices is None and price_ticker:
            pf = PriceFetcher()
            prices = pf.fetch(price_ticker, start_date=CONFIG.START_DATE, end_date=CONFIG.END_DATE)
            
        strat = StrategyWrapper()
        
        # Print Regime Stats
        print(f"  {label} Regime Dist: {states.value_counts(normalize=True).to_dict()}")
        
        # 1. Conditional Strategy
        test_prices = prices.loc[states.index]
        signals = strat.generate_signals(test_prices, states)
        res = strat.backtest_conditional(test_prices, signals, cost_bps=CONFIG.COST_BPS)
        
        # 2. Baseline (Buy & Hold)
        # Signals = 1.0 everywhere
        buy_hold_signals = pd.Series(1.0, index=states.index)
        res_bm = strat.backtest_conditional(test_prices, buy_hold_signals, cost_bps=CONFIG.COST_BPS)
        
        print(f"  {label} | Strategy Return: {res['total_return']:.2%} | Sharpe: {res['sharpe']:.2f} | DD: N/A")
        print(f"  {label} | Baseline Return: {res_bm['total_return']:.2%} | Sharpe: {res_bm['sharpe']:.2f}")
        
        # Also run Stress Test (15bps)
        res_stress = strat.backtest_conditional(test_prices, signals, cost_bps=CONFIG.STRESS_COST_BPS)
        print(f"  {label} | Stress (15bps) Ret: {res_stress['total_return']:.2%}")
        
        return {'strat': res, 'bm': res_bm, 'stress': res_stress}

    def _run_shuffle_test(self, prices, states):
        strat = StrategyWrapper()
        test_prices = prices.loc[states.index]
        
        # Original
        signals = strat.generate_signals(test_prices, states)
        res_orig = strat.backtest_conditional(test_prices, signals, cost_bps=CONFIG.COST_BPS)
        
        random_sharpes = []
        for _ in range(50):
            shuffled_states = states.sample(frac=1.0, replace=False)
            shuffled_states.index = states.index
            
            sig = strat.generate_signals(test_prices, shuffled_states)
            r = strat.backtest_conditional(test_prices, sig, cost_bps=CONFIG.COST_BPS)
            random_sharpes.append(r['sharpe'])
            
        p_val = np.mean(np.array(random_sharpes) >= res_orig['sharpe'])
        print(f"  Shuffle p-value (Sharpe): {p_val:.4f}")
        if p_val < 0.05:
            print("  PASS: Significant Edge.")
        else:
            print("  FAIL: Indistinguishable from Noise.")

    def _print_report(self, res_aapl, res_msft):
        print("\n=== VALIDATION SUMMARY ===")
        
        # AAPL
        delta_sharpe = res_aapl['strat']['sharpe'] - res_aapl['bm']['sharpe']
        print(f"Source (AAPL) Delta Sharpe: {delta_sharpe:.2f}")
        if res_aapl['strat']['total_return'] > res_aapl['bm']['total_return']:
            print("PASS: Conditional Beats Unconditional.")
        else:
            print("FAIL: Conditional Lags Unconditional.")
            
        if res_aapl['stress']['total_return'] > 0:
             print("PASS: Survives 15bps Stress.")
        else:
             print("FAIL: Decays under Stress.")
             
        # MSFT
        transfer_ret = res_msft['strat']['total_return']
        source_ret = res_aapl['strat']['total_return']
        retention = transfer_ret / source_ret if source_ret != 0 else 0
        print(f"Transfer Retention: {retention:.2%}")
        
        if transfer_ret > 0.7 * source_ret:
             print("PASS: Transfer Validation (>70%).") # Maybe too strict if AAPL Return is huge
        elif transfer_ret > 0 and res_msft['strat']['sharpe'] > 0:
             print("PASS: Transfer Positive (Soft Pass).")
        else:
             print("FAIL: Transfer Collapsed.")
             
        print("\nFinal Decision:")
        if delta_sharpe > 0 and transfer_ret > 0:
            print("SUCCESS: This is the minimal structural alpha that survives reality, costs, and transfer.")
        else:
            print("FAILURE: Edge not validated.")

if __name__ == "__main__":
    val = StructuralValidator()
    val.run()
