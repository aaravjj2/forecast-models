"""Walk-forward backtesting framework."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Robust imports
import sys
from pathlib import Path
try:
    from ..ensemble.meta_ensemble import MetaEnsemble
    from ..utils.config import DEFAULT_TRAIN_WINDOW_DAYS, DEFAULT_TEST_WINDOW_DAYS, INITIAL_CAPITAL, COMMISSION_RATE
    from ..utils.helpers import ensure_no_leakage, save_artifact
except ImportError:
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from ensemble.meta_ensemble import MetaEnsemble
    from utils.config import DEFAULT_TRAIN_WINDOW_DAYS, DEFAULT_TEST_WINDOW_DAYS, INITIAL_CAPITAL, COMMISSION_RATE
    from utils.helpers import ensure_no_leakage, save_artifact


class WalkForwardBacktest:
    """Walk-forward backtesting with rolling windows."""
    
    def __init__(self, ensemble: MetaEnsemble, 
                 train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
                 test_window_days: int = DEFAULT_TEST_WINDOW_DAYS,
                 initial_capital: float = INITIAL_CAPITAL,
                 commission_rate: float = COMMISSION_RATE):
        self.ensemble = ensemble
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        self.results = []
    
    def run(self, features: pd.DataFrame, prices: pd.DataFrame,
            target: pd.Series) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            features: Feature matrix (indexed by date)
            prices: Price data with 'close' column
            target: Target returns (forward-looking)
        
        Returns:
            Dictionary with backtest results
        """
        # Ensure dates are sorted
        features = features.sort_index()
        prices = prices.sort_index()
        target = target.sort_index()
        
        # Align all data
        common_dates = features.index.intersection(prices.index).intersection(target.index)
        features = features.loc[common_dates]
        prices = prices.loc[common_dates]
        target = target.loc[common_dates]
        
        # Calculate windows
        start_date = features.index.min()
        end_date = features.index.max()
        
        current_date = start_date + pd.Timedelta(days=self.train_window_days)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        window_num = 0
        
        while current_date + pd.Timedelta(days=self.test_window_days) <= end_date:
            window_num += 1
            
            # Define train and test windows (FIXED-LENGTH ROLLING)
            # Train: [current_date - train_window_days, current_date)
            # Test: [current_date, current_date + test_window_days)
            train_start = current_date - pd.Timedelta(days=self.train_window_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.Timedelta(days=self.test_window_days)
            
            # CRITICAL: Fixed-length rolling window (NOT expanding)
            train_mask = (features.index >= train_start) & (features.index < train_end)
            test_mask = (features.index >= test_start) & (features.index < test_end)
            
            train_features = features[train_mask]
            test_features = features[test_mask]
            test_target = target[test_mask]
            test_prices = prices.loc[prices.index.isin(test_features.index)]
            
            if len(train_features) < 50 or len(test_features) == 0:
                current_date += pd.Timedelta(days=self.test_window_days)
                continue
            
            print(f"\nWindow {window_num}: Train {train_features.index.min()} to {train_features.index.max()}, "
                  f"Test {test_features.index.min()} to {test_features.index.max()}")
            
            # Train ensemble (specialists should already be trained, but retrain meta-model)
            try:
                # Get market features for meta-training
                market_features = self._extract_market_features(train_features)
                
                # Train meta-model on training window
                train_target = target[train_mask]
                self.ensemble.train(train_features, train_target, market_features)
                
                # Predict on test window
                test_market_features = self._extract_market_features(test_features)
                predictions = self.ensemble.predict_batch(test_features, test_market_features)
                
                # Store results
                for idx, (date, pred_row) in enumerate(predictions.iterrows()):
                    if idx < len(test_target):
                        all_predictions.append({
                            'date': date,
                            'signal': pred_row['signal'],
                            'prob_up': pred_row['prob_up'],
                            'confidence': pred_row['confidence']
                        })
                        all_actuals.append({
                            'date': date,
                            'actual_return': test_target.iloc[idx] if idx < len(test_target) else 0,
                            'actual_direction': 0  # Ignored in Phase 3
                        })
                        all_dates.append(date)
                
            except Exception as e:
                print(f"Error in window {window_num}: {e}")
                continue
            
            # Move window forward
            current_date += pd.Timedelta(days=self.test_window_days)
        
        # Calculate metrics
        if not all_predictions:
            return {'error': 'No predictions generated'}
        
        results_df = pd.DataFrame(all_predictions).set_index('date')
        actuals_df = pd.DataFrame(all_actuals).set_index('date')
        
        # Align
        common_dates = results_df.index.intersection(actuals_df.index)
        results_df = results_df.loc[common_dates]
        actuals_df = actuals_df.loc[common_dates]
        
        # Calculate metrics
        metrics = self._calculate_metrics(results_df, actuals_df, prices.loc[common_dates])
        
        # Store results
        self.results = {
            'predictions': results_df,
            'actuals': actuals_df,
            'metrics': metrics,
            'n_windows': window_num
        }
        
        return self.results
    
    def _extract_market_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract market regime features."""
        market_features = pd.DataFrame(index=features.index)
        
        # Volatility
        vol_cols = [col for col in features.columns if 'volatility' in col.lower()]
        if vol_cols:
            market_features['volatility'] = features[vol_cols[0]]
        else:
            market_features['volatility'] = 0.0
        
        # News intensity
        news_cols = [col for col in features.columns if 'news_count' in col.lower()]
        if news_cols:
            market_features['news_intensity'] = features[news_cols[0]]
        else:
            market_features['news_intensity'] = 0.0
        
        return market_features.fillna(0)
    
    def _calculate_metrics(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                          prices: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        # Directional accuracy
        predicted_direction = np.where(predictions['signal'] != 0, 
                                      np.where(predictions['signal'] > 0, 1, -1), 
                                      0)
        actual_direction = actuals['actual_direction'].values
        
        # Only count non-abstained predictions
        non_abstain_mask = predictions['signal'] != 0
        if non_abstain_mask.sum() > 0:
            directional_accuracy = (predicted_direction[non_abstain_mask] == 
                                  actual_direction[non_abstain_mask]).mean()
        else:
            directional_accuracy = 0.0
        
        # Coverage (percentage of days traded)
        coverage = non_abstain_mask.mean()
        
        # Precision and recall
        buy_signals = predictions['signal'] == 1
        sell_signals = predictions['signal'] == -1
        
        if buy_signals.sum() > 0:
            buy_precision = (actual_direction[buy_signals] == 1).mean()
            buy_recall = (actual_direction[buy_signals] == 1).sum() / (actual_direction == 1).sum() if (actual_direction == 1).sum() > 0 else 0
        else:
            buy_precision = buy_recall = 0.0
        
        if sell_signals.sum() > 0:
            sell_precision = (actual_direction[sell_signals] == -1).mean()
            sell_recall = (actual_direction[sell_signals] == -1).sum() / (actual_direction == -1).sum() if (actual_direction == -1).sum() > 0 else 0
        else:
            sell_precision = sell_recall = 0.0
        
        # Confidence calibration
        if non_abstain_mask.sum() > 0:
            avg_confidence = predictions.loc[non_abstain_mask, 'confidence'].mean()
        else:
            avg_confidence = 0.0
        
        # PnL simulation
        pnl_results = self._simulate_pnl(predictions, actuals, prices)
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'coverage': float(coverage),
            'buy_precision': float(buy_precision),
            'buy_recall': float(buy_recall),
            'sell_precision': float(sell_precision),
            'sell_recall': float(sell_recall),
            'avg_confidence': float(avg_confidence),
            'total_return': float(pnl_results['total_return']),
            'sharpe_ratio': float(pnl_results['sharpe_ratio']),
            'max_drawdown': float(pnl_results['max_drawdown']),
            'win_rate': float(pnl_results['win_rate']),
            'expectancy': float(pnl_results['expectancy']),
            'profit_factor': float(pnl_results['profit_factor']),
            'cagr': float(pnl_results['cagr']),
            'returns': pnl_results['returns'],
            'equity_curve': pnl_results['equity_curve']
        }
    
    def _simulate_pnl(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                     prices: pd.DataFrame) -> Dict[str, float]:
        """Simulate PnL from predictions."""
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        returns = []
        equity_curve = [capital]
        
        # Execution Parameters (Phase C)
        SPREAD_BPS = 5     # 5 basis points (0.05%) bid/ask spread
        SLIPPAGE_STD = 0.0005 # 5 bps random slippage standard deviation
        EXECUTION_DELAY_PROB = 0.1 # 10% chance of T+2 execution becoming T+3 (filled next day)
        
        import random
        random.seed(42) # Deterministic for reproducibility
        
        for date in predictions.index:
            if date not in actuals.index:
                continue
            
            signal = predictions.loc[date, 'signal']
            period_return = actuals.loc[date, 'actual_return']
            
            # Phase C: Realistic Execution
            # 1. Spread Cost: We pay half the spread on entry and half on exit.
            # Effectively, any trade incurs full spread cost if round trip.
            # Strategy Return = Signal * Return - Spread
            # Cost applies on ENTRY and EXIT.
            # Simplification: Pay spread/2 on every turnover unit.
            
            # 2. Slippage: Random penalty proportional to volatility (or fixed std for now)
            slippage = abs(np.random.normal(0, SLIPPAGE_STD))
            
            # 3. Execution Delay:
            # If delayed, we might miss the T+1 Open and get T+2 Open (so we miss the return of day 1)
            # This logic is complex with pre-calculated returns.
            # Simplification: If delayed, we get 0 return for this period (missed trade) or get next period return.
            # Let's say we miss the trade if delayed.
            if random.random() < EXECUTION_DELAY_PROB:
                # Execution missed/delayed
                # Signal effectively becomes 0 for this period
                executed_signal = 0
            else:
                executed_signal = signal
            
            strategy_return = executed_signal * period_return
            
            # Transaction Costs
            turnover = abs(executed_signal - position)
            
            # Spread Cost (5bps * turnover)
            spread_cost = turnover * (SPREAD_BPS / 10000)
            
            # Slippage Cost (Slippage * turnover)
            slippage_cost = turnover * slippage
            
            # Commission
            comm_cost = turnover * self.commission_rate
            
            # Total Transaction Cost
            total_cost = spread_cost + slippage_cost + comm_cost
            
            # Net return
            net_return = strategy_return - total_cost
            
            # Update capital
            capital *= (1 + net_return)
            returns.append(net_return)
            equity_curve.append(capital)
            
            # Update position
            position = executed_signal
        
        # Calculate metrics
        if returns:
            returns_np = np.array(returns)
            
            # 1. Total Return
            total_return = (capital - self.initial_capital) / self.initial_capital
            
            # 2. Sharpe Ratio
            if returns_np.std() > 0:
                sharpe_ratio = (returns_np.mean() / returns_np.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 3. Max Drawdown
            equity_np = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_np)
            drawdown = (equity_np - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # 4. Win Rate
            win_rate = (returns_np > 0).mean()
            
            # 5. Expectancy (Average PnL per trade)
            expectancy = returns_np.mean()
            
            # 6. Profit Factor (Gross Win / Gross Loss)
            gross_win = returns_np[returns_np > 0].sum()
            gross_loss = abs(returns_np[returns_np < 0].sum())
            if gross_loss > 0:
                profit_factor = gross_win / gross_loss
            else:
                profit_factor = float('inf') if gross_win > 0 else 0.0
                
            # 7. CAGR (Annualized Return)
            days = (actuals.index[-1] - actuals.index[0]).days
            if days > 0:
                cagr = (capital / self.initial_capital) ** (365 / days) - 1
            else:
                cagr = 0.0
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
            expectancy = 0.0
            profit_factor = 0.0
            cagr = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'cagr': cagr,
            'final_capital': capital,
            'returns': returns,
            'equity_curve': equity_curve
        }
    
    def save_results(self, filepath: str):
        """Save backtest results."""
        if not self.results:
            raise ValueError("No results to save")
        
        save_artifact(self.results, filepath, format='pickle')

