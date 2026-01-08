"""
Unit tests for Confidence Building modules

Tests:
- Reconciliation report generation
- Shadow backtest parity detection
- Capital parking yield calculation
"""

import pytest
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.reconciliation import DailyReconciliation, DecisionRecord
from monitoring.shadow_backtest import ShadowBacktester, ShadowResult
from execution.capital_parking import CapitalParking, ParkingVehicle, FlatStateManager


class TestDailyReconciliation:
    """Tests for reconciliation module."""
    
    def test_aligned_decisions(self):
        recon = DailyReconciliation()
        
        recon.record_expected("SPY", "ENTER_LONG", "NEUTRAL", 450.0)
        recon.record_actual("SPY", "ENTER_LONG", "NEUTRAL", 450.05)
        
        report = recon.generate_report()
        
        assert report.aligned_decisions == 1
        assert report.regime_misclassifications == 0
    
    def test_action_mismatch(self):
        recon = DailyReconciliation()
        
        recon.record_expected("SPY", "EXIT_TO_CASH", "HOSTILE", 450.0)
        recon.record_actual("SPY", "HOLD", "HOSTILE", 450.0)
        
        report = recon.generate_report()
        
        assert report.aligned_decisions == 0
        assert report.missed_exits == 1
    
    def test_regime_mismatch(self):
        recon = DailyReconciliation()
        
        recon.record_expected("SPY", "HOLD", "NEUTRAL", 450.0)
        recon.record_actual("SPY", "HOLD", "HOSTILE", 450.0)
        
        report = recon.generate_report()
        
        assert report.aligned_decisions == 1  # Action matches
        assert report.regime_misclassifications == 1
    
    def test_slippage_calculation(self):
        recon = DailyReconciliation()
        
        # Buy at 450.10 when expected 450.00 = ~2.2 bps slippage
        recon.record_expected("SPY", "ENTER_LONG", "NEUTRAL", 450.0)
        recon.record_actual("SPY", "ENTER_LONG", "NEUTRAL", 450.10)
        
        report = recon.generate_report()
        
        assert abs(report.avg_slippage_bps - 2.2) < 1  # ~2.2 bps


class TestShadowBacktest:
    """Tests for shadow backtesting."""
    
    def test_aligned_actions(self):
        shadow = ShadowBacktester()
        
        shadow.record_live("SPY", "ENTER_LONG", "2026-01-07T10:00:00")
        shadow.shadow_actions["SPY_2026-01-07"] = {
            "action": "ENTER_LONG",
            "regime": "NEUTRAL",
            "timestamp": "2026-01-07T10:00:00"
        }
        shadow._compare("SPY_2026-01-07", "2026-01-07T10:00:00", "SPY")
        
        assert shadow.get_alignment_rate() == 1.0
        assert len(shadow.get_divergences()) == 0
    
    def test_divergent_actions(self):
        shadow = ShadowBacktester()
        
        shadow.record_live("SPY", "ENTER_LONG", "2026-01-07T10:00:00")
        shadow.shadow_actions["SPY_2026-01-07"] = {
            "action": "EXIT_TO_CASH",  # Different!
            "regime": "HOSTILE",
            "timestamp": "2026-01-07T10:00:00"
        }
        shadow._compare("SPY_2026-01-07", "2026-01-07T10:00:00", "SPY")
        
        assert shadow.get_alignment_rate() == 0.0
        assert len(shadow.get_divergences()) == 1
    
    def test_parity_assertion(self):
        shadow = ShadowBacktester()
        
        # Add aligned action
        shadow.results.append(ShadowResult(
            timestamp="2026-01-07",
            symbol="SPY",
            live_action="HOLD",
            shadow_action="HOLD",
            is_aligned=True
        ))
        
        assert shadow.assert_parity() is True
        
        # Add divergent action
        shadow.results.append(ShadowResult(
            timestamp="2026-01-07",
            symbol="GLD",
            live_action="EXIT",
            shadow_action="HOLD",
            is_aligned=False,
            divergence_reason="Mismatch"
        ))
        
        assert shadow.assert_parity() is False


class TestCapitalParking:
    """Tests for capital parking."""
    
    def test_park_capital(self):
        parking = CapitalParking()
        
        pos = parking.park(50000)
        
        assert pos.amount == 50000
        assert pos.vehicle == ParkingVehicle.MONEY_MARKET
        assert pos.apy_at_entry == 0.05
    
    def test_different_vehicles(self):
        parking = CapitalParking()
        
        pos1 = parking.park(25000, vehicle=ParkingVehicle.TBILLS)
        pos2 = parking.park(25000, vehicle=ParkingVehicle.CASH)
        
        assert pos1.apy_at_entry == 0.045
        assert pos2.apy_at_entry == 0.0
    
    def test_unpark_returns_principal(self):
        parking = CapitalParking()
        
        parking.park(50000, position_id="TEST")
        total = parking.unpark("TEST")
        
        # Same day = no yield
        assert total == 50000
    
    def test_total_parked(self):
        parking = CapitalParking()
        
        parking.park(25000, position_id="POS1")
        parking.park(25000, position_id="POS2")
        
        assert parking.get_total_parked() == 50000
    
    def test_unpark_all(self):
        parking = CapitalParking()
        
        parking.park(25000, position_id="POS1")
        parking.park(25000, position_id="POS2")
        
        total = parking.unpark_all()
        
        assert total == 50000
        assert len(parking.positions) == 0


class TestFlatStateManager:
    """Tests for flat state management."""
    
    def test_enter_flat(self):
        parking = CapitalParking()
        manager = FlatStateManager(parking)
        
        manager.enter_flat(50000)
        
        assert manager.is_flat is True
        assert manager.flat_since == date.today()
    
    def test_exit_flat(self):
        parking = CapitalParking()
        manager = FlatStateManager(parking)
        
        manager.enter_flat(50000)
        total = manager.exit_flat()
        
        assert total == 50000
        assert manager.is_flat is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
