"""
Capital Parking Logic

When the strategy is flat (EXIT_TO_CASH):
- Allocate to risk-free proxy (money market, T-bills)
- Zero interaction with regime logic
- Track parking yield

Paper-first. Capital efficiency only.
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkingVehicle(Enum):
    """Available parking vehicles."""
    MONEY_MARKET = "money_market"  # ~5% APY
    TBILLS = "tbills"  # ~4.5% APY
    CASH = "cash"  # 0% APY
    SPAXX = "SPAXX"  # Fidelity money market


@dataclass
class ParkingPosition:
    """Capital parking position."""
    vehicle: ParkingVehicle
    amount: float
    entry_date: date
    apy_at_entry: float
    current_value: float
    accrued_yield: float


class CapitalParking:
    """
    Manages capital when strategy is flat.
    
    CRITICAL: Zero interaction with regime logic.
    This is purely a treasury function.
    
    Usage:
        parking = CapitalParking(default_vehicle=ParkingVehicle.MONEY_MARKET)
        parking.park(50000)  # Park $50k
        
        # Later, when re-entering:
        amount = parking.unpark_all()
    """
    
    # Current APY estimates (as of 2026)
    APY_RATES = {
        ParkingVehicle.MONEY_MARKET: 0.050,  # 5.0%
        ParkingVehicle.TBILLS: 0.045,  # 4.5%
        ParkingVehicle.CASH: 0.000,  # 0%
        ParkingVehicle.SPAXX: 0.049,  # 4.9%
    }
    
    def __init__(
        self,
        default_vehicle: ParkingVehicle = ParkingVehicle.MONEY_MARKET,
        log_dir: Optional[Path] = None
    ):
        project_root = Path(__file__).parent.parent.parent
        self.log_dir = log_dir or project_root / "logs" / "parking"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_vehicle = default_vehicle
        self.positions: Dict[str, ParkingPosition] = {}
        self.total_yield_earned: float = 0.0
        
        logger.info(f"CapitalParking initialized (default: {default_vehicle.value})")
    
    def park(
        self,
        amount: float,
        vehicle: Optional[ParkingVehicle] = None,
        position_id: Optional[str] = None
    ) -> ParkingPosition:
        """
        Park capital in risk-free vehicle.
        
        Args:
            amount: Amount to park
            vehicle: Parking vehicle (default: money market)
            position_id: Optional identifier
        """
        vehicle = vehicle or self.default_vehicle
        pos_id = position_id or f"PARK_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        apy = self.APY_RATES.get(vehicle, 0)
        
        position = ParkingPosition(
            vehicle=vehicle,
            amount=amount,
            entry_date=date.today(),
            apy_at_entry=apy,
            current_value=amount,
            accrued_yield=0.0
        )
        
        self.positions[pos_id] = position
        
        logger.info(f"Parked ${amount:,.2f} in {vehicle.value} @ {apy*100:.1f}% APY")
        self._log_event("PARK", pos_id, amount, vehicle)
        
        return position
    
    def update_values(self):
        """Update current values based on time elapsed and APY."""
        today = date.today()
        
        for pos_id, pos in self.positions.items():
            days_parked = (today - pos.entry_date).days
            
            # Simple interest (daily accrual)
            daily_rate = pos.apy_at_entry / 365
            new_yield = pos.amount * daily_rate * days_parked
            
            pos.accrued_yield = new_yield
            pos.current_value = pos.amount + new_yield
    
    def unpark(self, position_id: str) -> float:
        """
        Unpark a specific position.
        
        Returns total value (principal + yield).
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return 0.0
        
        self.update_values()
        pos = self.positions.pop(position_id)
        
        self.total_yield_earned += pos.accrued_yield
        
        logger.info(f"Unparked ${pos.current_value:,.2f} (yield: ${pos.accrued_yield:.2f})")
        self._log_event("UNPARK", position_id, pos.current_value, pos.vehicle)
        
        return pos.current_value
    
    def unpark_all(self) -> float:
        """Unpark all positions and return total value."""
        self.update_values()
        
        total = 0.0
        for pos_id in list(self.positions.keys()):
            total += self.unpark(pos_id)
        
        return total
    
    def get_total_parked(self) -> float:
        """Get total parked capital including yields."""
        self.update_values()
        return sum(p.current_value for p in self.positions.values())
    
    def get_status(self) -> Dict:
        """Get parking status summary."""
        self.update_values()
        
        return {
            "positions": len(self.positions),
            "total_parked": self.get_total_parked(),
            "total_yield_earned": self.total_yield_earned,
            "vehicles": {
                v.value: sum(p.current_value for p in self.positions.values() 
                            if p.vehicle == v)
                for v in ParkingVehicle
            }
        }
    
    def _log_event(
        self,
        event_type: str,
        position_id: str,
        amount: float,
        vehicle: ParkingVehicle
    ):
        """Log parking event."""
        log_file = self.log_dir / f"parking_{date.today().isoformat()}.log"
        
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}|"
                   f"{event_type}|{position_id}|{amount:.2f}|{vehicle.value}\n")


class FlatStateManager:
    """
    Manages the flat state with automatic parking.
    """
    
    def __init__(self, parking: CapitalParking):
        self.parking = parking
        self.is_flat = False
        self.flat_since: Optional[date] = None
    
    def enter_flat(self, capital: float) -> ParkingPosition:
        """Enter flat state and park capital."""
        self.is_flat = True
        self.flat_since = date.today()
        
        # Park capital - NO regime logic here
        return self.parking.park(capital)
    
    def exit_flat(self) -> float:
        """Exit flat state and return capital."""
        if not self.is_flat:
            return 0.0
        
        self.is_flat = False
        flat_days = (date.today() - self.flat_since).days if self.flat_since else 0
        self.flat_since = None
        
        total = self.parking.unpark_all()
        logger.info(f"Exited flat state after {flat_days} days")
        
        return total
    
    def days_flat(self) -> int:
        """Get number of days in flat state."""
        if not self.is_flat or not self.flat_since:
            return 0
        return (date.today() - self.flat_since).days


if __name__ == "__main__":
    parking = CapitalParking()
    
    # Park capital
    pos = parking.park(50000)
    print(f"Parked: ${pos.amount:,.0f}")
    
    # Simulate time passing (update values)
    import time
    parking.update_values()
    
    status = parking.get_status()
    print(f"Total parked: ${status['total_parked']:,.2f}")
    
    # Unpark
    total = parking.unpark_all()
    print(f"Returned: ${total:,.2f}")
