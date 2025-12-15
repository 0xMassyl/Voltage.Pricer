from dataclasses import dataclass
from typing import Tuple


@dataclass
class MarketSettings:
    """
    Centralized configuration for the Belgian power market.
    Purpose: isolate all market assumptions (costs + calendar rules)
    to keep pricing logic clean and easily adjustable.
    """

    # ------------------------------------------------------------------
    # Non-commodity costs (€/MWh)
    # These costs are added on top of wholesale power prices
    # to approximate the final delivered electricity cost.
    # ------------------------------------------------------------------

    ELIA_GRID_FEE: float = 12.50
    # Transmission network cost (TSO – Elia)

    DISTRIBUTION_GRID_FEE: float = 8.00
    # Distribution network cost
    # Simulated average, since real values depend on location and voltage level

    GREEN_CERT_COST: float = 2.00
    # Cost linked to renewable energy obligations (green certificates)

    TAXES_AND_LEVIES: float = 3.50
    # Regulatory taxes and miscellaneous levies

    VAT_RATE: float = 0.21
    # Belgian VAT applied on top of all energy-related charges

    # ------------------------------------------------------------------
    # Market calendar definition (EEX-style Peak / Off-Peak)
    # Used to classify hours for pricing, risk, or load aggregation.
    # ------------------------------------------------------------------

    PEAK_START_HOUR: int = 8
    # Peak period starts at 08:00

    PEAK_END_HOUR: int = 20
    # Peak period ends at 20:00 (excluded)

    WEEKEND_DAYS: Tuple[int, int] = (5, 6)
    # Saturday (5) and Sunday (6) are always Off-Peak


# Global immutable-like settings instance
# Acts as a single source of truth across the project
SETTINGS = MarketSettings()
