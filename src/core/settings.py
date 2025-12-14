from dataclasses import dataclass
from typing import Dict

@dataclass
class MarketSettings:
    """
    Central configuration for the Belgian Power Market (EEX/Endex).
    Defines non-commodity costs and market calendar definitions.
    """
    # Non-Commodity Pass-Through Costs (€/MWh)
    ELIA_GRID_FEE: float = 12.50      # Transmission Network Cost
    DISTRIBUTION_GRID_FEE: float = 8.00 # Distribution Network Cost (simulated avg)
    GREEN_CERT_COST: float = 2.0      # Green Certificates Cost
    TAXES_AND_LEVIES: float = 3.50    # Regulatory Taxes
    VAT_RATE: float = 0.21            # VAT Rate
    
    # EEX Calendar Definition (Peak/Off-Peak)
    PEAK_START_HOUR: int = 8          # 08:00
    PEAK_END_HOUR: int = 20           # 20:00
    WEEKEND_DAYS: tuple = (5, 6)      # Saturday (5), Sunday (6)

# Global settings instance
SETTINGS = MarketSettings()