import pandas as pd
import numpy as np
from src.core.settings import MarketSettings

class RiskEngine:
    """
    Handles risk premiums calculation for Electricity B2B contracts.
    Focuses on Profiling Cost (Shape Risk) and Volume Uncertainty (Swing Risk).
    """

    def __init__(self, settings: MarketSettings, spot_volatility: float):
        self.settings = settings
        self.spot_volatility = spot_volatility # Reference volatility from Market Data

    def calculate_profiling_cost(self, load_curve: pd.Series, hpfc: pd.Series) -> float:
        """
        Calculates the Profiling Cost (Shape Risk) in €/MWh.
        
        Definition: The difference between the Client's Weighted Average Price
        and the Market Baseload Price.
        
        Formula: (Sum(Load * Price) / Sum(Load)) - Mean(Price)
        """
        if load_curve.empty or hpfc.empty:
            return 0.0

        total_volume = load_curve.sum()
        if total_volume == 0:
            return 0.0

        # 1. Market Reference (Baseload Price)
        # The simple average of the hourly curve
        market_baseload_price = hpfc.mean()

        # 2. Client Capture Price (Volume Weighted Average Price)
        # The actual price incurred to serve this specific profile
        client_capture_price = (load_curve * hpfc).sum() / total_volume

        # 3. Profiling Cost = Spread
        # If Positive: Client consumes during expensive hours -> Pays a premium
        # If Negative: Client consumes during cheap hours -> Gets a discount
        profiling_cost = client_capture_price - market_baseload_price
        
        return round(profiling_cost, 2)

    def calculate_volume_risk_premium(self, volume_mwh: float) -> float:
        """
        Calculates the Volume Risk Premium (Swing Risk) in €/MWh.
        
        This covers the risk that the client consumes +/- 10% vs forecast,
        forcing the supplier to trade on the volatile Spot market.
        """
        
        # 1. Base Risk (Balancing costs)
        base_risk_premium = 1.0 # €/MWh fixed component
        
        # 2. Volatility Component
        # Higher market volatility = Higher option cost
        vol_component = 5.0 * self.spot_volatility 
        
        # 3. Size Factor (Small clients are statistically more volatile/less predictable)
        size_factor = 1.0
        if volume_mwh < 2000:
            size_factor = 1.5
        elif volume_mwh < 500:
             size_factor = 2.0
        
        # Aggregated Premium
        premium = (base_risk_premium + vol_component) * size_factor
        
        # Cap to avoid non-commercial prices
        return round(min(premium, 15.0), 2)