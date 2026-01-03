import pandas as pd
import numpy as np
from src.core.settings import MarketSettings


class RiskEngine:
    """
    Risk premium engine for B2B electricity contracts.

    Covers two main sources of risk:
    - Profiling Cost (Shape Risk): mismatch between client load profile and baseload prices
    - Volume Risk (Swing Risk): uncertainty on actual consumption vs forecast
    """

    def __init__(self, settings: MarketSettings, spot_volatility: float):
        """
        settings:
        Market configuration and conventions.

        spot_volatility:
        Reference spot price volatility used to price volume optionality.
        """
        self.settings = settings
        self.spot_volatility = spot_volatility

    # ------------------------------------------------------------------
    # Profiling (Shape) risk
    # ------------------------------------------------------------------
    def calculate_profiling_cost(
        self, load_curve: pd.Series, hpfc: pd.Series
    ) -> float:
        """
        Computes the profiling cost in €/MWh.

        Definition:
        Difference between the client's load-weighted price
        and the simple market baseload price.

        Formula:
        (Sum(Load * Price) / Sum(Load)) - Mean(Price)
        """
        # Defensive checks
        if load_curve.empty or hpfc.empty:
            return 0.0

        total_volume = load_curve.sum()
        if total_volume == 0:
            return 0.0

        # 1. Market reference price
        # Baseload is approximated by the simple average of hourly prices
        market_baseload_price = hpfc.mean()

        # 2. Client capture price
        # Actual price paid given the specific consumption profile
        client_capture_price = (load_curve * hpfc).sum() / total_volume

        # 3. Profiling cost (spread)
        # Positive  -> consumption biased toward expensive hours
        # Negative  -> consumption biased toward cheap hours
        profiling_cost = client_capture_price - market_baseload_price

        return round(profiling_cost, 2)

    # ------------------------------------------------------------------
    # Volume (Swing) risk
    # ------------------------------------------------------------------
    def calculate_volume_risk_premium(self, volume_mwh: float) -> float:
        """
        Computes the volume risk premium (€/MWh).

        This premium compensates the supplier for volume uncertainty:
        deviations vs forecast are typically balanced on the spot market,
        where prices are volatile and asymmetric.
        """

        # 1. Fixed base premium
        # Covers standard balancing and operational costs
        base_risk_premium = 1.0  # €/MWh

        # 2. Volatility-driven component
        # Higher spot volatility increases the cost of volume optionality
        vol_component = 5.0 * self.spot_volatility

        # 3. Size adjustment
        # Smaller clients are statistically less predictable
        size_factor = 1.0
        if volume_mwh < 2000:
            size_factor = 1.5
        elif volume_mwh < 500:
            size_factor = 2.0

        # Aggregate premium
        premium = (base_risk_premium + vol_component) * size_factor

        # Hard cap to keep prices commercially realistic
        return round(min(premium, 15.0), 2)
