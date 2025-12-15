import pandas as pd
from dataclasses import dataclass
from typing import Dict

# ML forecasting engine used to generate the hourly price shape
from src.domain.ml_forecasting import MLPriceForecaster


@dataclass
class SourcingResult:
    """
    Output container for sourcing valuation results.
    Keeps volumes and costs explicit for reporting and auditability.
    """
    total_volume_mwh: float
    base_volume_mwh: float
    peak_volume_mwh: float
    weighted_average_price: float
    total_commodity_cost: float


class ElectricityPricingEngine:
    """
    Core pricing engine for B2B electricity contracts.
    Purpose: value a client's load curve against a smart HPFC
    generated via machine learning.
    """

    def __init__(self, market_prices: Dict[str, float]):
        """
        market_prices:
        Dictionary containing market references (e.g. CAL baseload).
        """
        self.market_prices = market_prices

        # Initialize ML forecaster using the current baseload price
        # as a level anchor for calibration
        self.forecaster = MLPriceForecaster(
            spot_reference=market_prices.get("CAL_BASE", 95.5)
        )

    # ------------------------------------------------------------------
    # HPFC generation
    # ------------------------------------------------------------------
    def generate_hpfc(self, dates: pd.Index) -> pd.Series:
        """
        Generates an Hourly Price Forward Curve (HPFC) for the given dates.
        The hourly shape is learned from long-term seasonality via XGBoost.
        """
        # Enforce proper datetime index
        dates = pd.to_datetime(dates)

        if dates.empty:
            # Defensive behavior: no dates, no prices
            return pd.Series(dtype=float)

        # HPFC is generated at yearly granularity
        target_year = dates[0].year

        # Call ML engine (training is triggered if needed)
        hpfc_full_year = self.forecaster.generate_forecast_curve(target_year)

        # Align the full-year curve to the exact requested timestamps
        # Nearest is acceptable since HPFC is hourly
        hpfc = hpfc_full_year.reindex(dates, method="nearest")

        return hpfc

    # ------------------------------------------------------------------
    # Commodity cost valuation
    # ------------------------------------------------------------------
    def compute_sourcing_cost(self, load_curve: pd.Series) -> SourcingResult:
        """
        Computes the raw commodity cost by marking the client's
        load curve against the ML-generated HPFC.
        """
        # Ensure the load curve is properly time-indexed
        if not isinstance(load_curve.index, pd.DatetimeIndex):
            load_curve.index = pd.to_datetime(load_curve.index)

        # 1. Generate HPFC corresponding to the load curve horizon
        hpfc = self.generate_hpfc(load_curve.index)

        # 2. Volume calculations
        total_volume = load_curve.sum()

        # Explicit conversion to guarantee datetime attributes
        hpfc_idx = pd.to_datetime(hpfc.index)

        # Peak definition (standard EEX convention)
        # Used only for volume reporting, not for pricing
        is_peak = (
            (hpfc_idx.hour >= 8)
            & (hpfc_idx.hour < 20)
            & (hpfc_idx.dayofweek < 5)
        )

        peak_volume = load_curve[is_peak].sum()

        # In sourcing terms, baseload volume covers the full profile
        base_volume = total_volume

        # 3. Valuation: hourly load marked to hourly forward prices
        total_cost = (load_curve * hpfc).sum()

        # Load-weighted average price
        avg_price = total_cost / total_volume if total_volume > 0 else 0.0

        return SourcingResult(
            total_volume_mwh=round(total_volume, 2),
            base_volume_mwh=round(base_volume, 2),
            peak_volume_mwh=round(peak_volume, 2),
            weighted_average_price=round(avg_price, 2),
            total_commodity_cost=round(total_cost, 2),
        )
