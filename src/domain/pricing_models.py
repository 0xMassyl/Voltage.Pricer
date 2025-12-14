import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
# L'importation fonctionne car ml_forecasting.py existe maintenant
from src.domain.ml_forecasting import MLPriceForecaster 

@dataclass
class SourcingResult:
    total_volume_mwh: float
    base_volume_mwh: float
    peak_volume_mwh: float
    weighted_average_price: float
    total_commodity_cost: float

class ElectricityPricingEngine:
    """
    Core Pricing Engine for B2B Electricity Contracts (Voltage Pricer).
    Integrates XGBoost Forecasting to value the commodity cost against a smart HPFC.
    """

    def __init__(self, market_prices: Dict[str, float]):
        self.market_prices = market_prices
        # Initialize the ML Forecaster with the current Baseload price anchor
        self.forecaster = MLPriceForecaster(spot_reference=market_prices.get("CAL_BASE", 95.5))

    def generate_hpfc(self, dates: pd.Index) -> pd.Series:
        """
        Generates an Hourly Price Forward Curve (HPFC).
        Uses XGBoost Machine Learning to predict the hourly shape based on 25 years of seasonality.
        """
        # Ensure dates is a proper index
        dates = pd.to_datetime(dates)

        # Extract target year
        if dates.empty:
            return pd.Series(dtype=float)
            
        target_year = dates[0].year
        
        # Call ML Engine (triggers training if not already done)
        hpfc_full_year = self.forecaster.generate_forecast_curve(target_year)
        
        # Filter to keep only the requested dates
        hpfc = hpfc_full_year.reindex(dates, method='nearest')
        
        return hpfc

    def compute_sourcing_cost(self, load_curve: pd.Series) -> SourcingResult:
        """
        Computes the raw commodity cost by marking the client's load curve against the ML-generated HPFC.
        """
        if not isinstance(load_curve.index, pd.DatetimeIndex):
             load_curve.index = pd.to_datetime(load_curve.index)
        
        # 1. Generate HPFC via ML
        hpfc = self.generate_hpfc(load_curve.index)

        # 2. Calcul des Volumes et Coûts
        total_volume = load_curve.sum()
        
        # FIX: On extrait l'index et on le convertit explicitement en DatetimeIndex
        # Cela garantit à Pylance que les attributs .hour et .dayofweek existent.
        hpfc_idx = pd.to_datetime(hpfc.index)

        # Peak Definition (Standard EEX: Mon-Fri, 08-20h)
        # Note: Cette définition est uniquement pour le reporting du volume Peak
        is_peak = ((hpfc_idx.hour >= 8) & (hpfc_idx.hour < 20) & (hpfc_idx.dayofweek < 5))
        
        peak_volume = load_curve[is_peak].sum()
        base_volume = total_volume # En comptabilité, Base couvre tout

        # Valuation (Volume * Price)
        total_cost = (load_curve * hpfc).sum()
        avg_price = total_cost / total_volume if total_volume > 0 else 0.0

        return SourcingResult(
            total_volume_mwh=round(total_volume, 2),
            base_volume_mwh=round(base_volume, 2),
            peak_volume_mwh=round(peak_volume, 2),
            weighted_average_price=round(avg_price, 2),
            total_commodity_cost=round(total_cost, 2)
        )