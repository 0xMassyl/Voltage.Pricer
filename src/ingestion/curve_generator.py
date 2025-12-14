import pandas as pd
import numpy as np
from src.core.settings import SETTINGS 

class LoadCurveGenerator:
    """
    Generates synthetic hourly load profiles (consumption curves) for testing.
    Simulates different B2B client typologies (Industry, Office, PPA).
    """

    def __init__(self, year: int = 2026):
        self.year = year
        # Create an hourly index for the full year
        self.dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h")
        self.n_hours = len(self.dates)

    def generate_profile(self, profile_type: str, annual_volume_mwh: float) -> pd.Series:
        """
        Creates a normalized load curve scaled to the target annual volume.
        """
        base_curve = np.zeros(self.n_hours)
        
        # 1. Définition de la forme (shape)
        if profile_type == "INDUSTRY_24_7":
            # Usine: Consommation plate avec bruit
            base_curve = np.random.normal(loc=1.0, scale=0.05, size=self.n_hours)
            # Utilisation des settings pour le weekend (jours 5 et 6)
            is_weekend = self.dates.dayofweek.isin(SETTINGS.WEEKEND_DAYS)
            base_curve[is_weekend] *= 0.85 # Légère baisse de production le weekend

        elif profile_type == "OFFICE_BUILDING":
            # Tertiaire: 9h-18h en semaine, veille le reste
            hour = self.dates.hour
            is_working_day = self.dates.dayofweek < 5
            
            # Utilisation des settings pour les heures Peak (8h-20h)
            is_active_hour = (hour >= SETTINGS.PEAK_START_HOUR) & (hour < SETTINGS.PEAK_END_HOUR)
            is_working = is_active_hour & is_working_day
            
            base_curve[is_working] = np.random.normal(1.0, 0.1, is_working.sum())
            base_curve[~is_working] = np.random.normal(0.1, 0.05, (~is_working).sum())

        elif profile_type == "SOLAR_PPA":
            # Production Solaire: Courbe en cloche le jour
            hour = self.dates.hour
            daylight = (hour >= 6) & (hour <= 20)
            base_curve[daylight] = np.sin((hour[daylight] - 6) * np.pi / 14) 
            base_curve = np.maximum(base_curve, 0)

        # 2. Normalisation au volume annuel cible
        total_units = base_curve.sum()
        if total_units == 0:
            return pd.Series(0.0, index=self.dates)
            
        normalized_curve = (base_curve / total_units) * annual_volume_mwh
        
        return pd.Series(normalized_curve, index=self.dates, name="Load (MWh)")