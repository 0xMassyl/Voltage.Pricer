import pandas as pd
import numpy as np
from src.core.settings import SETTINGS
from src.ingestion.elia_client import EliaDataConnector


class LoadCurveGenerator:
    """
    Hybrid load curve generator.

    Logic:
    1. Attempt to retrieve real load data from Elia (Belgian grid).
    2. Fall back to a synthetic profile if real data is unavailable.

    Purpose: generate realistic hourly consumption or production profiles
    suitable for pricing, risk, and sourcing analysis.
    """

    def __init__(self, year: int = 2026):
        self.year = year

        # Full hourly calendar for the target year
        self.dates = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31 23:00",
            freq="h",
        )
        self.n_hours = len(self.dates)

        # Connector to Elia Open Data
        self.elia = EliaDataConnector()

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------
    def generate_profile(
        self, profile_type: str, annual_volume_mwh: float
    ) -> pd.Series:
        """
        Generates an hourly load or production profile.

        profile_type:
        Defines the structural shape (industry, office, solar, etc.).

        annual_volume_mwh:
        Target annual energy volume used for normalization.
        """

        # --------------------------------------------------------------
        # Attempt to use real Elia data (only for 24/7 industrial profile)
        # --------------------------------------------------------------
        if profile_type == "INDUSTRY_24_7":
            print("Attempting to retrieve real Elia load profile...")

            # Use a short recent window as a representative shape
            real_curve = self.elia.fetch_real_load_curve(days=14)

            if not real_curve.empty:
                # Convert to numpy array for tiling
                pattern = real_curve.to_numpy()

                # Repeat the short pattern to cover the full year
                repeats = int((self.n_hours // len(pattern)) + 1)
                full_pattern = np.tile(pattern, repeats)[: self.n_hours]

                # Normalize to the requested annual volume
                total_units = full_pattern.sum()
                if total_units == 0:
                    normalized_curve = np.zeros(self.n_hours)
                else:
                    normalized_curve = (
                        full_pattern / total_units
                    ) * annual_volume_mwh

                print("Load profile generated using real Elia data.")
                return pd.Series(
                    normalized_curve,
                    index=self.dates,
                    name="Real Load (MWh)",
                )

        # --------------------------------------------------------------
        # Fallback: synthetic profile generation
        # --------------------------------------------------------------
        print("Using synthetic load generator (fallback).")

        base_curve = np.zeros(self.n_hours)

        if profile_type == "INDUSTRY_24_7":
            # Flat consumption with low noise and reduced weekends
            base_curve = np.random.normal(1.0, 0.05, self.n_hours)
            is_weekend = self.dates.dayofweek.isin(SETTINGS.WEEKEND_DAYS)
            base_curve[is_weekend] *= 0.85

        elif profile_type == "OFFICE_BUILDING":
            # Consumption concentrated on working hours
            hour = self.dates.hour
            is_working = (
                (hour >= 8)
                & (hour <= 18)
                & (self.dates.dayofweek < 5)
            )
            base_curve[is_working] = np.random.normal(
                1.0, 0.1, is_working.sum()
            )
            base_curve[~is_working] = 0.1

        elif profile_type == "SOLAR_PPA":
            # Solar production profile (daylight only)
            hour = self.dates.hour
            daylight = (hour >= 6) & (hour <= 20)
            base_curve[daylight] = np.sin(
                (hour[daylight] - 6) * np.pi / 14
            )
            base_curve = np.maximum(base_curve, 0)

        # Normalize synthetic curve to match annual volume
        total_units = base_curve.sum()
        if total_units > 0:
            normalized_curve = (base_curve / total_units) * annual_volume_mwh
        else:
            normalized_curve = base_curve

        return pd.Series(
            normalized_curve,
            index=self.dates,
            name="Synthetic Load",
        )
