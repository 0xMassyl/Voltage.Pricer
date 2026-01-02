import pandas as pd
import numpy as np
from src.core.settings import SETTINGS
from src.ingestion.elia_client import EliaDataConnector


class LoadCurveGenerator:
    """
    Hybrid load curve generator.

    The objective of this class is to produce hourly energy profiles
    that look realistic from a market and operational perspective.

    The emphasis is on structural shape and consistency,
    not on short-term forecasting accuracy.
    """

    def __init__(self, year: int = 2026):
        self.year = year
        self.dates = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31 23:00",
            freq="h",
        )
        self.n_hours = len(self.dates)
        self.elia = EliaDataConnector()

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------
    def generate_profile(
        self, profile_type: str, annual_volume_mwh: float
    ) -> pd.Series:
        """
        Generate an hourly load or production profile.

        The method follows two steps:
        1. Build a curve with the correct *shape*
           (daily, weekly and operational patterns)
        2. Scale the curve so that total annual energy
           matches the requested volume

        This separation ensures that shape and volume
        can be reasoned about independently.
        """

        # --------------------------------------------------------------
        # Attempt to use real Elia data
        # Only meaningful for a continuous industrial load
        # --------------------------------------------------------------
        if profile_type == "INDUSTRY_24_7":
            # We first try to reuse real historical system load data
            # because it naturally embeds:
            # - intraday cycles
            # - weekday / weekend effects
            # - operational inertia
            real_curve = self.elia.fetch_real_load_curve(days=14)

            if not real_curve.empty:
                # The absolute values of the real curve do not matter here.
                # We only keep its relative shape.
                pattern = real_curve.to_numpy()

                # The short real pattern is repeated to span the full year.
                # This assumes that the operational structure
                # is more important than long-term seasonality.
                repeats = int((self.n_hours // len(pattern)) + 1)
                full_pattern = np.tile(pattern, repeats)[: self.n_hours]

                # The repeated pattern is then normalized so that
                # the total yearly energy equals the target volume.
                total_units = full_pattern.sum()

                if total_units == 0:
                    # Defensive case: avoid division by zero
                    normalized_curve = np.zeros(self.n_hours)
                else:
                    normalized_curve = (
                        full_pattern / total_units
                    ) * annual_volume_mwh

                # At this point:
                # - the curve has a realistic shape
                # - the annual energy constraint is satisfied
                return pd.Series(
                    normalized_curve,
                    index=self.dates,
                    name="Real Load (MWh)",
                )

        # --------------------------------------------------------------
        # Synthetic profile generation
        # Used when real data is unavailable or not appropriate
        # --------------------------------------------------------------
        # The synthetic approach follows the same philosophy:
        # define a relative shape first, then scale it.
        base_curve = np.zeros(self.n_hours)

        # --------------------------------------------------------------
        # CASE 1: INDUSTRIAL SITE OPERATING 24/7
        # --------------------------------------------------------------
        if profile_type == "INDUSTRY_24_7":
            # Industrial processes typically run continuously
            # with limited short-term variability.
            # A near-flat signal with small noise reflects this behavior.
            base_curve = np.random.normal(1.0, 0.05, self.n_hours)

            # Even in continuous industries, activity is often reduced
            # during weekends due to staffing and maintenance constraints.
            is_weekend = self.dates.dayofweek.isin(SETTINGS.WEEKEND_DAYS)
            base_curve[is_weekend] *= 0.85

        # --------------------------------------------------------------
        # CASE 2: OFFICE BUILDING
        # --------------------------------------------------------------
        elif profile_type == "OFFICE_BUILDING":
            # Office consumption is strongly driven by human presence.
            # Load is concentrated on working days and working hours.
            hour = self.dates.hour
            is_working = (
                (hour >= 8)
                & (hour <= 18)
                & (self.dates.dayofweek < 5)
            )

            # During working hours, consumption fluctuates
            # due to occupancy, IT usage and HVAC behavior.
            base_curve[is_working] = np.random.normal(
                1.0, 0.1, is_working.sum()
            )

            # Outside working hours, a residual load remains
            # (servers, security systems, standby equipment).
            base_curve[~is_working] = 0.1

        # --------------------------------------------------------------
        # CASE 3: SOLAR PRODUCTION PROFILE
        # --------------------------------------------------------------
        elif profile_type == "SOLAR_PPA":
            # Solar production is driven by daylight availability.
            # No attempt is made here to model weather or seasonality.
            hour = self.dates.hour
            daylight = (hour >= 6) & (hour <= 20)

            # A sine function is used to approximate the typical
            # bell-shaped daily solar production profile.
            # It starts at zero at sunrise, peaks at midday
            # and returns to zero at sunset.
            base_curve[daylight] = np.sin(
                (hour[daylight] - 6) * np.pi / 14
            )

            # Negative values are clipped to zero
            # since production cannot be negative.
            base_curve = np.maximum(base_curve, 0)

        # --------------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------------
        # At this stage, base_curve only contains relative values.
        # The final step is to scale it so that total energy
        # matches the requested annual volume.
        total_units = base_curve.sum()

        if total_units > 0:
            normalized_curve = (base_curve / total_units) * annual_volume_mwh
        else:
            # Edge case protection: return a zero curve if needed
            normalized_curve = base_curve

        # The returned series is a fully specified hourly energy profile
        return pd.Series(
            normalized_curve,
            index=self.dates,
            name="Synthetic Load",
        )
        
        
        
        
