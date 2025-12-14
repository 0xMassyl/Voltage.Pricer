import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from typing import Optional

class MLPriceForecaster:
    """
    Moteur de prévision de prix basé sur XGBoost.
    Entraîné sur 25 ans d'historique (simulé) pour capter la saisonnalité long terme et intraday.
    """

    def __init__(self, spot_reference: float = 95.5):
        self.spot_reference = spot_reference
        # Configuration XGBoost
        self.model = XGBRegressor(
            n_estimators=200,     
            learning_rate=0.05,   
            max_depth=7,          
            n_jobs=-1,            
            objective='reg:squarederror'
        )
        self.is_trained = False

    def _create_features(self, dates: pd.Index) -> pd.DataFrame:
        """
        Feature Engineering: Transforms timestamps into numerical features.
        
        Note: Utilise pd.Index comme annotation pour satisfaire Pylance, 
        mais assure que c'est un DatetimeIndex à l'intérieur.
        """
        # FIX DU TYPAGE : On force l'objet à être un DatetimeIndex pour éviter l'erreur Pylance
        dates = pd.to_datetime(dates)
        
        df = pd.DataFrame(index=dates)
        
        # 1. Cyclical Features
        df['hour'] = dates.hour
        df['dayofweek'] = dates.dayofweek
        df['month'] = dates.month
        df['year'] = dates.year
        df['dayofyear'] = dates.dayofyear
        
        # 2. Boolean Logic
        df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
        df['is_peak'] = ((dates.hour >= 8) & (dates.hour < 20) & (dates.dayofweek < 5)).astype(int)
        
        return df

    def _generate_synthetic_history(self) -> pd.Series:
        """
        Creates a MASSIVE dataset (2000-2025) to train the model.
        ~219,000 hourly data points.
        """
        print(" Génération de 25 ans de données marché (2000-2025)...")
        dates = pd.date_range(start="2000-01-01", end="2025-12-31 23:00", freq="h")
        n = len(dates)
        
        # 1. Long Term Trend (Inflation)
        trend = np.linspace(20, 90, n)
        
        # 2. Yearly Seasonality (Winter Peak)
        yearly = 15 * -np.cos((dates.dayofyear / 365) * 2 * np.pi)
        
        # 3. Weekly Seasonality (Weekend drop)
        weekly = np.where(dates.dayofweek >= 5, -10, 5)
        
        # 4. Daily Profile (Duck Curve)
        hour = dates.hour
        daily = 12 * np.sin((dates.hour - 6) * np.pi / 12) + 8 * np.sin((dates.hour - 18) * np.pi / 6)
        
        # 5. Volatility / Noise
        noise = np.random.normal(0, 5, n)
        
        prices = trend + yearly + weekly + daily + noise
        prices = np.maximum(prices, 5.0)
        
        return pd.Series(prices, index=dates, name="Historical Spot")

    def train(self):
        """Entraîne le modèle XGBoost."""
        history = self._generate_synthetic_history()
        print(f" [ML Engine] Entraînement XGBoost sur {len(history)} points de données...")
        
        X = self._create_features(history.index)
        y = history.values
        
        self.model.fit(X, y)
        self.is_trained = True
        print(f" [ML Engine] Entraînement terminé.")

    def generate_forecast_curve(self, target_year: int = 2026) -> pd.Series:
        """Prédit la HPFC pour l'année cible."""
        if not self.is_trained:
            self.train()
            
        print(f"Forecasting Hourly Curve for {target_year}...")
        
        future_dates = pd.date_range(start=f"{target_year}-01-01", end=f"{target_year}-12-31 23:00", freq="h")
        X_future = self._create_features(future_dates)
        
        predicted_prices = self.model.predict(X_future)
        
        # Calibration / Ajustement au prix Forward actuel (spot_reference)
        current_mean = np.mean(predicted_prices)
        adjustment = self.spot_reference - current_mean
        final_curve = predicted_prices + adjustment
        final_curve = np.maximum(final_curve, 0.0)
        
        return pd.Series(final_curve, index=future_dates, name=f"HPFC {target_year} (ML)")