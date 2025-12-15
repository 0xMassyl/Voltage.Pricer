import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, cast


class MLPriceForecaster:
    """
    Electricity price forecasting engine based on XGBoost.
    Includes basic train/test diagnostics to detect overfitting
    and underfitting on a chronological split.
    """

    def __init__(self, spot_reference: float = 95.5):
        """
        spot_reference:
        External price anchor used to re-calibrate the forecasted level.
        """
        self.spot_reference = spot_reference

        # XGBoost configuration:
        # max_depth=7 is a trade-off between flexibility and generalization
        # Deeper trees -> higher overfitting risk
        # Shallower trees -> underfitting risk
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            n_jobs=-1,
            objective="reg:squarederror",
        )

        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _create_features(self, dates: pd.Index) -> pd.DataFrame:
        """
        Transforms timestamps into numerical calendar features.
        The model is purely time-structure driven (no fundamentals).
        """
        # Force DatetimeIndex to avoid typing / runtime issues
        dates = pd.to_datetime(dates)

        df = pd.DataFrame(index=dates)

        # Basic calendar features
        df["hour"] = dates.hour
        df["dayofweek"] = dates.dayofweek
        df["month"] = dates.month
        df["year"] = dates.year
        df["dayofyear"] = dates.dayofyear

        # Binary regime indicators
        df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
        df["is_peak"] = (
            (dates.hour >= 8) & (dates.hour < 20) & (dates.dayofweek < 5)
        ).astype(int)

        return df

    # ------------------------------------------------------------------
    # Synthetic market history generation
    # ------------------------------------------------------------------
    def _generate_synthetic_history(self) -> pd.Series:
        """
        Generates a large synthetic hourly spot price history (2000–2025).
        Purpose: provide enough data to train a stable ML model
        without relying on real historical prices.
        """
        print("Generating synthetic market history (2000–2025)...")

        dates = pd.date_range(
            start="2000-01-01",
            end="2025-12-31 23:00",
            freq="h",
        )
        n = len(dates)

        # 1. Long-term upward trend (inflation / structural increase)
        trend = np.linspace(20, 90, n)

        # 2. Yearly seasonality (winter price premium)
        yearly = 15 * -np.cos((dates.dayofyear / 365) * 2 * np.pi)

        # 3. Weekly seasonality (weekend demand drop)
        weekly = np.where(dates.dayofweek >= 5, -10, 5)

        # 4. Daily intraday profile (duck curve style)
        hour = dates.hour
        daily = (
            12 * np.sin((hour - 6) * np.pi / 12)
            + 8 * np.sin((hour - 18) * np.pi / 6)
        )

        # 5. Random noise to simulate volatility
        noise = np.random.normal(0, 5, n)

        prices = trend + yearly + weekly + daily + noise

        # Enforce a hard floor to avoid unrealistic negative prices
        prices = np.maximum(prices, 5.0)

        return pd.Series(prices, index=dates, name="Historical Spot")

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def train(self):
        """
        Trains the model using a chronological train/test split.
        No shuffling: time structure must be preserved.
        """
        history = self._generate_synthetic_history()

        # Chronological split:
        # First 90% -> training
        # Last 10%  -> validation (recent years)
        split_idx = int(len(history) * 0.9)

        y_all = history.values
        X_all = self._create_features(history.index)

        X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        print(
            f"Training on {len(X_train)} points, "
            f"validating on {len(X_test)} points..."
        )

        # Train only on the training window
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        self.is_trained = True

        # Compute RMSE on train and test sets
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        # Explicit casting for static type checkers
        y_train_arr = cast(np.ndarray, y_train)
        y_test_arr = cast(np.ndarray, y_test)

        rmse_train = np.sqrt(mean_squared_error(y_train_arr, train_preds))
        rmse_test = np.sqrt(mean_squared_error(y_test_arr, test_preds))

        self.metrics = {
            "RMSE_Train": round(float(rmse_train), 2),
            "RMSE_Test": round(float(rmse_test), 2),
            # Ratio used as a simple overfitting indicator
            "Overfitting_Ratio": round(
                float(rmse_test) / (float(rmse_train) + 1e-3), 2
            ),
        }

        print(
            f"RMSE Train: {rmse_train:.2f} € | "
            f"RMSE Test: {rmse_test:.2f} €"
        )

        # Simple automated diagnostics
        if rmse_test > rmse_train * 1.5:
            print("Warning: potential overfitting detected.")
        elif rmse_train > 20.0:
            print("Warning: potential underfitting detected.")
        else:
            print("Model generalization looks healthy.")

    # ------------------------------------------------------------------
    # Forecast generation
    # ------------------------------------------------------------------
    def generate_forecast_curve(self, target_year: int = 2026) -> pd.Series:
        """
        Generates an hourly price forward curve (HPFC) for a given year.
        """
        if not self.is_trained:
            self.train()

        print(f"Forecasting hourly curve for {target_year}...")

        # 1. Build future hourly timeline
        future_dates = pd.date_range(
            start=f"{target_year}-01-01",
            end=f"{target_year}-12-31 23:00",
            freq="h",
        )

        # 2. Create features for future timestamps
        X_future = self._create_features(future_dates)

        # 3. Raw ML prediction
        predicted_prices = self.model.predict(X_future)

        # 4. Level calibration to external spot reference
        # This avoids relying on the absolute ML level
        current_mean = np.mean(predicted_prices)
        adjustment = self.spot_reference - current_mean

        final_curve = predicted_prices + adjustment
        final_curve = np.maximum(final_curve, 0.0)

        return pd.Series(
            final_curve,
            index=future_dates,
            name=f"HPFC {target_year} (ML)",
        )

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """
        Returns model performance metrics after training.
        """
        return self.metrics
