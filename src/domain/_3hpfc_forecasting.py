import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, cast
from datetime import datetime
import sys
import os

# Gestion des imports pour le mode script vs module
try:
    from src.ingestion._1API_connector import EliaDataConnector
except ImportError:
    # Fallback pour exécution directe du fichier
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.ingestion._1API_connector import EliaDataConnector

class MLPriceForecaster:
    """
    Moteur de prévision Hybride (Voltage Pricer).
    
    Stratégie d'entraînement :
    1. 2000-2014 : Données Synthétiques (Apprentissage de la saisonnalité théorique pure).
    2. 2015-Aujourd'hui : Données Réelles ELIA (Apprentissage de la volatilité et des chocs récents).
    """

    def __init__(self, spot_reference: float = 95.5):
        self.spot_reference = spot_reference
        # Configuration XGBoost "Industrial Grade"
        self.model = XGBRegressor(
            n_estimators=300,      # Assez d'arbres pour capturer la complexité hybride
            learning_rate=0.03,    # Apprentissage lent pour la stabilité
            max_depth=8,           # Profondeur suffisante pour les interactions non-linéaires
            n_jobs=-1,             # Parallélisation
            objective='reg:squarederror'
        )
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        # Initialisation du connecteur pour les données réelles
        self.elia_client = EliaDataConnector()

    def _create_features(self, dates: pd.Index) -> pd.DataFrame:
        """
        Feature Engineering: Transforme les dates en indicateurs numériques.
        """
        # FIX TYPAGE : Conversion explicite en DatetimeIndex pour satisfaire Pylance
        dt_idx = pd.to_datetime(dates)
        
        df = pd.DataFrame(index=dt_idx)
        
        # Features Cycliques
        df['hour'] = dt_idx.hour
        df['dayofweek'] = dt_idx.dayofweek
        df['month'] = dt_idx.month
        df['year'] = dt_idx.year
        df['dayofyear'] = dt_idx.dayofyear
        
        # Features Booléennes (Logique de marché)
        df['is_weekend'] = (dt_idx.dayofweek >= 5).astype(int)
        # Heures de Pointe (08h-20h en semaine)
        df['is_peak'] = ((dt_idx.hour >= 8) & (dt_idx.hour < 20) & (dt_idx.dayofweek < 5)).astype(int)
        
        return df

    def _get_hybrid_history(self) -> pd.Series:
        """
        Construit le dataset d'entraînement ultime en fusionnant théorie et réalité.
        """
        print("🏗️ Construction du Dataset Hybride (Synthétique + Réel)...")
        
        # --- PHASE 1 : HISTORIQUE SYNTHÉTIQUE (2000-2014) ---
        print("   🔹 Génération données synthétiques (2000-2014)...")
        dates_syn = pd.date_range(start="2000-01-01", end="2014-12-31 23:00", freq="h")
        n_syn = len(dates_syn)
        
        # Modèle théorique stable pour apprendre les bases
        trend = np.linspace(20, 50, n_syn)
        yearly = 15 * -np.cos((dates_syn.dayofyear / 365) * 2 * np.pi)
        weekly = np.where(dates_syn.dayofweek >= 5, -10, 5)
        daily = 12 * np.sin((dates_syn.hour - 6) * np.pi / 12) + 8 * np.sin((dates_syn.hour - 18) * np.pi / 6)
        noise = np.random.normal(0, 3, n_syn)
        
        prices_syn = trend + yearly + weekly + daily + noise
        series_syn = pd.Series(np.maximum(prices_syn, 5.0), index=dates_syn, name="Spot")

        # --- PHASE 2 : HISTORIQUE RÉEL (2015-Maintenant) ---
        print("   🔹 Téléchargement données réelles ELIA (2015-Now)...")
        
        # Calcul du nombre de jours à récupérer
        days_since_2015 = (datetime.now() - datetime(2015, 1, 1)).days + 10
        
        # Appel à l'API ELIA (via le fichier elia_client.py)
        series_real = self.elia_client.fetch_real_spot_prices(days=days_since_2015)
        
        if series_real.empty:
            print("   ⚠️ Échec téléchargement réel ou données vides. Fallback sur 100% synthétique.")
            # Fallback : on étend le synthétique jusqu'à aujourd'hui pour ne pas crasher
            dates_fallback = pd.date_range(start="2015-01-01", end=datetime.now(), freq="h")
            series_real = pd.Series(50 + np.random.normal(0, 10, len(dates_fallback)), index=dates_fallback)

        # --- FUSION ET NETTOYAGE ---
        # On supprime les infos de fuseau horaire pour éviter les erreurs de concaténation
        series_syn.index = pd.to_datetime(series_syn.index).tz_localize(None)
        series_real.index = pd.to_datetime(series_real.index).tz_localize(None) 
        
        # Concaténation chronologique
        full_history = pd.concat([series_syn, series_real]).sort_index()
        
        # Suppression des doublons potentiels à la jonction
        full_history = full_history[~full_history.index.duplicated(keep='last')]
        
        # Interpolation finale pour boucher les trous éventuels de l'API
        full_history = full_history.interpolate(method='linear').ffill().bfill()
        
        print(f"✅ Dataset Hybride prêt : {len(full_history)} heures de trading.")
        return full_history

    def train(self):
        """
        Entraîne le modèle XGBoost sur l'historique hybride avec validation.
        """
        # Récupération des données
        history = self._get_hybrid_history()
        
        # Split Train (80%) / Test (20%)
        split_idx = int(len(history) * 0.8)
        
        y_all = history.values
        X_all = self._create_features(history.index)
        
        X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]
        
        print(f"🤖 [ML Engine] Training XGBoost (Train: {len(X_train)} | Test: {len(X_test)})...")
        
        # Entraînement
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        self.is_trained = True
        
        # Calcul des Métriques
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        y_train_arr = cast(np.ndarray, y_train)
        y_test_arr = cast(np.ndarray, y_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train_arr, train_preds))
        rmse_test = np.sqrt(mean_squared_error(y_test_arr, test_preds))
        
        self.metrics = {
            "RMSE_Train": round(float(rmse_train), 2),
            "RMSE_Test": round(float(rmse_test), 2),
            "Overfitting_Ratio": round(float(rmse_test) / (float(rmse_train) + 0.001), 2)
        }
        
        print(f"✅ [ML Stats] RMSE Train: {rmse_train:.2f}€ | RMSE Test: {rmse_test:.2f}€")

    def generate_forecast_curve(self, target_year: int = 2026) -> pd.Series:
        """
        Génère la courbe HPFC pour l'année cible.
        """
        if not self.is_trained:
            self.train()
            
        print(f"🔮 Forecasting Hourly Curve for {target_year}...")
        
        future_dates = pd.date_range(start=f"{target_year}-01-01", end=f"{target_year}-12-31 23:00", freq="h")
        X_future = self._create_features(future_dates)
        
        predicted_prices = self.model.predict(X_future)
        
        # Calibration (Level Shift)
        current_mean = np.mean(predicted_prices)
        adjustment = self.spot_reference - current_mean
        final_curve = predicted_prices + adjustment
        
        final_curve = np.maximum(final_curve, 0.0)
        
        return pd.Series(final_curve, index=future_dates, name=f"HPFC {target_year} (ML)")
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

# --- ZONE DE TEST INTÉGRÉE ---
if __name__ == "__main__":
    print("\n🔍 MODE TEST : VÉRIFICATION DU MOTEUR ML HYBRIDE")
    print("="*50)
    
    # Instanciation
    forecaster = MLPriceForecaster(spot_reference=100.0)
    
    # 1. Test du Dataset Hybride (Est-ce que j'ai du vrai et du faux ?)
    # On accède à la méthode privée pour le debug
    history = forecaster._get_hybrid_history()
    real_start = pd.Timestamp("2015-01-01")
    history_idx = pd.to_datetime(history.index)
    real_count = len(history[history_idx >= real_start])
    
    print(f"📊 Points réels (post-2015) trouvés : {real_count}")
    if real_count > 1000:
        print("✅ SUCCESS: L'API ELIA a bien fourni les données réelles.")
    else:
        print("⚠️ WARNING: Peu de données réelles, vérifiez la connexion API.")

    # 2. Test Entraînement
    print("\n🏃 Lancement de l'entraînement...")
    forecaster.train()
    
    # 3. Test Prévision
    print("\n🔮 Test de prévision 2026...")
    curve = forecaster.generate_forecast_curve(2026)
    print(f"✅ Courbe générée : {len(curve)} points. Moyenne : {curve.mean():.2f} €/MWh")
    print("="*50)