import sys
import os
import pandas as pd
import numpy as np

# Ajout du dossier racine au chemin de recherche pour permettre les imports depuis 'src'
# Utile si vous lancez le script depuis la racine du projet 'VOLTAGE PRICER'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

try:
    # Import du moteur de prévision hybride
    from src.domain.ml_forecasting import MLPriceForecaster
    print("✅ Importation réussie : MLPriceForecaster est prêt.")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("Conseil : Assurez-vous d'être à la racine du projet et que le dossier 'src' est présent.")
    sys.exit(1)

def run_diagnostic():
    """
    Exécute un diagnostic complet du moteur ML pour vérifier le 'catch' des données réelles.
    """
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DU DIAGNOSTIC HYBRIDE (VOLTAGE PRICER)")
    print("="*60)
    
    # Initialisation du moteur avec un prix de référence marché
    forecaster = MLPriceForecaster(spot_reference=95.5)
    
    try:
        # 1. Analyse de l'historique d'entraînement
        print("\n1. Analyse de la source de données d'entraînement...")
        # On appelle la méthode protégée pour inspecter le dataset avant l'entraînement
        history = forecaster._get_hybrid_history()
        
        total_hours = len(history)
        # Le point de bascule entre synthétique et réel est fixé au 1er Janvier 2015
        real_data_start = pd.Timestamp("2015-01-01")
        
        # On s'assure que l'index est au format datetime pour la comparaison
        history_index = pd.to_datetime(history.index)
        
        # Séparation des points pour vérification statistique
        real_points = history[history_index >= real_data_start]
        synthetic_points = history[history_index < real_data_start]
        
        print(f"   📊 Points totaux chargés : {total_hours} heures")
        print(f"   🧬 Données Synthétiques (2000-2014) : {len(synthetic_points)} points")
        print(f"   📡 Données Réelles ELIA (2015-Now)  : {len(real_points)} points")
        
        if len(real_points) > 0:
            avg_real = real_points.mean()
            print(f"   ✅ SUCCESS : Données réelles ELIA captées (Moyenne réelle : {avg_real:.2f} €/MWh)")
        else:
            print("   ⚠️ WARNING : Aucune donnée réelle détectée. Le moteur est en mode FALLBACK complet.")

        # 2. Lancement de l'entraînement
        print("\n2. Entraînement du modèle XGBoost sur le dataset hybride...")
        forecaster.train()
        
        # 3. Récupération des métriques (RMSE)
        metrics = forecaster.get_metrics()
        print("\n3. Métriques de performance du modèle :")
        for label, value in metrics.items():
            print(f"   📈 {label}: {value}")
            
        # 4. Test de génération de courbe
        target_year = 2026
        print(f"\n4. Test de génération d'une courbe HPFC pour {target_year}...")
        curve = forecaster.generate_forecast_curve(target_year=target_year)
        
        print("\n" + "="*60)
        print("✅ DIAGNOSTIC TERMINÉ AVEC SUCCÈS")
        print(f"   - Nombre d'heures prévues : {len(curve)} points")
        print(f"   - Prix moyen Cal-{target_year} : {curve.mean():.2f} €/MWh")
        print(f"   - Écart-type (Volatilité) : {curve.std():.2f} €")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE DURANT LE DIAGNOSTIC : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()