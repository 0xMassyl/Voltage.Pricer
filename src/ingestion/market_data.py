import pandas as pd
from typing import Dict

class MarketDataManager:
    """
    Manages Market Price Data (Forward Curves).
    Simule les prix futures EEX pour le modèle de pricing B2B.
    Dans un environnement réel, cette classe se connecterait à une API ETRM ou Bloomberg.
    """

    def get_forward_prices(self) -> Dict[str, float]:
        """
        Retourne la vue actuelle du marché pour l'année calendrier prochaine (CAL 2026).
        Les prix sont en EUR/MWh.
        """
        # Simulated EEX Belgian Power Futures
        return {
            "CAL_BASE": 95.50,  # Prix du Ruban (Baseload 24/7)
            "CAL_PEAK": 112.75, # Prix de la Pointe (Peakload 8h-20h Lun-Ven)
            "SPOT_VOLATILITY": 0.25 # Volatilité annualisée des prix Spot (pour le calcul du risque)
        }