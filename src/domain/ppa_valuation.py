from dataclasses import dataclass
from typing import Optional

@dataclass
class PPAResult:
    fair_price: float       # Prix PPA suggéré (€/MWh)
    capture_rate: float     # % de prix de marché capturé par la technologie
    cannibalization_impact: float # Perte de valeur due à la corrélation

def price_renewable_ppa(technology: str, baseload_price: float) -> PPAResult:
    """
    Estime le prix fixe équitable pour un PPA en mode "Pay-As-Produced".
    
    Logique : 
    Les renouvelables (solaire/éolien) souffrent de 'Cannibalisation'. 
    Quand il y a beaucoup de soleil/vent, la production est forte mais les prix Spot chutent.
    Le prix PPA doit donc être décoté par rapport au prix Baseload du marché.
    """
    
    # Taux de Capture (Estimations standards du marché belge/européen)
    capture_map = {
        "SOLAR": 0.88,        # Solaire : produit à midi quand les prix sont souvent plus bas
        "ONSHORE_WIND": 0.92, # Éolien terrestre : profil plus réparti mais corrélé
        "OFFSHORE_WIND": 0.96 # Éolien offshore : profil très stable et puissant
    }
    
    # Récupération du taux, défaut à 90% si technologie inconnue
    capture_rate = capture_map.get(technology.upper(), 0.90)
    
    # Prix PPA = (Baseload * Taux de Capture) - Buffer de Risque
    # Le risk_buffer couvre les coûts d'équilibrage (Imbalance costs)
    risk_buffer = 1.5 # €/MWh 
    
    fair_price = (baseload_price * capture_rate) - risk_buffer
    
    # Impact de Cannibalisation : Valeur perdue par rapport à un profil plat (Baseload)
    cannibalization = baseload_price * (1.0 - capture_rate)
    
    return PPAResult(
        fair_price=round(fair_price, 2),
        capture_rate=round(capture_rate * 100, 1),
        cannibalization_impact=round(cannibalization, 2)
    )