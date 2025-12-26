# src/regimes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


def royalty_rate_from_gold_price(price: float) -> float:
    """
    Barème simple inspiré de ta feuille "Données fiscales":
    - <= 1000 : 3%
    - (1000, 1500] : 4%
    - > 1500 : 5%

    Tu pourras ajuster les seuils ensuite (si tu veux coller exactement à la loi).
    """
    if price <= 1000:
        return 0.03
    if price <= 1500:
        return 0.04
    return 0.05


@dataclass(frozen=True)
class FiscalRegime:
    name: Literal["CM2003", "CM2015"]
    cit_rate_default: float                 # IS
    state_participation: float              # % participation Etat (dividendes)
    dividend_wht_rate: float                # retenue sur dividendes (IRCM)
    imf_rate: float                         # minimum tax (ex: 0.5% du CA)
    local_dev_rate: float                   # CM2015: fonds dev local (ex: 1% du CA)
    royalty_rate_func: Callable[[float], float] = royalty_rate_from_gold_price


CM2003 = FiscalRegime(
    name="CM2003",
    cit_rate_default=0.35,
    state_participation=0.10,
    dividend_wht_rate=0.0625,
    imf_rate=0.005,
    local_dev_rate=0.0,
)

CM2015 = FiscalRegime(
    name="CM2015",
    cit_rate_default=0.275,
    state_participation=0.10,
    dividend_wht_rate=0.0625,
    imf_rate=0.005,
    local_dev_rate=0.01,   # (ta feuille fiscale met 1%)
)
