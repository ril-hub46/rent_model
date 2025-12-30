from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.model import default_regime, load_project_inputs_from_excel, run_model


def sweep_price_from_excel(
    excel_path: str,
    prices: Iterable[float],
    regime_code: str = "CM2003",
    mine_sheet: Optional[str] = None,
    amort_sheet: str = "Amortissement",
    discount_override: Optional[float] = None,
    royalty_rate_override: Optional[float] = None,
    cit_rate_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Runs the model for a grid of gold prices and returns indicators per scenario.
    """
    reg = default_regime(regime_code)
    inputs = load_project_inputs_from_excel(
        excel_path, regime=regime_code, mine_sheet=mine_sheet, amort_sheet=amort_sheet
    )

    rows = []
    for p in prices:
        _, ind = run_model(
            inputs,
            reg,
            gold_price=float(p),
            discount_rate=discount_override,
            royalty_rate_override=royalty_rate_override,
            cit_rate_override=cit_rate_override,
        )
        rows.append(ind)

    df = pd.DataFrame(rows)
    cols = [
        "regime",
        "gold_price",
        "discount_rate",
        "royalty_rate",
        "cit_rate",
        "NPV_pre_tax",
        "NPV_post_tax",
        "Gov_NPV",
        "TEMI",
    ]
    return df[cols]
