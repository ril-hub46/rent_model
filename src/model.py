from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _is_year(x) -> bool:
    """True if x looks like a year (e.g. 2011, 2011.0, '2011')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    try:
        xi = int(float(str(x).strip()))
        return 1900 <= xi <= 2200
    except Exception:
        return False


def _excel_like_npv(rate: float, cashflows: np.ndarray) -> float:
    """
    Replicates Excel NPV(): discounts cashflows starting at period 1.
    cashflows[t] discounted by (1+rate)^(t+1).
    """
    cf = np.asarray(cashflows, dtype=float)
    if cf.size == 0:
        return 0.0
    disc = np.power(1.0 + float(rate), np.arange(1, cf.size + 1))
    return float(np.nansum(cf / disc))


def _read_sheet_year_matrix(
    excel_path: str,
    sheet_name: str,
    label_col: int = 0,
    unit_col: int = 1,
    data_start_col: int = 2,
    header_row: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Reads a sheet shaped like:
      | label | unit | 2011 | 2012 | ... |
    Robust to non-year value such as "Unités" in col=1.

    Returns:
      - DataFrame columns: ["label","unit", <year columns>]
      - years as int array
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    header = list(raw.iloc[header_row, :].values)
    year_cols: List[int] = []
    years: List[int] = []

    for j in range(data_start_col, raw.shape[1]):
        v = header[j] if j < len(header) else None
        if _is_year(v):
            year_cols.append(j)
            years.append(int(float(v)))

    if not years:
        raise ValueError(
            f"Could not detect year columns on sheet '{sheet_name}'. "
            f"Expected years on row {header_row} starting at column {data_start_col}."
        )

    out = pd.DataFrame({
        "label": raw.iloc[header_row + 1 :, label_col].astype(object).values,
        "unit": raw.iloc[header_row + 1 :, unit_col].astype(object).values,
    })

    for idx, y in zip(year_cols, years):
        out[y] = pd.to_numeric(raw.iloc[header_row + 1 :, idx], errors="coerce").values

    out = out[~out["label"].isna()].copy()
    return out, np.asarray(years, dtype=int)


def _pick_row(series_df: pd.DataFrame, pattern: str) -> Optional[pd.Series]:
    """Pick first row where label matches regex pattern (case-insensitive)."""
    lab = series_df["label"].astype(str)
    m = lab.str.contains(pattern, case=False, na=False, regex=True)
    if not bool(m.any()):
        return None
    row = series_df.loc[m].iloc[0]
    year_cols = [c for c in series_df.columns if isinstance(c, int)]
    return row[year_cols]


def _read_scalar_from_sheet(df: pd.DataFrame, label_regex: str, fallback: float = np.nan) -> float:
    """
    Reads scalar value stored in unit column for a given label.
    Useful for discount rate / exchange rate / base gold price when stored as a constant.
    """
    lab = df["label"].astype(str)
    m = lab.str.contains(label_regex, case=False, na=False, regex=True)
    if not bool(m.any()):
        return float(fallback)
    v = df.loc[m, "unit"].iloc[0]
    try:
        return float(v)
    except Exception:
        return float(fallback)


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class ProjectInputs:
    years: np.ndarray
    revenue: np.ndarray          # USD
    opex: np.ndarray             # USD
    capex: np.ndarray            # USD
    depreciation: np.ndarray     # USD
    base_gold_price: float       # USD/oz in the sheet
    discount_rate: float         


@dataclass(frozen=True)
class FiscalRegime:
    name: str
    cit_rate: float

    # Royalty
    royalty_mode: str = "progressive"  # "flat" or "progressive"
    royalty_rate_flat: float = 0.03
    royalty_bands: Tuple[Tuple[float, float], ...] = (
        (1000.0, 0.03),
        (1300.0, 0.04),
        (float("inf"), 0.05),
    )

    # Other levies
    local_development_levy: float = 0.0   # e.g. 1% turnover
    imf_rate: float = 0.0                 # e.g. 0.5% turnover when loss-making
    loss_carry_forward: bool = True


def default_regime(regime_code: str) -> FiscalRegime:
    rc = regime_code.strip().upper()
    if rc in {"CM2003", "2003"}:
        return FiscalRegime(
            name="CM2003",
            cit_rate=0.175,
            royalty_mode="progressive",
            local_development_levy=0.0,
            imf_rate=0.0,
        )
    if rc in {"CM2015", "2015"}:
        return FiscalRegime(
            name="CM2015",
            cit_rate=0.275,
            royalty_mode="progressive",
            local_development_levy=0.01,
            imf_rate=0.005,
        )
    raise ValueError(f"Unknown regime_code='{regime_code}'. Use CM2003 or CM2015.")


# ----------------------------
# Excel loader
# ----------------------------

def load_project_inputs_from_excel(
    excel_path: str,
    regime: str = "CM2003",
    mine_sheet: Optional[str] = None,
    amort_sheet: str = "Amortissement",
) -> ProjectInputs:
    """
    Default sheets:
      - CM2003 -> 'Données de la mine'
      - CM2015 -> 'Code Minier 2015'
    """
    rc = regime.strip().upper()
    if mine_sheet is None:
        mine_sheet = "Données de la mine" if rc in {"CM2003", "2003"} else "Code Minier 2015"

    mine_df, years = _read_sheet_year_matrix(excel_path, mine_sheet)

    revenue = _pick_row(mine_df, r"Chiffre\s*d.?Affaire")
    opex = _pick_row(mine_df, r"Coûts\s*d['’]exploitation\s*\(Opex\)")
    capex = _pick_row(mine_df, r"TOTAL\s+CAPITAL")

    if revenue is None:
        raise ValueError(f"Revenue row not found on '{mine_sheet}' (expected 'Chiffre d\\'Affaire').")

    if opex is None:
        opex = _pick_row(mine_df, r"Opex|exploitation")

    if capex is None:
        capex = _pick_row(mine_df, r"Capital|CAPEX|Investissement")

    base_gold_price = _read_scalar_from_sheet(mine_df, r"Cours\s+de\s+l['’]or", fallback=np.nan)
    discount_rate = _read_scalar_from_sheet(mine_df, r"Taux\s+d['’]actualisation", fallback=np.nan)

    amort_df, amort_years = _read_sheet_year_matrix(excel_path, amort_sheet)

    dep = _pick_row(amort_df, r"Total\s+des\s+charges\s+d['’]amortissements")
    if dep is None:
        dep1 = _pick_row(amort_df, r"Amortissements\s+des\s+constructions")
        dep2 = _pick_row(amort_df, r"Amortissements\s+des\s+biens")
        if dep1 is None and dep2 is None:
            raise ValueError(f"Depreciation rows not found on '{amort_sheet}'.")
        dep = (dep1.fillna(0) if dep1 is not None else 0) + (dep2.fillna(0) if dep2 is not None else 0)

    common_years = np.intersect1d(years, amort_years)
    if common_years.size == 0:
        raise ValueError("No overlapping years between mine sheet and amortization sheet.")

    def _align(s: pd.Series) -> np.ndarray:
        s2 = s.reindex(common_years).astype(float)
        return np.nan_to_num(s2.values, nan=0.0)

    revenue_a = _align(revenue)
    opex_a = _align(opex) if opex is not None else np.zeros_like(revenue_a)
    capex_a = _align(capex) if capex is not None else np.zeros_like(revenue_a)
    dep_a = _align(dep)

    if not np.isfinite(base_gold_price):
        price_row = _pick_row(mine_df, r"Cours\s+de\s+l['’]or")
        if price_row is not None:
            base_gold_price = float(pd.to_numeric(price_row.dropna().iloc[0], errors="coerce"))
        else:
            base_gold_price = 1300.0

    if not np.isfinite(discount_rate):
        discount_rate = 0.10

    return ProjectInputs(
        years=common_years.astype(int),
        revenue=revenue_a,
        opex=opex_a,
        capex=capex_a,
        depreciation=dep_a,
        base_gold_price=float(base_gold_price),
        discount_rate=float(discount_rate),
    )


# ----------------------------
# Fiscal calculations
# ----------------------------

def royalty_rate_for_price(regime: FiscalRegime, gold_price: float) -> float:
    if regime.royalty_mode == "flat":
        return float(regime.royalty_rate_flat)

    p = float(gold_price)
    for threshold, rate in regime.royalty_bands:
        if p < float(threshold):
            return float(rate)
    return float(regime.royalty_bands[-1][1])


def run_model(
    inputs: ProjectInputs,
    regime: FiscalRegime,
    gold_price: Optional[float] = None,
    discount_rate: Optional[float] = None,
    royalty_rate_override: Optional[float] = None,
    cit_rate_override: Optional[float] = None,
):
    """
    Returns:
      annual_table (DataFrame)
      indicators (dict)
    """
    years = inputs.years
    n = years.size

    price = float(gold_price) if gold_price is not None else float(inputs.base_gold_price)
    scale = price / float(inputs.base_gold_price) if inputs.base_gold_price else 1.0

    revenue = np.asarray(inputs.revenue, dtype=float) * scale
    opex = np.asarray(inputs.opex, dtype=float)
    capex = np.asarray(inputs.capex, dtype=float)
    dep = np.asarray(inputs.depreciation, dtype=float)

    dr = float(discount_rate) if discount_rate is not None else float(inputs.discount_rate)

    rr = float(royalty_rate_override) if royalty_rate_override is not None else royalty_rate_for_price(regime, price)
    cit_rate = float(cit_rate_override) if cit_rate_override is not None else float(regime.cit_rate)

    royalty = revenue * rr
    local_dev = revenue * float(regime.local_development_levy)

    taxable_raw = revenue - opex - dep - royalty - local_dev

    taxable = np.zeros(n, dtype=float)
    loss_cf = 0.0
    for t in range(n):
        ti = taxable_raw[t] - loss_cf
        if ti >= 0:
            taxable[t] = ti
            loss_cf = 0.0
        else:
            taxable[t] = 0.0
            loss_cf = -ti if regime.loss_carry_forward else 0.0

    cit = taxable * cit_rate

    imf = np.zeros(n, dtype=float)
    if regime.imf_rate and regime.imf_rate > 0:
        imf = revenue * float(regime.imf_rate)
        imf = np.where(cit > 0, 0.0, imf)

    gov_revenue = royalty + local_dev + cit + imf

    cf_pre_tax = revenue - opex - capex
    cf_post_tax = cf_pre_tax - gov_revenue

    npv_pre = _excel_like_npv(dr, cf_pre_tax)
    npv_post = _excel_like_npv(dr, cf_post_tax)
    gov_npv = _excel_like_npv(dr, gov_revenue)

    aetr = float(gov_npv / npv_pre) if npv_pre > 0 else float("nan")
    temi = float(1.0 - (npv_post / npv_pre)) if npv_pre > 0 else float("nan")

    tbl = pd.DataFrame({
        "Year": years,
        "Revenue_CA": revenue,
        "OPEX": opex,
        "CAPEX": capex,
        "Depreciation": dep,
        "Royalty": royalty,
        "Local_dev_levy": local_dev,
        "Taxable_income": taxable,
        "CIT": cit,
        "IMF": imf,
        "CF_pre_tax": cf_pre_tax,
        "CF_post_tax": cf_post_tax,
        "Gov_revenue": gov_revenue,
    })

    ind = {
        "regime": regime.name,
        "gold_price": price,
        "discount_rate": dr,
        "royalty_rate": rr,
        "cit_rate": cit_rate,
        "NPV_pre_tax": npv_pre,
        "NPV_post_tax": npv_post,
        "Gov_NPV": gov_npv,
        "AETR": aetr,
        "TEMI": temi,
    }
    return tbl, ind
