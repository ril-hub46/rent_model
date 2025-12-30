# src/results_store.py
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class ScenarioMeta:
    run_id: str
    created_at: str

    excel_path: str
    mine_sheet: str | None
    amort_sheet: str | None

    regime_key: str
    gold_price: float
    royalty_rate: float
    cit_rate: float
    discount_rate: float


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _hash_payload(payload: Dict[str, Any]) -> str:
    # Hash stable (sorted keys) -> run_id court
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def ensure_results_dir(root: Path) -> Path:
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    return out


def _index_path(results_dir: Path) -> Path:
    return results_dir / "index.csv"


def _run_dir(results_dir: Path, run_id: str) -> Path:
    d = results_dir / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_runs(results_dir: Path) -> pd.DataFrame:
    idx = _index_path(results_dir)
    if not idx.exists():
        return pd.DataFrame()
    df = pd.read_csv(idx)
    # Tri: plus récent d'abord si colonne existe
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    return df


def save_run(
    results_dir: Path,
    *,
    created_at_iso: str,
    excel_path: str,
    mine_sheet: str | None,
    amort_sheet: str | None,
    regime_key: str,
    gold_price: float,
    royalty_rate: float,
    cit_rate: float,
    discount_rate: float,
    indicators: Dict[str, Any],
    annual_df: pd.DataFrame,
) -> str:
    """
    Sauvegarde un scénario:
      - meta + indicateurs en JSON
      - annual table en pickle
      - index.csv mis à jour
    Renvoie run_id.
    """
    results_dir = ensure_results_dir(results_dir)

    payload = {
        "excel_path": str(excel_path),
        "mine_sheet": mine_sheet or "",
        "amort_sheet": amort_sheet or "",
        "regime_key": str(regime_key),
        "gold_price": _safe_float(gold_price),
        "royalty_rate": _safe_float(royalty_rate),
        "cit_rate": _safe_float(cit_rate),
        "discount_rate": _safe_float(discount_rate),
    }
    run_id = _hash_payload(payload)

    meta = ScenarioMeta(
        run_id=run_id,
        created_at=created_at_iso,
        excel_path=str(excel_path),
        mine_sheet=mine_sheet,
        amort_sheet=amort_sheet,
        regime_key=str(regime_key),
        gold_price=_safe_float(gold_price),
        royalty_rate=_safe_float(royalty_rate),
        cit_rate=_safe_float(cit_rate),
        discount_rate=_safe_float(discount_rate),
    )

    rdir = _run_dir(results_dir, run_id)
    # 1) Meta
    (rdir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) Indicators
    (rdir / "indicators.json").write_text(
        json.dumps(indicators, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) Annual DF
    annual_df.to_pickle(rdir / "annual.pkl")

    # 4) Index append/update
    idx = _index_path(results_dir)
    row = asdict(meta)
    df_idx = pd.DataFrame([row])

    if idx.exists():
        old = pd.read_csv(idx)
        # si run_id existe déjà: on remplace (update) la ligne, sinon append
        if "run_id" in old.columns and (old["run_id"] == run_id).any():
            old = old[old["run_id"] != run_id]
        out = pd.concat([df_idx, old], ignore_index=True)
    else:
        out = df_idx

    out.to_csv(idx, index=False)
    return run_id


def load_run(
    results_dir: Path, run_id: str
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Retourne (meta, annual_df, indicators)
    """
    rdir = _run_dir(results_dir, run_id)

    meta_path = rdir / "meta.json"
    ind_path = rdir / "indicators.json"
    annual_path = rdir / "annual.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json introuvable pour run_id={run_id}")
    if not ind_path.exists():
        raise FileNotFoundError(f"indicators.json introuvable pour run_id={run_id}")
    if not annual_path.exists():
        raise FileNotFoundError(f"annual.pkl introuvable pour run_id={run_id}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    indicators = json.loads(ind_path.read_text(encoding="utf-8"))
    annual_df = pd.read_pickle(annual_path)

    return meta, annual_df, indicators
