from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pandas.testing as pdt
from src.results_store import (
    _safe_float,
    _hash_payload,
    ensure_results_dir,
    list_runs,
    save_run,
    load_run,
)


def test_safe_float_valid_values() -> None:
    assert _safe_float(1) == 1.0
    assert _safe_float(1.5) == 1.5
    assert _safe_float("2.3") == 2.3
    assert _safe_float("  4.5  ") == 4.5


def test_safe_float_invalid_returns_nan() -> None:
    v = _safe_float("not-a-number")
    assert math.isnan(v)

    v2 = _safe_float(None)
    assert math.isnan(v2)


def test_hash_payload_deterministic_and_order_independent() -> None:
    payload1: Dict[str, Any] = {"a": 1, "b": 2}
    payload2: Dict[str, Any] = {"b": 2, "a": 1}
    payload3: Dict[str, Any] = {"a": 1, "b": 3}

    h1 = _hash_payload(payload1)
    h2 = _hash_payload(payload2)
    h3 = _hash_payload(payload3)

    assert isinstance(h1, str)
    assert len(h1) == 16
    assert h1 == h2
    assert h1 != h3


def test_ensure_results_dir_creates_structure(tmp_path: Path) -> None:
    root = tmp_path / "project_root"
    results_dir = ensure_results_dir(root)

    assert results_dir.exists()
    assert results_dir.is_dir()
    runs_dir = results_dir / "runs"
    assert runs_dir.exists()
    assert runs_dir.is_dir()


def test_list_runs_empty_when_no_index(tmp_path: Path) -> None:
    root = tmp_path / "project_root"
    results_dir = ensure_results_dir(root)
    df = list_runs(results_dir)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def _build_dummy_annual_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2025, 2026],
            "Revenue_CA": [100.0, 110.0],
            "Gov_revenue": [20.0, 22.0],
        }
    )


def _build_dummy_indicators() -> Dict[str, Any]:
    return {
        "regime": "CM2003",
        "gold_price": 1300.0,
        "discount_rate": 0.10,
        "NPV_pre_tax": 1000.0,
        "NPV_post_tax": 700.0,
        "Gov_NPV": 300.0,
        "AETR": 0.30,
        "TEMI": 0.30,
    }


def test_save_and_load_run_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "project_root"
    results_dir = ensure_results_dir(root)
    annual_df = _build_dummy_annual_df()
    indicators = _build_dummy_indicators()
    run_id = save_run(
        root,
        created_at_iso="2025-01-01T12:00:00",
        excel_path="my_model.xlsx",
        mine_sheet="Données de la mine",
        amort_sheet="Amortissement",
        regime_key="CM2003",
        gold_price=1300.0,
        royalty_rate=0.04,
        cit_rate=0.35,
        discount_rate=0.10,
        indicators=indicators,
        annual_df=annual_df,
    )

    assert isinstance(run_id, str)
    assert len(run_id) == 16
    run_dir = results_dir / "runs" / run_id
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "indicators.json").exists()
    assert (run_dir / "annual.pkl").exists()
    idx_df = list_runs(results_dir)
    assert not idx_df.empty
    assert "run_id" in idx_df.columns
    assert run_id in idx_df["run_id"].tolist()

    meta_loaded, annual_loaded, indicators_loaded = load_run(results_dir, run_id)

    assert isinstance(meta_loaded, dict)
    assert meta_loaded["run_id"] == run_id
    assert meta_loaded["excel_path"] == "my_model.xlsx"
    assert meta_loaded["regime_key"] == "CM2003"
    assert meta_loaded["gold_price"] == 1300.0
    assert meta_loaded["cit_rate"] == 0.35
    assert meta_loaded["discount_rate"] == 0.10

    pdt.assert_frame_equal(
        annual_loaded.reset_index(drop=True), annual_df.reset_index(drop=True)
    )
    assert indicators_loaded == indicators


def test_save_run_updates_index_when_same_payload(tmp_path: Path) -> None:
    """
    We verify that the index only keeps one line (replacement),
    and that created_at is updated to the latest value.
    """
    root = tmp_path / "project_root"
    results_dir = ensure_results_dir(root)

    annual_df = _build_dummy_annual_df()
    indicators = _build_dummy_indicators()
    run_id1 = save_run(
        root,
        created_at_iso="2025-01-01T10:00:00",
        excel_path="my_model.xlsx",
        mine_sheet="Données de la mine",
        amort_sheet="Amortissement",
        regime_key="CM2003",
        gold_price=1300.0,
        royalty_rate=0.04,
        cit_rate=0.35,
        discount_rate=0.10,
        indicators=indicators,
        annual_df=annual_df,
    )

    run_id2 = save_run(
        root,
        created_at_iso="2025-01-02T09:00:00",
        excel_path="my_model.xlsx",
        mine_sheet="Données de la mine",
        amort_sheet="Amortissement",
        regime_key="CM2003",
        gold_price=1300.0,
        royalty_rate=0.04,
        cit_rate=0.35,
        discount_rate=0.10,
        indicators=indicators,
        annual_df=annual_df,
    )

    assert run_id1 == run_id2
    idx_df = list_runs(results_dir)
    assert len(idx_df) == 1
    row = idx_df.iloc[0]
    assert row["run_id"] == run_id1
    assert row["created_at"] == "2025-01-02T09:00:00"
