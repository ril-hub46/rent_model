from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pytest
from src import storage as mod


@dataclass
class DummyIndicators:
    npv: float
    gov_npv: float


def test_to_jsonable_with_dataclass() -> None:
    obj = DummyIndicators(npv=1000.0, gov_npv=500.0)
    res = mod._to_jsonable(obj)
    assert isinstance(res, dict)
    assert res["npv"] == 1000.0
    assert res["gov_npv"] == 500.0


def test_to_jsonable_with_normal_object() -> None:
    obj = {"a": 1, "b": 2}
    res = mod._to_jsonable(obj)
    assert res is obj


def test_save_run_and_list_runs_roundtrip(tmp_path: Path) -> None:
    """Test fonctionnel : save_run puis list_runs sur un vrai fichier SQLite."""
    db_path = tmp_path / "db" / "runs.db"

    annual = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "CF_pre_tax": [100.0, 110.0, 120.0],
            "CF_post_tax": [80.0, 90.0, 100.0],
        }
    )
    indicators: Dict[str, Any] = {
        "NPV_pre_tax": 1000.0,
        "NPV_post_tax": 800.0,
        "Gov_NPV": 200.0,
        "TEMI": 0.25,
    }

    run_id = mod.save_run(
        db_path=str(db_path),
        scenario_name="Base case",
        created_at="2025-01-01T12:00:00",
        excel_path="model.xlsx",
        regime_key="CM2015",
        gold_price=1600.0,
        discount_rate=0.10,
        royalty_rate=0.05,
        cit_rate=0.275,
        indicators=indicators,
        annual_table=annual,
        notes="Test notes",
    )
    assert isinstance(run_id, int)
    assert run_id > 0
    df_runs = mod.list_runs(str(db_path))
    assert df_runs.shape[0] == 1

    row = df_runs.iloc[0]
    assert row["run_id"] == run_id
    assert row["scenario_name"] == "Base case"
    assert row["excel_path"] == "model.xlsx"
    assert row["regime_key"] == "CM2015"
    assert row["gold_price"] == 1600.0
    assert row["discount_rate"] == 0.10
    assert row["royalty_rate"] == 0.05
    assert row["cit_rate"] == 0.275
    assert row["notes"] == "Test notes"


def test_load_run_reconstructs_indicators_and_annual(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "runs.db"

    annual = pd.DataFrame(
        {
            "Year": [3, 1, 2],
            "CF_pre_tax": [120.0, 100.0, 110.0],
            "CF_post_tax": [100.0, 80.0, 90.0],
        }
    )
    indicators = {"NPV_pre_tax": 1000.0, "TEMI": 0.3}

    run_id = mod.save_run(
        db_path=str(db_path),
        scenario_name="Scenario sorted",
        created_at="2025-01-02T12:00:00",
        excel_path=None,
        regime_key="CM2003",
        gold_price=1500.0,
        discount_rate=0.12,
        royalty_rate=0.04,
        cit_rate=0.175,
        indicators=indicators,
        annual_table=annual,
        notes="",
    )

    loaded_ind, df_annual = mod.load_run(str(db_path), run_id)
    assert loaded_ind["NPV_pre_tax"] == 1000.0
    assert loaded_ind["TEMI"] == 0.3
    assert set(df_annual.columns) == {"Year", "CF_pre_tax", "CF_post_tax"}
    assert list(df_annual["Year"]) == [1, 2, 3]


def test_save_run_raises_if_no_year_column(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "runs.db"

    annual = pd.DataFrame(
        {
            "CF_pre_tax": [100.0, 110.0],
            "CF_post_tax": [80.0, 90.0],
        }
    )

    with pytest.raises(ValueError) as exc:
        mod.save_run(
            db_path=str(db_path),
            scenario_name="No year",
            created_at="2025-01-03T12:00:00",
            excel_path=None,
            regime_key="CM2015",
            gold_price=1600.0,
            discount_rate=0.10,
            royalty_rate=0.05,
            cit_rate=0.275,
            indicators={"x": 1},
            annual_table=annual,
        )

    assert "annual_table doit contenir une colonne 'Year' (ou 'year')" in str(exc.value)


def test_load_run_unknown_id_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "runs.db"
    mod.init_db(str(db_path))
    with pytest.raises(ValueError) as exc:
        mod.load_run(str(db_path), run_id=999)
    assert "run_id introuvable" in str(exc.value)
