from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest
from src import interface as mod


class DummyUploadedFile:
    """Simule un fichier uploadé Streamlit."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


def test_ensure_dirs_creates_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)
    data_dir = tmp_path / "data"
    uploads_dir = data_dir / "uploads"
    results_dir = data_dir / "results"
    assert not data_dir.exists()
    mod.ensure_dirs()
    assert data_dir.is_dir()
    assert uploads_dir.is_dir()
    assert results_dir.is_dir()


def test_list_saved_runs_returns_sorted_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)

    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    f1 = results_dir / "scenario_20240101_120000.parquet"
    f2 = results_dir / "scenario_20250101_100000.parquet"
    f3 = results_dir / "scenario_20230101_090000.parquet"
    for f in (f1, f2, f3):
        f.touch()
    runs = mod.list_saved_runs(results_dir)
    assert runs == [f2, f1, f3]


def test_save_and_load_scenario_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)

    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "CF_pre_tax": [100.0, 110.0, 120.0],
            "CF_post_tax": [80.0, 90.0, 100.0],
            "Gov_revenue": [20.0, 20.0, 20.0],
        }
    )
    indicators: Dict[str, Any] = {
        "NPV_pre_tax": 1000.0,
        "NPV_post_tax": 800.0,
        "Gov_NPV": 200.0,
        "TEMI": 0.25,
    }
    meta = {
        "regime": "CM2015",
        "params": {
            "gold_price": 1600.0,
            "discount_rate": 0.1,
            "royalty_rate": 0.05,
            "cit_rate": 0.275,
        },
    }

    out_parquet = mod.save_scenario(results_dir, df, indicators, meta)

    assert out_parquet.exists()
    json_path = out_parquet.with_suffix(".json")
    assert json_path.exists()
    df_loaded, payload = mod.load_saved_scenario(out_parquet)
    pd.testing.assert_frame_equal(df_loaded, df)
    assert payload["indicators"] == indicators
    assert payload["meta"]["regime"] == "CM2015"
    assert payload["meta"]["params"]["gold_price"] == 1600.0


def test_save_sweep_creates_csv_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)

    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_sweep = pd.DataFrame(
        {
            "gold_price": [1200.0, 1600.0],
            "discount_rate": [0.1, 0.1],
            "royalty_rate": [0.05, 0.05],
            "cit_rate": [0.275, 0.275],
            "Gov_NPV": [100.0, 200.0],
        }
    )
    meta = {"regime": "CM2015", "scenarios": df_sweep.to_dict(orient="records")}
    out_csv = mod.save_sweep(results_dir, df_sweep, meta)
    assert out_csv.exists()
    assert out_csv.suffix == ".csv"
    json_path = out_csv.with_suffix(".json")
    assert json_path.exists()
    loaded_meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_meta["regime"] == "CM2015"
    assert len(loaded_meta["scenarios"]) == 2


def test_persist_uploaded_excel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)
    content = b"dummy excel bytes"
    uploaded = DummyUploadedFile(name="test.xlsx", content=content)
    out_path = mod.persist_uploaded_excel(uploaded)
    assert out_path.exists()
    assert out_path.parent == tmp_path / "data" / "uploads"
    assert out_path.read_bytes() == content
    assert out_path.name.endswith("_test.xlsx")


def test_run_scenario_row_calls_run_model(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_default_regime(code: str) -> str:
        called["regime_code"] = code
        return f"REGIME-{code}"

    def fake_run_model(
        inputs,
        regime,
        gold_price,
        discount_rate,
        royalty_rate_override,
        cit_rate_override,
    ):
        called["inputs"] = inputs
        called["regime"] = regime
        called["gold_price"] = gold_price
        called["discount_rate"] = discount_rate
        called["royalty_rate_override"] = royalty_rate_override
        called["cit_rate_override"] = cit_rate_override

        dummy_df = pd.DataFrame({"Year": [1], "CF_pre_tax": [100.0]})
        indicators = {
            "NPV_pre_tax": 1000.0,
            "NPV_post_tax": 800.0,
            "Gov_NPV": 200.0,
            "TEMI": 0.3,
        }
        return dummy_df, indicators

    monkeypatch.setattr(mod, "default_regime", fake_default_regime, raising=True)
    monkeypatch.setattr(mod, "run_model", fake_run_model, raising=True)

    inputs0 = object()
    result = mod.run_scenario_row(
        inputs0=inputs0,
        regime_code="CM2015",
        gold_price=1500.0,
        discount_rate=0.1,
        royalty_rate=0.05,
        cit_rate=0.275,
    )

    assert called["regime_code"] == "CM2015"
    assert called["regime"] == "REGIME-CM2015"
    assert called["inputs"] is inputs0
    assert called["gold_price"] == 1500.0
    assert called["discount_rate"] == 0.1
    assert called["royalty_rate_override"] == 0.05
    assert called["cit_rate_override"] == 0.275

    assert result["gold_price"] == 1500.0
    assert result["NPV_pre_tax"] == 1000.0
    assert result["NPV_post_tax"] == 800.0
    assert result["Gov_NPV"] == 200.0
    assert np.isclose(result["TEMI"], 0.3)


def test_run_scenarios_table_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_scenario_row(**kwargs) -> Dict[str, Any]:
        return {
            "gold_price": float(kwargs["gold_price"]),
            "discount_rate": float(kwargs["discount_rate"]),
            "royalty_rate": float(kwargs["royalty_rate"]),
            "cit_rate": float(kwargs["cit_rate"]),
            "NPV_pre_tax": 1.0,
            "NPV_post_tax": 2.0,
            "Gov_NPV": 3.0,
            "TEMI": 0.1,
        }

    monkeypatch.setattr(mod, "run_scenario_row", fake_run_scenario_row, raising=True)
    inputs0 = object()
    regime_code = "CM2015"

    scenarios = pd.DataFrame(
        [
            {
                "gold_price": 1200.0,
                "discount_rate": 0.1,
                "royalty_rate": 0.05,
                "cit_rate": 0.275,
            },
            {
                "gold_price": 1600.0,
                "discount_rate": 0.12,
                "royalty_rate": 0.05,
                "cit_rate": 0.275,
            },
        ]
    )

    df_out = mod.run_scenarios_table(inputs0, regime_code, scenarios)

    assert df_out.shape[0] == 2
    assert set(df_out.columns) == {
        "gold_price",
        "discount_rate",
        "royalty_rate",
        "cit_rate",
        "NPV_pre_tax",
        "NPV_post_tax",
        "Gov_NPV",
        "TEMI",
    }
    assert list(df_out["gold_price"]) == [1200.0, 1600.0]


def test_run_scenarios_table_missing_column_raises() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "gold_price": 1200.0,
                "discount_rate": 0.1,
                "royalty_rate": 0.05,
            }
        ]
    )

    with pytest.raises(ValueError) as exc:
        mod.run_scenarios_table(
            inputs0=object(), regime_code="CM2015", scenarios=scenarios
        )

    assert "Colonnes manquantes" in str(exc.value)
    assert "cit_rate" in str(exc.value)


def test_plot_timeseries_calls_streamlit_pyplot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"count": 0}

    def fake_pyplot(fig):
        called["count"] += 1

    monkeypatch.setattr(mod.st, "pyplot", fake_pyplot, raising=True)
    df = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "CF_pre_tax": [100.0, 110.0, 120.0],
            "CF_post_tax": [80.0, 90.0, 100.0],
            "Gov_revenue": [20.0, 25.0, 30.0],
        }
    )
    mod.plot_timeseries(df)
    assert called["count"] == 1


def test_plot_xy_calls_streamlit_pyplot(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}

    def fake_pyplot(fig):
        called["count"] += 1

    monkeypatch.setattr(mod.st, "pyplot", fake_pyplot, raising=True)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})

    mod.plot_xy(df, "x", "y", "titre test")

    assert called["count"] == 1
