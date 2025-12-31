from __future__ import annotations
from typing import Any, Dict
import pytest
from unittest.mock import Mock
import src.simulations as sim
from src.simulations import sweep_price_from_excel


def _dummy_indicators(price: float, discount: float | None) -> Dict[str, Any]:
    """
    Fabrique un dict d'indicateurs minimal cohérent avec run_model.
    """
    return {
        "regime": "TEST_REGIME",
        "gold_price": price,
        "discount_rate": discount,
        "royalty_rate": 0.05,
        "cit_rate": 0.3,
        "NPV_pre_tax": 1000.0 + price,
        "NPV_post_tax": 700.0 + price,
        "Gov_NPV": 300.0 + price,
        "AETR": 0.30,
        "TEMI": 0.30,
    }


def test_sweep_price_runs_model_for_each_price(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = [1200.0, 1300.0, 1400.0]
    fake_regime = object()
    fake_inputs = object()
    default_regime_mock = Mock(return_value=fake_regime)
    load_inputs_mock = Mock(return_value=fake_inputs)

    def run_model_side_effect(
        inputs,
        regime,
        gold_price,
        discount_rate=None,
        royalty_rate_override=None,
        cit_rate_override=None,
    ):
        assert inputs is fake_inputs
        assert regime is fake_regime
        assert discount_rate == 0.12
        assert royalty_rate_override == 0.08
        assert cit_rate_override == 0.27
        return None, _dummy_indicators(gold_price, discount_rate)

    run_model_mock = Mock(side_effect=run_model_side_effect)

    monkeypatch.setattr(sim, "default_regime", default_regime_mock)
    monkeypatch.setattr(sim, "load_project_inputs_from_excel", load_inputs_mock)
    monkeypatch.setattr(sim, "run_model", run_model_mock)
    df = sweep_price_from_excel(
        excel_path="dummy.xlsx",
        prices=prices,
        regime_code="CM2003",
        mine_sheet="MineSheet",
        amort_sheet="AmortSheet",
        discount_override=0.12,
        royalty_rate_override=0.08,
        cit_rate_override=0.27,
    )

    default_regime_mock.assert_called_once_with("CM2003")
    load_inputs_mock.assert_called_once_with(
        "dummy.xlsx", regime="CM2003", mine_sheet="MineSheet", amort_sheet="AmortSheet"
    )
    assert run_model_mock.call_count == len(prices)
    assert len(df) == len(prices)
    expected_cols = [
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
    assert list(df.columns) == expected_cols
    assert df["gold_price"].tolist() == prices
    assert set(df["discount_rate"].tolist()) == {0.12}


def test_sweep_price_without_overrides_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = [1000.0, 1100.0]
    fake_regime = object()
    fake_inputs = object()
    monkeypatch.setattr(sim, "default_regime", Mock(return_value=fake_regime))
    monkeypatch.setattr(
        sim, "load_project_inputs_from_excel", Mock(return_value=fake_inputs)
    )

    calls: list[Dict[str, Any]] = []

    def run_model_side_effect(
        inputs,
        regime,
        gold_price,
        discount_rate=None,
        royalty_rate_override=None,
        cit_rate_override=None,
    ):
        calls.append(
            {
                "inputs": inputs,
                "regime": regime,
                "gold_price": gold_price,
                "discount_rate": discount_rate,
                "royalty_rate_override": royalty_rate_override,
                "cit_rate_override": cit_rate_override,
            }
        )
        return None, _dummy_indicators(gold_price, discount_rate)

    monkeypatch.setattr(sim, "run_model", Mock(side_effect=run_model_side_effect))

    df = sweep_price_from_excel(
        excel_path="dummy.xlsx",
        prices=prices,
        regime_code="CM2015",
        mine_sheet=None,
        amort_sheet="Amortissement",
    )

    assert len(calls) == len(prices)
    assert df.shape[0] == len(prices)
    for call, price in zip(calls, prices):
        assert call["gold_price"] == price
        assert call["discount_rate"] is None
        assert call["royalty_rate_override"] is None
        assert call["cit_rate_override"] is None

    assert "NPV_pre_tax" in df.columns
    assert "Gov_NPV" in df.columns
