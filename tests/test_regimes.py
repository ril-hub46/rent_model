from __future__ import annotations
import pytest
from src.regimes import (royalty_rate_from_gold_price,FiscalRegime,CM2003,CM2015)

def test_royalty_rate_low_price() -> None:
    assert royalty_rate_from_gold_price(800.0) == pytest.approx(0.03)
    assert royalty_rate_from_gold_price(1000.0) == pytest.approx(0.03)

def test_royalty_rate_mid_price() -> None:
    assert royalty_rate_from_gold_price(1000.01) == pytest.approx(0.04)
    assert royalty_rate_from_gold_price(1200.0) == pytest.approx(0.04)
    assert royalty_rate_from_gold_price(1500.0) == pytest.approx(0.04)

def test_royalty_rate_high_price() -> None:
    assert royalty_rate_from_gold_price(1500.01) == pytest.approx(0.05)
    assert royalty_rate_from_gold_price(2000.0) == pytest.approx(0.05)

def test_cm2003_parameters() -> None:
    assert isinstance(CM2003, FiscalRegime)
    assert CM2003.name == "CM2003"
    assert CM2003.cit_rate_default == pytest.approx(0.35)
    assert CM2003.state_participation == pytest.approx(0.10)
    assert CM2003.dividend_wht_rate == pytest.approx(0.0625)
    assert CM2003.imf_rate == pytest.approx(0.005)
    assert CM2003.local_dev_rate == pytest.approx(0.0)

def test_cm2015_parameters() -> None:
    assert isinstance(CM2015, FiscalRegime)
    assert CM2015.name == "CM2015"
    assert CM2015.cit_rate_default == pytest.approx(0.275)
    assert CM2015.state_participation == pytest.approx(0.10)
    assert CM2015.dividend_wht_rate == pytest.approx(0.0625)
    assert CM2015.imf_rate == pytest.approx(0.005)
    assert CM2015.local_dev_rate == pytest.approx(0.01)


def test_regime_uses_royalty_rate_func() -> None:
    price = 1400.0
    expected = royalty_rate_from_gold_price(price)

    assert CM2003.royalty_rate_func is royalty_rate_from_gold_price
    assert CM2015.royalty_rate_func is royalty_rate_from_gold_price
    assert CM2003.royalty_rate_func(price) == pytest.approx(expected)
    assert CM2015.royalty_rate_func(price) == pytest.approx(expected)
