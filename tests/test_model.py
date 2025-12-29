from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.model import (_is_year,_excel_like_npv,_pick_row,_read_scalar_from_sheet,ProjectInputs,FiscalRegime,default_regime,royalty_rate_for_price,run_model)


class TestIsYear:
    def test_valid_year_int(self) -> None:
        assert _is_year(2011)
        assert _is_year(1900)
        assert _is_year(2200)

    def test_valid_year_float_and_str(self) -> None:
        assert _is_year(2011.0)
        assert _is_year("2011")
        assert _is_year("  2015  ")

    def test_invalid_year_out_of_bounds(self) -> None:
        assert not _is_year(1899)
        assert not _is_year(2201)

    def test_invalid_year_nan_or_none_or_text(self) -> None:
        assert not _is_year(None)
        assert not _is_year(float("nan"))
        assert not _is_year("foo")
        assert not _is_year("20xx")


class TestExcelLikeNPV:
    def test_empty_cashflows_returns_zero(self) -> None:
        assert _excel_like_npv(0.1, np.array([])) == 0.0

    def test_zero_rate_is_simple_sum(self) -> None:
        cf = np.array([100.0, 100.0, 100.0])
        assert _excel_like_npv(0.0, cf) == pytest.approx(300.0)

    def test_positive_rate_discounts_from_period_1(self) -> None:
        cf = np.array([100.0])
        npv = _excel_like_npv(0.1, cf)
        assert npv == pytest.approx(100.0 / 1.1)


class TestPickRow:
    def test_pick_existing_row_by_regex(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["Revenue", "Total CAPEX"],
                "unit": ["USD", "USD"],
                2010: [1.0, 2.0],
                2011: [3.0, 4.0],
            }
        )

        row = _pick_row(df, r"CAPEX")
        assert row is not None
        assert list(row.index) == [2010, 2011]
        assert row.values.tolist() == [2.0, 4.0]

    def test_return_none_when_no_match(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["Revenue", "Total CAPEX"],
                "unit": ["USD", "USD"],
                2010: [1.0, 2.0],
            }
        )
        row = _pick_row(df, r"DoesNotExist")
        assert row is None


class TestReadScalarFromSheet:
    def test_reads_scalar_from_unit_column(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["Taux d'actualisation", "Autre"],
                "unit": ["0.12", "x"],
            }
        )
        v = _read_scalar_from_sheet(df, r"Taux\s+d['’]actualisation")
        assert v == pytest.approx(0.12)

    def test_returns_fallback_when_no_match(self) -> None:
        df = pd.DataFrame({"label": ["Foo"], "unit": ["1.23"]})
        v = _read_scalar_from_sheet(df, r"Bar", fallback=0.5)
        assert v == pytest.approx(0.5)

    def test_non_numeric_unit_returns_fallback(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["Taux d'actualisation"],
                "unit": ["not-a-number"],
            }
        )
        v = _read_scalar_from_sheet(df, r"Taux\s+d['’]actualisation", fallback=0.1)
        assert v == pytest.approx(0.1)

class TestDefaultRegime:
    def test_cm2003_properties(self) -> None:
        r = default_regime("CM2003")
        assert r.name == "CM2003"
        assert r.cit_rate == pytest.approx(0.175)
        assert r.royalty_mode == "progressive"
        assert r.local_development_levy == pytest.approx(0.0)
        assert r.imf_rate == pytest.approx(0.0)

    def test_cm2015_properties_and_short_code(self) -> None:
        r1 = default_regime("CM2015")
        r2 = default_regime("2015")
        assert r1.name == "CM2015"
        assert r2.name == "CM2015"
        assert r1.cit_rate == pytest.approx(0.275)
        assert r1.local_development_levy == pytest.approx(0.01)
        assert r1.imf_rate == pytest.approx(0.005)

    def test_unknown_regime_raises(self) -> None:
        with pytest.raises(ValueError):
            default_regime("UNKNOWN")


class TestRoyaltyRateForPrice:
    def test_flat_mode_always_returns_flat_rate(self) -> None:
        regime = FiscalRegime(
            name="TEST_FLAT",
            cit_rate=0.3,
            royalty_mode="flat",
            royalty_rate_flat=0.07,
        )
        for p in [800.0, 1000.0, 2000.0]:
            assert royalty_rate_for_price(regime, p) == pytest.approx(0.07)

    def test_progressive_mode_uses_bands(self) -> None:
        regime = FiscalRegime(name="TEST_PROG",cit_rate=0.3,royalty_mode="progressive")
        
        assert royalty_rate_for_price(regime, 900.0) == pytest.approx(0.03)
        assert royalty_rate_for_price(regime, 1000.0) == pytest.approx(0.04)
        assert royalty_rate_for_price(regime, 1299.9) == pytest.approx(0.04)
        assert royalty_rate_for_price(regime, 1300.0) == pytest.approx(0.05)
        assert royalty_rate_for_price(regime, 2000.0) == pytest.approx(0.05)


def _build_simple_inputs() -> ProjectInputs:
    years = np.array([2025, 2026, 2027], dtype=int)
    return ProjectInputs(
        years=years,
        revenue=np.array([100.0, 110.0, 120.0]),
        opex=np.array([30.0, 35.0, 40.0]),
        capex=np.array([50.0, 20.0, 0.0]),
        depreciation=np.array([10.0, 10.0, 10.0]),
        base_gold_price=1000.0,
        discount_rate=0.10,
    )


def _build_flat_regime() -> FiscalRegime:
    return FiscalRegime(
        name="TEST",
        cit_rate=0.30,
        royalty_mode="flat",
        royalty_rate_flat=0.05,
        local_development_levy=0.01,
        imf_rate=0.0,
    )


class TestRunModelBasic:
    def test_basic_shapes_and_columns(self) -> None:
        inputs = _build_simple_inputs()
        regime = _build_flat_regime()

        tbl, ind = run_model(inputs, regime)

        # Table shape
        assert len(tbl) == 3
        expected_cols = {
            "Year",
            "Revenue_CA",
            "OPEX",
            "CAPEX",
            "Depreciation",
            "Royalty",
            "Local_dev_levy",
            "Taxable_income",
            "CIT",
            "IMF",
            "CF_pre_tax",
            "CF_post_tax",
            "Gov_revenue",
        }
        assert expected_cols.issubset(tbl.columns)
        row0 = tbl.iloc[0]
        assert row0["Year"] == 2025
        assert row0["Revenue_CA"] == pytest.approx(100.0)
        assert row0["Royalty"] == pytest.approx(5.0)
        assert row0["Local_dev_levy"] == pytest.approx(1.0)
        expected_taxable0 = 100.0 - 30.0 - 10.0 - 5.0 - 1.0
        assert row0["Taxable_income"] == pytest.approx(expected_taxable0)
        assert row0["CIT"] == pytest.approx(expected_taxable0 * 0.30)
        expected_cf_pre0 = 100.0 - 30.0 - 50.0
        assert row0["CF_pre_tax"] == pytest.approx(expected_cf_pre0)
        expected_gov0 = row0["Royalty"] + row0["Local_dev_levy"] + row0["CIT"]
        assert row0["Gov_revenue"] == pytest.approx(expected_gov0)
        assert row0["CF_post_tax"] == pytest.approx(
            row0["CF_pre_tax"] - row0["Gov_revenue"]
        )

        assert ind["regime"] == "TEST"
        assert ind["gold_price"] == pytest.approx(inputs.base_gold_price)
        assert ind["discount_rate"] == pytest.approx(inputs.discount_rate)
        assert ind["royalty_rate"] == pytest.approx(0.05)
        assert ind["cit_rate"] == pytest.approx(0.30)
        assert ind["NPV_pre_tax"] > 0
        assert ind["NPV_post_tax"] > 0
        assert ind["Gov_NPV"] > 0
        
        assert ind["AETR"] == pytest.approx(ind["Gov_NPV"] / ind["NPV_pre_tax"])
        assert ind["TEMI"] == pytest.approx(
            1.0 - ind["NPV_post_tax"] / ind["NPV_pre_tax"]
        )


class TestRunModelOverrides:
    def test_gold_price_scales_revenue(self) -> None:
        inputs = _build_simple_inputs()
        regime = _build_flat_regime()

        tbl_base, ind_base = run_model(inputs, regime)
        tbl_double, ind_double = run_model(inputs, regime, gold_price=2000.0)

        assert ind_double["gold_price"] == pytest.approx(2000.0)
        assert tbl_double["Revenue_CA"].iloc[0] == pytest.approx(2.0 * tbl_base["Revenue_CA"].iloc[0])

    def test_royalty_and_cit_rate_overrides(self) -> None:
        inputs = _build_simple_inputs()
        regime = _build_flat_regime()
        tbl, ind = run_model(inputs, regime, royalty_rate_override=0.10, cit_rate_override=0.50)
        assert ind["royalty_rate"] == pytest.approx(0.10)
        assert ind["cit_rate"] == pytest.approx(0.50)
        taxable = tbl["Taxable_income"].values
        expected_cit = taxable * 0.50
        assert np.allclose(tbl["CIT"].values, expected_cit)


class TestRunModelLossCarryForwardAndIMF:
    def _build_loss_inputs(self) -> ProjectInputs:
        years = np.array([1, 2, 3, 4], dtype=int)
        return ProjectInputs(years=years,
            revenue=np.array([50.0, 80.0, 80.0, 80.0]),
            opex=np.array([40.0, 40.0, 40.0, 40.0]),
            capex=np.array([0.0, 0.0, 0.0, 0.0]),
            depreciation=np.array([20.0, 20.0, 20.0, 20.0]),
            base_gold_price=1000.0,
            discount_rate=0.10)

    def test_loss_carry_forward_reduces_future_taxable(self) -> None:
        inputs = self._build_loss_inputs()
        regime = FiscalRegime(name="TEST2",
            cit_rate=0.30,
            royalty_mode="flat",
            royalty_rate_flat=0.0,
            local_development_levy=0.0,
            imf_rate=0.0,
            loss_carry_forward=True)

        tbl, _ = run_model(inputs, regime)
        expected_taxable = np.array([0.0, 10.0, 20.0, 20.0])
        expected_cit = expected_taxable * 0.30

        assert np.allclose(tbl["Taxable_income"].values, expected_taxable)
        assert np.allclose(tbl["CIT"].values, expected_cit)

    def test_imf_applied_only_when_no_cit(self) -> None:
        inputs = self._build_loss_inputs()
        regime = FiscalRegime(name="IMF",
            cit_rate=0.30,
            royalty_mode="flat",
            royalty_rate_flat=0.0,
            local_development_levy=0.0,
            imf_rate=0.01, 
            loss_carry_forward=True)
        tbl, _ = run_model(inputs, regime)

        assert tbl["CIT"].iloc[0] == pytest.approx(0.0)
        assert tbl["IMF"].iloc[0] == pytest.approx(0.01 * 50.0)
        assert tbl["CIT"].iloc[1] > 0.0
        assert tbl["IMF"].iloc[1] == pytest.approx(0.0)
        assert tbl["IMF"].iloc[2] == pytest.approx(0.0)
        assert tbl["IMF"].iloc[3] == pytest.approx(0.0)
