from __future__ import annotations
import sys
import pandas as pd
import pytest
from src import main as main_mod


def test_main_sweep_price_without_prices_raises_system_exit(monkeypatch):
    """
    Verify that main() crashes cleanly if --sweep_price is used without --prices.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--excel", "dummy.xlsx", "--sweep_price"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()
    assert "did not pass --prices" in str(excinfo.value)


def test_main_sweep_price_calls_sweep_and_prints(monkeypatch, capsys, tmp_path):
    """
    Check the path with --sweep_price + --prices:
    - sweep_price_from_excel is called with the correct parameters
    - the result is printed
    - if --out is provided, the CSV is written.
    """
    fake_excel = tmp_path / "mine.xlsx"
    fake_excel.write_text("dummy", encoding="utf-8")
    df_fake = pd.DataFrame({"gold_price": [1500.0, 1600.0], "NPV": [1.0, 2.0]})

    called_args: dict | None = None

    def fake_sweep_price_from_excel(
        *,
        excel_path: str,
        prices: list[float],
        regime_code: str,
        mine_sheet: str | None,
        amort_sheet: str,
        discount_override: float | None,
        royalty_rate_override: float | None,
        cit_rate_override: float | None,
    ):
        nonlocal called_args
        called_args = {
            "excel_path": excel_path,
            "prices": prices,
            "regime_code": regime_code,
            "mine_sheet": mine_sheet,
            "amort_sheet": amort_sheet,
            "discount_override": discount_override,
            "royalty_rate_override": royalty_rate_override,
            "cit_rate_override": cit_rate_override,
        }
        return df_fake

    monkeypatch.setattr(main_mod, "sweep_price_from_excel", fake_sweep_price_from_excel)

    out_path = tmp_path / "results.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--excel",
            str(fake_excel),
            "--regime",
            "CM2015",
            "--sweep_price",
            "--prices",
            "1500",
            "1600",
            "--out",
            str(out_path),
        ],
    )

    main_mod.main()
    assert called_args is not None
    assert called_args["excel_path"] == str(fake_excel)
    assert called_args["prices"] == [1500.0, 1600.0]
    assert called_args["regime_code"] == "CM2015"

    captured = capsys.readouterr()
    assert "gold_price" in captured.out
    assert "NPV" in captured.out

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert list(written.columns) == ["gold_price", "NPV"]
