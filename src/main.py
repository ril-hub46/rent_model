from __future__ import annotations

import argparse
from pathlib import Path

from src.model import default_regime, load_project_inputs_from_excel, run_model
from src.simulations import sweep_price_from_excel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rent-sharing model (Burkina Faso)")
    p.add_argument("--excel", required=True, help="Path to Excel workbook ")
    p.add_argument(
        "--regime", default="CM2003", choices=["CM2003", "CM2015"], help="Fiscal regime"
    )
    p.add_argument("--mine_sheet", default=None, help="Override mine sheet name")
    p.add_argument(
        "--amort_sheet", default="Amortissement", help="Amortization sheet name"
    )

    p.add_argument(
        "--gold_price", type=float, default=None, help="Override gold price (USD/oz)"
    )
    p.add_argument(
        "--discount",
        type=float,
        default=None,
        help="Override discount rate (e.g., 0.10)",
    )
    p.add_argument(
        "--royalty_rate",
        type=float,
        default=None,
        help="Override royalty rate (e.g., 0.05)",
    )
    p.add_argument(
        "--cit_rate", type=float, default=None, help="Override CIT rate (e.g., 0.275)"
    )

    p.add_argument("--sweep_price", action="store_true", help="Run sweep over --prices")
    p.add_argument(
        "--prices", type=float, nargs="+", default=None, help="Gold prices for sweep"
    )
    p.add_argument("--out", default=None, help="Output CSV path for sweep")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    excel_path = str(Path(args.excel))

    if args.sweep_price:
        if not args.prices:
            raise SystemExit("You used --sweep_price but did not pass --prices ...")

        df = sweep_price_from_excel(
            excel_path=excel_path,
            prices=args.prices,
            regime_code=args.regime,
            mine_sheet=args.mine_sheet,
            amort_sheet=args.amort_sheet,
            discount_override=args.discount,
            royalty_rate_override=args.royalty_rate,
            cit_rate_override=args.cit_rate,
        )
        print(df.to_string(index=False))

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}")
        return

    regime = default_regime(args.regime)
    inputs = load_project_inputs_from_excel(
        excel_path,
        regime=args.regime,
        mine_sheet=args.mine_sheet,
        amort_sheet=args.amort_sheet,
    )

    table, ind = run_model(
        inputs,
        regime,
        gold_price=args.gold_price,
        discount_rate=args.discount,
        royalty_rate_override=args.royalty_rate,
        cit_rate_override=args.cit_rate,
    )

    print(f"--- Regime: {ind['regime']} ---")
    for k in [
        "gold_price",
        "discount_rate",
        "royalty_rate",
        "cit_rate",
        "NPV_pre_tax",
        "NPV_post_tax",
        "Gov_NPV",
        "TEMI",
    ]:
        print(f"{k}: {ind[k]}")

    print("\n--- Annual table (head) ---")
    print(table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
