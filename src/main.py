from __future__ import annotations

import argparse
from pathlib import Path

from src.model import default_regime, load_project_inputs_from_excel, run_model
from src.simulations import sweep_price_from_excel  # IMPORTANT: import + expose


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--excel", required=True, help="Path to the Excel input file.")
    # regime est optionnel au parsing, mais on le rend obligatoire si pas sweep
    p.add_argument("--regime", choices=["CM2003", "CM2015"], required=False)

    p.add_argument("--mine_sheet", default=None)
    p.add_argument("--amort_sheet", default="Amortissement")

    p.add_argument("--gold_price", type=float, default=None)
    p.add_argument("--discount", type=float, default=None)
    p.add_argument("--royalty_rate", type=float, default=None)
    p.add_argument("--cit_rate", type=float, default=None)

    p.add_argument("--sweep_price", action="store_true")
    p.add_argument("--prices", type=float, nargs="*")
    p.add_argument("--out", default=None, help="Optional output CSV file for sweep.")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    excel_path = str(Path(args.excel))

    # -----------------------------
    # 1) Mode sweep (sensibilité prix)
    # -----------------------------
    if args.sweep_price:
        # tests attendent ce message dans str(SystemExit)
        if not args.prices:
            raise SystemExit("did not pass --prices")

        # si regime absent en sweep, on met une valeur par défaut
        regime_code = args.regime or "CM2015"

        df = sweep_price_from_excel(
            excel_path=excel_path,
            prices=[float(x) for x in args.prices],
            regime_code=regime_code,
            mine_sheet=args.mine_sheet,
            amort_sheet=args.amort_sheet,
            discount_override=args.discount,
            royalty_rate_override=args.royalty_rate,
            cit_rate_override=args.cit_rate,
        )

        # impression console
        print(df.to_string(index=False))

        # sortie éventuelle
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)

        return

    # -----------------------------
    # 2) Mode scénario unique
    # -----------------------------
    if not args.regime:
        raise SystemExit("did not pass --regime")

    inputs0 = load_project_inputs_from_excel(
        excel_path=excel_path,
        regime=args.regime,
        mine_sheet=args.mine_sheet,
        amort_sheet=args.amort_sheet,
    )

    regime = default_regime(args.regime)

    gold_price = (
        float(args.gold_price)
        if args.gold_price is not None
        else float(getattr(inputs0, "base_gold_price", 1600.0))
    )
    discount_rate = (
        float(args.discount)
        if args.discount is not None
        else float(getattr(inputs0, "discount_rate", 0.10))
    )

    df, ind = run_model(
        inputs=inputs0,
        regime=regime,
        gold_price=gold_price,
        discount_rate=discount_rate,
        royalty_rate_override=args.royalty_rate,
        cit_rate_override=args.cit_rate,
    )

    print(f"--- Regime: {args.regime} ---")
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
        if k in ind:
            print(f"{k}: {ind[k]}")

    print("\n--- Annual table (head) ---")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
