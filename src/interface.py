# src/interface.py
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- Fix imports when running: streamlit run src/interface.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import default_regime, load_project_inputs_from_excel, run_model  # noqa: E402


# ----------------------------
# Utilities: folders & persistence
# ----------------------------
def ensure_dirs() -> None:
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "results").mkdir(parents=True, exist_ok=True)


def list_saved_runs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("scenario_*.parquet"), reverse=True)


def save_scenario(results_dir: Path, df: pd.DataFrame, ind: dict, meta: dict) -> Path:
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = results_dir / f"scenario_{ts}.parquet"
    out_json = results_dir / f"scenario_{ts}.json"

    df.to_parquet(out_parquet, index=False)
    payload = {"indicators": ind, "meta": meta}
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_parquet


def save_sweep(results_dir: Path, df_sweep: pd.DataFrame, meta: dict) -> Path:
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"sweep_{ts}.csv"
    out_json = results_dir / f"sweep_{ts}.json"

    df_sweep.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_csv


def load_saved_scenario(parquet_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(parquet_path)
    json_path = parquet_path.with_suffix(".json")
    payload = {}
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    return df, payload


def persist_uploaded_excel(uploaded_file) -> Path:
    """Sauvegarde le fichier uploadé dans data/uploads/ et renvoie le chemin."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    out_path = ROOT / "data" / "uploads" / f"{ts}_{safe_name}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


# ----------------------------
# Plots
# ----------------------------
def plot_timeseries(df: pd.DataFrame) -> None:
    fig = plt.figure()
    plt.plot(df["Year"], df["CF_pre_tax"], label="CF pré-tax")
    plt.plot(df["Year"], df["CF_post_tax"], label="CF post-tax")
    plt.plot(df["Year"], df["Gov_revenue"], label="Recettes État")
    plt.legend()
    plt.xlabel("Année")
    plt.ylabel("Montants")
    st.pyplot(fig)


def plot_xy(df: pd.DataFrame, xcol: str, ycol: str, title: str) -> None:
    fig = plt.figure()
    plt.plot(df[xcol], df[ycol])
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    st.pyplot(fig)


# ----------------------------
# Metric helper (compatible old/new streamlit)
# ----------------------------
def metric(label: str, value: str, help_text: str, delta: str | None = None) -> None:
    """
    st.metric a un paramètre help= dans les versions récentes.
    On garde une compatibilité si l'utilisateur a une version plus ancienne.
    """
    try:
        st.metric(label=label, value=value, delta=delta, help=help_text)
    except TypeError:
        st.metric(label=label, value=value, delta=delta)
        st.caption(f"❓ {help_text}")


# ----------------------------
# Core: run multiple scenarios table
# ----------------------------
def run_scenario_row(
    inputs0,
    regime_code: str,
    gold_price: float,
    discount_rate: float,
    royalty_rate: float,
    cit_rate: float,
) -> Dict[str, Any]:
    regime = default_regime(regime_code)
    _, ind = run_model(
        inputs=inputs0,
        regime=regime,
        gold_price=float(gold_price),
        discount_rate=float(discount_rate),
        royalty_rate_override=float(royalty_rate),
        cit_rate_override=float(cit_rate),
    )

    return {
        "gold_price": float(gold_price),
        "discount_rate": float(discount_rate),
        "royalty_rate": float(royalty_rate),
        "cit_rate": float(cit_rate),
        "NPV_pre_tax": float(ind.get("NPV_pre_tax", np.nan)),
        "NPV_post_tax": float(ind.get("NPV_post_tax", np.nan)),
        "Gov_NPV": float(ind.get("Gov_NPV", np.nan)),
        "TEMI": float(ind.get("TEMI", np.nan))
        if ind.get("TEMI") is not None
        else np.nan,
    }


def run_scenarios_table(
    inputs0, regime_code: str, scenarios: pd.DataFrame
) -> pd.DataFrame:
    required_cols = {"gold_price", "discount_rate", "royalty_rate", "cit_rate"}
    missing = required_cols - set(scenarios.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la table scénarios : {missing}")

    rows: list[dict] = []
    for _, r in scenarios.iterrows():
        rows.append(
            run_scenario_row(
                inputs0=inputs0,
                regime_code=regime_code,
                gold_price=float(r["gold_price"]),
                discount_rate=float(r["discount_rate"]),
                royalty_rate=float(r["royalty_rate"]),
                cit_rate=float(r["cit_rate"]),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Rent_share", layout="wide")
    st.title("Rent_share")

    ensure_dirs()
    results_dir = ROOT / "data" / "results"

    # Glossaire
    with st.expander("📌 Glossaire", expanded=False):
        st.markdown(
            "- **NPV / VAN** : valeur actuelle nette des flux futurs actualisés.\n"
            "- **Pré-tax / Post-tax** : avant / après fiscalité.\n"
            "- **TEMI** : taux effectif moyen d’imposition (part de la rente captée via l’ensemble des prélèvements).\n"
            "- **Redevance minière (royalty)** : prélèvement calculé souvent sur le chiffre d’affaires (revenus bruts).\n"
            "- **CIT / IS** : impôt sur les bénéfices (Corporate Income Tax).\n"
            "- **Taux d’actualisation** : reflète la valeur du temps et le risque.\n"
        )

    st.sidebar.header("1) Base de données (Excel)")
    uploaded = st.sidebar.file_uploader(
        "Importer le fichier Excel du projet",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Charge le fichier contenant les données économiques du projet (production, coûts, CAPEX/OPEX...).",
    )

    if not uploaded:
        st.info(
            "Importe ton fichier Excel (.xlsx) dans la barre latérale pour démarrer."
        )
        st.stop()

    excel_path = persist_uploaded_excel(uploaded)

    st.sidebar.header("2) Régime fiscal")
    regime_code = st.sidebar.selectbox(
        "Régime fiscal",
        ["CM2003", "CM2015"],
        index=1,
        help="Choix du régime fiscal (paramètres par défaut du code minier / règles fiscales).",
    )

    with st.sidebar.expander("Avancé: noms des feuilles", expanded=False):
        mine_sheet = st.text_input("Feuille mine (optionnel)", value="").strip() or None
        amort_sheet = (
            st.text_input("Feuille amortissement", value="Amortissement").strip()
            or "Amortissement"
        )

    # Charger inputs (Excel)
    try:
        inputs0 = load_project_inputs_from_excel(
            excel_path=str(excel_path),
            regime=regime_code,
            mine_sheet=mine_sheet,
            amort_sheet=amort_sheet,
        )
    except Exception as e:
        st.error(f"Erreur lecture Excel : {e}")
        st.stop()

    # Defaults
    default_gold = float(getattr(inputs0, "base_gold_price", 1600.0))
    default_disc = float(getattr(inputs0, "discount_rate", 0.10))
    default_royalty = 0.05
    default_cit = 0.175 if regime_code == "CM2003" else 0.275

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Scénario unique", "Table annuelle", "Simulations", "Résultats sauvegardés"]
    )

    # ----------------------------
    # TAB 1: single scenario
    # ----------------------------
    with tab1:
        st.subheader("Scénario de base")

        cA, cB, cC, cD = st.columns(4)

        gold_price = cA.number_input(
            "Cours de l'or (USD/oz)",
            min_value=0.0,
            max_value=100000.0,
            value=float(default_gold),
            step=50.0,
            format="%.2f",
            help="Prix de référence de l’or utilisé pour calculer les revenus du projet.",
        )
        royalty_rate = cB.number_input(
            "Redevance minière",
            min_value=0.0,
            max_value=1.0,
            value=float(default_royalty),
            step=0.001,
            format="%.6f",
            help="Taux de redevance minière (souvent appliqué sur le chiffre d’affaires).",
        )
        cit_rate = cC.number_input(
            "Impôt sur les sociétés (CIT/IS)",
            min_value=0.0,
            max_value=1.0,
            value=float(default_cit),
            step=0.001,
            format="%.6f",
            help="Taux d’impôt sur les bénéfices (Corporate Income Tax).",
        )
        discount_rate = cD.number_input(
            "Taux d'actualisation",
            min_value=0.0,
            max_value=1.0,
            value=float(default_disc),
            step=0.001,
            format="%.6f",
            help="Taux utilisé pour actualiser les flux futurs en valeur présente (VAN/NPV).",
        )

        autosave = st.checkbox(
            "Enregistrer automatiquement ce scénario",
            value=False,
            help="Si activé, le scénario (table annuelle + indicateurs) est stocké dans data/results/.",
        )

        regime = default_regime(regime_code)
        df, ind = run_model(
            inputs=inputs0,
            regime=regime,
            gold_price=float(gold_price),
            discount_rate=float(discount_rate),
            royalty_rate_override=float(royalty_rate),
            cit_rate_override=float(cit_rate),
        )

        k1, k2, k3, k4, k5 = st.columns(5)

        temi_val = float(ind.get("TEMI", np.nan))
        npv_pre = float(ind.get("NPV_pre_tax", np.nan))
        npv_post = float(ind.get("NPV_post_tax", np.nan))
        gov_npv = float(ind.get("Gov_NPV", np.nan))

        with k1:
            metric(
                "TEMI",
                f"{temi_val*100:.2f}%" if np.isfinite(temi_val) else "NA",
                "Taux effectif moyen d’imposition : part de la rente captée par l’État (tous prélèvements).",
            )
        with k3:
            metric(
                "VAN pré-tax",
                f"{npv_pre:,.0f}" if np.isfinite(npv_pre) else "NA",
                "VAN des flux avant fiscalité (projet “brut”).",
            )
        with k4:
            metric(
                "VAN post-tax",
                f"{npv_post:,.0f}" if np.isfinite(npv_post) else "NA",
                "VAN des flux après fiscalité (investisseur).",
            )
        with k5:
            metric(
                "Gov VAN (recettes)",
                f"{gov_npv:,.0f}" if np.isfinite(gov_npv) else "NA",
                "VAN des recettes publiques (redevances + impôts + autres).",
            )

        st.subheader("Graphique : cash-flows & recettes publiques (annuel)")
        st.caption(
            "❓ CF = cash-flow (flux de trésorerie). Pré-tax = avant. Post-tax = après."
        )
        plot_timeseries(df)

        with st.expander("Indicateurs (détail)", expanded=False):
            st.json(ind)

        if autosave:
            meta = {
                "excel_uploaded_name": uploaded.name,
                "excel_saved_path": str(excel_path.relative_to(ROOT)),
                "regime": regime_code,
                "mine_sheet": mine_sheet,
                "amort_sheet": amort_sheet,
                "params": {
                    "gold_price": float(gold_price),
                    "discount_rate": float(discount_rate),
                    "royalty_rate": float(royalty_rate),
                    "cit_rate": float(cit_rate),
                },
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_scenario(results_dir, df, ind, meta)
            st.success("Scénario sauvegardé dans data/results/")

    # ----------------------------
    # TAB 2: annual table
    # ----------------------------
    with tab2:
        st.subheader("Tableau annuel (cash-flows & prélèvements)")
        st.caption("Affiche le tableau annuel du scénario calculé dans l’onglet 1.")
        st.dataframe(df, use_container_width=True)

    # ----------------------------
    # TAB 3: multi scenarios
    # ----------------------------
    with tab3:
        st.subheader("Simulations (table de scénarios modifiable)")
        st.markdown(
            "👉 Modifie la table ci-dessous (ajoute/supprime des lignes). Chaque ligne = **un scénario**.\n\n"
            "- **gold_price** : cours de l’or (USD/oz)\n"
            "- **royalty_rate** : redevance minière\n"
            "- **cit_rate** : impôt sur les sociétés\n"
            "- **discount_rate** : taux d’actualisation\n"
        )

        default_table = pd.DataFrame(
            [
                {
                    "gold_price": 1300.0,
                    "royalty_rate": 0.03,
                    "cit_rate": default_cit,
                    "discount_rate": default_disc,
                },
                {
                    "gold_price": 1400.0,
                    "royalty_rate": 0.04,
                    "cit_rate": default_cit,
                    "discount_rate": default_disc,
                },
                {
                    "gold_price": 1500.0,
                    "royalty_rate": 0.04,
                    "cit_rate": default_cit,
                    "discount_rate": default_disc,
                },
                {
                    "gold_price": 1600.0,
                    "royalty_rate": 0.05,
                    "cit_rate": default_cit,
                    "discount_rate": default_disc,
                },
                {
                    "gold_price": 1700.0,
                    "royalty_rate": 0.05,
                    "cit_rate": default_cit,
                    "discount_rate": default_disc,
                },
            ]
        )

        if "scenarios_table" not in st.session_state:
            st.session_state["scenarios_table"] = default_table

        edited = st.data_editor(
            st.session_state["scenarios_table"],
            num_rows="dynamic",
            use_container_width=True,
            key="scenarios_editor",
            column_config={
                "gold_price": st.column_config.NumberColumn(
                    "gold_price",
                    help="Cours de l’or (USD/oz).",
                    format="%.2f",
                ),
                "royalty_rate": st.column_config.NumberColumn(
                    "royalty_rate",
                    help="Taux de redevance. Ex: 0.05 = 5%.",
                    format="%.6f",
                ),
                "cit_rate": st.column_config.NumberColumn(
                    "cit_rate",
                    help="Taux d’IS/CIT. Ex: 0.275 = 27.5%.",
                    format="%.6f",
                ),
                "discount_rate": st.column_config.NumberColumn(
                    "discount_rate",
                    help="Taux d’actualisation. Ex: 0.10 = 10%.",
                    format="%.6f",
                ),
            },
        )
        st.session_state["scenarios_table"] = edited

        run_btn = st.button(
            "Lancer les scénarios",
            help="Calcule les indicateurs (NPV, Gov NPV, TEMI) pour chaque ligne/scénario.",
        )

        if run_btn:
            try:
                df_sweep = run_scenarios_table(inputs0, regime_code, edited)
            except Exception as e:
                st.error(f"Erreur scénarios : {e}")
                st.stop()

            df_sweep = df_sweep.sort_values("gold_price").reset_index(drop=True)

            st.subheader("Résultats (par scénario)")
            st.dataframe(df_sweep, use_container_width=True)

            st.subheader("Graphiques")
            st.caption(
                "❓ Fiscalité “progressive” : TEMI augmente quand gold_price augmente."
            )
            plot_xy(df_sweep, "gold_price", "TEMI", "TEMI vs cours de l'or")

            if st.checkbox(
                "Enregistrer ces résultats (scénarios)",
                value=False,
                help="Sauvegarde le tableau des résultats + un JSON meta dans data/results/.",
            ):
                meta = {
                    "excel_uploaded_name": uploaded.name,
                    "excel_saved_path": str(excel_path.relative_to(ROOT)),
                    "regime": regime_code,
                    "mine_sheet": mine_sheet,
                    "amort_sheet": amort_sheet,
                    "scenarios": edited.to_dict(orient="records"),
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
                out = save_sweep(results_dir, df_sweep, meta)
                st.success(f"Sauvegardé: {out}")

    # ----------------------------
    # TAB 4: load saved runs
    # ----------------------------
    with tab4:
        st.subheader("Charger des résultats sauvegardés")
        runs = list_saved_runs(results_dir)
        if not runs:
            st.info("Aucun résultat sauvegardé pour l’instant.")
        else:
            pick = st.selectbox(
                "Choisir un scénario", options=[p.name for p in runs], index=0
            )
            chosen = results_dir / pick
            df_old, payload = load_saved_scenario(chosen)

            st.markdown("### Métadonnées")
            st.json(payload.get("meta", {}))

            st.markdown("### Indicateurs")
            st.json(payload.get("indicators", {}))

            st.markdown("### Table annuelle")
            st.dataframe(df_old, use_container_width=True)

            st.markdown("### Graphique")
            plot_timeseries(df_old)


if __name__ == "__main__":
    main()
