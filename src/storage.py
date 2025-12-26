# src/storage.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def init_db(db_path: str) -> None:
    p = Path(db_path)
    _ensure_parent_dir(p)

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                excel_path TEXT,
                regime_key TEXT NOT NULL,
                gold_price REAL,
                discount_rate REAL,
                royalty_rate REAL,
                cit_rate REAL,
                notes TEXT,
                indicators_json TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS annual (
                run_id INTEGER NOT NULL,
                year INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                PRIMARY KEY (run_id, year),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
            """
        )

        con.commit()


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def save_run(
    db_path: str,
    scenario_name: str,
    created_at: str,
    excel_path: Optional[str],
    regime_key: str,
    gold_price: Optional[float],
    discount_rate: Optional[float],
    royalty_rate: Optional[float],
    cit_rate: Optional[float],
    indicators: Dict[str, Any],
    annual_table: pd.DataFrame,
    notes: str = "",
) -> int:
    init_db(db_path)

    indicators_clean = json.dumps(_to_jsonable(indicators), ensure_ascii=False)

    # annual_table -> store each year row as json
    if "Year" in annual_table.columns:
        year_col = "Year"
    elif "year" in annual_table.columns:
        year_col = "year"
    else:
        raise ValueError("annual_table doit contenir une colonne 'Year' (ou 'year').")

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                created_at, scenario_name, excel_path, regime_key,
                gold_price, discount_rate, royalty_rate, cit_rate, notes, indicators_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                scenario_name,
                excel_path,
                regime_key,
                regime_key,
                float(gold_price) if gold_price is not None else None,
                float(discount_rate) if discount_rate is not None else None,
                float(royalty_rate) if royalty_rate is not None else None,
                float(cit_rate) if cit_rate is not None else None,
                notes,
                indicators_clean,
            ),
        )
        run_id = int(cur.lastrowid)

        rows = []
        for _, r in annual_table.iterrows():
            y = int(r[year_col])
            d = r.to_dict()
            d[year_col] = y
            rows.append((run_id, y, json.dumps(d, ensure_ascii=False)))

        cur.executemany(
            "INSERT OR REPLACE INTO annual (run_id, year, data_json) VALUES (?, ?, ?)",
            rows,
        )

        con.commit()

    return run_id


def list_runs(db_path: str) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT
                run_id, created_at, scenario_name, excel_path, regime_key,
                gold_price, discount_rate, royalty_rate, cit_rate, notes
            FROM runs
            ORDER BY run_id DESC
            """,
            con,
        )
    return df


def load_run(db_path: str, run_id: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        run = pd.read_sql_query(
            "SELECT * FROM runs WHERE run_id = ?",
            con,
            params=(run_id,),
        )
        if run.empty:
            raise ValueError(f"run_id introuvable: {run_id}")

        annual = pd.read_sql_query(
            "SELECT year, data_json FROM annual WHERE run_id = ? ORDER BY year",
            con,
            params=(run_id,),
        )

    indicators = json.loads(run.loc[0, "indicators_json"])

    # reconstruct annual table
    rows = [json.loads(s) for s in annual["data_json"].tolist()]
    df_annual = pd.DataFrame(rows).sort_values("Year" if "Year" in rows[0] else "year")

    return indicators, df_annual
# ---------------------------