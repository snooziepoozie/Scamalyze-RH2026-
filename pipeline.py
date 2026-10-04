# pipeline.py — all TigerData access lives here
import os
from pathlib import Path
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam

# getting the ip addr
import urllib.request

ip = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
print(f"Your current public IP is: {ip}")
print(f"Add this to TigerData trusted sources as: {ip}/32")

# Load the project-local .env regardless of the working directory used to launch Streamlit.
load_dotenv(Path(__file__).resolve().with_name(".env"))


def _db_url() -> str:
    url = os.environ.get("TIGER_DB_URL")
    if not url:
        raise RuntimeError("Set TIGER_DB_URL in your .env (see setup note).")
    # SQLAlchemy needs an explicit driver; your string starts with postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(_db_url(), pool_pre_ping=True)


# ── Build a WHERE clause + params from a filters dict ──────────────────────
def _where(filters: dict):
    clauses, params, expanding = [], {}, []

    def add_in(col, key):
        vals = filters.get(key)
        if vals:
            clauses.append(f"{col} IN :{key}")
            params[key] = list(vals)
            expanding.append(key)

    add_in("transaction_type", "txn_types")
    add_in("location", "locations")
    add_in("merchant_category", "merchant_categories")

    ff = filters.get("fraud_filter", "all")
    if ff == "fraud":
        clauses.append("is_fraud = true")
    elif ff == "legit":
        clauses.append("is_fraud = false")

    if filters.get("start_date"):
        clauses.append("timestamp >= :start_date")
        params["start_date"] = filters["start_date"]
    if filters.get("end_date"):
        clauses.append("timestamp <= :end_date")
        params["end_date"] = filters["end_date"]
    if filters.get("min_amount") is not None:
        clauses.append("amount >= :min_amount")
        params["min_amount"] = filters["min_amount"]
    if filters.get("max_amount") is not None:
        clauses.append("amount <= :max_amount")
        params["max_amount"] = filters["max_amount"]

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params, expanding


def _run(sql: str, params: dict, expanding: list) -> pd.DataFrame:
    stmt = text(sql)
    for k in expanding:  # IN (:list) needs expanding binds
        stmt = stmt.bindparams(bindparam(k, expanding=True))
    return pd.read_sql(stmt, get_engine(), params=params)


# ── Dropdown options + slider bounds (run once, cached on the UI side) ─────
def get_filter_options() -> dict:
    eng = get_engine()
    out = {}
    for col in ("transaction_type", "location", "merchant_category"):
        df = pd.read_sql(
            text(f"SELECT DISTINCT {col} FROM transactions ORDER BY 1"), eng
        )
        out[col] = df[col].dropna().tolist()
    return out


def get_bounds() -> pd.Series:
    sql = """SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts,
                    MIN(amount) AS min_amt, MAX(amount) AS max_amt
             FROM transactions"""
    return _run(sql, {}, []).iloc[0]


# ── Aggregations for the dashboard (all return tiny frames) ────────────────
def get_summary(filters: dict) -> pd.Series:
    where, p, e = _where(filters)
    sql = f"""
        SELECT COUNT(*)                              AS total_txns,
               COUNT(*) FILTER (WHERE is_fraud)      AS fraud_txns,
               COALESCE(SUM(amount), 0)              AS total_amount,
               COALESCE(SUM(amount) FILTER (WHERE is_fraud), 0) AS fraud_amount
        FROM transactions {where}
    """
    return _run(sql, p, e).iloc[0]


def get_fraud_by_type(filters: dict) -> pd.DataFrame:
    where, p, e = _where(filters)
    sql = f"""
        SELECT transaction_type,
               COUNT(*)                         AS txns,
               COUNT(*) FILTER (WHERE is_fraud) AS fraud_txns
        FROM transactions {where}
        GROUP BY transaction_type ORDER BY txns DESC
    """
    return _run(sql, p, e)


def get_daily_volume(filters: dict) -> pd.DataFrame:
    where, p, e = _where(filters)
    sql = f"""
        SELECT date_trunc('day', timestamp) AS day,
               COUNT(*)                         AS txns,
               COUNT(*) FILTER (WHERE is_fraud) AS fraud_txns
        FROM transactions {where}
        GROUP BY 1 ORDER BY 1
    """
    return _run(sql, p, e)


def get_sample(filters: dict, limit: int = 500) -> pd.DataFrame:
    where, p, e = _where(filters)
    sql = f"""
        SELECT transaction_id, timestamp, sender_account, receiver_account,
               amount, transaction_type, merchant_category, location, is_fraud
        FROM transactions {where}
        ORDER BY timestamp DESC
        LIMIT :limit
    """
    return _run(sql, {**p, "limit": limit}, e)
