# pipeline.py — all TigerData access lives here
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

# Load the project-local .env regardless of where Streamlit was launched from.
load_dotenv(Path(__file__).resolve().with_name(".env"))


# ── Connection (shared engine = module-level state) ────────────────────────
def _db_url() -> str:
    url = os.environ.get("TIGER_DB_URL")
    if not url:
        raise RuntimeError("Set TIGER_DB_URL in your .env file.")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine():
    """Single shared engine — built once, reused for every query."""
    return create_engine(
        _db_url(),
        pool_pre_ping=True,  # silently revives a dropped connection
        pool_size=5,
        max_overflow=2,
        connect_args={"connect_timeout": 10},
    )


def check_connection() -> tuple[bool, str]:
    """Probe the DB once and report state. Call at app startup."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as exc:
        return False, str(exc)


def current_public_ip() -> str:
    """Call only when debugging an allow-list issue — not at import time."""
    import urllib.request

    ip = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
    block = ip[: ip.rfind(".")] + ".0/24"
    print(f"Current public IP: {ip}  (add to TigerData allow list as {block})")
    return ip


def get_solana_transactions(wallet: str, limit: int = 100) -> pd.DataFrame:
    """Fetch a wallet's parsed transactions from Helius into the app's schema.

    Returns the 8 columns the model/dashboard expect. The three off-chain
    columns (transaction_type is real-ish; merchant_category, location) are
    filled with placeholders since Solana has no equivalent — the model
    ignores them anyway.
    """
    import requests

    api_key = os.environ.get("HELIUS_API_KEY")
    if not api_key:
        raise RuntimeError("Set HELIUS_API_KEY in your .env file.")

    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    resp = requests.get(url, params={"api-key": api_key, "limit": limit}, timeout=15)
    resp.raise_for_status()
    txns = resp.json()

    rows = []
    for tx in txns:
        sig = tx.get("signature")
        ts = (
            pd.to_datetime(tx.get("timestamp"), unit="s")
            if tx.get("timestamp")
            else pd.NaT
        )
        tx_type = tx.get("type", "UNKNOWN")

        transfers = tx.get("nativeTransfers") or []
        if transfers:
            for t in transfers:
                rows.append(
                    {
                        "transaction_id": sig,
                        "timestamp": ts,
                        "sender_account": t.get("fromUserAccount", "unknown"),
                        "receiver_account": t.get("toUserAccount", "unknown"),
                        "amount": (t.get("amount", 0) or 0) / 1e9,  # lamports -> SOL
                        "transaction_type": tx_type,
                        "merchant_category": None,  # no on-chain equivalent
                        "location": None,  # no on-chain equivalent
                        "is_fraud": False,  # unlabeled — placeholder
                    }
                )
        else:
            # Keep non-transfer transactions (swaps, etc.) as a single row.
            rows.append(
                {
                    "transaction_id": sig,
                    "timestamp": ts,
                    "sender_account": tx.get("feePayer", "unknown"),
                    "receiver_account": "unknown",
                    "amount": (tx.get("fee", 0) or 0) / 1e9,
                    "transaction_type": tx_type,
                    "merchant_category": None,
                    "location": None,
                    "is_fraud": False,
                }
            )

    cols = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "transaction_type",
        "merchant_category",
        "location",
        "is_fraud",
    ]
    return pd.DataFrame(rows, columns=cols)


# ── Filter handling ────────────────────────────────────────────────────────
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

    fraud_filter = filters.get("fraud_filter", "all")
    if fraud_filter == "fraud":
        clauses.append("is_fraud = true")
    elif fraud_filter == "legit":
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
    for key in expanding:  # IN (:list) needs an expanding bind
        stmt = stmt.bindparams(bindparam(key, expanding=True))
    return pd.read_sql(stmt, get_engine(), params=params)


# ── Filter options and bounds (rarely change) ──────────────────────────────
def get_filter_options() -> dict:
    engine = get_engine()
    options = {}
    for col in ("transaction_type", "location", "merchant_category"):
        df = pd.read_sql(
            text(f"SELECT DISTINCT {col} FROM transactions ORDER BY 1"), engine
        )
        options[col] = df[col].dropna().tolist()
    return options


def get_bounds() -> pd.Series:
    sql = """
        SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts,
               MIN(amount)    AS min_amt, MAX(amount)   AS max_amt
        FROM transactions
    """
    return _run(sql, {}, []).iloc[0]


# ── Aggregations (all return small frames) ─────────────────────────────────
def get_summary(filters: dict) -> pd.Series:
    where, params, expanding = _where(filters)
    sql = f"""
        SELECT COUNT(*)                                         AS total_txns,
               COUNT(*) FILTER (WHERE is_fraud)                 AS fraud_txns,
               COALESCE(SUM(amount), 0)                         AS total_amount,
               COALESCE(SUM(amount) FILTER (WHERE is_fraud), 0) AS fraud_amount
        FROM transactions {where}
    """
    return _run(sql, params, expanding).iloc[0]


def get_fraud_by_type(filters: dict) -> pd.DataFrame:
    where, params, expanding = _where(filters)
    sql = f"""
        SELECT transaction_type,
               COUNT(*)                         AS txns,
               COUNT(*) FILTER (WHERE is_fraud) AS fraud_txns
        FROM transactions {where}
        GROUP BY transaction_type
        ORDER BY txns DESC
    """
    return _run(sql, params, expanding)


def get_daily_volume(filters: dict) -> pd.DataFrame:
    where, params, expanding = _where(filters)
    sql = f"""
        SELECT date_trunc('day', timestamp)     AS day,
               COUNT(*)                         AS txns,
               COUNT(*) FILTER (WHERE is_fraud) AS fraud_txns
        FROM transactions {where}
        GROUP BY 1
        ORDER BY 1
    """
    return _run(sql, params, expanding)


def get_sample(filters: dict, limit: int = 500) -> pd.DataFrame:
    where, params, expanding = _where(filters)
    sql = f"""
        SELECT transaction_id, timestamp, sender_account, receiver_account,
               amount, transaction_type, merchant_category, location, is_fraud
        FROM transactions {where}
        ORDER BY timestamp DESC
        LIMIT :limit
    """
    return _run(sql, {**params, "limit": limit}, expanding)


# incase we have no address
def mock_solana_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": "3A7d8F9g",
                "timestamp": pd.Timestamp.now(),
                "sender_account": "3uv7g8",
                "receiver_account": "8vY1b9",
                "amount": 0.05,
                "transaction_type": "transfer",
                "merchant_category": None,
                "location": None,
                "is_fraud": False,
            },
            {
                "transaction_id": "8Hj2Kl9s",
                "timestamp": pd.Timestamp.now(),
                "sender_account": "9xR4f1",
                "receiver_account": "3uv7g8",
                "amount": 12.4,
                "transaction_type": "transfer",
                "merchant_category": None,
                "location": None,
                "is_fraud": False,
            },
        ]
    )


# DOWNLOAD NOW!!!!


def get_filtered_export(filters: dict, max_rows: int = 100_000) -> pd.DataFrame:
    """Full filtered set for CSV download, capped so the browser doesn't choke."""
    where, params, expanding = _where(filters)
    sql = f"""
        SELECT transaction_id, timestamp, sender_account, receiver_account,
               amount, transaction_type, merchant_category, location, is_fraud
        FROM transactions {where}
        ORDER BY timestamp DESC
        LIMIT :limit
    """
    return _run(sql, {**params, "limit": max_rows}, expanding)
