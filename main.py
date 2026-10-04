# run via `streamlit run main.py`

from functools import lru_cache
from sqlalchemy import create_engine
import streamlit as st
import pipeline as pl


st.set_page_config(page_title="Fraud Analytics", layout="wide")
st.title("Transaction Fraud Dashboard")


# error handling
@lru_cache(maxsize=1)
def get_engine():
    return create_engine(
        pl._db_url(),
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )

# Options/bounds change rarely — cache so reruns don't re-query them.
@st.cache_data(ttl=3600)
def load_options():
    return pl.get_filter_options()


@st.cache_data(ttl=3600)
def load_bounds():
    return pl.get_bounds()


# Aggregations depend on filters — cache keyed on the filter dict.
@st.cache_data(ttl=300)
def summary(f):
    return pl.get_summary(f)


@st.cache_data(ttl=300)
def by_type(f):
    return pl.get_fraud_by_type(f)


@st.cache_data(ttl=300)
def daily(f):
    return pl.get_daily_volume(f)


@st.cache_data(ttl=300)
def sample(f, n):
    return pl.get_sample(f, n)


opts, bounds = load_options(), load_bounds()

# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Filters")
txn_types = st.sidebar.multiselect("Transaction type", opts["transaction_type"])
locations = st.sidebar.multiselect("Location", opts["location"])
cats = st.sidebar.multiselect("Merchant category", opts["merchant_category"])
fraud_filter = st.sidebar.radio(
    "Show",
    ["all", "fraud", "legit"],
    format_func=lambda x: {
        "all": "All transactions",
        "fraud": "Fraud only",
        "legit": "Legitimate only",
    }[x],
)
date_range = st.sidebar.date_input(
    "Date range", value=(bounds["min_ts"], bounds["max_ts"])
)
min_amt, max_amt = st.sidebar.slider(
    "Amount range",
    float(bounds["min_amt"]),
    float(bounds["max_amt"]),
    (float(bounds["min_amt"]), float(bounds["max_amt"])),
)

filters = {
    "txn_types": txn_types,
    "locations": locations,
    "merchant_categories": cats,
    "fraud_filter": fraud_filter,
    "min_amount": min_amt,
    "max_amount": max_amt,
}
if isinstance(date_range, tuple) and len(date_range) == 2:
    filters["start_date"], filters["end_date"] = str(date_range[0]), str(date_range[1])

# ── Metrics ────────────────────────────────────────────────────────────────
s = summary(filters)
total, fraud = int(s["total_txns"]), int(s["fraud_txns"])
rate = (fraud / total * 100) if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{total:,}")
c2.metric("Fraudulent", f"{fraud:,}")
c3.metric("Fraud rate", f"{rate:.2f}%")
c4.metric("Total volume", f"${s['total_amount']:,.0f}")

if total == 0:
    st.info("No transactions match these filters.")
    st.stop()

# ── Charts ─────────────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("By transaction type")
    t = by_type(filters).set_index("transaction_type")[["txns", "fraud_txns"]]
    st.bar_chart(t)
with right:
    st.subheader("Daily volume")
    d = daily(filters).set_index("day")[["txns", "fraud_txns"]]
    st.line_chart(d)

# ── Row-level sample (capped) ──────────────────────────────────────────────
st.subheader("Sample transactions (most recent 500 matching)")
st.dataframe(sample(filters, 500), use_container_width=True)
