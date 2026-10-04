# run via `streamlit run main.py`
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Fraud Analytics", layout="wide")
st.title("Fraud Prediction Dashboard")


# ── Connection state (checked once per session) ────────────────────────────
if "db_ok" not in st.session_state:
    st.session_state.db_ok, st.session_state.db_msg = pl.check_connection()

if not st.session_state.db_ok:
    st.error("Can't connect to the TigerData database.")
    msg = st.session_state.db_msg.lower()
    if "password authentication failed" in msg:
        st.info("The password is being rejected — check TIGER_DB_URL in your .env.")
    elif "timed out" in msg:
        st.info("Connection timed out — your IP may not be on the allow list.")
    elif "refused" in msg:
        st.info(
            "Connection refused — check the service is Running and the port is correct."
        )
    st.caption(st.session_state.db_msg)
    st.stop()


# ── Cached data loaders ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_options():
    return pl.get_filter_options()


@st.cache_data(ttl=3600)
def load_bounds():
    return pl.get_bounds()


@st.cache_data(ttl=300)
def summary(filters):
    return pl.get_summary(filters)


@st.cache_data(ttl=300)
def by_type(filters):
    return pl.get_fraud_by_type(filters)


@st.cache_data(ttl=300)
def daily(filters):
    return pl.get_daily_volume(filters)


@st.cache_data(ttl=300)
def sample(filters, n):
    return pl.get_sample(filters, n)


options, bounds = load_options(), load_bounds()


# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Filters")
txn_types = st.sidebar.multiselect("Transaction type", options["transaction_type"])
locations = st.sidebar.multiselect("Location", options["location"])
cats = st.sidebar.multiselect("Merchant category", options["merchant_category"])
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
    filters["start_date"] = str(date_range[0])
    filters["end_date"] = str(date_range[1])


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
    chart = by_type(filters).set_index("transaction_type")[["txns", "fraud_txns"]]
    st.bar_chart(chart)
with right:
    st.subheader("Daily volume")
    chart = daily(filters).set_index("day")[["txns", "fraud_txns"]]
    st.line_chart(chart)


# ── Row-level sample (capped) ──────────────────────────────────────────────
st.subheader("Sample transactions (most recent 500 matching)")
st.dataframe(sample(filters, 500), use_container_width=True)
