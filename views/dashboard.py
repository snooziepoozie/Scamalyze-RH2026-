# views/dashboard.py
import streamlit as st

import pipeline as pl

st.title("Dashboard")


# ── Cached data loaders ──────────────────────────────────────────────────────
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


# ── Export dialog ────────────────────────────────────────────────────────────
@st.dialog("Export Data")
def export_dialog(filters):
    with st.spinner("Querying filtered transactions…"):
        export_df = pl.get_filtered_export(filters, 100_000)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    n = len(export_df)
    size_kb = len(csv_bytes) / 1024
    st.success("Download is ready!")
    st.markdown(f"**{n:,} rows**  ·  {size_kb:,.1f} KB")
    if n == 100_000:
        st.caption(
            "Capped at 100,000 rows — narrow your filters for a complete export."
        )

    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )


# ── Filters (in a bordered panel on the main page) ───────────────────────────
with st.container(border=True):
    st.markdown("**Filters**")
    f1, f2, f3 = st.columns(3)
    txn_types = f1.multiselect("Transaction type", options["transaction_type"])
    locations = f2.multiselect("Location", options["location"])
    cats = f3.multiselect("Merchant category", options["merchant_category"])

    f4, f5 = st.columns([2, 3])
    fraud_filter = f4.radio(
        "Show",
        ["all", "fraud", "legit"],
        format_func=lambda x: {
            "all": "All transactions",
            "fraud": "Fraud only",
            "legit": "Legitimate only",
        }[x],
        horizontal=True,
    )
    date_range = f5.date_input("Date range", value=(bounds["min_ts"], bounds["max_ts"]))

    min_amt, max_amt = st.slider(
        "Amount range",
        float(bounds["min_amt"]),
        float(bounds["max_amt"]),
        (float(bounds["min_amt"]), float(bounds["max_amt"])),
    )


# ── Assemble filters dict ────────────────────────────────────────────────────
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

# Share filters with the model page.
st.session_state["filters"] = filters


# ── Metrics ──────────────────────────────────────────────────────────────────
s = summary(filters)
total, fraud = int(s["total_txns"]), int(s["fraud_txns"])
rate = (fraud / total * 100) if total else 0.0

m1, m2, m3, m4 = st.columns(4)
with m1.container(border=True):
    st.metric("Transactions", f"{total:,}")
with m2.container(border=True):
    st.metric("Fraudulent", f"{fraud:,}")
with m3.container(border=True):
    st.metric("Fraud rate", f"{rate:.2f}%")
with m4.container(border=True):
    st.metric("Total volume", f"${s['total_amount']:,.0f}")

if total == 0:
    st.info("No transactions match these filters.")
    st.stop()


# ── Charts ───────────────────────────────────────────────────────────────────
ch1, ch2 = st.columns(2)
with ch1.container(border=True):
    st.subheader("By transaction type")
    chart = by_type(filters).set_index("transaction_type")[["txns", "fraud_txns"]]
    st.bar_chart(chart)
with ch2.container(border=True):
    st.subheader("Daily volume")
    chart = daily(filters).set_index("day")[["txns", "fraud_txns"]]
    st.line_chart(chart)

# table
st.subheader("Sample (Top 500)")
st.dataframe(sample(filters, 500), use_container_width=True)


# ── Export dialog ────────────────────────────────────────────────────────────
@st.dialog("Export to CSV")
def export_dialog(filters):
    # Runs inside the modal. Query first, showing a spinner, then reveal download.
    with st.spinner("Querying filtered transactions…"):
        export_df = pl.get_filtered_export(filters, 100_000)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    n = len(export_df)
    size_kb = len(csv_bytes) / 1024

    st.success("Download ready!")
    st.markdown(f"**{n:,} rows**  ·  {size_kb:,.1f} KB")
    if n == 100_000:
        st.caption("Capped at 100,000 rows. Narrow your filters for a complete export.")

    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )


# In the sidebar, the trigger:
if st.button("Export Data", icon=":material/download:", use_container_width=True):
    export_dialog(filters)
