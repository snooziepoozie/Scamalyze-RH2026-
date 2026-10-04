# views/runmodel.py
import streamlit as st

import pipeline as pl
import model as ml

st.title("Fraud Model")


# Cache the Tiger scoring on the filters dict so re-runs are instant.
# (Solana isn't cached — a live wallet should re-fetch each run.)
@st.cache_data(ttl=300)
def score_tiger(filters):
    return ml.run_model(filters=filters, use_solana=False)


source = st.radio(
    "Score which data?", ["Tiger Database", "Solana Wallet"], horizontal=True
)

wallet = ""
if source == "Solana Wallet":

    wallet = st.text_input(
        "Wallet Address",
        placeholder="7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU (Enter no input for mock data)",
    )
    st.caption(
        "Solana scoring uses numerical data only"
        "Certain columns don't exist on-chain, so these predictions are illustrative."
    )
else:
    # Show what filters will be applied, inherited from the Dashboard page.
    active = st.session_state.get("filters", {})
    if active:
        st.caption("Scoring the Dashboard's current filtered selection.")
    else:
        st.caption("No dashboard filters set. Scoring the most recent transactions.")


if st.button("Run model", type="primary"):
    try:
        if source == "Solana Wallet":
            with st.spinner("Fetching and scoring Solana transactions…"):
                data = (
                    pl.get_solana_transactions(wallet)
                    if wallet
                    else pl.mock_solana_transactions()
                )
                if data.empty:
                    st.warning("No transactions found for that wallet.")
                    st.stop()
                scored = ml.run_model(use_solana=True, solana_df=data)
        else:
            filters = st.session_state.get("filters", {})
            with st.spinner("Scoring transactions…"):
                scored = score_tiger(filters)

        if scored.empty:
            st.warning("No transactions to score.")
            st.stop()

        flagged = int(scored["predicted_fraud"].sum())
        total = len(scored)
        rate = (flagged / total * 100) if total else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Scored", f"{total:,}")
        c2.metric("Flagged as fraud", f"{flagged:,}")
        c3.metric("Flag rate", f"{rate:.2f}%")

        st.dataframe(
            scored.sort_values("fraud_probability", ascending=False).head(200),
            use_container_width=True,
        )

    except FileNotFoundError as exc:
        st.error(str(exc))
    except RuntimeError as exc:
        # e.g. missing HELIUS_API_KEY
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
