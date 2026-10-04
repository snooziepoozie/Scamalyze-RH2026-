import streamlit as st
import pipeline as pl
import model as ml

st.title("Fraud Model")

source = st.radio(
    "Score which data?", ["Tiger (filtered)", "Solana account"], horizontal=True
)

solana_df = None
if source == "Solana account":
    st.caption(
        "Solana scoring uses amount only — type, merchant, and location "
        "don't exist on-chain."
    )
    wallet = st.text_input("Wallet address (blank = mock data)")

if st.button("Run model", type="primary"):
    try:
        if source == "Solana account":
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
            # Dashboard filters live in session_state if you stash them there;
            # otherwise score unfiltered or add filters to this page.
            scored = ml.run_model(filters={}, use_solana=False)

        flagged = int(scored["predicted_fraud"].sum())
        st.success(f"Scored {len(scored):,} transactions — {flagged:,} flagged.")
        st.dataframe(
            scored.sort_values("fraud_probability", ascending=False).head(200),
            use_container_width=True,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
