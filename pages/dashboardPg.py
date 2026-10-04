import streamlit as st
import pipeline as pl


st.title("Transaction sample")
st.caption("Showing the most recent transactions from TigerData.")
st.dataframe(pl.get_sample({}), use_container_width=True)