import streamlit as st
import pipeline as pl
from model import train_model

st.set_page_config(page_title="Scamalyze", layout="wide")

# Connection check runs once, before any page loads.
if "db_ok" not in st.session_state:
    st.session_state.db_ok, st.session_state.db_msg = pl.check_connection()

if not st.session_state.db_ok:
    st.error("Can't connect to the TigerData database.")
    st.caption(st.session_state.db_msg)
    st.stop()

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
    st.Page("views/runmodel.py", title="Fraud Model", icon=":material/smart_toy:"),
]
st.navigation(pages).run()
