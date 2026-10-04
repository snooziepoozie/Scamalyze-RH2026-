import streamlit as st
import pipeline as pl
from pathlib import Path

# inside static (just incase)
IMAGE_DIR = Path(__file__).resolve().parent / "static"

st.set_page_config(page_title="Scamalyze", layout="wide")

# Connection check runs once, before any page loads.
if "db_ok" not in st.session_state:
    st.session_state.db_ok, st.session_state.db_msg = pl.check_connection()

if not st.session_state.db_ok:
    st.error("Can't connect to the TigerData database.")
    st.caption(st.session_state.db_msg)
    st.stop()

# sidebar
with st.sidebar:
    st.title("Scamalyze")
    st.image(
        "https://media1.tenor.com/m/viWX9VvemBkAAAAd/cat-silly.gif",
        width=300,
    )

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
    st.Page("views/runmodel.py", title="Fraud Model", icon=":material/smart_toy:"),
]


st.navigation(pages).run()
