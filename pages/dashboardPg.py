import streamlit as st
from pipeline import DATASETS


# model dropdown
dataset_name = st.selectbox("Choose a dataset to view", list(DATASETS))