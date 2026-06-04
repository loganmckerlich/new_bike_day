"""Streamlit entry point – sets up navigation between app pages."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(
    page_title="New Bike Day",
    page_icon=":material/pedal_bike:",
    layout="wide",
)

with st.sidebar:
    st.session_state.setdefault("use_metric", True)
    st.session_state["use_metric"] = st.toggle(
        "🌍 Metric units",
        value=st.session_state["use_metric"],
        help="Toggle between metric (km, m) and imperial (mi, ft).",
    )

pg = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
        st.Page(
            "app_pages/data_collection.py",
            title="1 · Data Collection",
            icon=":material/cloud_download:",
        ),
        st.Page(
            "app_pages/data_cleaning.py",
            title="2 · Data Cleaning",
            icon=":material/cleaning_services:",
        ),
        st.Page(
            "app_pages/bike_comparison.py",
            title="3 · Bike Comparison",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "app_pages/final_conclusions.py",
            title="4 · Final Conclusions",
            icon=":material/flag:",
        ),
    ]
)
pg.run()
