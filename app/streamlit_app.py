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
    initial_sidebar_state = "auto"
)

with st.sidebar:
    metric_toggle = st.toggle(
        "🌍 Metric units",
        value=st.session_state.get("use_metric", True),
        help="Toggle between metric (km, m) and imperial (mi, ft).",
    )
    if metric_toggle != st.session_state.get("use_metric", True):
        st.session_state["use_metric"] = metric_toggle
        st.rerun()

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
            "app_pages/bike_comparison_segmented.py",
            title="3 · Bike Comparison (Segmented)",
            icon=":material/laps:",
        ),
        st.Page(
            "app_pages/bike_comparison_overall.py",
            title="4 · Bike Comparison (Overall)",
            icon=":material/sports:",
        ),
        st.Page(
            "app_pages/final_conclusions.py",
            title="5 · Final Conclusions",
            icon=":material/flag:",
        ),
    ]
)
pg.run()
