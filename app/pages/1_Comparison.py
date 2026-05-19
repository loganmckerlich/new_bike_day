"""Streamlit comparison page – coming soon."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure `src` imports work when launching Streamlit from different working directories.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(page_title="Comparison – New Bike Day", layout="wide")
st.title("🏆 Bike Comparison")

activities = st.session_state.get("activities")
if activities is None or (hasattr(activities, "empty") and activities.empty):
    st.info("Head to the **Home** page to sign in with Strava and load your activities first.")
else:
    st.info("Comparison features are coming soon. Stay tuned!")
