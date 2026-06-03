"""Bike Comparison page – tab host for Overall and Segmented analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app.app_pages.bike_comparison_overall as _overall
import app.app_pages.bike_comparison_segmented as _segmented
from app.app_pages._ui_helpers import (
    gear_label,
)

# ── Page title ────────────────────────────────────────────────────────────

st.title("📊 Step 3 — Bike Comparison")
st.markdown(
    "Filters and cleaning are already applied (configured in **Step 2 — Data Cleaning**). "
    "Select bikes and segments below to compare performance."
)

# ── Guards ────────────────────────────────────────────────────────────────────
if st.session_state.get("efforts") is None:
    st.info("👈 Head to **Step 1 — Data Collection** to sign in with Strava and load your data first.")
    st.stop()

_efforts = st.session_state.get("cleaned_efforts")
if _efforts is None or (hasattr(_efforts, "empty") and _efforts.empty):
    st.info("👈 Head to **Step 2 — Data Cleaning** to configure and apply data filters first.")
    st.stop()

_segments = st.session_state.get("segments")
if _segments is None or (hasattr(_segments, "empty") and _segments.empty):
    st.warning("No starred segments found. Star some segments on Strava and reload.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_segmented, tab_overall = st.tabs(["📍 Segmented","📈 Overall"])

bikes = st.session_state.get("bikes", {})
efforts = st.session_state.get("cleaned_efforts")
watt_efforts = efforts[efforts["average_watts"].notna()].copy()
watt_efforts["bike_name"] = watt_efforts["gear_id"].map(lambda g: gear_label(g, bikes))
available_bikes = sorted(watt_efforts["bike_name"].dropna().unique().tolist())
bikes_to_compare = st.sidebar.multiselect(
            "Bikes to compare",
            options=available_bikes,
            default=available_bikes[:2],
            max_selections=5,
            help="Select 2–5 bikes to compare.",
        )

min_efforts = st.sidebar.number_input(
    "Min efforts per bike per segment",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    help="Both bikes must have at least this many power-measured efforts on a segment.",
)

with tab_segmented:
    _segmented.show(bikes_to_compare, min_efforts)

with tab_overall:
    _overall.show(bikes_to_compare, min_efforts)