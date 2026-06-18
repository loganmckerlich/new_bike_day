"""Bike Comparison page – tab host for Overall and Segmented analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app.app_pages.subpages.bike_comparison_overall as _overall
import app.app_pages.subpages.bike_comparison_segmented as _segmented
from src._ui_helpers import (
    gear_label,
)
from src.utils import navigator, page_guard

# ── Page title ────────────────────────────────────────────────────────────

def comp_inputs():
    bikes = st.session_state.get("bikes", {})
    efforts = st.session_state.get("cleaned_efforts")
    watt_efforts = efforts[efforts["average_watts"].notna()].copy()
    watt_efforts["bike_name"] = watt_efforts["gear_id"].map(lambda g: gear_label(g, bikes))
    available_bikes = sorted(watt_efforts["bike_name"].dropna().unique().tolist())
    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=available_bikes[:2],
        max_selections=2,
        help="Select 2 bikes to compare.",
        # in future allow this to be more than 2, spider plots already ready for that
    )

    min_efforts = st.number_input(
        "Min efforts per bike per segment",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Both bikes must have at least this many power-measured efforts on a segment.",
    )
    return bikes_to_compare, min_efforts

def main() -> None:
    st.title("📊 Step 3 — Bike Comparison")
    st.markdown(
        "Filters and cleaning are already applied (configured in **Step 2 — Data Cleaning**). "
        "Select bikes and segments below to compare performance."
    )

    bikes_to_compare, min_efforts = comp_inputs()

    page_guard("bike_comparison")

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_segmented, tab_overall = st.tabs(["📍 Segmented","📈 Overall"])

    with tab_segmented:
        _segmented.show(bikes_to_compare, min_efforts)

    with tab_overall:
        _overall.show(bikes_to_compare, min_efforts)

navigator("bike_comparison1")
main()
navigator("bike_comparison2")