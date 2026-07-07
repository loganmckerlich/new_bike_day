"""Home page: landing page explaining the New Bike Day concept."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from src.utils import navigator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    st.title("🚴 New Bike Day")
    st.markdown(
        """
        **New Bike Day** helps cyclists answer one question: *does your new bike actually make you faster?*

        Unless you're a professional cyclist you'll probably never be able to test your bike in the wind 
        tunnel and get a straight up answer on which one is faster. With this project I am aiming to use machine learning to help you
        determine which of the bikes you have ridden is the fastest one, and by how much.

        ---
        They all claim to be the fastest, but there can only be one!
        """)
    st.image("src/bike_comp.jpeg")
    st.markdown(
        """
        ---
        ### How it works

        The analysis is broken into 3 steps:

        1. **Data Collection** — Sign in with Strava. We pull your segment efforts, bikes, and starred
           segments from the Strava API and cache them locally.

        2. **Data Cleaning** — Before any analysis, we remove noisy efforts: those with suspiciously low
           power (e.g. coasting, technical issues) and statistical outliers detected by comparing each
           effort's speed watt effeciency against the segment average. You can trim the thresholds based on
           the visualizations to see what feels right.

        3. **Bike Comparison Segment Mode** — Spider charts and head-to-head tables let you compare your bikes across
           segment types (sprints, flats, climbs, descents). Which bike is strongest on which terrain?
               
        4. **Bike Comparison Overall Mode** — Treating residuals as counterfactuals in a pseudo DML style analysis to compare
           bikes controlling for as many factors as we can get data on. Which bike is strongest overall?

           - The idea here is that ```Speed = <A bunch of factors> + bike``` so if we can represent as many of those
           factors as possible using data from strava, then we can solve for ```bike```.
           - This is a more technical bike v bike comparison.

        5. **Final Conclusions** — What does fast really mean?

        ---
        First let us know if you want metric or imperial units:
        """
    )
    st.session_state.setdefault("use_metric", True)
    metric_toggle = st.toggle(
        "🌍 Metric units",
        value=st.session_state.get("use_metric", True),
        help="Toggle between metric (km, m) and imperial (mi, ft).",
    )
    if metric_toggle != st.session_state["use_metric"]: 
        st.session_state["use_metric"] = metric_toggle
        st.rerun()

navigator("home1")
main()
navigator("home2")
