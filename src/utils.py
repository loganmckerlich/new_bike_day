from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import streamlit as st
import pandas as pd


def normalized_redirect_uri(raw_value: str) -> str:
    """Normalise a redirect URI: ensures a trailing slash if no path is given."""
    value = raw_value.strip() if raw_value else ""
    if not value:
        return "http://localhost:8501/"
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc and not parsed.path:
        return urlunsplit((parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment))
    return value


def link_button_no_tab(label: str, url: str):
    st.markdown(
        f"""<a href="{url}" target="_self" style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            background-color: #FC4C02;;
            color: inherit;
            text-decoration: none;
            font-size: 0.875rem;
            font-family: sans-serif;
            cursor: pointer;
        ">{label}</a>""",
        unsafe_allow_html=True,
    )

def navigator(on_raw):
    on = on_raw[:-1]
    order = [
        "home",
        "data_collection",
        "data_cleaning",
        "bike_comparison",
        "final_conclusions",
    ]
    if on not in order:
        raise ValueError(f"Unknown page '{on}' for navigator. Expected one of: {order}")
    index = order.index(on)
    next_page = order[index + 1] if index + 1 < len(order) else None
    prev_page = order[index - 1] if index - 1 >= 0 else None

    sideways = st.container(horizontal=True)
    with sideways:
        if prev_page and st.button("←",use_container_width=True,key=f"back_{on_raw}"):
            st.switch_page(f"app_pages/{prev_page}.py")
        if on != "home" and st.button("🏠",use_container_width=True,key=f"home_{on_raw}"):
            st.switch_page("app_pages/home.py")
        if next_page and st.button("→",use_container_width=True,key=f"forward_{on_raw}"):
            st.switch_page(f"app_pages/{next_page}.py")

def page_guard():
    # ── Guard: data must be loaded ─────────────────────────────────────────────
    raw_efforts: pd.DataFrame | None = st.session_state.get("efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if raw_efforts is None or (hasattr(raw_efforts, "empty") and raw_efforts.empty):
        st.info("Head to **Step 1 — Data Collection** to sign in with Strava and load your data first.")
        if st.button("Go to Step 1"):
            st.switch_page("app_pages/data_collection.py")
        st.stop()

    cleaned_efforts = st.session_state.get("cleaned_efforts")
    if cleaned_efforts is None or (hasattr(cleaned_efforts, "empty") and cleaned_efforts.empty):
        st.info("Head to **Step 2 — Data Cleaning** to configure and apply data filters first.")
        if st.button("Go to Step 2"):
            st.switch_page("app_pages/data_cleaning.py")
        st.stop()

    if segments is None or segments.empty:
        st.warning("No starred segments found. Star some segments on Strava and reload from Step 1.")
        st.stop()

    # Keep only power efforts
    efforts_with_power = raw_efforts[raw_efforts["average_watts"].notna()].copy()
    if efforts_with_power.empty:
        st.warning("No efforts with power data found. Ensure your rides are recorded with a power meter.")
        st.stop()