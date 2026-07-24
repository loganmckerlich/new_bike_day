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
        "bike_comparison_segmented",
        "bike_comparison_overall",
        "final_conclusions",
    ]
    if on not in order:
        raise ValueError(f"Unknown page '{on}' for navigator. Expected one of: {order}")
    index = order.index(on)
    next_page = order[index + 1] if index + 1 < len(order) else None
    prev_page = order[index - 1] if index - 1 >= 0 else None

    sideways = st.container(horizontal=True)
    with sideways:
        if prev_page and st.button("←",width='stretch',key=f"back_{on_raw}"):
            st.switch_page(f"app_pages/{prev_page}.py")
        if on != "home" and st.button("🏠",width='stretch',key=f"home_{on_raw}"):
            st.switch_page("app_pages/home.py")
        if next_page and st.button("→",width='stretch',key=f"forward_{on_raw}"):
            st.switch_page(f"app_pages/{next_page}.py")
    if on_raw[-1] == "2":
        st.markdown("All Data Comes From Strava.")
        st.image("src/api_logo_pwrdBy_strava_horiz_orange.png", width=150, link="https://www.strava.com/")

def _redirect(message: str, button_text: str, page: str) -> None:
    st.info(message)
    if st.button(button_text):
        st.switch_page(page)
    st.stop()


def page_guard(page_name: str) -> None:
    requirements = {
        "data_cleaning": ["data_loaded"],
        "bike_comparison_segmented": ["data_loaded", "data_cleaned"],
        "bike_comparison_overall": ["data_loaded", "data_cleaned"],
        "final_conclusions": ["data_loaded", "data_cleaned"],
    }

    raw_efforts = st.session_state.get("efforts")
    cleaned_efforts = st.session_state.get("cleaned_efforts")
    segments = st.session_state.get("segments")

    checks = {
        "data_loaded": (
            raw_efforts is not None
            and not raw_efforts.empty
        ),
        "data_cleaned": (
            cleaned_efforts is not None
            and not cleaned_efforts.empty
        ),
    }

    needed = requirements.get(page_name, [])

    if "data_loaded" in needed and not checks["data_loaded"]:
        _redirect(
            "Head to **Step 1 — Data Collection** to load your Strava data first.",
            "Go to Step 1",
            "app_pages/data_collection.py",
        )

    if "data_cleaned" in needed and not checks["data_cleaned"]:
        _redirect(
            "Head to **Step 2 — Data Cleaning** to configure and apply filters first.",
            "Go to Step 2",
            "app_pages/data_cleaning.py",
        )

    if segments is None or segments.empty:
        st.warning(
            "No starred segments found. Star some segments on Strava and reload from Step 1."
        )
        st.stop()

    efforts_with_power = raw_efforts[
        raw_efforts["average_watts"].notna()
    ]

    if efforts_with_power.empty:
        st.warning(
            "No efforts with power data found. Ensure your rides are recorded with a power meter."
        )
        st.stop()