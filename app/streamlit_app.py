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

    st.divider()
    st.markdown("### ⚙️ Analysis settings")

    # Z-score threshold — shared between Segment comparison and Causal analysis
    st.session_state.setdefault("outlier_z_threshold", 2.0)
    st.slider(
        "Outlier z-score threshold",
        min_value=0.25,
        max_value=3.5,
        step=0.25,
        key="outlier_z_threshold",
        help=(
            "Z-score = how many standard deviations an effort's speed/W^(1/3) sits "
            "from that bike's mean on this segment. Efforts beyond this threshold "
            "are removed as outliers. Lower = more aggressive filtering. "
            "Applies to both Segment comparison and Causal analysis."
        ),
    )

    # Minimum watts — efforts below this threshold are excluded
    ftp: int | None = st.session_state.get("ftp")
    _default_min_watts = int(round(ftp * 0.75)) if ftp is not None and ftp > 0 else 0
    st.session_state.setdefault("min_watts", _default_min_watts)
    st.number_input(
        "Minimum watts",
        min_value=0,
        max_value=999,
        step=5,
        key="min_watts",
        help=(
            "Efforts with average power below this threshold are excluded from "
            "analysis. Default is 75 % of your FTP. Set to 0 to disable."
        ),
    )

    # Descent exemption — descents can bypass the watts minimum
    st.session_state.setdefault("descents_exempt_watts", False)
    st.checkbox(
        "Descents exempt from min watts",
        key="descents_exempt_watts",
        help=(
            "When checked, descent segments are not subject to the minimum-watts "
            "filter (useful because descents naturally have lower power output)."
        ),
    )

    # Exclude descents — remove descent segments from all analysis
    st.session_state.setdefault("exclude_descents", False)
    st.checkbox(
        "Exclude descent segments",
        key="exclude_descents",
        help=(
            "Remove descent segments entirely from Segment comparison and "
            "Causal analysis. Descents are often wind-dominated and can skew results."
        ),
    )

pg = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
        st.Page(
            "app_pages/segment_comparison.py",
            title="Segment comparison",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pages/causal_analysis.py",
            title="Causal analysis",
            icon=":material/science:",
        ),
        st.Page(
            "pages/cda_ranking.py",
            title="CdA ranking",
            icon=":material/air:",
        ),
    ]
)
pg.run()
