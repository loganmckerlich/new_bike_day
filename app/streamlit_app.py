"""Streamlit entry point – sets up navigation between app pages."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auth import exchange_code
from src.database import init_db, save_athlete_token
from src.utils import normalized_redirect_uri

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

# ---------------------------------------------------------------------------
# OAuth callback — intercepts ?code= on any page so the token is captured
# even when Strava redirects to the root URL (Home page).
# ---------------------------------------------------------------------------
_oauth_code = st.query_params.get("code")
_oauth_error = st.query_params.get("error")

if _oauth_error:
    st.session_state["oauth_error"] = str(_oauth_error)
    st.query_params.clear()
elif _oauth_code and str(_oauth_code) != st.session_state.get("last_processed_code"):
    _client_id = st.secrets.get("STRAVA_CLIENT_ID", "")
    _client_secret = st.secrets.get("STRAVA_CLIENT_SECRET", "")
    _redirect_uri = normalized_redirect_uri(
        st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501")
    )
    if _client_id and _client_secret:
        try:
            _token = exchange_code(_client_id, _client_secret, str(_oauth_code), _redirect_uri)
            st.session_state["access_token"] = _token["access_token"]
            st.session_state["last_processed_code"] = str(_oauth_code)
            if _token.get("athlete_firstname"):
                st.session_state["athlete_name"] = _token["athlete_firstname"]
            if _token.get("athlete_id"):
                init_db()
                save_athlete_token(
                    athlete_id=_token["athlete_id"],
                    access_token=_token["access_token"],
                    refresh_token=_token["refresh_token"],
                    expires_at=_token["expires_at"],
                )
        except Exception:  # noqa: BLE001
            pass
    st.query_params.clear()

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
        # st.Page(
        #     "pages/causal_analysis.py",
        #     title="4 · Bike Head to Head",
        #     icon=":material/science:",
        # ),
        # st.Page(
        #     "pages/cda_ranking.py",
        #     title="5 · CdA Estimation",
        #     icon=":material/air:",
        # ),
        st.Page(
            "app_pages/final_conclusions.py",
            title="4 · Final Conclusions",
            icon=":material/flag:",
        ),
    ]
)
pg.run()
