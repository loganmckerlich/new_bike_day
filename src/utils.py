from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import streamlit as st


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
