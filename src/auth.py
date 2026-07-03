"""Authentication helpers for Strava API access."""

from __future__ import annotations

from urllib.parse import urlencode
from typing import Final
import streamlit as st
import requests

TOKEN_URL: Final[str] = "https://www.strava.com/oauth/token"
AUTHORIZE_URL: Final[str] = "https://www.strava.com/oauth/authorize"

def link_button_same_tab(
    label: str,
    url: str,
    *,
    key: str | None = None,
    use_container_width: bool = False,
    type: str = "secondary",  # "primary" | "secondary"
    disabled: bool = False,
) -> None:
    """
    Drop-in-ish replacement for st.link_button that navigates in the SAME tab
    instead of opening a new one, while keeping the native button styling.

    Implementation notes:
    - st.link_button renders an <a data-testid="stLinkButton"> with target="_blank".
    - st.html() inserts HTML via innerHTML, so <script> tags are inert -- they
      won't execute. We route the JS through an onerror handler on a hidden
      <img> instead, since attribute-based event handlers DO run on insertion.
    - We poll briefly for the matching anchor (Streamlit's React render can
      lag a tick behind our script), match on href + visible text so we only
      patch the right button even if multiple link buttons share a URL, mark
      it as patched, strip target="_blank", and intercept the click to do a
      plain window.location.href navigation instead.
    """
    # Render the real, natively-styled link button.
    st.link_button(
        label,
        url,
        use_container_width=use_container_width,
        type=type,
        disabled=disabled,
    )

    if disabled:
        return  # nothing to intercept

    uid = f"lb_{key or uuid.uuid4().hex[:8]}"
    safe_url = html_lib.escape(url, quote=True)
    safe_label = html_lib.escape(label, quote=True)

    st.html(f"""
    <img src="x" alt="" style="display:none" data-lb-marker="{uid}" onerror="
        (function() {{
            var target = '{safe_url}';
            var wantedLabel = '{safe_label}';
            var attempts = 0;

            function tryPatch() {{
                attempts++;
                var anchors = document.querySelectorAll('[data-testid=\"stLinkButton\"] a');
                for (var i = 0; i < anchors.length; i++) {{
                    var a = anchors[i];
                    if (a.dataset.lbPatched) continue;
                    var hrefMatches = a.href === target || a.getAttribute('href') === target;
                    var textMatches = (a.textContent || '').trim() === wantedLabel;
                    if (hrefMatches && textMatches) {{
                        a.dataset.lbPatched = 'true';
                        a.removeAttribute('target');
                        a.removeAttribute('rel');
                        a.addEventListener('click', function(e) {{
                            e.preventDefault();
                            e.stopPropagation();
                            window.location.href = target;
                        }});
                        return true;
                    }}
                }}
                return false;
            }}

            if (!tryPatch() && attempts < 20) {{
                var iv = setInterval(function() {{
                    if (tryPatch() || attempts >= 20) clearInterval(iv);
                }}, 50);
            }}
        }})();
    ">
    """)

def custom_auth_button() -> None:
    auth_url = get_authorization_url(
        client_id=st.secrets["STRAVA_CLIENT_ID"],
        redirect_uri=st.secrets["STRAVA_REDIRECT_URI"],
        scope="read,activity:read_all,profile:read_all",
    )

    st.markdown("### Connect your Strava account")
    link_button_same_tab("🚴 Connect Strava", auth_url, type="primary")
    st.link_button("🚴 Connect Strava", auth_url, type="primary")
    st.html(
        f"""
        <a href="{auth_url}"
        style="
            display: inline-block;
            padding: 0.5rem 1.2rem;
            background-color: #FC4C02;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 6px;
            text-decoration: none;
        ">
        🚴 Connect Strava
        </a>
        """
    )


def get_authorization_url(
    client_id: str,
    redirect_uri: str,
    scope: str = "read,activity:read_all,profile:read_all",
) -> str:
    """Build the Strava OAuth authorization URL for SSO."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": scope,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def handle_redirect() -> None:
    query_params = st.query_params

    if "code" not in query_params:
        return

    code = query_params["code"]
    st.info("Authorizing with Strava...")

    token_response = requests.post(
        TOKEN_URL,
        data={
            "client_id": st.secrets["STRAVA_CLIENT_ID"],
            "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
            "code": code,
            "grant_type": "authorization_code",
        },
    )

    token_data = token_response.json()

    if "access_token" in token_data:
        st.session_state["strava_token"] = token_data["access_token"]
        st.session_state["strava_athlete"] = token_data.get("athlete", {})
        athlete_id = token_data.get("athlete", {}).get("id")
        if athlete_id is not None:
            from src.database import touch_user  # noqa: PLC0415 — lazy to avoid init at import
            touch_user(athlete_id)
        st.success("✅ Connected to Strava!")
        st.query_params.clear()
        st.rerun()
    else:
        st.error("❌ Failed to authenticate")
        st.json(token_data)