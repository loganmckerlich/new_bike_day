import streamlit as st
import html as html_lib
import uuid


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


# --- demo usage ---
st.title("Same-tab link button demo")

link_button_same_tab(
    "Open new_bike_day app",
    "https://newbikeday-nashwnuvekoy9fmgp3bvln.streamlit.app/",
    key="nbd",
    type="primary",
)

link_button_same_tab(
    "Open repo",
    "https://github.com/loganmckerlich/new_bike_day",
    key="repo",
)