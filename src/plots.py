from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

FIG_B_SHARED_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    legend={"orientation": "h", "y": -0.25},
    height=300,
    margin={"t": 10, "b": 10},
)

def make_fig_b_h(bike_color, spw_b, b_lo, b_hi, b_mean, nbins, spd, z_threshold) -> go.Figure:
    fig_b_h = go.Figure()
    for is_out, bar_color, bar_name in [
        (False, bike_color, "Normal"),
        (True, "#ef5350", "Outlier"),
    ]:
        pts_h = spw_b[spw_b["is_outlier"] == is_out]
        if pts_h.empty:
            continue
        fig_b_h.add_trace(go.Histogram(
            x=pts_h["speed_per_cbrt_watt"],
            name=bar_name,
            marker_color=bar_color,
            opacity=0.8,
            nbinsx=nbins,
            hovertemplate="Speed/W¹⁄³: %{x:.4f}<br>Count: %{y}<extra>" + bar_name + "</extra>",
        ))

    xlo = float(spw_b["speed_per_cbrt_watt"].min())
    xhi = float(spw_b["speed_per_cbrt_watt"].max())
    xpad = max((xhi - xlo) * 0.05, 1e-6)
    for sx0, sx1 in [(xlo - xpad, b_lo), (b_hi, xhi + xpad)]:
        fig_b_h.add_vrect(
            x0=sx0, x1=sx1,
            fillcolor="rgba(239,83,80,0.12)",
            line_width=0, layer="below",
        )
    for vx, vlabel in [(b_lo, f"−{z_threshold:.1f}σ"), (b_hi, f"+{z_threshold:.1f}σ")]:
        fig_b_h.add_vline(
            x=vx, line_dash="dash", line_color="#ef5350",
            annotation_text=vlabel, annotation_position="top",
            annotation_font_color="#ef5350",
        )
    fig_b_h.add_vline(
        x=b_mean, line_dash="dot",
        line_color="rgba(128,128,128,0.7)",
        annotation_text="μ", annotation_position="top",
    )
    fig_b_h.update_layout(
        barmode="overlay",
        xaxis_title=f"Speed/W¹⁄³ ({spd}/W¹⁄³)",
        yaxis_title="Efforts",
        **FIG_B_SHARED_LAYOUT
    )

    return fig_b_h


def make_fig_b_sc(bike_color, bdata, spd, z_threshold, b_lo, b_hi, b_mean) -> go.Figure:
    fig_b_sc = go.Figure()
    for is_out, dot_color, dot_name in [
        (False, bike_color, "Normal"),
        (True, "#ef5350", "Outlier"),
    ]:
        pts = bdata[bdata["is_outlier"] == is_out]
        if pts.empty:
            continue
        fig_b_sc.add_trace(go.Scatter(
            x=pts["average_watts"],
            y=pts["speed_kmh"],
            mode="markers",
            name=dot_name,
            marker={"color": dot_color, "size": 10,
                    "line": {"width": 1, "color": "white"}},
            text=pts["z_label"],
            hovertemplate=(
                "Power: %{x:.0f} W<br>"
                f"Speed: %{{y:.1f}} {spd}<br>"
                "Z-score: %{text}<extra>" + dot_name + "</extra>"
            ),
        ))
    sc_spw = bdata.dropna(subset=["speed_per_cbrt_watt", "average_watts"])
    if len(sc_spw) >= 2:
        w_min = float(sc_spw["average_watts"].min())
        w_max = float(sc_spw["average_watts"].max())
        w_pad = (w_max - w_min) * 0.05
        wx = list(np.linspace(w_min - w_pad, w_max + w_pad, 60))
        wx_rev = list(reversed(wx))
        fig_b_sc.add_trace(go.Scatter(
            x=wx + wx_rev,
            y=[b_lo * np.cbrt(w) for w in wx] + [b_hi * np.cbrt(w) for w in wx_rev],
            fill="toself",
            fillcolor="rgba(239,83,80,0.10)",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_b_sc.add_trace(go.Scatter(
            x=wx,
            y=[b_mean * np.cbrt(w) for w in wx],
            mode="lines",
            line={"color": "rgba(128,128,128,0.6)", "dash": "dot", "width": 1.5},
            name="μ (speed/W¹⁄³)",
            hovertemplate=f"μ = {b_mean:.4f} {spd}/W¹⁄³<extra>mean</extra>",
        ))
        for slope, slabel in [(b_lo, f"−{z_threshold:.2g}σ"), (b_hi, f"+{z_threshold:.2g}σ")]:
            fig_b_sc.add_trace(go.Scatter(
                x=wx,
                y=[slope * np.cbrt(w) for w in wx],
                mode="lines",
                line={"color": "#ef5350", "dash": "dash", "width": 1.5},
                name=slabel,
                hovertemplate=f"{slabel} = {slope:.4f} {spd}/W¹⁄³<extra>{slabel}</extra>",
            ))
    fig_b_sc.update_layout(
        xaxis_title="Avg power (W)",
        yaxis_title=f"Speed ({spd})",
        **FIG_B_SHARED_LAYOUT
    )
    return fig_b_sc