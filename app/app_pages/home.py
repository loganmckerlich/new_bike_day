"""Home page: landing page explaining the New Bike Day concept."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    st.title("🚴 New Bike Day")
    st.markdown(
        """
        **New Bike Day** helps cyclists answer one question: *does your new bike actually make you faster?*

        When you buy a new road bike, it's tempting to think every ride feels quicker. But is it the bike,
        or is it your fitness, the weather, the route, or just motivation? Strava gives you a treasure trove
        of ride data — but raw segment times don't control for any of those variables.

        This tool takes your Strava segment efforts across multiple bikes and applies rigorous data science
        to isolate the bike's contribution to your speed.

        ---

        ### How it works

        The analysis is broken into six steps, each on its own page:

        1. **Data Collection** — Sign in with Strava. We pull your segment efforts, bikes, and starred
           segments from the Strava API and cache them locally.

        2. **Data Cleaning** — Before any analysis, we remove noisy efforts: those with suspiciously low
           power (e.g. coasting, technical issues) and statistical outliers detected by comparing each
           effort's speed-per-watt against the segment average. You control the thresholds.

        3. **Segment Comparison** — Spider charts and head-to-head tables let you compare your bikes across
           segment types (sprints, flats, climbs, descents). Which bike is strongest on which terrain?

        4. **Bike Head to Head** — A doubly-robust causal inference model estimates the true speed
           difference between two bikes, controlling for segment type, gradient, and weather. This is the
           closest we can get to a controlled experiment without a wind tunnel.

        5. **CdA Estimation** — Using flat-segment efforts and basic physics, we back-calculate an estimate
           of your aerodynamic drag coefficient (CdA) per bike. A lower CdA means a more aerodynamic
           position. *(Work in progress)*

        6. **Final Conclusions** — A summary of findings across all analyses. *(Coming soon)*

        ---

        ### Get started

        👈 Use the navigation on the left to begin with **Step 1 — Data Collection**.
        """
    )


main()
