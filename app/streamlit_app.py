"""Streamlit app entrypoint for new-bike-day analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(data_path: str = "data/activities.csv") -> pd.DataFrame:
    """Load activities data for display from a cached CSV file."""
    if not Path(data_path).exists():
        return pd.DataFrame()
    return pd.read_csv(data_path)


def main() -> None:
    """Render the Streamlit dashboard scaffold."""
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("Compare ride performance between bikes using Strava data.")
    st.info("Run `python src/ingest.py` first to populate `data/activities.csv`.")

    data = load_data()
    if data.empty:
        st.warning("No activity data found yet.")
        return

    st.subheader("Activity Preview")
    st.dataframe(data.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
