"""Streamlit app entrypoint for new-bike-day analysis."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


def load_data(db_path: str = "data/strava.db") -> pd.DataFrame:
    """Load activities data for display from SQLite."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM activities", conn)


def main() -> None:
    """Render the Streamlit dashboard scaffold."""
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("Compare ride performance between bikes using Strava data.")
    st.info("Run `python src/ingest.py` first to populate `data/strava.db`.")

    data = load_data()
    if data.empty:
        st.warning("No activity data found yet.")
        return

    st.subheader("Activity Preview")
    st.dataframe(data.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
