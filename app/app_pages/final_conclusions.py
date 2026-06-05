"""Final Conclusions page — summary of findings across all analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import navigator

def main() -> None:
    st.title("🏁 Step 4 — Final Conclusions")
    st.info("🚧 Coming soon — this page will summarise findings from all analysis steps.")

navigator("final_conclusions1")
main()
navigator("final_conclusions2")