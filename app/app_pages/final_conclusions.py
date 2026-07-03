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
    st.markdown("""
    The verdict is in. But honestly, that was never really the point.
    Yes, the data has something to say. Yes, one of these bikes edges out the other and maybe we were able to find
    that edge. But the numbers don't catch it all. At the end of the day the best bike is the one you're riding. So go back,
    manipulate the data cleaning step until your favorite bike comes out on top, and then go ride it. 

    Appreciate the parts of the ride that don't show up in the data, the speed, the hurt, the riding partners.
                

    I built this app because I love bikes and I love data, and it turns out those two things have a lot to say to each other. 
    If you made it this far, I suspect you feel the same way. It would be great if you shared this app with anyone who you think
    may find it interesting, and reach out if you have any questions or comments. I would love feedback.
                
    I can be reached via my personal website which I'll link below. You can also see other projects I have worked on /plan to work on there.
    """)
    
    st.link_button("🌐 My Personal Website", "https://www.loganmckerlich.com", type="primary")

navigator("final_conclusions1")
main()
navigator("final_conclusions2")