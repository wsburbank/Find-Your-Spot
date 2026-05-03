"""
Navigation component for Find Your Spot app
"""
import streamlit as st


def render_navigation():
    """Render navigation links in the sidebar."""
    with st.sidebar:
        st.markdown("## Navigation")

        st.page_link("find_your_spot.py", label="Home", icon="🏠")
        st.page_link("pages/1_Quiz.py", label="Take Quiz", icon="📝")
        st.page_link("pages/4_City_Explorer.py", label="City Explorer", icon="🔍")
        st.page_link("pages/5_Data_Documentation.py", label="Data & Documentation", icon="📊")
        st.page_link("pages/6_Cities_Dashboard.py", label="Cities Dashboard", icon="🗺️")

        st.divider()
