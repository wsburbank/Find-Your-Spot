"""
Find Your Spot - City Recommendation App
Main landing page
"""
import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utilities.st_base import initialize_session_state
from components.quiz_questions import get_total_question_count, get_categories
from components.navigation import render_navigation

initialize_session_state("Find Your Spot", show_header_title=False)
render_navigation()

# Hero section
st.title("Find Your Perfect City")
st.markdown(
    """
    Discover the US cities that match your lifestyle, climate preferences, and priorities.
    Answer a few questions and we'll recommend your top 10 city matches from 700+ real US cities,
    all scored with real data from the Census Bureau, NOAA, FBI, and more.
    """
)

st.divider()

# Stats about the quiz/database
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Questions", get_total_question_count())
with col2:
    st.metric("Categories", len(get_categories()))
with col3:
    try:
        import pandas as pd
        _cities = pd.read_parquet(Path(__file__).parent / "data" / "cities.parquet")
        st.metric("Cities", f"{len(_cities):,}")
    except Exception:
        st.metric("Cities", "700+")

st.divider()

# Categories preview
st.subheader("What We'll Ask About")

categories = get_categories()
cols = st.columns(3)
for i, category in enumerate(categories):
    with cols[i % 3]:
        st.markdown(f"**{category}**")

st.divider()

# How it works
st.subheader("How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Take the Quiz")
    st.markdown(
        "Answer questions about your ideal climate, city size, cost of living, "
        "outdoor activities, and more."
    )

with col2:
    st.markdown("### 2. Get Matched")
    st.markdown(
        "Our algorithm scores over 700 real US cities based on your preferences "
        "and finds your best matches."
    )

with col3:
    st.markdown("### 3. Explore Results")
    st.markdown(
        "View your top 10 cities on an interactive map, compare scores, "
        "and dive deep into city profiles."
    )

st.divider()

# Call to action
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Start the Quiz", type="primary", use_container_width=True):
        st.switch_page("pages/1_Quiz.py")

    # Resume quiz if in progress
    if "quiz_answers" in st.session_state and st.session_state.quiz_answers:
        answered = sum(1 for v in st.session_state.quiz_answers.values() if v)
        total = get_total_question_count()
        if answered > 0:
            st.info(f"You have {answered}/{total} questions answered.")
            if st.button("Continue Quiz", use_container_width=True):
                st.switch_page("pages/1_Quiz.py")
            if answered >= total * 0.5:
                if st.button("View Results", use_container_width=True):
                    st.session_state.quiz_section = "My Results"
                    st.switch_page("pages/1_Quiz.py")

st.divider()

# Footer
st.caption(
    "Find Your Spot uses public data to help you discover cities that match your lifestyle. "
    "Data includes climate, cost of living, outdoor recreation, culture, and more."
)
