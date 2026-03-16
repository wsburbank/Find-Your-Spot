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
        st.page_link("pages/2_Results.py", label="My Results", icon="🗺️")
        st.page_link("pages/3_City_Details.py", label="City Details", icon="🏙️")
        st.page_link("pages/4_City_Explorer.py", label="City Explorer", icon="🔍")
        st.page_link("pages/5_Data_Documentation.py", label="Data & Documentation", icon="📊")

        st.divider()

        # Show quiz progress if available
        if "quiz_answers" in st.session_state and st.session_state.quiz_answers:
            from components.quiz_questions import get_total_question_count
            answered = sum(1 for v in st.session_state.quiz_answers.values() if v)
            total = get_total_question_count()
            st.caption(f"Quiz Progress: {answered}/{total}")
            st.progress(answered / total if total > 0 else 0)
