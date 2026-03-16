"""
Quiz Page - Answer questions to find your ideal city
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.quiz_questions import (
    QUIZ_QUESTIONS,
    get_all_questions,
    get_categories,
    get_total_question_count,
)
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - Quiz", show_header_title=False)
render_navigation()

# Initialize quiz answers in session state
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False


def count_answered():
    """Count how many questions have been answered."""
    count = 0
    for q in get_all_questions():
        q_id = q["id"]
        q_type = q.get("type")

        if q_type == "sliders":
            # For slider groups, check if any slider has been modified
            if q_id in st.session_state.quiz_answers:
                count += 1
        elif q_type == "slider":
            # Single sliders count as answered if in state
            if q_id in st.session_state.quiz_answers:
                count += 1
        else:
            # Standard questions
            if q_id in st.session_state.quiz_answers and st.session_state.quiz_answers[q_id]:
                count += 1
    return count


def render_slider_question(question):
    """Render a single slider question."""
    q_id = question["id"]
    q_type = question.get("type")

    st.markdown(f"**{question['question']}**")

    if q_type == "sliders":
        # Multiple sliders (like commute preferences)
        sliders = question.get("sliders", [])

        if q_id not in st.session_state.quiz_answers:
            st.session_state.quiz_answers[q_id] = {}

        cols = st.columns(len(sliders))
        for i, slider in enumerate(sliders):
            with cols[i]:
                slider_id = slider["id"]
                current_val = st.session_state.quiz_answers[q_id].get(
                    slider_id, slider.get("default", 5)
                )
                new_val = st.slider(
                    slider["label"],
                    min_value=slider.get("min", 0),
                    max_value=slider.get("max", 10),
                    value=current_val,
                    key=f"slider_{q_id}_{slider_id}",
                )
                st.session_state.quiz_answers[q_id][slider_id] = new_val
    else:
        # Single slider
        default = question.get("default", question.get("min", 0))
        current_val = st.session_state.quiz_answers.get(q_id, default)

        format_str = question.get("format", "%d")

        new_val = st.slider(
            "",
            min_value=question.get("min", 0),
            max_value=question.get("max", 100),
            value=current_val,
            step=question.get("step", 1),
            format=format_str,
            key=f"slider_{q_id}",
            label_visibility="collapsed",
        )
        st.session_state.quiz_answers[q_id] = new_val

    st.divider()


def render_question(question, category_key):
    """Render a single question with options."""
    q_id = question["id"]
    q_type = question.get("type")

    # Handle slider questions separately
    if q_type in ["slider", "sliders"]:
        render_slider_question(question)
        return

    # Initialize answer if not exists
    if q_id not in st.session_state.quiz_answers:
        st.session_state.quiz_answers[q_id] = [] if question.get("multi_select") else None

    st.markdown(f"**{question['question']}**")

    if question.get("multi_select"):
        # Multi-select with checkboxes
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Select All", key=f"sel_all_{q_id}", use_container_width=True):
                st.session_state.quiz_answers[q_id] = [opt[0] for opt in question["options"]]
                st.rerun()
            if st.button("Clear", key=f"clear_{q_id}", use_container_width=True):
                st.session_state.quiz_answers[q_id] = []
                st.rerun()

        with col1:
            current_selections = st.session_state.quiz_answers[q_id] or []
            num_cols = min(len(question["options"]), 3)
            cols = st.columns(num_cols)
            for i, (value, label) in enumerate(question["options"]):
                col_idx = i % num_cols
                with cols[col_idx]:
                    checked = value in current_selections
                    if st.checkbox(label, value=checked, key=f"{q_id}_{value}"):
                        if value not in current_selections:
                            current_selections.append(value)
                    else:
                        if value in current_selections:
                            current_selections.remove(value)
            st.session_state.quiz_answers[q_id] = current_selections
    else:
        # Single select with radio buttons
        options = [opt[1] for opt in question["options"]]
        option_values = [opt[0] for opt in question["options"]]

        current_value = st.session_state.quiz_answers[q_id]
        current_index = option_values.index(current_value) if current_value in option_values else None

        selected = st.radio(
            "Select one:",
            options,
            index=current_index,
            key=f"radio_{q_id}",
            horizontal=True,
            label_visibility="collapsed",
        )

        if selected:
            selected_idx = options.index(selected)
            st.session_state.quiz_answers[q_id] = option_values[selected_idx]

    st.divider()


# Page header
st.title("Find Your Perfect City")
st.markdown("Answer these questions to discover cities that match your lifestyle preferences.")

# Progress bar
total_questions = get_total_question_count()
answered = count_answered()
progress = answered / total_questions if total_questions > 0 else 0

col1, col2 = st.columns([4, 1])
with col1:
    st.progress(progress, text=f"Progress: {answered}/{total_questions} questions answered")
with col2:
    if answered == total_questions:
        if st.button("See Results", type="primary", use_container_width=True):
            st.session_state.quiz_completed = True
            st.switch_page("pages/2_Results.py")

# Reset button
if st.button("Reset All Answers"):
    st.session_state.quiz_answers = {}
    st.session_state.quiz_completed = False
    st.rerun()

st.divider()

# Render questions by category using expanders
for category in get_categories():
    questions = QUIZ_QUESTIONS[category]

    # Count answered in this category
    category_answered = 0
    for q in questions:
        q_id = q["id"]
        q_type = q.get("type")
        if q_type in ["slider", "sliders"]:
            if q_id in st.session_state.quiz_answers:
                category_answered += 1
        elif q_id in st.session_state.quiz_answers and st.session_state.quiz_answers[q_id]:
            category_answered += 1

    with st.expander(
        f"{category} ({category_answered}/{len(questions)})",
        expanded=category_answered < len(questions),
    ):
        for question in questions:
            render_question(question, category)

# Bottom navigation
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if answered == total_questions:
        st.success("All questions answered! Click 'See Results' to find your perfect city.")
        if st.button("View My Results", type="primary", use_container_width=True, key="bottom_results"):
            st.session_state.quiz_completed = True
            st.switch_page("pages/2_Results.py")
    elif answered > total_questions * 0.5:
        st.info(f"Good progress! {total_questions - answered} questions remaining. You can view preliminary results now.")
        if st.button("View Preliminary Results", use_container_width=True):
            st.switch_page("pages/2_Results.py")
    elif answered > 0:
        st.info(f"Keep going! {total_questions - answered} questions remaining.")
    else:
        st.info("Start answering questions above to discover your ideal cities.")
