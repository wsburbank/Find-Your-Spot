"""
Results Page - Display top matching cities on a map
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import calculate_city_scores
from components.quiz_questions import get_total_question_count
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - Results", show_header_title=False)
render_navigation()

# Check if quiz has been completed
if "quiz_answers" not in st.session_state or not st.session_state.quiz_answers:
    st.title("No Quiz Results Yet")
    st.warning("Please complete the quiz first to see your city recommendations.")
    if st.button("Take the Quiz", type="primary"):
        st.switch_page("pages/1_Quiz.py")
    st.stop()

# Count answered questions
answered = sum(1 for v in st.session_state.quiz_answers.values() if v)
total = get_total_question_count()

if answered < total * 0.5:  # Require at least 50% answered
    st.title("Quiz Incomplete")
    st.warning(f"Please answer more questions to get accurate results. ({answered}/{total} answered)")
    if st.button("Continue Quiz", type="primary"):
        st.switch_page("pages/1_Quiz.py")
    st.stop()

# Page title
st.title("Your Top City Matches")
st.markdown(f"Based on your {answered} answers, here are the cities that best match your preferences.")

# Calculate scores
with st.spinner("Finding your perfect cities..."):
    top_cities = calculate_city_scores(st.session_state.quiz_answers, top_n=10)

# Store results in session state for city details page
st.session_state.top_cities = top_cities

# Create dataframe for map
df_map = pd.DataFrame(top_cities)

# Add rank column
df_map["rank"] = range(1, len(df_map) + 1)

# Create marker sizes based on rank (higher rank = larger marker)
df_map["marker_size"] = 30 - (df_map["rank"] - 1) * 2

# Create the map
fig = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    size="marker_size",
    color="total_score",
    color_continuous_scale="Greens",
    range_color=[df_map["total_score"].min() - 5, df_map["total_score"].max()],
    hover_name="name",
    hover_data={
        "state": True,
        "total_score": ":.1f",
        "rank": True,
        "population": ":,",
        "marker_size": False,
        "lat": False,
        "lon": False,
    },
    labels={
        "total_score": "Match Score",
        "state": "State",
        "rank": "Rank",
        "population": "Population",
    },
    zoom=3,
    center={"lat": 39.8283, "lon": -98.5795},  # Center of US
    mapbox_style="carto-darkmatter",
)

fig.update_layout(
    height=500,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar={
        "title": "Match<br>Score",
        "ticksuffix": "%",
    },
)

# Display the map
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True,
    "displayModeBar": True,
})

st.divider()

# Display results as cards
st.subheader("Your Top 10 Cities")

# Create columns for the results
for i, city in enumerate(top_cities):
    rank = i + 1

    col1, col2, col3 = st.columns([1, 4, 2])

    with col1:
        # Rank badge
        if rank == 1:
            st.markdown(f"### :first_place_medal:")
        elif rank == 2:
            st.markdown(f"### :second_place_medal:")
        elif rank == 3:
            st.markdown(f"### :third_place_medal:")
        else:
            st.markdown(f"### #{rank}")

    with col2:
        # City name and basic info
        st.markdown(f"### {city['name']}, {city['state']}")

        # Score breakdown in compact form
        scores = city["scores"]
        score_cols = st.columns(4)
        with score_cols[0]:
            st.metric("Climate", f"{scores['climate']:.0f}")
        with score_cols[1]:
            st.metric("Cost", f"{scores['cost_taxes']:.0f}")
        with score_cols[2]:
            st.metric("Outdoor", f"{scores['outdoor']:.0f}")
        with score_cols[3]:
            st.metric("Lifestyle", f"{scores['lifestyle']:.0f}")

        # Key stats
        stats = city["stats"]
        st.caption(
            f"Population: {city['population']:,} | "
            f"Weather: {stats['avg_temp_summer']:.0f}F/{stats['avg_temp_winter']:.0f}F | "
            f"COL: {stats['cost_of_living_index']:.0f} | "
            f"Home Price: ${stats['median_home_price']:,}"
        )

    with col3:
        # Overall score and details button
        st.markdown(f"### {city['total_score']:.0f}%")
        st.caption("Match Score")
        if st.button("View Details", key=f"details_{city['city_id']}", use_container_width=True):
            st.session_state.selected_city = city["city_id"]
            st.switch_page("pages/3_City_Details.py")

    st.divider()

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Retake Quiz"):
        st.switch_page("pages/1_Quiz.py")
with col3:
    if st.button("Modify Answers"):
        st.switch_page("pages/1_Quiz.py")
