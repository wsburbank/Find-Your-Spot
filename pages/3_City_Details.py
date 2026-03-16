"""
City Details Page - Deep dive on a selected city
"""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import get_city_details, calculate_city_scores
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - City Details", show_header_title=False)
render_navigation()

# Check if a city is selected
if "selected_city" not in st.session_state:
    st.title("No City Selected")
    st.warning("Please select a city from your results to view details.")
    if st.button("View Results", type="primary"):
        st.switch_page("pages/2_Results.py")
    st.stop()

# Get city details
city_id = st.session_state.selected_city
city = get_city_details(city_id)

if city is None:
    st.error("City not found. Please try again.")
    if st.button("Back to Results"):
        st.switch_page("pages/2_Results.py")
    st.stop()

# Find this city's score data from top_cities
city_score_data = None
if "top_cities" in st.session_state:
    for c in st.session_state.top_cities:
        if c["city_id"] == city_id:
            city_score_data = c
            break

# Recalculate if needed
if city_score_data is None and "quiz_answers" in st.session_state:
    all_scores = calculate_city_scores(st.session_state.quiz_answers, top_n=100)
    for c in all_scores:
        if c["city_id"] == city_id:
            city_score_data = c
            break

# Page header
col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"{city['name']}, {city['state']}")
    st.markdown(f"**{city['region']} Region** | Population: {city['population']:,}")
with col2:
    if city_score_data:
        st.metric("Match Score", f"{city_score_data['total_score']:.0f}%")

st.divider()

# Score breakdown radar chart
if city_score_data:
    st.subheader("Score Breakdown")

    scores = city_score_data["scores"]
    categories = list(scores.keys())
    values = list(scores.values())

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=[c.replace("_", " ").title() for c in categories] + [categories[0].replace("_", " ").title()],
        fill='toself',
        fillcolor='rgba(0, 176, 80, 0.3)',
        line=dict(color='rgb(0, 176, 80)', width=2),
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )
        ),
        showlegend=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

# City stats in columns
st.subheader("City Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Climate")
    st.metric("Summer Temp", f"{city['avg_temp_summer']:.0f}°F")
    st.metric("Winter Temp", f"{city['avg_temp_winter']:.0f}°F")
    st.metric("Sunny Days/Year", f"{city['sunny_days']}")
    st.metric("Annual Rainfall", f"{city['annual_rainfall']:.0f} in")
    st.metric("Annual Snowfall", f"{city['annual_snow']:.0f} in")

with col2:
    st.markdown("#### Cost of Living")
    st.metric("COL Index", f"{city['cost_of_living_index']:.0f}", help="100 = National Average")
    st.metric("Median Home Price", f"${city['median_home_price']:,}")
    st.metric("State Income Tax", "None" if city['no_income_tax_state'] else f"{city['state_income_tax_rate']:.1f}%")
    st.metric("Property Tax Rate", f"{city['avg_property_tax_rate']:.2f}%")
    st.metric("Sales Tax", f"{city['state_sales_tax_rate']:.1f}%")

with col3:
    st.markdown("#### Livability")
    st.metric("Walkability", f"{city['walkability_score']}/100")
    st.metric("Transit Score", f"{city['transit_score']}/100")
    st.metric("Crime Rate", f"{city['crime_rate_per_1000']:.1f}/1000")
    st.metric("School Rating", f"{city['avg_school_rating']:.1f}/10")
    st.metric("Unemployment", f"{city['unemployment_rate']:.1f}%")

st.divider()

# Geographic features
st.subheader("Geography & Recreation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Landscape Features")
    features = []
    if city["has_mountains"]:
        features.append("Mountains nearby")
    if city["has_ocean"]:
        features.append("Ocean/beach access")
    if city["has_lakes"]:
        features.append("Lakes nearby")
    if city["has_desert"]:
        features.append("Desert landscape")

    for feature in features:
        st.markdown(f"- {feature}")

    if not features:
        st.markdown("- Plains/prairie landscape")

with col2:
    st.markdown("#### Outdoor Recreation")
    st.markdown(f"- Ski resort: {city['ski_resort_distance_miles']:.0f} miles")
    st.markdown(f"- Hiking trails: {city['hiking_trails_count']}")
    st.markdown(f"- Mountain biking: {city['mountain_biking_trails']} trails")
    st.markdown(f"- Rock climbing areas: {city['rock_climbing_areas_nearby']}")
    st.markdown(f"- National parks (100mi): {city['national_parks_within_100mi']}")
    st.markdown(f"- State parks nearby: {city['state_parks_nearby']}")

st.divider()

# Culture & Entertainment
st.subheader("Culture & Entertainment")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Arts & Entertainment")
    st.markdown(f"- Museums: {city['museums_count']}")
    st.markdown(f"- Performing arts venues: {city['performing_arts_venues']}")
    st.markdown(f"- Broadway tours: {'Yes' if city['broadway_tour_stop'] else 'No'}")
    st.markdown(f"- Concert venue capacity: {city['concert_venue_capacity']:,}")

with col2:
    st.markdown("#### Sports & Education")
    st.markdown(f"- Pro sports teams: {city['pro_sports_teams']}")
    st.markdown(f"- Universities: {city['university_count']}")
    st.markdown(f"- Major university: {'Yes' if city['has_major_university'] else 'No'}")
    st.markdown(f"- Community college: {'Yes' if city['community_college_nearby'] else 'No'}")

st.divider()

# Transportation & Economy
st.subheader("Transportation & Economy")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Transportation")
    st.markdown(f"- Airport distance: {city['airport_distance_miles']:.0f} miles")
    st.markdown(f"- Airline hub: {'Yes' if city['is_airline_hub'] else 'No'}")
    st.markdown(f"- Direct flight destinations: {city['direct_flight_destinations_count']}")

with col2:
    st.markdown("#### Economy")
    st.markdown(f"- Median income: ${city['median_household_income']:,}")
    st.markdown(f"- Job growth rate: {city['job_growth_rate']:.1f}%")
    st.markdown(f"- Economic diversity: {city['economic_diversity_index']:.2f}")
    st.markdown(f"- Municipal bond rating: {city['municipal_bond_rating']}")

    # Major industries
    industries = city["major_industries"]
    if isinstance(industries, str):
        industries = industries.split(",")
    st.markdown(f"- Major industries: {', '.join(industries)}")

st.divider()

# External links
st.subheader("Learn More")

col1, col2, col3 = st.columns(3)

city_name_encoded = city["name"].replace(" ", "+")
state_encoded = city["state"]

with col1:
    st.link_button(
        "Search on Zillow",
        f"https://www.zillow.com/homes/{city_name_encoded},-{state_encoded}",
        use_container_width=True,
    )

with col2:
    st.link_button(
        "Google Maps",
        f"https://www.google.com/maps/place/{city_name_encoded},+{state_encoded}",
        use_container_width=True,
    )

with col3:
    st.link_button(
        "Wikipedia",
        f"https://en.wikipedia.org/wiki/{city['name'].replace(' ', '_')},_{state_encoded}",
        use_container_width=True,
    )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("Back to Results"):
        st.switch_page("pages/2_Results.py")

with col3:
    if st.button("Retake Quiz"):
        st.switch_page("pages/1_Quiz.py")
