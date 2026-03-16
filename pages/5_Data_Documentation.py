"""
Data Documentation Page - View raw data and understand the database
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import load_cities
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - Data Documentation", show_header_title=False)
render_navigation()

# Load city data
@st.cache_data
def get_cities_df():
    return load_cities()

cities_df = get_cities_df()

st.title("Data Documentation")
st.markdown("""
This page documents all data fields in our city database and lets you explore the raw data.

**Data Sources:**
- **City Names & Population**: US Census Bureau (2022 ACS 5-Year Estimates)
- **Tax Rates**: Tax Foundation (2024 state tax data)
- **Climate Data**: NOAA Climate Normals by state
- **Crime Statistics**: FBI Uniform Crime Reporting (UCR) 2022 state-level data
- **Airport Data**: FAA enplanement data and hub classifications
- **Education Data**: NCES (National Center for Education Statistics) - universities, graduation rates, school quality

**Note:** Some attribute values (walkability, outdoor recreation counts) are estimated based on
population and regional patterns. For production use, these could be enhanced with
data from Walk Score API, trail databases, etc.
""")

# Database overview
st.header("Database Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Cities", len(cities_df))
with col2:
    st.metric("States Covered", cities_df["state"].nunique())
with col3:
    st.metric("Data Fields", len(cities_df.columns))
with col4:
    st.metric("Regions", cities_df["region"].nunique())

st.divider()

# Data dictionary
st.header("Data Dictionary")

data_dict = {
    "Identity & Location": {
        "city_id": "Unique identifier for the city (lowercase name_state)",
        "name": "City name",
        "state": "Two-letter state abbreviation",
        "lat": "Latitude coordinate",
        "lon": "Longitude coordinate",
        "population": "City population (approximate)",
        "metro_pop": "Metropolitan area population",
        "region": "Geographic region (Northeast, Southeast, Midwest, Southwest, Mountain, West, Alaska, Hawaii)",
    },
    "Climate & Weather": {
        "avg_temp_summer": "Average summer temperature (°F)",
        "avg_temp_winter": "Average winter temperature (°F)",
        "sunny_days": "Average sunny days per year",
        "annual_rainfall": "Annual rainfall (inches)",
        "annual_snow": "Annual snowfall (inches)",
    },
    "Cost of Living & Taxes": {
        "cost_of_living_index": "Cost of living index (100 = national average)",
        "median_home_price": "Median home price ($)",
        "state_income_tax_rate": "State income tax rate (%)",
        "state_sales_tax_rate": "State sales tax rate (%)",
        "avg_property_tax_rate": "Average property tax rate (%)",
        "no_income_tax_state": "Whether the state has no income tax (True/False)",
    },
    "Geography & Landscape": {
        "has_mountains": "Mountains nearby (True/False)",
        "has_ocean": "Ocean/beach access (True/False)",
        "has_lakes": "Within 30 miles of navigable waterway for boating (True/False)",
        "has_desert": "Desert landscape (True/False)",
        "nearest_waterway": "Name of nearest navigable waterway (river/lake)",
        "waterway_distance_miles": "Distance to nearest navigable waterway (miles)",
    },
    "Outdoor Recreation": {
        "ski_resort_distance_miles": "Distance to nearest ski resort (miles)",
        "ski_resorts_within_100mi": "Number of ski resorts within 100 miles",
        "hiking_trails_count": "Number of hiking trails in area",
        "mountain_biking_trails": "Number of mountain biking trails",
        "rock_climbing_areas_nearby": "Number of rock climbing areas nearby",
        "swimming_access": "Primary swimming access type (ocean/lake/pool)",
        "national_parks_within_100mi": "National parks within 100 miles",
        "state_parks_nearby": "State parks nearby",
        "camping_areas_count": "Number of camping areas",
    },
    "Transportation": {
        "nearest_major_airport": "Nearest major airport name",
        "airport_distance_miles": "Distance to nearest major airport (miles)",
        "is_airline_hub": "Whether city has an airline hub (True/False)",
        "direct_flight_destinations_count": "Number of direct flight destinations",
        "walkability_score": "Walkability score (0-100)",
        "transit_score": "Public transit score (0-100)",
    },
    "Education": {
        "school_quality_score": "Overall school quality score (0-10)",
        "university_count": "Number of universities/colleges",
        "has_major_university": "Has a major research university (True/False)",
        "community_college_nearby": "Community college nearby (True/False)",
        "avg_school_rating": "Average K-12 school rating (0-10)",
    },
    "Culture & Entertainment": {
        "pro_sports_teams": "Number of professional sports teams",
        "performing_arts_venues": "Number of performing arts venues",
        "broadway_tour_stop": "Broadway tour stop city (True/False)",
        "museums_count": "Number of museums",
        "concert_venue_capacity": "Largest concert venue capacity",
    },
    "Financial Health": {
        "median_household_income": "Median household income ($)",
        "unemployment_rate": "Unemployment rate (%)",
        "poverty_rate": "Poverty rate (%)",
        "job_growth_rate": "Annual job growth rate (%)",
        "median_home_value_growth": "Median home value growth rate (%)",
        "municipal_bond_rating": "Municipal bond rating (AAA to BBB)",
        "economic_diversity_index": "Economic diversity index (0-1)",
    },
    "Safety": {
        "crime_rate_per_1000": "Total crime rate per 1,000 residents",
        "violent_crime_rate": "Violent crime rate per 1,000 residents",
        "property_crime_rate": "Property crime rate per 1,000 residents",
    },
    "Economy": {
        "major_industries": "Major industries (comma-separated)",
    },
}

# Display data dictionary with expandable sections
for category, fields in data_dict.items():
    with st.expander(f"**{category}** ({len(fields)} fields)"):
        for field, description in fields.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.code(field)
            with col2:
                st.write(description)

st.divider()

# Raw data viewer
st.header("Raw Data Viewer")

# Column selection
all_columns = cities_df.columns.tolist()
default_cols = ["name", "state", "population", "region", "cost_of_living_index", "median_home_price"]

selected_cols = st.multiselect(
    "Select columns to display",
    options=all_columns,
    default=default_cols,
)

if not selected_cols:
    selected_cols = default_cols

# Filters
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

with col1:
    regions = ["All"] + sorted(cities_df["region"].unique().tolist())
    selected_region = st.selectbox("Region", regions)

with col2:
    states = ["All"] + sorted(cities_df["state"].unique().tolist())
    selected_state = st.selectbox("State", states)

with col3:
    search = st.text_input("Search city name")

# Apply filters
filtered_df = cities_df.copy()

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state"] == selected_state]

if search:
    filtered_df = filtered_df[filtered_df["name"].str.lower().str.contains(search.lower())]

# Sort options
col1, col2 = st.columns(2)
with col1:
    sort_col = st.selectbox("Sort by", options=selected_cols, index=0)
with col2:
    sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

filtered_df = filtered_df.sort_values(sort_col, ascending=(sort_order == "Ascending"))

# Display data
st.subheader(f"Data ({len(filtered_df)} cities)")

# Configure column display
column_config = {
    "population": st.column_config.NumberColumn(format="%d"),
    "metro_pop": st.column_config.NumberColumn(format="%d"),
    "median_home_price": st.column_config.NumberColumn(format="$%d"),
    "median_household_income": st.column_config.NumberColumn(format="$%d"),
    "cost_of_living_index": st.column_config.NumberColumn(format="%.1f"),
    "avg_temp_summer": st.column_config.NumberColumn(format="%.0f°F"),
    "avg_temp_winter": st.column_config.NumberColumn(format="%.0f°F"),
    "annual_rainfall": st.column_config.NumberColumn(format="%.1f in"),
    "annual_snow": st.column_config.NumberColumn(format="%.1f in"),
    "state_income_tax_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "state_sales_tax_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "avg_property_tax_rate": st.column_config.NumberColumn(format="%.2f%%"),
    "unemployment_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "poverty_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "job_growth_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "crime_rate_per_1000": st.column_config.NumberColumn(format="%.1f"),
    "school_quality_score": st.column_config.NumberColumn(format="%.1f"),
    "avg_school_rating": st.column_config.NumberColumn(format="%.1f"),
    "walkability_score": st.column_config.NumberColumn(format="%d"),
    "transit_score": st.column_config.NumberColumn(format="%d"),
    "lat": st.column_config.NumberColumn(format="%.4f"),
    "lon": st.column_config.NumberColumn(format="%.4f"),
}

st.dataframe(
    filtered_df[selected_cols],
    use_container_width=True,
    hide_index=True,
    column_config={k: v for k, v in column_config.items() if k in selected_cols},
    height=500,
)

# Download button
csv = filtered_df[selected_cols].to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="cities_data.csv",
    mime="text/csv",
)

st.divider()

# Data statistics
st.header("Data Statistics")

# Select numeric columns for statistics
numeric_cols = cities_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
stat_cols = st.multiselect(
    "Select columns for statistics",
    options=numeric_cols,
    default=["population", "cost_of_living_index", "median_home_price", "avg_temp_summer", "crime_rate_per_1000"],
)

if stat_cols:
    st.dataframe(
        filtered_df[stat_cols].describe().round(2),
        use_container_width=True,
    )

st.divider()

# Data sources note
st.header("Data Sources")
st.markdown("""
**Current data sources used in this application:**

| Category | Source | Status |
|----------|--------|--------|
| Population & Demographics | US Census Bureau (2022 ACS 5-Year) | ✅ Real data |
| Climate | NOAA Climate Normals by state | ✅ Real data |
| Tax Rates | Tax Foundation (2024) | ✅ Real data |
| Crime Statistics | FBI UCR (2022 state-level) | ✅ Real data |
| Airport/Transportation | FAA enplanement data | ✅ Real data |
| Education | NCES (universities, school quality) | ✅ Real data |
| Housing Prices | Estimated from cost of living | ⚠️ Estimates |
| Walkability | Estimated from population density | ⚠️ Estimates |
| Outdoor Recreation | Estimated from geography | ⚠️ Estimates |

**Potential enhancements:**
- Walk Score API for accurate walkability scores (paid)
- GreatSchools API for school-level ratings (paid)
- Zillow/Redfin for housing price data
- Trail databases (MTB Project, AllTrails) for recreation data
""")
