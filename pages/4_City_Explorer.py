"""
City Explorer Page - Search and compare all available cities
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import load_cities
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - City Explorer", show_header_title=False)
render_navigation()

# Load city data
@st.cache_data
def get_cities_df():
    return load_cities()

cities_df = get_cities_df()

st.title("City Explorer")
st.markdown("Search, filter, and compare cities from our database of nearly 2,000 US cities.")

# Sidebar filters
st.sidebar.header("Filter Cities")

# Region filter
regions = ["All"] + sorted(cities_df["region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

# State filter
if selected_region == "All":
    states = ["All"] + sorted(cities_df["state"].unique().tolist())
else:
    states = ["All"] + sorted(cities_df[cities_df["region"] == selected_region]["state"].unique().tolist())
selected_state = st.sidebar.selectbox("State", states)

# Population filter
pop_min, pop_max = int(cities_df["population"].min()), int(cities_df["population"].max())
pop_range = st.sidebar.slider(
    "Population Range",
    min_value=pop_min,
    max_value=pop_max,
    value=(pop_min, pop_max),
    format="%d",
)

# Cost of living filter
col_min, col_max = float(cities_df["cost_of_living_index"].min()), float(cities_df["cost_of_living_index"].max())
col_range = st.sidebar.slider(
    "Cost of Living Index",
    min_value=col_min,
    max_value=col_max,
    value=(col_min, col_max),
    help="100 = National Average",
)

# Home price filter
price_min = int(cities_df["median_home_price"].min())
price_max = int(cities_df["median_home_price"].max())
price_range = st.sidebar.slider(
    "Median Home Price",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    format="$%d",
)

# Climate filters
st.sidebar.subheader("Climate")
temp_summer_range = st.sidebar.slider(
    "Summer Temp (F)",
    min_value=50,
    max_value=120,
    value=(50, 120),
)
temp_winter_range = st.sidebar.slider(
    "Winter Temp (F)",
    min_value=-20,
    max_value=80,
    value=(-20, 80),
)
sunny_days_range = st.sidebar.slider(
    "Sunny Days/Year",
    min_value=80,
    max_value=320,
    value=(80, 320),
)

# Geographic features
st.sidebar.subheader("Geography")
has_mountains = st.sidebar.checkbox("Mountains nearby")
has_ocean = st.sidebar.checkbox("Ocean access")
has_lakes = st.sidebar.checkbox("Lakes/rivers for boating")
has_climbing = st.sidebar.checkbox("Rock climbing areas nearby")
ski_nearby = st.sidebar.selectbox(
    "Ski resort access",
    ["Any", "Within 60 mi (~1 hr)", "Within 150 mi (day trip)"],
    help="Distance to nearest ski resort"
)
airport_nearby = st.sidebar.selectbox(
    "Airport access",
    ["Any", "Within 15 mi", "Within 30 mi", "Within 60 mi", "Major hub only"],
    help="Distance to nearest major airport"
)
no_income_tax = st.sidebar.checkbox("No state income tax")

# Apply filters
filtered_df = cities_df.copy()

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state"] == selected_state]

filtered_df = filtered_df[
    (filtered_df["population"] >= pop_range[0]) &
    (filtered_df["population"] <= pop_range[1]) &
    (filtered_df["cost_of_living_index"] >= col_range[0]) &
    (filtered_df["cost_of_living_index"] <= col_range[1]) &
    (filtered_df["median_home_price"] >= price_range[0]) &
    (filtered_df["median_home_price"] <= price_range[1]) &
    (filtered_df["avg_temp_summer"] >= temp_summer_range[0]) &
    (filtered_df["avg_temp_summer"] <= temp_summer_range[1]) &
    (filtered_df["avg_temp_winter"] >= temp_winter_range[0]) &
    (filtered_df["avg_temp_winter"] <= temp_winter_range[1])
]

if has_mountains:
    filtered_df = filtered_df[filtered_df["has_mountains"] == True]
if has_ocean:
    filtered_df = filtered_df[filtered_df["has_ocean"] == True]
if has_lakes:
    filtered_df = filtered_df[filtered_df["has_lakes"] == True]
if has_climbing:
    filtered_df = filtered_df[filtered_df["rock_climbing_areas_nearby"] >= 5]
if ski_nearby == "Within 60 mi (~1 hr)":
    filtered_df = filtered_df[filtered_df["ski_resort_distance_miles"] <= 60]
elif ski_nearby == "Within 150 mi (day trip)":
    filtered_df = filtered_df[filtered_df["ski_resort_distance_miles"] <= 150]
if airport_nearby == "Within 15 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 15]
elif airport_nearby == "Within 30 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 30]
elif airport_nearby == "Within 60 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 60]
elif airport_nearby == "Major hub only":
    filtered_df = filtered_df[filtered_df["is_airline_hub"] == True]
if no_income_tax:
    filtered_df = filtered_df[filtered_df["no_income_tax_state"] == True]
filtered_df = filtered_df[
    (filtered_df["sunny_days"] >= sunny_days_range[0]) &
    (filtered_df["sunny_days"] <= sunny_days_range[1])
]

# Display count
st.info(f"Showing {len(filtered_df):,} cities matching your filters")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Map View", "Table View", "Compare Cities"])

with tab1:
    # Map visualization
    if len(filtered_df) > 0:
        # Define numeric columns available for size/color with friendly names
        numeric_columns = {
            "population": "Population",
            "metro_pop": "Metro Population",
            "cost_of_living_index": "Cost of Living Index",
            "median_home_price": "Median Home Price",
            "avg_temp_summer": "Summer Temp (F)",
            "avg_temp_winter": "Winter Temp (F)",
            "sunny_days": "Sunny Days/Year",
            "annual_rainfall": "Annual Rainfall (in)",
            "annual_snow": "Annual Snow (in)",
            "crime_rate_per_1000": "Crime Rate (per 1000)",
            "violent_crime_rate": "Violent Crime Rate",
            "property_crime_rate": "Property Crime Rate",
            "walkability_score": "Walkability Score",
            "transit_score": "Transit Score",
            "avg_school_rating": "School Rating",
            "avg_property_tax_rate": "Property Tax Rate",
            "state_income_tax_rate": "State Income Tax Rate",
            "state_sales_tax_rate": "State Sales Tax Rate",
            "median_household_income": "Median Household Income",
            "unemployment_rate": "Unemployment Rate",
            "ski_resort_distance_miles": "Ski Resort Distance (mi)",
            "airport_distance_miles": "Airport Distance (mi)",
            "waterway_distance_miles": "Waterway Distance (mi)",
            "rock_climbing_areas_nearby": "Rock Climbing Areas Nearby",
            "national_parks_within_100mi": "National Parks within 100mi",
        }

        # Selectboxes for map customization
        col1, col2 = st.columns(2)
        with col1:
            size_col = st.selectbox(
                "Dot Size",
                options=list(numeric_columns.keys()),
                format_func=lambda x: numeric_columns[x],
                index=0,  # Default: population
            )
        with col2:
            color_col = st.selectbox(
                "Dot Color",
                options=list(numeric_columns.keys()),
                format_func=lambda x: numeric_columns[x],
                index=2,  # Default: cost_of_living_index
            )

        # Determine color scale based on metric
        # Use reversed scale (red=high) for: costs, crime, distances, and temperatures (warm=red)
        red_when_high = ["cost_of_living_index", "median_home_price", "crime_rate_per_1000",
                         "violent_crime_rate", "property_crime_rate", "unemployment_rate",
                         "avg_property_tax_rate", "state_income_tax_rate", "state_sales_tax_rate",
                         "ski_resort_distance_miles", "airport_distance_miles", "waterway_distance_miles",
                         "avg_temp_summer", "avg_temp_winter"]
        color_scale = "RdYlGn_r" if color_col in red_when_high else "RdYlGn"

        # Normalize size column to 10-100 range for consistent dot sizing
        map_df = filtered_df.copy()
        size_values = map_df[size_col]
        size_min = size_values.min()
        size_max = size_values.max()

        if size_max > size_min:
            # Normalize to 0-1 then scale to 10-100
            map_df["_normalized_size"] = 10 + 90 * (size_values - size_min) / (size_max - size_min)
        else:
            # All values are the same
            map_df["_normalized_size"] = 50

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="_normalized_size",
            color=color_col,
            color_continuous_scale=color_scale,
            hover_name="name",
            hover_data={
                "state": True,
                "population": ":,",
                "cost_of_living_index": ":.0f",
                "median_home_price": ":$,",
                "lat": ":.4f",
                "lon": ":.4f",
                size_col: True,
                color_col: True,
                "_normalized_size": False,  # Hide normalized size from hover
            },
            labels=numeric_columns,
            zoom=3,
            center={"lat": 39.8283, "lon": -98.5795},
            mapbox_style="carto-darkmatter",
            size_max=20,
        )

        # Create short label for colorbar
        color_label = numeric_columns[color_col].replace(" ", "<br>")
        if len(color_label) > 15:
            color_label = color_label[:15] + "..."

        fig.update_layout(
            height=500,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar={"title": color_label},
        )

        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
        })
    else:
        st.warning("No cities match your current filters. Try adjusting the filters.")

with tab2:
    # Search box
    search_term = st.text_input("Search cities by name", placeholder="Type city name...")

    display_df = filtered_df.copy()
    if search_term:
        display_df = display_df[
            display_df["name"].str.lower().str.contains(search_term.lower())
        ]

    # Select columns to display
    display_cols = [
        "name", "state", "lat", "lon", "population", "region",
        "avg_temp_summer", "avg_temp_winter", "sunny_days",
        "cost_of_living_index", "median_home_price",
        "no_income_tax_state", "avg_property_tax_rate",
        "walkability_score", "transit_score",
        "crime_rate_per_1000", "avg_school_rating",
        "nearest_waterway", "waterway_distance_miles",
    ]

    col_rename = {
        "name": "City",
        "state": "State",
        "lat": "Latitude",
        "lon": "Longitude",
        "population": "Population",
        "region": "Region",
        "avg_temp_summer": "Summer (F)",
        "avg_temp_winter": "Winter (F)",
        "sunny_days": "Sunny Days",
        "cost_of_living_index": "COL Index",
        "median_home_price": "Home Price",
        "no_income_tax_state": "No Income Tax",
        "avg_property_tax_rate": "Property Tax %",
        "walkability_score": "Walkability",
        "transit_score": "Transit",
        "crime_rate_per_1000": "Crime Rate",
        "avg_school_rating": "School Rating",
        "nearest_waterway": "Nearest Waterway",
        "waterway_distance_miles": "Waterway Dist (mi)",
    }

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        options=display_cols,
        format_func=lambda x: col_rename.get(x, x),
        index=display_cols.index("population"),
    )
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    display_df = display_df.sort_values(
        sort_col,
        ascending=(sort_order == "Ascending")
    )

    # Format and display
    st.dataframe(
        display_df[display_cols].rename(columns=col_rename),
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "Latitude": st.column_config.NumberColumn(format="%.4f"),
            "Longitude": st.column_config.NumberColumn(format="%.4f"),
            "Population": st.column_config.NumberColumn(format="%d"),
            "Home Price": st.column_config.NumberColumn(format="$%d"),
            "Summer (F)": st.column_config.NumberColumn(format="%.0f"),
            "Winter (F)": st.column_config.NumberColumn(format="%.0f"),
            "COL Index": st.column_config.NumberColumn(format="%.0f"),
            "Property Tax %": st.column_config.NumberColumn(format="%.2f%%"),
            "Crime Rate": st.column_config.NumberColumn(format="%.1f"),
            "School Rating": st.column_config.NumberColumn(format="%.1f"),
            "Waterway Dist (mi)": st.column_config.NumberColumn(format="%.1f"),
        }
    )

    st.caption(f"Showing all {len(display_df):,} cities. Use filters or search to narrow results.")

with tab3:
    st.subheader("Compare Cities Side by Side")

    # City selection
    city_options = filtered_df.apply(lambda x: f"{x['name']}, {x['state']}", axis=1).tolist()

    col1, col2 = st.columns(2)
    with col1:
        city1_name = st.selectbox("Select first city", [""] + city_options, key="city1")
    with col2:
        city2_name = st.selectbox("Select second city", [""] + city_options, key="city2")

    if city1_name and city2_name:
        # Parse city names
        city1_parts = city1_name.rsplit(", ", 1)
        city2_parts = city2_name.rsplit(", ", 1)

        city1 = filtered_df[(filtered_df["name"] == city1_parts[0]) & (filtered_df["state"] == city1_parts[1])].iloc[0]
        city2 = filtered_df[(filtered_df["name"] == city2_parts[0]) & (filtered_df["state"] == city2_parts[1])].iloc[0]

        # Comparison metrics
        compare_metrics = [
            ("Latitude", "lat", "{:.4f}"),
            ("Longitude", "lon", "{:.4f}"),
            ("Population", "population", "{:,}"),
            ("Cost of Living Index", "cost_of_living_index", "{:.0f}"),
            ("Median Home Price", "median_home_price", "${:,}"),
            ("Summer Temp (F)", "avg_temp_summer", "{:.0f}"),
            ("Winter Temp (F)", "avg_temp_winter", "{:.0f}"),
            ("Sunny Days/Year", "sunny_days", "{:.0f}"),
            ("Annual Rainfall (in)", "annual_rainfall", "{:.1f}"),
            ("Annual Snow (in)", "annual_snow", "{:.1f}"),
            ("Walkability Score", "walkability_score", "{:.0f}"),
            ("Transit Score", "transit_score", "{:.0f}"),
            ("Crime Rate (per 1000)", "crime_rate_per_1000", "{:.1f}"),
            ("School Rating", "avg_school_rating", "{:.1f}"),
            ("Unemployment Rate", "unemployment_rate", "{:.1f}%"),
            ("Property Tax Rate", "avg_property_tax_rate", "{:.2f}%"),
            ("State Income Tax", "state_income_tax_rate", "{:.1f}%"),
        ]

        # Create comparison table
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.markdown(f"### {city1['name']}, {city1['state']}")
        with col2:
            st.markdown("### Metric")
        with col3:
            st.markdown(f"### {city2['name']}, {city2['state']}")

        for label, col, fmt in compare_metrics:
            col1, col2, col3 = st.columns([2, 2, 2])

            val1 = city1[col]
            val2 = city2[col]

            # Determine which is "better" (lower cost/crime is better, higher scores are better)
            # Skip highlighting for neutral metrics like coordinates
            neutral_metrics = ["lat", "lon"]
            lower_is_better = col in ["cost_of_living_index", "median_home_price", "crime_rate_per_1000",
                                       "unemployment_rate", "avg_property_tax_rate", "state_income_tax_rate"]

            if col in neutral_metrics:
                better1 = False
                better2 = False
            elif lower_is_better:
                better1 = val1 < val2
                better2 = val2 < val1
            else:
                better1 = val1 > val2
                better2 = val2 > val1

            with col1:
                formatted1 = fmt.format(val1)
                if better1:
                    st.markdown(f"**:green[{formatted1}]**")
                else:
                    st.markdown(formatted1)

            with col2:
                st.markdown(f"*{label}*")

            with col3:
                formatted2 = fmt.format(val2)
                if better2:
                    st.markdown(f"**:green[{formatted2}]**")
                else:
                    st.markdown(formatted2)

        # Radar chart comparison
        st.markdown("---")
        st.subheader("Visual Comparison")

        # Normalize metrics for radar chart (0-100 scale)
        radar_metrics = [
            ("Affordability", 100 - (city1["cost_of_living_index"] - 70), 100 - (city2["cost_of_living_index"] - 70)),
            ("Walkability", city1["walkability_score"], city2["walkability_score"]),
            ("Transit", city1["transit_score"], city2["transit_score"]),
            ("Safety", 100 - city1["crime_rate_per_1000"] * 2, 100 - city2["crime_rate_per_1000"] * 2),
            ("Schools", city1["avg_school_rating"] * 10, city2["avg_school_rating"] * 10),
            ("Weather", min(100, city1["sunny_days"] / 3), min(100, city2["sunny_days"] / 3)),
        ]

        categories = [m[0] for m in radar_metrics]
        city1_vals = [max(0, min(100, m[1])) for m in radar_metrics]
        city2_vals = [max(0, min(100, m[2])) for m in radar_metrics]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=city1_vals + [city1_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{city1['name']}, {city1['state']}",
            line=dict(color='rgb(0, 176, 80)'),
        ))

        fig.add_trace(go.Scatterpolar(
            r=city2_vals + [city2_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{city2['name']}, {city2['state']}",
            line=dict(color='rgb(99, 110, 250)'),
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Geographic features comparison
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{city1['name']} Features:**")
            features1 = []
            if city1["has_mountains"]: features1.append("Mountains")
            if city1["has_ocean"]: features1.append("Ocean")
            if city1["has_lakes"]: features1.append("Lakes")
            if city1["has_desert"]: features1.append("Desert")
            if city1["no_income_tax_state"]: features1.append("No State Income Tax")
            if city1["has_major_university"]: features1.append("Major University")
            st.markdown(", ".join(features1) if features1 else "No special features")

        with col2:
            st.markdown(f"**{city2['name']} Features:**")
            features2 = []
            if city2["has_mountains"]: features2.append("Mountains")
            if city2["has_ocean"]: features2.append("Ocean")
            if city2["has_lakes"]: features2.append("Lakes")
            if city2["has_desert"]: features2.append("Desert")
            if city2["no_income_tax_state"]: features2.append("No State Income Tax")
            if city2["has_major_university"]: features2.append("Major University")
            st.markdown(", ".join(features2) if features2 else "No special features")

    elif city1_name or city2_name:
        st.info("Select two cities to compare them side by side.")
    else:
        st.info("Select two cities from the dropdowns above to compare them.")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Take the Quiz"):
        st.switch_page("pages/1_Quiz.py")
with col2:
    if "quiz_answers" in st.session_state and st.session_state.quiz_answers:
        if st.button("View My Results"):
            st.switch_page("pages/2_Results.py")
