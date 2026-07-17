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
from utilities.tax_estimate import add_estimated_taxes_column

initialize_session_state("Find Your Spot - City Explorer", show_header_title=False)
render_navigation()

# Load city data
@st.cache_data
def get_cities_df():
    return load_cities()

cities_df = get_cities_df()

st.title("City Explorer")
st.markdown("Search, filter, and compare cities from our database of nearly 2,000 US cities.")

# --- Helper for range sliders from data ---
def _range_slider(label: str, col: str, fmt: str = "%d", **kwargs):
    """Create a sidebar range slider from column min/max."""
    series = cities_df[col].dropna()
    lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        return (lo, hi)
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), format=fmt, **kwargs)


def _int_range_slider(label: str, col: str, fmt: str = "%d", **kwargs):
    """Create a sidebar range slider with integer bounds."""
    series = cities_df[col].dropna()
    lo, hi = int(series.min()), int(series.max())
    if lo == hi:
        return (lo, hi)
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), format=fmt, **kwargs)


# ── Sidebar controls ─────────────────────────────────────────
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.header("Filter Cities")

# Location
st.sidebar.subheader("Location")
regions = ["All"] + sorted(cities_df["region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

if selected_region == "All":
    states = ["All"] + sorted(cities_df["state"].unique().tolist())
else:
    states = ["All"] + sorted(cities_df[cities_df["region"] == selected_region]["state"].unique().tolist())
selected_state = st.sidebar.selectbox("State", states)

pop_range = _int_range_slider("Population (city)", "population")
metro_pop_range = _int_range_slider("Metro Population", "metro_pop")
land_area_range = _range_slider("Land Area (sq mi)", "land_area_sq_mi", fmt="%.0f")

# Climate
st.sidebar.subheader("Climate")
temp_summer_range = _range_slider("Avg Summer Temp (F)", "avg_temp_summer", fmt="%.0f")
temp_winter_range = _range_slider("Avg Winter Temp (F)", "avg_temp_winter", fmt="%.0f")
avg_high_july_range = _range_slider("Avg High July (F)", "avg_high_july", fmt="%.0f")
avg_low_jan_range = _range_slider("Avg Low January (F)", "avg_low_january", fmt="%.0f")
sunny_days_range = _range_slider("Sunny Days/Year", "sunny_days", fmt="%.0f")
rainfall_range = _range_slider("Annual Rainfall (in)", "annual_rainfall", fmt="%.0f")
snowfall_range = _range_slider("Annual Snowfall (in)", "annual_snow", fmt="%.0f")

# Economics
st.sidebar.subheader("Economics")
col_range = _range_slider(
    "Housing Cost Index", "cost_of_living_index", fmt="%.0f",
    help="Based on home prices + rent (100 = National Average). Not a full COL measure.",
)
price_range = _int_range_slider("Median Home Price", "median_home_price", fmt="$%d")
rent_range = _range_slider("Median Gross Rent", "median_gross_rent", fmt="$%.0f")
income_range = _range_slider("Median Household Income", "median_household_income", fmt="$%.0f")
unemployment_range = _range_slider("Unemployment Rate (%)", "unemployment_rate", fmt="%.1f")
property_tax_range = _range_slider("Property Tax Rate (%)", "avg_property_tax_rate", fmt="%.2f")
state_income_tax_range = _range_slider("State Income Tax (%)", "state_income_tax_rate", fmt="%.1f")
state_sales_tax_range = _range_slider("Combined Sales Tax (%)", "state_sales_tax_rate", fmt="%.2f")
no_income_tax = st.sidebar.checkbox("No state income tax")

# Tax Estimator
st.sidebar.subheader("Tax Estimator")
st.sidebar.caption("Enter your financials to calculate estimated taxes per city")
est_income = st.sidebar.slider("Your Annual Income", min_value=0, max_value=500000, value=75000, step=5000, format="$%d")
est_home_value = st.sidebar.slider("Your Home Value", min_value=0, max_value=2000000, value=400000, step=25000, format="$%d")
est_expenses = st.sidebar.slider("Your Annual Taxable Spending", min_value=0, max_value=200000, value=40000, step=5000, format="$%d")

# Crime / Safety
st.sidebar.subheader("Crime / Safety")
crime_range = _range_slider("Total Crime Rate (per 1K)", "crime_rate_per_1000", fmt="%.1f")
violent_crime_range = _range_slider("Violent Crime Rate", "violent_crime_rate", fmt="%.1f")
property_crime_range = _range_slider("Property Crime Rate", "property_crime_rate", fmt="%.1f")

# Air Quality
st.sidebar.subheader("Air Quality")
median_aqi_range = _range_slider("Median AQI (lower=better)", "median_aqi", fmt="%.0f")
pct_good_days_range = _range_slider("Good Air Days (%)", "pct_good_days", fmt="%.0f")

# Government Spending
st.sidebar.subheader("Government Spending ($/capita)")
police_spending_range = _range_slider("Police ($/capita)", "police_spending_pc", fmt="$%.0f")
parks_spending_range = _range_slider("Parks ($/capita)", "parks_spending_pc", fmt="$%.0f")

# Education
st.sidebar.subheader("Education")
school_rating_range = _range_slider("Avg School Rating", "avg_school_rating", fmt="%.1f")
university_count_range = _int_range_slider("University Count", "university_count")
has_major_university = st.sidebar.checkbox("Major university")
has_community_college = st.sidebar.checkbox("Community college nearby")

# Transportation
st.sidebar.subheader("Transportation")
walkability_range = _int_range_slider("Walkability Score", "walkability_score")
transit_range = _int_range_slider("Transit Score", "transit_score")
airport_nearby = st.sidebar.selectbox(
    "Airport access",
    ["Any", "Within 15 mi", "Within 30 mi", "Within 60 mi", "Major hub only"],
    help="Distance to nearest major airport",
)
flight_dest_range = _int_range_slider("Direct Flight Destinations", "direct_flight_destinations_count")

# Geography / Outdoors
st.sidebar.subheader("Geography / Outdoors")
has_mountains = st.sidebar.checkbox("Mountains nearby")
has_ocean = st.sidebar.checkbox("Ocean access")
has_desert = st.sidebar.checkbox("Desert")
has_climbing = st.sidebar.checkbox("Rock climbing areas nearby")
ski_dist_range = _range_slider("Ski Resort Distance (mi)", "ski_resort_distance_miles", fmt="%.0f")
ocean_dist_range = _range_slider("Ocean Distance (mi)", "ocean_distance_miles", fmt="%.0f")
mountain_dist_range = _range_slider("Mountain Distance (mi)", "mountain_distance_miles", fmt="%.0f")
lake_dist_range = _range_slider("Boatable Lake Distance (mi)", "nearest_boatable_lake_miles", fmt="%.0f")
lakes_50_range = _int_range_slider("Boatable Lakes within 50 mi", "boatable_lakes_within_50mi")
hiking_range = _int_range_slider("Hiking Trails", "hiking_trails_count")
mt_biking_range = _int_range_slider("Mountain Biking Trails", "mountain_biking_trails")
camping_range = _int_range_slider("Camping Areas", "camping_areas_count")
state_parks_range = _int_range_slider("State Parks Nearby", "state_parks_nearby")
nat_parks_range = _int_range_slider("National Parks within 100 mi", "national_parks_within_100mi")
swimming_options = ["Any"] + sorted(cities_df["swimming_access"].dropna().unique().tolist())
swimming_filter = st.sidebar.selectbox("Swimming Access", swimming_options)

# Population Trend
st.sidebar.subheader("Population Trend")
if "pop_growth_5yr" in cities_df.columns:
    pop_growth_5yr_range = _range_slider("5-Year Growth (%)", "pop_growth_5yr", fmt="%.1f")
if "pop_growth_10yr" in cities_df.columns:
    pop_growth_10yr_range = _range_slider("10-Year Growth (%)", "pop_growth_10yr", fmt="%.1f")

# Restaurants / Food Scene
st.sidebar.subheader("Restaurants / Food")
if "restaurants_per_10k" in cities_df.columns:
    restaurants_per_10k_range = _range_slider("Restaurants per 10K", "restaurants_per_10k", fmt="%.0f")
if "bars_per_10k" in cities_df.columns:
    bars_per_10k_range = _range_slider("Bars per 10K", "bars_per_10k", fmt="%.0f")
if "breweries" in cities_df.columns:
    breweries_range = _int_range_slider("Breweries", "breweries")

# Natural Disasters
st.sidebar.subheader("Natural Disasters")
if "disaster_declarations_20yr" in cities_df.columns:
    disaster_range = _int_range_slider("Disaster Declarations (20yr)", "disaster_declarations_20yr")

# Healthcare
st.sidebar.subheader("Healthcare")
if "preventable_hospital_stays" in cities_df.columns:
    hospital_stays_range = _range_slider("Preventable Hospital Stays", "preventable_hospital_stays", fmt="%.0f")
if "obesity_pct" in cities_df.columns:
    obesity_range = _range_slider("Obesity Rate (%)", "obesity_pct", fmt="%.1f")
if "poor_mental_health_days_pct" in cities_df.columns:
    mental_health_range = _range_slider("Poor Mental Health Days (%)", "poor_mental_health_days_pct", fmt="%.1f")

# Water Quality
st.sidebar.subheader("Water Quality")
if "water_violation_rate" in cities_df.columns:
    water_viol_range = _range_slider("Violations/System/Year", "water_violation_rate", fmt="%.2f")

# Culture / Entertainment
st.sidebar.subheader("Culture / Entertainment")
museums_range = _int_range_slider("Museums", "museums_count")
performing_arts_range = _int_range_slider("Performing Arts Venues", "performing_arts_venues")
concert_range = _int_range_slider("Concert Venues", "concert_venue_count")
pro_sports_range = _int_range_slider("Pro Sports Teams (Major)", "major_pro_teams")

# ── Apply filters ─────────────────────────────────────────────
def _between(df, col, rng):
    """Filter df where col is between rng[0] and rng[1], keeping nulls."""
    return df[df[col].isna() | (df[col].between(rng[0], rng[1]))]

filtered_df = cities_df.copy()

# Location
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state"] == selected_state]

# Range filters — each keeps nulls so cities with missing data aren't silently dropped
filtered_df = _between(filtered_df, "population", pop_range)
filtered_df = _between(filtered_df, "metro_pop", metro_pop_range)
filtered_df = _between(filtered_df, "land_area_sq_mi", land_area_range)

# Climate
filtered_df = _between(filtered_df, "avg_temp_summer", temp_summer_range)
filtered_df = _between(filtered_df, "avg_temp_winter", temp_winter_range)
filtered_df = _between(filtered_df, "avg_high_july", avg_high_july_range)
filtered_df = _between(filtered_df, "avg_low_january", avg_low_jan_range)
filtered_df = _between(filtered_df, "sunny_days", sunny_days_range)
filtered_df = _between(filtered_df, "annual_rainfall", rainfall_range)
filtered_df = _between(filtered_df, "annual_snow", snowfall_range)

# Economics
filtered_df = _between(filtered_df, "cost_of_living_index", col_range)
filtered_df = _between(filtered_df, "median_home_price", price_range)
filtered_df = _between(filtered_df, "median_gross_rent", rent_range)
filtered_df = _between(filtered_df, "median_household_income", income_range)
filtered_df = _between(filtered_df, "unemployment_rate", unemployment_range)
filtered_df = _between(filtered_df, "avg_property_tax_rate", property_tax_range)
filtered_df = _between(filtered_df, "state_income_tax_rate", state_income_tax_range)
filtered_df = _between(filtered_df, "state_sales_tax_rate", state_sales_tax_range)
if no_income_tax:
    filtered_df = filtered_df[filtered_df["no_income_tax_state"] == True]

# Crime
filtered_df = _between(filtered_df, "crime_rate_per_1000", crime_range)
filtered_df = _between(filtered_df, "violent_crime_rate", violent_crime_range)
filtered_df = _between(filtered_df, "property_crime_rate", property_crime_range)

# Air Quality
filtered_df = _between(filtered_df, "median_aqi", median_aqi_range)
filtered_df = _between(filtered_df, "pct_good_days", pct_good_days_range)

# Government Spending
filtered_df = _between(filtered_df, "police_spending_pc", police_spending_range)
filtered_df = _between(filtered_df, "parks_spending_pc", parks_spending_range)

# Education
filtered_df = _between(filtered_df, "avg_school_rating", school_rating_range)
filtered_df = _between(filtered_df, "university_count", university_count_range)
if has_major_university:
    filtered_df = filtered_df[filtered_df["has_major_university"] == True]
if has_community_college:
    filtered_df = filtered_df[filtered_df["has_community_college"] == True]

# Transportation
filtered_df = _between(filtered_df, "walkability_score", walkability_range)
filtered_df = _between(filtered_df, "transit_score", transit_range)
filtered_df = _between(filtered_df, "direct_flight_destinations_count", flight_dest_range)
if airport_nearby == "Within 15 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 15]
elif airport_nearby == "Within 30 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 30]
elif airport_nearby == "Within 60 mi":
    filtered_df = filtered_df[filtered_df["airport_distance_miles"] <= 60]
elif airport_nearby == "Major hub only":
    filtered_df = filtered_df[filtered_df["is_airline_hub"] == True]

# Geography / Outdoors
if has_mountains:
    filtered_df = filtered_df[filtered_df["has_mountains"] == True]
if has_ocean:
    filtered_df = filtered_df[filtered_df["has_ocean"] == True]
if has_desert:
    filtered_df = filtered_df[filtered_df["has_desert"] == True]
if has_climbing:
    filtered_df = filtered_df[filtered_df["rock_climbing_areas_nearby"] >= 5]
filtered_df = _between(filtered_df, "ski_resort_distance_miles", ski_dist_range)
filtered_df = _between(filtered_df, "ocean_distance_miles", ocean_dist_range)
filtered_df = _between(filtered_df, "mountain_distance_miles", mountain_dist_range)
filtered_df = _between(filtered_df, "nearest_boatable_lake_miles", lake_dist_range)
filtered_df = _between(filtered_df, "boatable_lakes_within_50mi", lakes_50_range)
filtered_df = _between(filtered_df, "hiking_trails_count", hiking_range)
filtered_df = _between(filtered_df, "mountain_biking_trails", mt_biking_range)
filtered_df = _between(filtered_df, "camping_areas_count", camping_range)
filtered_df = _between(filtered_df, "state_parks_nearby", state_parks_range)
filtered_df = _between(filtered_df, "national_parks_within_100mi", nat_parks_range)
if swimming_filter != "Any":
    filtered_df = filtered_df[filtered_df["swimming_access"] == swimming_filter]

# Population Trend
if "pop_growth_5yr" in cities_df.columns:
    filtered_df = _between(filtered_df, "pop_growth_5yr", pop_growth_5yr_range)
if "pop_growth_10yr" in cities_df.columns:
    filtered_df = _between(filtered_df, "pop_growth_10yr", pop_growth_10yr_range)

# Restaurants / Food
if "restaurants_per_10k" in cities_df.columns:
    filtered_df = _between(filtered_df, "restaurants_per_10k", restaurants_per_10k_range)
if "bars_per_10k" in cities_df.columns:
    filtered_df = _between(filtered_df, "bars_per_10k", bars_per_10k_range)
if "breweries" in cities_df.columns:
    filtered_df = _between(filtered_df, "breweries", breweries_range)

# Natural Disasters
if "disaster_declarations_20yr" in cities_df.columns:
    filtered_df = _between(filtered_df, "disaster_declarations_20yr", disaster_range)

# Healthcare
if "preventable_hospital_stays" in cities_df.columns:
    filtered_df = _between(filtered_df, "preventable_hospital_stays", hospital_stays_range)
if "obesity_pct" in cities_df.columns:
    filtered_df = _between(filtered_df, "obesity_pct", obesity_range)
if "poor_mental_health_days_pct" in cities_df.columns:
    filtered_df = _between(filtered_df, "poor_mental_health_days_pct", mental_health_range)

# Water Quality
if "water_violation_rate" in cities_df.columns:
    filtered_df = _between(filtered_df, "water_violation_rate", water_viol_range)

# Culture / Entertainment
filtered_df = _between(filtered_df, "museums_count", museums_range)
filtered_df = _between(filtered_df, "performing_arts_venues", performing_arts_range)
filtered_df = _between(filtered_df, "concert_venue_count", concert_range)
filtered_df = _between(filtered_df, "major_pro_teams", pro_sports_range)

# Compute estimated taxes column
has_financials = est_income > 0 or est_home_value > 0 or est_expenses > 0
if has_financials:
    add_estimated_taxes_column(filtered_df, est_income, est_home_value, est_expenses)

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
            "cost_of_living_index": "Housing Cost Index",
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
            "state_sales_tax_rate": "Combined Sales Tax Rate",
            "median_household_income": "Median Household Income",
            "unemployment_rate": "Unemployment Rate",
            "ski_resort_distance_miles": "Ski Resort Distance (mi)",
            "airport_distance_miles": "Airport Distance (mi)",
            "rock_climbing_areas_nearby": "Rock Climbing Areas Nearby",
            "national_parks_within_100mi": "National Parks within 100mi",
            "median_aqi": "Median AQI (Air Quality)",
            "pct_good_days": "Good Air Quality Days (%)",
            "police_spending_pc": "Police Spending ($/capita)",
            "parks_spending_pc": "Parks Spending ($/capita)",
            "roads_spending_pc": "Roads Spending ($/capita)",
        }
        if "climbing_routes_nearby" in filtered_df.columns:
            numeric_columns["climbing_routes_nearby"] = "Climbing Routes Nearby"
        if "goods_rpp" in filtered_df.columns:
            numeric_columns["goods_rpp"] = "Goods Price Index (RPP)"
        if "pop_growth_5yr" in filtered_df.columns:
            numeric_columns["pop_growth_5yr"] = "5-Year Pop Growth (%)"
        if "pop_growth_10yr" in filtered_df.columns:
            numeric_columns["pop_growth_10yr"] = "10-Year Pop Growth (%)"
        if "restaurants_per_10k" in filtered_df.columns:
            numeric_columns["restaurants_per_10k"] = "Restaurants per 10K"
        if "bars_per_10k" in filtered_df.columns:
            numeric_columns["bars_per_10k"] = "Bars per 10K"
        if "breweries" in filtered_df.columns:
            numeric_columns["breweries"] = "Breweries"
        if "disaster_declarations_20yr" in filtered_df.columns:
            numeric_columns["disaster_declarations_20yr"] = "Disaster Declarations (20yr)"
        if "water_violation_rate" in filtered_df.columns:
            numeric_columns["water_violation_rate"] = "Water Violations/System/Year"
        if "water_pop_pct_affected" in filtered_df.columns:
            numeric_columns["water_pop_pct_affected"] = "Pop Affected by Water Violations (%)"
        if "obesity_pct" in filtered_df.columns:
            numeric_columns["obesity_pct"] = "Obesity Rate (%)"
        if "diabetes_pct" in filtered_df.columns:
            numeric_columns["diabetes_pct"] = "Diabetes Rate (%)"
        if "poor_mental_health_days_pct" in filtered_df.columns:
            numeric_columns["poor_mental_health_days_pct"] = "Poor Mental Health Days (%)"
        if "poor_physical_health_days_pct" in filtered_df.columns:
            numeric_columns["poor_physical_health_days_pct"] = "Poor Physical Health Days (%)"
        if "preventable_hospital_stays" in filtered_df.columns:
            numeric_columns["preventable_hospital_stays"] = "Preventable Hospital Stays"
        if "dental_visit_pct" in filtered_df.columns:
            numeric_columns["dental_visit_pct"] = "Dental Visit (%)"
        if "annual_checkup_pct" in filtered_df.columns:
            numeric_columns["annual_checkup_pct"] = "Annual Checkup (%)"
        if "debt_outstanding_pc" in filtered_df.columns:
            numeric_columns["debt_outstanding_pc"] = "Govt Debt ($/capita)"
        if "total_revenue_pc" in filtered_df.columns:
            numeric_columns["total_revenue_pc"] = "Govt Revenue ($/capita)"
        if "effective_interest_rate" in filtered_df.columns:
            numeric_columns["effective_interest_rate"] = "Govt Debt Interest Rate (%)"
        if "fire_spending_pc" in filtered_df.columns:
            numeric_columns["fire_spending_pc"] = "Fire Spending ($/capita)"
        if "sewerage_spending_pc" in filtered_df.columns:
            numeric_columns["sewerage_spending_pc"] = "Sewerage Spending ($/capita)"
        if "health_hospital_spending_pc" in filtered_df.columns:
            numeric_columns["health_hospital_spending_pc"] = "Health/Hospital Spending ($/capita)"
        if "diversity_index" in filtered_df.columns:
            numeric_columns["diversity_index"] = "Diversity Index"
        if "mean_commute_minutes" in filtered_df.columns:
            numeric_columns["mean_commute_minutes"] = "Mean Commute (min)"
        if "poverty_rate" in filtered_df.columns:
            numeric_columns["poverty_rate"] = "Poverty Rate (%)"
        if has_financials and "estimated_annual_taxes" in filtered_df.columns:
            numeric_columns["estimated_annual_taxes"] = "Estimated Annual Taxes ($)"

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
                         "estimated_annual_taxes", "median_aqi",
                         "ski_resort_distance_miles", "airport_distance_miles",
                         "avg_temp_summer", "avg_temp_winter",
                         "disaster_declarations_20yr", "water_violation_rate",
                         "water_pop_pct_affected", "obesity_pct", "diabetes_pct",
                         "poor_mental_health_days_pct", "poor_physical_health_days_pct",
                         "preventable_hospital_stays", "debt_outstanding_pc",
                         "effective_interest_rate", "mean_commute_minutes", "poverty_rate"]
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
        "cost_of_living_index": "Housing Index",
        "median_home_price": "Home Price",
        "no_income_tax_state": "No Income Tax",
        "avg_property_tax_rate": "Property Tax %",
        "walkability_score": "Walkability",
        "transit_score": "Transit",
        "crime_rate_per_1000": "Crime Rate",
        "avg_school_rating": "School Rating",
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
            "Housing Index": st.column_config.NumberColumn(format="%.0f"),
            "Property Tax %": st.column_config.NumberColumn(format="%.2f%%"),
            "Crime Rate": st.column_config.NumberColumn(format="%.1f"),
            "School Rating": st.column_config.NumberColumn(format="%.1f"),
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

        # --- Helper to render a comparison row ---
        _LOWER_IS_BETTER = {
            "cost_of_living_index", "median_home_price", "median_gross_rent",
            "crime_rate_per_1000", "violent_crime_rate", "property_crime_rate",
            "unemployment_rate", "poverty_rate", "pct_uninsured",
            "avg_property_tax_rate", "state_income_tax_rate", "state_sales_tax_rate",
            "avg_frl_rate", "avg_pupil_teacher_ratio",
            "airport_distance_miles", "nearest_hub_distance_miles",
            "ski_resort_distance_miles", "ocean_distance_miles", "mountain_distance_miles",
            "mean_commute_minutes", "estimated_annual_taxes",
            "disaster_declarations_20yr", "water_violation_rate", "water_pop_pct_affected",
            "water_systems_with_violations", "water_systems_health_violations",
            "obesity_pct", "diabetes_pct", "poor_mental_health_days_pct",
            "poor_physical_health_days_pct", "preventable_hospital_stays",
        }
        _NEUTRAL = {
            "lat", "lon", "fips_state", "fips_place", "fips_county",
            "cbsa_code", "cbsa_name", "region", "name", "state", "city_id",
            "nearest_airport_code", "pro_leagues", "major_industries",
            "swimming_access", "interpolated", "incorporated_places",
            "median_age", "pct_white", "pct_black", "pct_hispanic", "pct_asian",
        }

        def _fmt_val(val, fmt):
            if pd.isna(val):
                return "N/A"
            if isinstance(val, bool):
                return "Yes" if val else "No"
            try:
                return fmt.format(val)
            except (ValueError, TypeError):
                return str(val)

        def _render_rows(metrics, c1, c2):
            """Render comparison rows. metrics: list of (label, col, fmt)."""
            for label, col_name, fmt in metrics:
                v1 = c1.get(col_name)
                v2 = c2.get(col_name)
                f1 = _fmt_val(v1, fmt)
                f2 = _fmt_val(v2, fmt)

                # Determine "better" value
                better1, better2 = False, False
                if col_name not in _NEUTRAL and pd.notna(v1) and pd.notna(v2):
                    if not isinstance(v1, (bool, str)):
                        if col_name in _LOWER_IS_BETTER:
                            better1 = v1 < v2
                            better2 = v2 < v1
                        else:
                            better1 = v1 > v2
                            better2 = v2 > v1

                left, mid, right = st.columns([2, 2, 2])
                with left:
                    st.markdown(f"**:green[{f1}]**" if better1 else f1)
                with mid:
                    st.markdown(f"*{label}*")
                with right:
                    st.markdown(f"**:green[{f2}]**" if better2 else f2)

        # --- Headers ---
        left, mid, right = st.columns([2, 2, 2])
        with left:
            st.markdown(f"### {city1['name']}, {city1['state']}")
        with mid:
            st.markdown("### Metric")
        with right:
            st.markdown(f"### {city2['name']}, {city2['state']}")

        # --- Grouped metrics in expanders ---
        with st.expander("Location & Overview", expanded=True):
            _render_rows([
                ("Region", "region", "{}"),
                ("Population", "population", "{:,}"),
                ("Metro Population", "metro_pop", "{:,}"),
                ("Land Area (sq mi)", "land_area_sq_mi", "{:.1f}"),
                ("Latitude", "lat", "{:.4f}"),
                ("Longitude", "lon", "{:.4f}"),
            ], city1, city2)

        with st.expander("Climate & Weather", expanded=True):
            _render_rows([
                ("Avg Summer Temp (F)", "avg_temp_summer", "{:.0f}"),
                ("Avg Winter Temp (F)", "avg_temp_winter", "{:.0f}"),
                ("Avg High July (F)", "avg_high_july", "{:.0f}"),
                ("Avg Low January (F)", "avg_low_january", "{:.0f}"),
                ("Sunny Days/Year", "sunny_days", "{:.0f}"),
                ("Annual Rainfall (in)", "annual_rainfall", "{:.1f}"),
                ("Annual Snowfall (in)", "annual_snow", "{:.1f}"),
            ], city1, city2)

        with st.expander("Housing, Goods & Taxes", expanded=True):
            cost_tax_rows = [
                ("Housing Cost Index", "cost_of_living_index", "{:.0f}"),
                ("Goods Price Index (RPP)", "goods_rpp", "{:.1f}"),
                ("Median Home Price", "median_home_price", "${:,.0f}"),
                ("Median Gross Rent", "median_gross_rent", "${:,.0f}"),
                ("Median Household Income", "median_household_income", "${:,.0f}"),
                ("Property Tax Rate", "avg_property_tax_rate", "{:.2f}%"),
                ("State Income Tax", "state_income_tax_rate", "{:.1f}%"),
                ("No State Income Tax", "no_income_tax_state", "{}"),
                ("Combined Sales Tax", "state_sales_tax_rate", "{:.2f}%"),
            ]
            if has_financials and "estimated_annual_taxes" in filtered_df.columns:
                cost_tax_rows.append(("Est. Annual Taxes", "estimated_annual_taxes", "${:,.0f}"))
            _render_rows(cost_tax_rows, city1, city2)

        with st.expander("Crime & Safety"):
            _render_rows([
                ("Total Crime Rate (per 1K)", "crime_rate_per_1000", "{:.1f}"),
                ("Violent Crime Rate", "violent_crime_rate", "{:.1f}"),
                ("Property Crime Rate", "property_crime_rate", "{:.1f}"),
            ], city1, city2)

        with st.expander("Air Quality"):
            _render_rows([
                ("Median AQI (lower=better)", "median_aqi", "{:.0f}"),
                ("Good Air Days (%)", "pct_good_days", "{:.1f}%"),
                ("Unhealthy Air Days/Year", "days_unhealthy_total", "{:.0f}"),
                ("Max AQI", "max_aqi", "{:.0f}"),
            ], city1, city2)

        with st.expander("Government Spending ($/capita)"):
            _render_rows([
                ("Police", "police_spending_pc", "${:,.0f}"),
                ("Fire", "fire_spending_pc", "${:,.0f}"),
                ("Parks & Recreation", "parks_spending_pc", "${:,.0f}"),
                ("Roads/Highways", "roads_spending_pc", "${:,.0f}"),
                ("Sewerage", "sewerage_spending_pc", "${:,.0f}"),
                ("Health & Hospitals", "health_hospital_spending_pc", "${:,.0f}"),
            ], city1, city2)

        with st.expander("Education"):
            _render_rows([
                ("Avg School Rating (1-10)", "avg_school_rating", "{:.1f}"),
                ("Pupil-Teacher Ratio", "avg_pupil_teacher_ratio", "{:.1f}"),
                ("Free/Reduced Lunch Rate", "avg_frl_rate", "{:.1%}"),
                ("K-12 Schools Nearby", "k12_school_count", "{:,}"),
                ("K-12 Total Enrollment", "total_k12_enrollment", "{:,}"),
                ("Universities", "university_count", "{}"),
                ("Community Colleges", "community_college_count", "{}"),
                ("Major University", "has_major_university", "{}"),
                ("R1/R2 Research Univ.", "has_r1_r2", "{}"),
                ("Community College", "has_community_college", "{}"),
                ("Total Institutions", "total_institutions", "{}"),
                ("College Town Score", "college_town_score", "{:.3f}"),
                ("Largest Enrollment", "largest_enrollment", "{:,}"),
                ("Total Student Pop.", "total_student_population", "{:,}"),
            ], city1, city2)

        with st.expander("Transportation & Commute"):
            _render_rows([
                ("Walkability Score", "walkability_score", "{:.0f}"),
                ("Transit Score", "transit_score", "{:.0f}"),
                ("Mean Commute (min)", "mean_commute_minutes", "{:.0f}"),
                ("Drove to Work (%)", "pct_drove", "{:.1f}%"),
                ("Public Transit (%)", "pct_public_transit", "{:.1f}%"),
                ("Walked (%)", "pct_walked", "{:.1f}%"),
                ("Work from Home (%)", "pct_work_from_home", "{:.1f}%"),
                ("Nearest Airport", "nearest_airport_code", "{}"),
                ("Airport Distance (mi)", "airport_distance_miles", "{:.0f}"),
                ("Airline Hub", "is_airline_hub", "{}"),
                ("Direct Destinations", "direct_flight_destinations_count", "{}"),
                ("Nearest Hub Distance (mi)", "nearest_hub_distance_miles", "{:.0f}"),
            ], city1, city2)

        with st.expander("Geography & Outdoors"):
            _render_rows([
                ("Mountains Nearby", "has_mountains", "{}"),
                ("Ocean Access", "has_ocean", "{}"),
                ("Desert", "has_desert", "{}"),
                ("Ocean Distance (mi)", "ocean_distance_miles", "{:.0f}"),
                ("Mountain Distance (mi)", "mountain_distance_miles", "{:.0f}"),
                ("Boatable Lake Distance (mi)", "nearest_boatable_lake_miles", "{:.0f}"),
                ("Boatable Lakes within 50 mi", "boatable_lakes_within_50mi", "{}"),
                ("Ski Resort Distance (mi)", "ski_resort_distance_miles", "{:.0f}"),
                ("Hiking Trails", "hiking_trails_count", "{}"),
                ("Mountain Biking Trails", "mountain_biking_trails", "{}"),
                ("Rock Climbing Areas", "rock_climbing_areas_nearby", "{}"),
                ("Climbing Routes", "climbing_routes_nearby", "{:,}"),
                ("Camping Areas", "camping_areas_count", "{}"),
                ("State Parks Nearby", "state_parks_nearby", "{}"),
                ("National Parks (100 mi)", "national_parks_within_100mi", "{}"),
                ("Swimming Access", "swimming_access", "{}"),
            ], city1, city2)

        with st.expander("Culture & Entertainment"):
            _render_rows([
                ("Museums (IMLS)", "museums_count", "{}"),
                ("Performing Arts Venues", "performing_arts_venues", "{}"),
                ("Concert Venues", "concert_venue_count", "{}"),
                ("Major Pro Teams", "major_pro_teams", "{}"),
                ("Minor Pro Teams", "minor_pro_teams", "{}"),
                ("Total Pro Teams", "total_pro_teams", "{}"),
                ("Pro Leagues", "pro_leagues", "{}"),
            ], city1, city2)

        with st.expander("Demographics"):
            _render_rows([
                ("Median Age", "median_age", "{:.1f}"),
                ("Under 18 (%)", "pct_under_18", "{:.1f}%"),
                ("Over 65 (%)", "pct_over_65", "{:.1f}%"),
                ("Diversity Index", "diversity_index", "{:.3f}"),
                ("White (%)", "pct_white", "{:.1f}%"),
                ("Black (%)", "pct_black", "{:.1f}%"),
                ("Hispanic (%)", "pct_hispanic", "{:.1f}%"),
                ("Asian (%)", "pct_asian", "{:.1f}%"),
                ("Bachelor's Degree+ (%)", "pct_bachelors_plus", "{:.1f}%"),
                ("High School+ (%)", "pct_high_school_plus", "{:.1f}%"),
            ], city1, city2)

        with st.expander("Economy & Employment"):
            _render_rows([
                ("Unemployment Rate", "unemployment_rate", "{:.1f}%"),
                ("Poverty Rate", "poverty_rate", "{:.1f}%"),
                ("Uninsured (%)", "pct_uninsured", "{:.1f}%"),
                ("Job Growth Rate", "job_growth_rate", "{:.1f}%"),
                ("5-Year Pop Growth", "pop_growth_5yr", "{:+.1f}%"),
                ("10-Year Pop Growth", "pop_growth_10yr", "{:+.1f}%"),
            ], city1, city2)

        with st.expander("Restaurants & Food"):
            _render_rows([
                ("Restaurants (Full-Service)", "restaurants_fullservice", "{:,}"),
                ("Restaurants (Quick-Service)", "restaurants_quickservice", "{:,}"),
                ("Coffee/Snack Bars", "coffee_snack_bars", "{:,}"),
                ("Bars/Pubs", "bars", "{:,}"),
                ("Breweries", "breweries", "{}"),
                ("Restaurants per 10K", "restaurants_per_10k", "{:.1f}"),
                ("Bars per 10K", "bars_per_10k", "{:.1f}"),
            ], city1, city2)

        with st.expander("Natural Disasters"):
            _render_rows([
                ("Disaster Declarations (20yr)", "disaster_declarations_20yr", "{}"),
            ], city1, city2)

        with st.expander("Healthcare & Wellness"):
            _render_rows([
                ("Preventable Hospital Stays", "preventable_hospital_stays", "{:.0f}"),
                ("Obesity Rate (%)", "obesity_pct", "{:.1f}%"),
                ("Diabetes Rate (%)", "diabetes_pct", "{:.1f}%"),
                ("Poor Mental Health Days (%)", "poor_mental_health_days_pct", "{:.1f}%"),
                ("Poor Physical Health Days (%)", "poor_physical_health_days_pct", "{:.1f}%"),
                ("Annual Checkup (%)", "annual_checkup_pct", "{:.1f}%"),
                ("Dental Visit (%)", "dental_visit_pct", "{:.1f}%"),
            ], city1, city2)

        with st.expander("Water Quality"):
            _render_rows([
                ("Water Systems (Total)", "water_systems_total", "{}"),
                ("Systems with Violations", "water_systems_with_violations", "{}"),
                ("Health Violations", "water_systems_health_violations", "{}"),
                ("Violations/System/Year", "water_violation_rate", "{:.2f}"),
                ("Pop Affected (%)", "water_pop_pct_affected", "{:.1f}%"),
            ], city1, city2)

        # Radar chart comparison
        st.divider()
        st.subheader("Visual Comparison")

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

    elif city1_name or city2_name:
        st.info("Select two cities to compare them side by side.")
    else:
        st.info("Select two cities from the dropdowns above to compare them.")

