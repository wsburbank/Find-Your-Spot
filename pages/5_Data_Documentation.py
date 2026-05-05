"""
Data Documentation Page - View raw data and understand the database
"""
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import load_cities
from components.navigation import render_navigation

PROJECT_ROOT = Path(__file__).parent.parent

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
- **Climate Data**: NOAA ACIS 1991-2020 Climate Normals
- **Sunshine (Sunny Days)**: NOAA NCEI 1991-2020 Hourly Climate Normals (cloud cover)
- **Crime Statistics**: FBI Uniform Crime Reporting (UCR) 2022 state-level data
- **Airport Data**: OurAirports public dataset + BTS route data (2024)
- **Walkability & Transit**: EPA Smart Location Database V3 (January 2021)
- **Education Data**: NCES IPEDS (colleges/universities), NCES CCD 2022-23 (K-12)
- **Employment**: US Census ACS 5-Year (B23025) 2021-2022 (unemployment, job growth)
- **Outdoor Recreation**: PAD-US (state parks), USFS EDW + NPS (trails/campgrounds), OpenBeta (climbing)
- **Demographics**: US Census ACS 2022 (diversity, age, education, health insurance)
- **Entertainment**: Census CBP 2022 (venues), IMLS 2018 (museums), official league rosters (sports)
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

# ---------------------------------------------------------------------------
# Interactive Data Map
# ---------------------------------------------------------------------------

st.header("Data Map")
st.markdown("Explore geographic data layers on an interactive map. Select datasets to display.")

# Layer definitions: name -> (color_rgb, loader_function)
# Colors chosen from BPX brand palette for visual distinction

LAYER_COLORS = {
    "Cities": [0, 127, 0],           # bp green
    "Ski Resorts": [0, 122, 201],     # bp blue
    "National Parks": [102, 0, 153],  # bp purple
    "Airports": [255, 153, 0],        # bp yellow orange
    "Lakes": [0, 0, 153],             # bp dark blue
    "State Parks": [153, 204, 0],     # bp light green
    "Campgrounds": [210, 70, 20],     # bp orange
    "Climbing Areas": [102, 102, 102],  # bp dark grey
    "Museums": [255, 230, 0],         # bp yellow
}


@st.cache_data(show_spinner=False)
def _load_layer(layer_name: str) -> pd.DataFrame:
    """Load a geographic dataset and return a DataFrame with name, lat, lon."""
    if layer_name == "Cities":
        df = load_cities()
        return df[["name", "state", "lat", "lon"]].assign(
            label=df["name"] + ", " + df["state"]
        )

    if layer_name == "Ski Resorts":
        from scripts.collect_geography import SKI_RESORTS
        records = [{"name": r[0], "lat": r[1], "lon": r[2]} for r in SKI_RESORTS]
        return pd.DataFrame(records).assign(label=lambda d: d["name"])

    if layer_name == "National Parks":
        from scripts.collect_geography import NATIONAL_PARKS
        records = [{"name": r[0], "lat": r[1], "lon": r[2]} for r in NATIONAL_PARKS]
        return pd.DataFrame(records).assign(label=lambda d: d["name"])

    if layer_name == "Airports":
        csv_path = PROJECT_ROOT / "data" / "cache" / "ourairports.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        raw = pd.read_csv(csv_path)
        us = raw[
            (raw["iso_country"] == "US")
            & (raw["scheduled_service"] == "yes")
            & raw["type"].isin(["large_airport", "medium_airport", "small_airport"])
        ].dropna(subset=["latitude_deg", "longitude_deg", "iata_code"]).copy()
        return pd.DataFrame({
            "name": us["name"],
            "lat": us["latitude_deg"],
            "lon": us["longitude_deg"],
            "label": us["iata_code"] + " - " + us["name"],
        })

    if layer_name == "Lakes":
        parquet = PROJECT_ROOT / "data" / "lakes.parquet"
        if not parquet.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        df = pd.read_parquet(parquet)
        # Only centroids, skip boundary sample points
        df = df[~df["is_boundary_point"]].copy()
        area = df["area_acres"].round(0).astype(int).astype(str)
        return pd.DataFrame({
            "name": df["name"],
            "lat": df["lat"],
            "lon": df["lon"],
            "label": df["name"].fillna("Unnamed") + " (" + area + " acres)",
        })

    if layer_name == "State Parks":
        parquet = PROJECT_ROOT / "data" / "raw" / "outdoor_recreation" / "padus_state_parks.parquet"
        if not parquet.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        df = pd.read_parquet(parquet)
        return pd.DataFrame({
            "name": df["name"],
            "lat": df["lat"],
            "lon": df["lon"],
            "label": df["name"] + " (" + df["state"] + ")",
        })

    if layer_name == "Campgrounds":
        parquet = PROJECT_ROOT / "data" / "raw" / "outdoor_recreation" / "usfs_campgrounds.parquet"
        if not parquet.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        df = pd.read_parquet(parquet)
        # Filter to valid US coordinates
        df = df[(df["lat"] > 17) & (df["lat"] < 72) & (df["lon"] > -180) & (df["lon"] < -65)].copy()
        return pd.DataFrame({
            "name": df["name"],
            "lat": df["lat"],
            "lon": df["lon"],
            "label": df["name"],
        })

    if layer_name == "Climbing Areas":
        parquet = PROJECT_ROOT / "data" / "raw" / "outdoor_recreation" / "openbeta_climbing.parquet"
        if not parquet.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        df = pd.read_parquet(parquet)
        climbs = df["total_climbs"].fillna(0).astype(int).astype(str)
        return pd.DataFrame({
            "name": df["name"],
            "lat": df["lat"],
            "lon": df["lon"],
            "label": df["name"] + " (" + climbs + " routes)",
        })

    if layer_name == "Museums":
        csv_path = PROJECT_ROOT / "data" / "cache" / "imls_museums.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=["name", "lat", "lon", "label"])
        raw = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        raw["LATITUDE"] = pd.to_numeric(raw["LATITUDE"], errors="coerce")
        raw["LONGITUDE"] = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
        raw = raw.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
        return pd.DataFrame({
            "name": raw["COMMONNAME"],
            "lat": raw["LATITUDE"],
            "lon": raw["LONGITUDE"],
            "label": raw["COMMONNAME"].fillna("Museum"),
        })

    return pd.DataFrame(columns=["name", "lat", "lon", "label"])


# Layer selection
available_layers = list(LAYER_COLORS.keys())
selected_layers = st.multiselect(
    "Select data layers to display",
    options=available_layers,
    default=["Cities", "Ski Resorts", "National Parks"],
)

if selected_layers:
    # Build pydeck layers
    deck_layers = []
    legend_items = []

    for layer_name in selected_layers:
        with st.spinner(f"Loading {layer_name}..."):
            df = _load_layer(layer_name)

        if df.empty:
            st.warning(f"No data available for {layer_name}.")
            continue

        color = LAYER_COLORS[layer_name]
        legend_items.append((layer_name, color, len(df)))

        # Scale point size: cities and large datasets get smaller points
        if layer_name == "Cities":
            radius = 6000
        elif len(df) > 5000:
            radius = 2000
        elif len(df) > 1000:
            radius = 3000
        else:
            radius = 4500

        layer_df = df[["lat", "lon", "label"]].copy()
        layer_df["color"] = [color + [200]] * len(layer_df)  # RGBA with alpha

        deck_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=layer_df,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=radius,
                pickable=True,
                auto_highlight=True,
            )
        )

    if deck_layers:
        tooltip = {
            "html": "<b>{label}</b>",
            "style": {
                "backgroundColor": "#333",
                "color": "white",
                "fontSize": "13px",
                "padding": "8px",
            },
        }

        view_state = pdk.ViewState(
            latitude=39.0,
            longitude=-98.0,
            zoom=3.5,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=deck_layers,
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            ),
            height=550,
        )

        # Legend
        legend_html = " &nbsp;|&nbsp; ".join(
            f'<span style="color:rgb({c[0]},{c[1]},{c[2]});font-weight:bold;">'
            f"&#9679;</span> {name} ({count:,})"
            for name, c, count in legend_items
        )
        st.markdown(legend_html, unsafe_allow_html=True)
    else:
        st.info("No data loaded for selected layers.")
else:
    st.info("Select one or more data layers above to display them on the map.")

st.divider()

# Data dictionary
st.header("Data Dictionary")

data_dict = {
    "Identity & Location": {
        "city_id": "Unique identifier for the city",
        "name": "City name",
        "state": "Two-letter state abbreviation",
        "lat": "Latitude coordinate",
        "lon": "Longitude coordinate",
        "population": "City proper population (Census 2022 ACS)",
        "area_population": "City + nearby eliminated cities' population (20-mile roll-up)",
        "metro_pop": "CBSA metropolitan/micropolitan area population (OMB 2023)",
        "incorporated_places": "Names of nearby cities rolled into this city's area population",
        "cbsa_code": "Core Based Statistical Area code (OMB 2023)",
        "cbsa_name": "CBSA metropolitan/micropolitan area name",
        "region": "Geographic region (Northeast, South, Midwest, West)",
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
        "has_lakes": "Boatable lake within 50 miles (True/False) — derived from NHD distance",
        "has_desert": "Desert landscape (True/False)",
        "nearest_boatable_lake_miles": "Distance to nearest lake/reservoir >= 300 acres (miles) — USGS NHD",
        "nearest_lake_area_acres": "Area of the nearest boatable lake (acres) — USGS NHD",
        "boatable_lakes_within_50mi": "Count of lakes/reservoirs >= 300 acres within 50 miles — USGS NHD",
    },
    "Outdoor Recreation": {
        "ski_resort_distance_miles": "Distance to nearest ski resort (miles) — NSAA resort coordinates",
        "national_parks_within_100mi": "National parks within 100 miles — NPS coordinates",
        "state_parks_nearby": "State parks within 50 miles — PAD-US 4.1 (USGS)",
        "camping_areas_count": "Campgrounds within 60 miles — USFS EDW + NPS API",
        "hiking_trails_count": "Hiking trail segments within 50 miles — USFS EDW + NPS Public Trails",
        "mountain_biking_trails": "Bicycle-managed trail segments within 50 miles — USFS EDW + NPS Public Trails",
        "rock_climbing_areas_nearby": "Rock climbing areas within 75 miles — OpenBeta",
        "swimming_access": "Primary swimming access type (ocean/lake/pool) — derived",
    },
    "Transportation": {
        "nearest_major_airport": "Nearest major airport name",
        "airport_distance_miles": "Distance to nearest major airport (miles)",
        "is_airline_hub": "Whether city has an airline hub (True/False)",
        "direct_flight_destinations_count": "Number of direct flight destinations",
        "walkability_score": "Walkability score (0-100) — EPA Smart Location Database V3 (2021)",
        "transit_score": "Public transit score (0-100) — EPA Smart Location Database V3 (2021)",
    },
    "Education": {
        "university_count": "Number of universities/colleges — NCES IPEDS",
        "community_college_count": "Number of community colleges — NCES IPEDS",
        "has_major_university": "Has a major research university (True/False) — NCES IPEDS",
        "has_r1_r2": "Has an R1 or R2 research university (True/False) — NCES IPEDS",
        "has_community_college": "Community college nearby (True/False) — NCES IPEDS",
        "total_institutions": "Total higher-ed institutions — NCES IPEDS",
        "largest_enrollment": "Largest institution enrollment — NCES IPEDS",
        "total_student_population": "Total student population — NCES IPEDS",
        "college_town_score": "College town score (0-100) — NCES IPEDS",
        "avg_school_rating": "Average K-12 school rating (1-10) — NCES CCD 2022-23 (PTR + FRL composite)",
    },
    "Culture & Entertainment": {
        "major_pro_teams": "Number of major professional sports teams — official league rosters",
        "minor_pro_teams": "Number of minor league sports teams — official league rosters",
        "total_pro_teams": "Total professional sports teams",
        "has_major_pro_team": "Has at least one major pro team (True/False)",
        "pro_leagues": "Leagues represented (comma-separated)",
        "performing_arts_venues": "Performing arts venues — Census CBP 2022",
        "concert_venue_count": "Concert/event venues — Census CBP 2022",
        "museums_count": "Number of museums — IMLS 2018 Museum Data Files",
    },
    "Demographics": {
        "diversity_index": "Simpson's Diversity Index (0-1) — Census ACS 2022 B03002",
        "pct_white": "% White (non-Hispanic) — Census ACS 2022 B03002",
        "pct_black": "% Black/African American — Census ACS 2022 B03002",
        "pct_asian": "% Asian — Census ACS 2022 B03002",
        "pct_hispanic": "% Hispanic/Latino — Census ACS 2022 B03002",
        "median_age": "Median age — Census ACS 2022 S0101",
        "pct_under_18": "% population under 18 — Census ACS 2022 S0101",
        "pct_over_65": "% population 65 and over — Census ACS 2022 S0101",
        "pct_bachelors_plus": "% with bachelor's degree or higher (pop 25+) — Census ACS 2022 S1501",
        "pct_high_school_plus": "% with high school diploma or higher (pop 25+) — Census ACS 2022 S1501",
        "pct_uninsured": "% without health insurance — Census ACS 2022 S2701",
    },
    "Economy & Employment": {
        "median_household_income": "Median household income ($) — Census ACS 2022 B19013",
        "unemployment_rate": "Unemployment rate (%) — Census ACS 5-Year B23025 (county-level, 2022)",
        "poverty_rate": "Poverty rate (%) — Census ACS 2022 DP03",
        "major_industries": "Top 3 industries by employment (%) — Census ACS 2022 DP03",
        "job_growth_rate": "Annual job growth rate (%) — Census ACS 5-Year B23025 (2021 vs 2022 YoY)",
    },
    "Safety": {
        "crime_rate_per_1000": "Total crime rate per 1,000 residents",
        "violent_crime_rate": "Violent crime rate per 1,000 residents",
        "property_crime_rate": "Property crime rate per 1,000 residents",
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
    width="stretch",
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
        width="stretch",
    )

st.divider()

# Data sources note
st.header("Parquet File Sources")
try:
    from utilities.data_io import list_parquet_sources
    sources = list_parquet_sources()
    if not sources.empty:
        st.dataframe(sources, width="stretch", hide_index=True)
    else:
        st.info("No parquet files found in data directory.")
except Exception as e:
    st.warning(f"Could not load parquet metadata: {e}")

st.divider()

st.header("Data Sources")
st.markdown("""
**Current data sources used in this application:**

| Category | Source | Year | Coverage |
|----------|--------|------|----------|
| Population & Coordinates | US Census Bureau ACS 5-Year + Gazetteer | 2022-2023 | 100% |
| Metro Populations (CBSA) | OMB Delineation Files + Census ACS | 2022-2023 | 95% |
| Climate Normals | NOAA ACIS 1991-2020 Normals | 2020 | 99% |
| Sunshine (Sunny Days) | NOAA NCEI 1991-2020 Hourly Normals (cloud cover) | 2020 | 100% |
| Home Prices & Income | US Census Bureau ACS 5-Year (B25077, B19013) | 2022 | 99.8% |
| State Tax Rates | Tax Foundation published data | 2024 | 100% |
| Crime Rates | FBI UCR Crime in the US (state-level) | 2022 | 100% |
| Airport Locations | OurAirports public dataset (scheduled service) | 2024 | 100% |
| Walkability & Transit | EPA Smart Location Database V3 (NatWalkInd, D4A) | 2021 | 100% |
| Geographic Features | USGS GNIS (mountains) + Natural Earth (coastline) | 2024 | 100% |
| National Parks | NPS park coordinates (static reference) | 2024 | 100% |
| Ski Resorts | NSAA resort locations (static reference) | 2024 | 100% |
| Lakes | USGS National Hydrography Dataset (NHD) | 2024 | 100% |
| State Parks | PAD-US 4.1 ArcGIS REST API (USGS) | 2024 | 100% |
| Campgrounds | USFS EDW recreation sites + NPS API | 2024 | 100% |
| Hiking/Biking Trails | USFS EDW + NPS Public Trails (106K segments) | 2024 | 100% |
| Rock Climbing Areas | OpenBeta GraphQL API (CC0 license) | 2024 | 100% |
| Commute & Transportation | Census ACS 5-Year (B08301, B08303) | 2022 | 99.9% |
| K-12 School Ratings | NCES CCD 2022-23 (pupil-teacher ratio + free lunch) | 2023 | 99% |
| Higher Education | NCES IPEDS (colleges/universities) | 2022 | 89% |
| Employment | Census ACS 5-Year B23025 (unemployment + job growth) | 2021-2022 | 99% |
| Museums | IMLS 2018 Museum Data Files | 2018 | 98% |
| Pro Sports Teams | Official league rosters (Wikipedia) | 2024 | 100% |
| Entertainment Venues | Census County Business Patterns (CBP) | 2022 | 46% |
| Demographics & Diversity | Census ACS 5-Year (B03002, S0101, S1501, S2701, DP03) | 2022 | 99.9% |

**Static reference data** (hardcoded with source citations, verified 2026-05-02):

| Data | Source | Notes |
|------|--------|-------|
| State Tax Rates | Tax Foundation 2024 | Income, sales, property rates per state |
| Ski Resort Locations | NSAA / public resort data | ~120 resorts with lat/lon coordinates |
| National Park Locations | NPS.gov | ~60 parks with lat/lon coordinates |
| Airport Hub Destinations | BTS / airline route maps 2024 | Approximate nonstop destination counts |
| Desert Regions | Geographic reference | Bounding circles for 5 major US deserts |

**Derived fields:**

| Field | Method | Notes |
|-------|--------|-------|
| swimming_access | Derived from ocean/lake proximity | "ocean" if coastal, "lake" if within 50mi, else "pool" |
| cost_of_living_index | Derived from median home price + income | Composite index (100 = national average) |
| crime rates | FBI UCR state-level data applied to cities | State-level rates, not city-specific |
""")
