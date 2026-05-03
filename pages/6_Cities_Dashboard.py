"""
Cities Dashboard - Interactive map and data explorer for all cities in the database.
"""

import json
import streamlit as st
import pandas as pd
import pydeck as pdk
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.scoring import load_cities
from components.navigation import render_navigation

initialize_session_state("Find Your Spot - Cities Dashboard", show_header_title=False)
render_navigation()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def get_cities_df() -> pd.DataFrame:
    return load_cities()


@st.cache_data
def get_state_boundaries() -> dict | None:
    """Load US state boundaries GeoJSON for map overlay."""
    path = DATA_DIR / "us_states.geojson"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


cities_df = get_cities_df()
state_geojson = get_state_boundaries()

st.title("Cities Dashboard")

# ---------------------------------------------------------------------------
# Metric configuration for map visualization
# ---------------------------------------------------------------------------

# Numeric columns available for color/size mapping
METRIC_OPTIONS: dict[str, str] = {}
for col in cities_df.select_dtypes(include=["float64", "int64", "Int64"]).columns:
    if col in ("city_id", "fips_state", "fips_place", "fips_county"):
        continue
    non_null = cities_df[col].notna().sum()
    if non_null > len(cities_df) * 0.5:
        METRIC_OPTIONS[col] = col

# Friendly labels for common columns
FRIENDLY_LABELS = {
    "population": "Population",
    "area_population": "Area Population (with roll-up)",
    "metro_pop": "Metro Population (CBSA)",
    "avg_temp_summer": "Avg Summer Temp (F)",
    "avg_temp_winter": "Avg Winter Temp (F)",
    "sunny_days": "Sunny Days / Year",
    "annual_rainfall": "Annual Rainfall (in)",
    "annual_snow": "Annual Snowfall (in)",
    "cost_of_living_index": "Cost of Living Index",
    "median_home_price": "Median Home Price ($)",
    "median_household_income": "Median Household Income ($)",
    "state_income_tax_rate": "State Income Tax Rate (%)",
    "avg_property_tax_rate": "Avg Property Tax Rate (%)",
    "crime_rate_per_1000": "Crime Rate (per 1,000)",
    "walkability_score": "Walkability Score",
    "transit_score": "Transit Score",
    "airport_distance_miles": "Airport Distance (mi)",
    "ski_resort_distance_miles": "Ski Resort Distance (mi)",
    "national_parks_within_100mi": "National Parks within 100mi",
    "hiking_trails_count": "Hiking Trails",
    "lat": "Latitude",
    "lon": "Longitude",
    "land_area_sq_mi": "Land Area (sq mi)",
}


def label_for(col: str) -> str:
    return FRIENDLY_LABELS.get(col, col)


def metric_selectbox(label: str, key: str, default: str) -> str:
    options = list(METRIC_OPTIONS.keys())
    default_idx = options.index(default) if default in options else 0
    return st.selectbox(
        label,
        options=options,
        index=default_idx,
        format_func=label_for,
        key=key,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

with st.container(border=True):
    filter_cols = st.columns(4)
    with filter_cols[0]:
        regions = ["All"] + sorted(cities_df["region"].unique().tolist())
        sel_region = st.selectbox("Region", regions, key="dash_region")
    with filter_cols[1]:
        if sel_region != "All":
            state_options = sorted(cities_df[cities_df["region"] == sel_region]["state"].unique().tolist())
        else:
            state_options = sorted(cities_df["state"].unique().tolist())
        states = ["All"] + state_options
        sel_state = st.selectbox("State", states, key="dash_state")
    with filter_cols[2]:
        pop_min, pop_max = int(cities_df["population"].min()), int(cities_df["population"].max())
        pop_range = st.slider(
            "Population range",
            min_value=pop_min,
            max_value=pop_max,
            value=(pop_min, pop_max),
            step=10_000,
            format="%d",
            key="dash_pop",
        )
    with filter_cols[3]:
        search = st.text_input("Search city name", key="dash_search")

# Apply filters
filtered = cities_df.copy()
if sel_region != "All":
    filtered = filtered[filtered["region"] == sel_region]
if sel_state != "All":
    filtered = filtered[filtered["state"] == sel_state]
filtered = filtered[
    (filtered["population"] >= pop_range[0]) & (filtered["population"] <= pop_range[1])
]
if search:
    filtered = filtered[filtered["name"].str.lower().str.contains(search.lower())]

st.caption(f"Showing {len(filtered):,} of {len(cities_df):,} cities")

# ---------------------------------------------------------------------------
# Map controls
# ---------------------------------------------------------------------------

with st.container(border=True):
    map_cols = st.columns(3)
    with map_cols[0]:
        size_col = metric_selectbox("Dot size", "dash_size", "population")
    with map_cols[1]:
        color_col = metric_selectbox("Dot color", "dash_color", "cost_of_living_index")
    with map_cols[2]:
        base_radius = st.slider("Base dot size", 1, 20, 5, key="dash_base_radius")

# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------


def build_map_data(df: pd.DataFrame, size_metric: str, color_metric: str) -> pd.DataFrame:
    """Prepare data for pydeck rendering with normalized size and color values."""
    map_df = df[["name", "state", "lat", "lon", "population"]].copy()

    # Size: normalize to 0-1 range, then scale
    size_vals = df[size_metric].fillna(0).astype(float)
    s_min, s_max = size_vals.min(), size_vals.max()
    if s_max > s_min:
        map_df["size_norm"] = (size_vals.values - s_min) / (s_max - s_min)
    else:
        map_df["size_norm"] = 0.5

    # Color: normalize to 0-1, map to a green-yellow-red gradient
    color_vals = df[color_metric].fillna(0).astype(float)
    c_min, c_max = color_vals.min(), color_vals.max()
    if c_max > c_min:
        map_df["color_norm"] = (color_vals.values - c_min) / (c_max - c_min)
    else:
        map_df["color_norm"] = 0.5

    # Build RGBA columns (green=low, yellow=mid, red=high)
    def gradient_color(t: float) -> list:
        if t < 0.5:
            r = int(255 * (t * 2))
            g = 200
        else:
            r = 255
            g = int(200 * (1 - (t - 0.5) * 2))
        return [r, g, 30, 180]

    colors = map_df["color_norm"].apply(gradient_color)
    map_df["color_r"] = colors.apply(lambda c: c[0])
    map_df["color_g"] = colors.apply(lambda c: c[1])
    map_df["color_b"] = colors.apply(lambda c: c[2])
    map_df["color_a"] = colors.apply(lambda c: c[3])

    # Add display values for tooltip
    map_df["size_val"] = size_vals.values
    map_df["color_val"] = color_vals.values

    return map_df


if len(filtered) > 0:
    map_data = build_map_data(filtered, size_col, color_col)

    # Compute center of visible cities
    center_lat = map_data["lat"].mean()
    center_lon = map_data["lon"].mean()

    # Zoom: tighter for fewer/closer cities
    lat_range = map_data["lat"].max() - map_data["lat"].min()
    lon_range = map_data["lon"].max() - map_data["lon"].min()
    max_range = max(lat_range, lon_range, 1)
    if max_range > 40:
        zoom = 3
    elif max_range > 20:
        zoom = 4
    elif max_range > 10:
        zoom = 5
    elif max_range > 5:
        zoom = 6
    else:
        zoom = 7

    layers = []

    # State boundaries layer (rendered first, underneath cities)
    if state_geojson is not None:
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=state_geojson,
            stroked=True,
            filled=False,
            get_line_color=[80, 80, 80, 200],
            get_line_width=1,
            line_width_min_pixels=1,
            pickable=False,
        ))

    # City dots
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_radius=f"(size_norm * 40000 + 3000) * {base_radius / 5}",
        get_fill_color="[color_r, color_g, color_b, color_a]",
        pickable=True,
        auto_highlight=True,
    ))

    # City name labels (scale with zoom)
    layers.append(pdk.Layer(
        "TextLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_text="name",
        get_size=18000,
        size_units='"meters"',
        size_min_pixels=0,
        size_max_pixels=18,
        get_color=[30, 30, 30, 230],
        get_text_anchor='"middle"',
        get_alignment_baseline='"top"',
        get_pixel_offset=[0, 8],
        pickable=False,
    ))

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{name}, {state}</b><br/>"
            f"Pop: {{population:,}}<br/>"
            f"{label_for(size_col)}: {{size_val}}<br/>"
            f"{label_for(color_col)}: {{color_val}}"
        ),
        "style": {
            "backgroundColor": "#333",
            "color": "white",
            "fontSize": "13px",
            "padding": "8px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        ),
        height=550,
    )

    # Color legend
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.caption(
            f"Color: {label_for(color_col)} "
            f"(green = low, yellow = mid, red = high) | "
            f"Size: {label_for(size_col)}"
        )
else:
    st.info("No cities match the current filters.")

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"City Data ({len(filtered):,} cities)")

# Column selection
default_display = [
    "name", "state", "population", "area_population", "metro_pop",
    "region", "incorporated_places",
]
default_display = [c for c in default_display if c in filtered.columns]

all_cols = filtered.columns.tolist()
selected_cols = st.multiselect(
    "Columns to display",
    options=all_cols,
    default=default_display,
    key="dash_cols",
)

if not selected_cols:
    selected_cols = default_display

# Sort
sort_cols = st.columns(2)
with sort_cols[0]:
    sort_by = st.selectbox("Sort by", options=selected_cols, key="dash_sort")
with sort_cols[1]:
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True, key="dash_order")

display_df = filtered[selected_cols].sort_values(
    sort_by, ascending=(sort_order == "Ascending")
)

# Column formatting config
column_config = {
    "population": st.column_config.NumberColumn(format="%d"),
    "area_population": st.column_config.NumberColumn(format="%d"),
    "metro_pop": st.column_config.NumberColumn(format="%d"),
    "median_home_price": st.column_config.NumberColumn(format="$%d"),
    "median_household_income": st.column_config.NumberColumn(format="$%d"),
    "cost_of_living_index": st.column_config.NumberColumn(format="%.1f"),
    "avg_temp_summer": st.column_config.NumberColumn(format="%.0f"),
    "avg_temp_winter": st.column_config.NumberColumn(format="%.0f"),
    "annual_rainfall": st.column_config.NumberColumn(format="%.1f"),
    "annual_snow": st.column_config.NumberColumn(format="%.1f"),
    "state_income_tax_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "state_sales_tax_rate": st.column_config.NumberColumn(format="%.1f%%"),
    "avg_property_tax_rate": st.column_config.NumberColumn(format="%.2f%%"),
    "crime_rate_per_1000": st.column_config.NumberColumn(format="%.1f"),
    "walkability_score": st.column_config.NumberColumn(format="%d"),
    "transit_score": st.column_config.NumberColumn(format="%d"),
    "lat": st.column_config.NumberColumn(format="%.4f"),
    "lon": st.column_config.NumberColumn(format="%.4f"),
    "airport_distance_miles": st.column_config.NumberColumn(format="%.1f"),
    "ski_resort_distance_miles": st.column_config.NumberColumn(format="%.0f"),
    "land_area_sq_mi": st.column_config.NumberColumn(format="%.1f"),
}

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={k: v for k, v in column_config.items() if k in selected_cols},
    height=500,
)

# Download
csv = display_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="cities_dashboard.csv",
    mime="text/csv",
)
