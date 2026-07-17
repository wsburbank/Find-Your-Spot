"""
Geocoding and distance calculations for Find Your Spot.

Provides haversine distance, city separation enforcement (partitioned by state
for efficiency), and optional geocoding via the free US Census Geocoder API.
"""
import logging
import math
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Earth radius in miles (mean radius).
_EARTH_RADIUS_MI = 3958.8

# Neighboring states lookup — used by enforce_separation to check cross-border
# proximity without comparing every city to every other city nationally.
# Each state maps to its set of bordering states (including itself).
NEIGHBORING_STATES: dict[str, set[str]] = {
    "AL": {"AL", "FL", "GA", "MS", "TN"},
    "AK": {"AK"},
    "AZ": {"AZ", "CA", "CO", "NM", "NV", "UT"},
    "AR": {"AR", "LA", "MO", "MS", "OK", "TN", "TX"},
    "CA": {"CA", "AZ", "NV", "OR"},
    "CO": {"CO", "AZ", "KS", "NE", "NM", "OK", "UT", "WY"},
    "CT": {"CT", "MA", "NY", "RI"},
    "DE": {"DE", "MD", "NJ", "PA"},
    "FL": {"FL", "AL", "GA"},
    "GA": {"GA", "AL", "FL", "NC", "SC", "TN"},
    "HI": {"HI"},
    "ID": {"ID", "MT", "NV", "OR", "UT", "WA", "WY"},
    "IL": {"IL", "IA", "IN", "KY", "MO", "WI"},
    "IN": {"IN", "IL", "KY", "MI", "OH"},
    "IA": {"IA", "IL", "MN", "MO", "NE", "SD", "WI"},
    "KS": {"KS", "CO", "MO", "NE", "OK"},
    "KY": {"KY", "IL", "IN", "MO", "OH", "TN", "VA", "WV"},
    "LA": {"LA", "AR", "MS", "TX"},
    "ME": {"ME", "NH"},
    "MD": {"MD", "DC", "DE", "PA", "VA", "WV"},
    "MA": {"MA", "CT", "NH", "NY", "RI", "VT"},
    "MI": {"MI", "IN", "OH", "WI"},
    "MN": {"MN", "IA", "ND", "SD", "WI"},
    "MS": {"MS", "AL", "AR", "LA", "TN"},
    "MO": {"MO", "AR", "IA", "IL", "KS", "KY", "NE", "OK", "TN"},
    "MT": {"MT", "ID", "ND", "SD", "WY"},
    "NE": {"NE", "CO", "IA", "KS", "MO", "SD", "WY"},
    "NV": {"NV", "AZ", "CA", "ID", "OR", "UT"},
    "NH": {"NH", "MA", "ME", "VT"},
    "NJ": {"NJ", "DE", "NY", "PA"},
    "NM": {"NM", "AZ", "CO", "OK", "TX", "UT"},
    "NY": {"NY", "CT", "MA", "NJ", "PA", "VT"},
    "NC": {"NC", "GA", "SC", "TN", "VA"},
    "ND": {"ND", "MN", "MT", "SD"},
    "OH": {"OH", "IN", "KY", "MI", "PA", "WV"},
    "OK": {"OK", "AR", "CO", "KS", "MO", "NM", "TX"},
    "OR": {"OR", "CA", "ID", "NV", "WA"},
    "PA": {"PA", "DE", "MD", "NJ", "NY", "OH", "WV"},
    "RI": {"RI", "CT", "MA"},
    "SC": {"SC", "GA", "NC"},
    "SD": {"SD", "IA", "MN", "MT", "ND", "NE", "WY"},
    "TN": {"TN", "AL", "AR", "GA", "KY", "MO", "MS", "NC", "VA"},
    "TX": {"TX", "AR", "LA", "NM", "OK"},
    "UT": {"UT", "AZ", "CO", "ID", "NM", "NV", "WY"},
    "VT": {"VT", "MA", "NH", "NY"},
    "VA": {"VA", "DC", "KY", "MD", "NC", "TN", "WV"},
    "WA": {"WA", "ID", "OR"},
    "WV": {"WV", "KY", "MD", "OH", "PA", "VA"},
    "WI": {"WI", "IA", "IL", "MI", "MN"},
    "WY": {"WY", "CO", "ID", "MT", "NE", "SD", "UT"},
    "DC": {"DC", "MD", "VA"},
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in miles.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees).
        lat2, lon2: Latitude and longitude of point 2 (degrees).

    Returns:
        Distance in statute miles.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def find_nearby_cities(
    target_lat: float,
    target_lon: float,
    cities_df: pd.DataFrame,
    radius_miles: float = 20.0,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Find all cities within *radius_miles* of a target point.

    Returns a copy of the matching rows with an added ``distance_miles`` column,
    sorted by distance ascending.
    """
    distances = cities_df.apply(
        lambda row: haversine_miles(target_lat, target_lon, row[lat_col], row[lon_col]),
        axis=1,
    )
    mask = distances <= radius_miles
    result = cities_df.loc[mask].copy()
    result["distance_miles"] = distances.loc[mask]
    return result.sort_values("distance_miles")


def enforce_separation(
    cities_df: pd.DataFrame,
    min_distance_miles: float = 20.0,
    lat_col: str = "lat",
    lon_col: str = "lon",
    priority_col: str = "population",
    state_col: str = "state",
    return_rollup: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[int, int]]:
    """Remove cities that are too close together, keeping the higher-priority one.

    Uses state-based partitioning: only compares cities within the same state or
    neighboring states, which reduces comparisons dramatically.

    Args:
        cities_df: DataFrame with city data (must include lat, lon, state, priority columns).
        min_distance_miles: Minimum allowed distance between any two retained cities.
        lat_col: Column name for latitude.
        lon_col: Column name for longitude.
        priority_col: Column used to decide which city to keep (higher value wins).
        state_col: Column name for 2-letter state abbreviation.
        return_rollup: If True, also return a dict mapping removed row index
            to the nearest surviving row index (for population roll-up).

    Returns:
        Filtered DataFrame with close duplicates removed. If return_rollup is True,
        returns (DataFrame, rollup_dict) where rollup_dict maps removed_idx -> nearest_kept_idx.
    """
    df = cities_df.copy()
    df["_orig_idx"] = range(len(df))
    df = df.sort_values(priority_col, ascending=False).reset_index(drop=True)
    keep = set(df.index)
    # Track which kept city removed each city (the one that caused removal)
    removed_by: dict[int, int] = {}

    for idx in df.index:
        if idx not in keep:
            continue

        city_state = df.at[idx, state_col]
        neighbors = NEIGHBORING_STATES.get(city_state, {city_state})
        city_lat = df.at[idx, lat_col]
        city_lon = df.at[idx, lon_col]

        # Only check cities that are in neighboring states and still kept.
        for other_idx in df.index:
            if other_idx <= idx or other_idx not in keep:
                continue
            if df.at[other_idx, state_col] not in neighbors:
                continue

            dist = haversine_miles(
                city_lat, city_lon,
                df.at[other_idx, lat_col], df.at[other_idx, lon_col],
            )
            if dist < min_distance_miles:
                keep.discard(other_idx)
                removed_by[other_idx] = idx

    result = df.loc[sorted(keep)].copy()
    log.info(
        "enforce_separation: %d -> %d cities (min %.0f mi)",
        len(df), len(result), min_distance_miles,
    )

    if not return_rollup:
        result = result.drop(columns=["_orig_idx"]).reset_index(drop=True)
        return result

    # Build rollup: for each removed city, find the *nearest* surviving city.
    # Uses _orig_idx for stable identity across index resets.
    kept_indices = sorted(keep)

    # Map from _orig_idx -> kept row position for the surviving cities
    kept_orig_indices = {df.at[k, "_orig_idx"]: k for k in kept_indices}

    rollup: dict[int, int] = {}  # removed _orig_idx -> kept _orig_idx
    for removed_idx in removed_by:
        best_dist = float("inf")
        best_kept_orig = -1
        r_lat = df.at[removed_idx, lat_col]
        r_lon = df.at[removed_idx, lon_col]
        r_state = df.at[removed_idx, state_col]
        r_neighbors = NEIGHBORING_STATES.get(r_state, {r_state})

        for k_idx in kept_indices:
            if df.at[k_idx, state_col] not in r_neighbors:
                continue
            dist = haversine_miles(
                r_lat, r_lon, df.at[k_idx, lat_col], df.at[k_idx, lon_col],
            )
            if dist < best_dist:
                best_dist = dist
                best_kept_orig = df.at[k_idx, "_orig_idx"]
        if best_kept_orig >= 0:
            removed_orig = df.at[removed_idx, "_orig_idx"]
            rollup[removed_orig] = best_kept_orig

    result = result.drop(columns=["_orig_idx"]).reset_index(drop=True)
    return result, rollup


def geocode_city(city_name: str, state: str) -> tuple[float, float] | None:
    """Geocode a city using the free US Census Geocoder API.

    Args:
        city_name: City name (e.g. "Denver").
        state: State abbreviation or full name (e.g. "CO" or "Colorado").

    Returns:
        (latitude, longitude) tuple or None if geocoding fails.
    """
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": f"{city_name}, {state}",
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0]["coordinates"]
            return float(coords["y"]), float(coords["x"])
    except Exception:
        log.warning("Geocoding failed for %s, %s", city_name, state, exc_info=True)
    return None
