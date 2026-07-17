"""
Collect EPA Air Quality Index (AQI) data at county level.

Source: US EPA Pre-Generated Data Files
  - https://aqs.epa.gov/aqsweb/airdata/download_files.html
  - Annual AQI by county (most recent full year: 2024)
  - No API key required for pre-generated files

Metrics collected:
  - median_aqi: Median daily AQI over the year
  - max_aqi: Maximum daily AQI observed
  - days_good: Days with AQI 0-50 (Good)
  - days_moderate: Days with AQI 51-100 (Moderate)
  - days_unhealthy_sensitive: Days with AQI 101-150
  - days_unhealthy: Days with AQI 151+ (Unhealthy or worse)
  - pct_good_days: Percentage of monitored days rated Good

Strategy:
  1. Download annual_aqi_by_county_2024.zip from EPA
  2. Match to cities via FIPS state + county codes
  3. For cities spanning multiple counties, use the county with most coverage

Output: data/air_quality.parquet
"""

import io
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# EPA pre-generated file URL
EPA_AQI_URL = "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2024.zip"
EPA_AQI_URL_2023 = "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2023.zip"

# Census FIPS county codes
FIPS_COUNTY_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"

# State name to FIPS mapping
STATE_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09", "delaware": "10",
    "district of columbia": "11", "florida": "12", "georgia": "13", "hawaii": "15",
    "idaho": "16", "illinois": "17", "indiana": "18", "iowa": "19",
    "kansas": "20", "kentucky": "21", "louisiana": "22", "maine": "23",
    "maryland": "24", "massachusetts": "25", "michigan": "26", "minnesota": "27",
    "mississippi": "28", "missouri": "29", "montana": "30", "nebraska": "31",
    "nevada": "32", "new hampshire": "33", "new jersey": "34", "new mexico": "35",
    "new york": "36", "north carolina": "37", "north dakota": "38", "ohio": "39",
    "oklahoma": "40", "oregon": "41", "pennsylvania": "42", "rhode island": "44",
    "south carolina": "45", "south dakota": "46", "tennessee": "47", "texas": "48",
    "utah": "49", "vermont": "50", "virginia": "51", "washington": "53",
    "west virginia": "54", "wisconsin": "55", "wyoming": "56",
}


def _build_state_county_fips_map() -> dict[tuple[str, str], str]:
    """Download Census FIPS county codes and build (state_name, county_name) -> fips_county map."""
    cache_path = CACHE_DIR / "national_county2020.txt"

    if cache_path.exists():
        text = cache_path.read_text(encoding="latin-1")
    else:
        log.info("Downloading Census FIPS county codes...")
        resp = requests.get(FIPS_COUNTY_URL, timeout=30)
        if resp.status_code != 200:
            log.error("Failed to download FIPS codes (HTTP %d)", resp.status_code)
            return {}
        text = resp.text
        cache_path.write_text(text, encoding="latin-1")

    # Parse: STATE_ABBR|STATEFP|COUNTYFP|COUNTYNS|COUNTYNAME|CLASSFP|FUNCSTAT
    # EPA uses full state names and county names without "County" suffix.
    # Census file uses state abbreviations. We map via STATE_FIPS dict.
    # Build reverse: fips_code -> state_name for joining
    fips_to_state = {v: k for k, v in STATE_FIPS.items()}

    fips_map = {}
    for line in text.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 5:
            continue
        state_fp = parts[1].strip()
        county_fp = parts[2].strip()
        county_name = parts[4].strip().lower()

        # Get full state name from FIPS code
        state_name = fips_to_state.get(state_fp, "")
        if not state_name:
            continue

        # Remove " county", " parish", " borough", etc. suffixes for matching
        county_clean = county_name
        for suffix in [" county", " parish", " borough", " census area",
                       " municipality", " city and borough", " city"]:
            if county_clean.endswith(suffix):
                county_clean = county_clean[: -len(suffix)]
                break

        fips_county = state_fp + county_fp
        fips_map[(state_name, county_clean)] = fips_county
        # Also store with full name for exact matches
        fips_map[(state_name, county_name)] = fips_county

    log.info("Built FIPS map with %d entries", len(fips_map))
    return fips_map


def download_epa_aqi(url: str) -> pd.DataFrame | None:
    """Download and parse EPA annual AQI by county CSV from zip."""
    log.info("Downloading EPA AQI data from %s", url)
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        log.warning("Failed to download %s (HTTP %d)", url, resp.status_code)
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            log.error("No CSV found in zip archive")
            return None
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    log.info("Downloaded %d county-year records", len(df))
    return df


def _fill_from_nearest_county(
    result: pd.DataFrame,
    county_aqi: pd.DataFrame,
    cities: pd.DataFrame,
    aqi_cols: list[str],
) -> pd.DataFrame:
    """Fill missing AQI values by assigning the nearest monitored county's data.

    For each city without AQI data (county has no EPA monitor), find the
    geographically nearest county that DOES have a monitor and use its values.
    This is not interpolation — it assigns a single real measurement.
    """
    # Build county centroid lookup from cities that have lat/lon
    # (use cities with AQI data to approximate monitored county locations)
    cities_with_fips = cities[["city_id", "lat", "lon", "fips_county_full"]].copy()
    result_geo = result.merge(cities_with_fips[["city_id", "lat", "lon"]], on="city_id", how="left")

    # Get centroid of each monitored county (average lat/lon of cities in that county)
    monitored = result_geo[result_geo["aqi_source"] == "own_county"].copy()
    county_centroids = monitored.groupby("fips_county").agg(
        county_lat=("lat", "mean"),
        county_lon=("lon", "mean"),
    ).reset_index()

    # Merge AQI values to county centroids
    county_centroids = county_centroids.merge(county_aqi, on="fips_county", how="inner")

    # Vectorized haversine
    def haversine(lat1, lon1, lat2_arr, lon2_arr):
        R = 3959.0  # miles
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2_arr), np.radians(lon2_arr)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    county_lats = county_centroids["county_lat"].values
    county_lons = county_centroids["county_lon"].values

    # For each missing city, find nearest monitored county
    missing_mask = result["median_aqi"].isna()
    missing_cities = result_geo[missing_mask].copy()

    fill_cols = [c for c in aqi_cols if c in county_aqi.columns]

    for idx, city in missing_cities.iterrows():
        dists = haversine(city["lat"], city["lon"], county_lats, county_lons)
        nearest_idx = dists.argmin()
        nearest_dist = dists[nearest_idx]

        for col in fill_cols:
            result.loc[idx, col] = county_centroids.iloc[nearest_idx][col]

        result.loc[idx, "aqi_source"] = "nearest_county"
        result.loc[idx, "aqi_station_distance_mi"] = round(nearest_dist, 1)

    # Set distance to 0 for own_county matches
    result.loc[result["aqi_source"] == "own_county", "aqi_station_distance_mi"] = 0.0

    return result


def main():
    # Try 2024 first, fall back to 2023
    aqi_df = download_epa_aqi(EPA_AQI_URL)
    data_year = "2024"
    if aqi_df is None:
        log.info("2024 not available, trying 2023...")
        aqi_df = download_epa_aqi(EPA_AQI_URL_2023)
        data_year = "2023"
    if aqi_df is None:
        log.error("Could not download EPA AQI data")
        sys.exit(1)

    # Standardize column names
    aqi_df.columns = aqi_df.columns.str.strip().str.lower().str.replace(" ", "_")
    log.info("Columns: %s", list(aqi_df.columns))

    # EPA file uses state/county NAMES, not FIPS codes.
    # We need to map state+county name to FIPS county codes.
    # Load our cities master to get the FIPS county mapping.
    # We'll use Census FIPS codes from a standard crosswalk.
    fips_map = _build_state_county_fips_map()

    # Normalize state and county names for matching
    aqi_df["state_lower"] = aqi_df["state"].str.strip().str.lower()
    aqi_df["county_lower"] = aqi_df["county"].str.strip().str.lower()
    aqi_df["fips_county"] = aqi_df.apply(
        lambda r: fips_map.get((r["state_lower"], r["county_lower"]), ""), axis=1
    )
    unmatched = (aqi_df["fips_county"] == "").sum()
    if unmatched > 0:
        log.warning("%d counties could not be matched to FIPS codes", unmatched)
    aqi_df = aqi_df[aqi_df["fips_county"] != ""].copy()

    # Select and rename relevant columns
    col_map = {
        "fips_county": "fips_county",
        "median_aqi": "median_aqi",
        "max_aqi": "max_aqi",
        "days_with_aqi": "days_monitored",
        "good_days": "days_good",
        "moderate_days": "days_moderate",
        "unhealthy_for_sensitive_groups_days": "days_unhealthy_sensitive",
        "unhealthy_days": "days_unhealthy",
        "very_unhealthy_days": "days_very_unhealthy",
        "hazardous_days": "days_hazardous",
    }

    # Check which columns exist
    available = {k: v for k, v in col_map.items() if k in aqi_df.columns}
    missing = set(col_map.keys()) - set(available.keys())
    if missing:
        log.warning("Missing columns in EPA data: %s", missing)

    county_aqi = aqi_df[list(available.keys())].rename(columns=available).copy()

    # Compute percentage of good days
    if "days_monitored" in county_aqi.columns and "days_good" in county_aqi.columns:
        county_aqi["pct_good_days"] = (
            county_aqi["days_good"] / county_aqi["days_monitored"] * 100
        ).round(1)

    # Compute combined unhealthy days (all categories above moderate)
    unhealthy_cols = ["days_unhealthy_sensitive", "days_unhealthy",
                      "days_very_unhealthy", "days_hazardous"]
    existing_unhealthy = [c for c in unhealthy_cols if c in county_aqi.columns]
    if existing_unhealthy:
        county_aqi["days_unhealthy_total"] = county_aqi[existing_unhealthy].sum(axis=1)

    # Deduplicate: some counties may have multiple entries (shouldn't, but safeguard)
    county_aqi = county_aqi.drop_duplicates(subset=["fips_county"], keep="first")
    log.info("Unique counties with AQI data: %d", len(county_aqi))

    # Load master city list to match by FIPS state+county
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    # Cities have separate fips_state (2-digit) and fips_county (3-digit county part)
    # Build full 5-digit FIPS county code for join
    cities["fips_county_full"] = (
        cities["fips_state"].astype(str).str.zfill(2)
        + cities["fips_county"].astype(str).str.zfill(3)
    )
    county_aqi["fips_county"] = county_aqi["fips_county"].astype(str).str.zfill(5)

    # Merge city to county AQI
    aqi_cols = [c for c in county_aqi.columns if c != "fips_county"]
    result = cities[["city_id", "name", "state", "fips_county_full"]].merge(
        county_aqi,
        left_on="fips_county_full",
        right_on="fips_county",
        how="left",
    )

    # Track source method
    result["aqi_source"] = np.where(result["median_aqi"].notna(), "own_county", None)

    direct_matched = result["median_aqi"].notna().sum()
    log.info("Direct county match: %d/%d (%.1f%%)",
             direct_matched, len(result), direct_matched / len(result) * 100)

    # --- Fill missing cities via nearest monitored county ---
    # AQI is regional (driven by weather/geography), so the nearest monitored
    # county is a valid proxy. We assign the real measured value from that county.
    missing_mask = result["median_aqi"].isna()
    if missing_mask.any():
        result = _fill_from_nearest_county(result, county_aqi, cities, aqi_cols)

    total_matched = result["median_aqi"].notna().sum()
    nearest_filled = total_matched - direct_matched
    log.info("After nearest-county fill: %d/%d (%.1f%%) — %d from nearest county",
             total_matched, len(result), total_matched / len(result) * 100, nearest_filled)

    # Keep only city_id + AQI metrics + source for output
    output_cols = ["city_id", "aqi_source", "aqi_station_distance_mi"] + [
        c for c in aqi_cols if c in result.columns
    ]
    output = result[[c for c in output_cols if c in result.columns]].copy()

    # Write output
    write_parquet_with_metadata(
        output,
        DATA_DIR / "air_quality.parquet",
        source_name=f"US EPA Air Quality System (AQS) - Annual AQI by County {data_year}",
        source_url="https://aqs.epa.gov/aqsweb/airdata/download_files.html",
        notes=(
            f"Annual AQI summary by county ({data_year}). "
            f"Direct county match: {direct_matched}/{len(result)} cities. "
            f"Nearest-county fill: {nearest_filled} additional cities. "
            f"Total coverage: {total_matched}/{len(result)}. "
            "Metrics: median AQI, max AQI, good/moderate/unhealthy days, pct good days. "
            "aqi_source indicates 'own_county' (direct FIPS match) vs 'nearest_county'."
        ),
    )

    # Print summary stats
    print(f"\nAir Quality Data Summary ({data_year}):")
    print(f"  Direct county match: {direct_matched}/{len(result)} ({direct_matched/len(result)*100:.1f}%)")
    print(f"  Nearest-county fill: {nearest_filled}")
    print(f"  Total coverage: {total_matched}/{len(result)} ({total_matched/len(result)*100:.1f}%)")
    if total_matched > 0:
        print(f"  Median AQI range: {output['median_aqi'].min():.0f} - {output['median_aqi'].max():.0f}")
        print(f"  Mean median AQI: {output['median_aqi'].mean():.1f}")
        print(f"  Mean pct good days: {output['pct_good_days'].mean():.1f}%")
    # Distance stats for nearest-county fills
    nearest_dists = output.loc[output["aqi_source"] == "nearest_county", "aqi_station_distance_mi"]
    if len(nearest_dists) > 0:
        print(f"  Nearest-county distances: median {nearest_dists.median():.0f} mi, "
              f"max {nearest_dists.max():.0f} mi")


if __name__ == "__main__":
    main()
