"""
Collect K-12 school quality data from NCES Common Core of Data (CCD).

Source:
  - Urban Institute Education Data Portal — CCD School Directory 2022-23
    https://educationdata.urban.org/api/v1/schools/ccd/directory/2022/
    Fields used: enrollment, teachers_fte, free_or_reduced_price_lunch,
                 latitude, longitude, school_level, school_type, school_status

Methodology:
  1. Fetch school-level data for all 50 states from the API (paginated).
  2. Filter to open, regular K-12 schools with valid enrollment and coordinates.
  3. Compute per-school pupil-teacher ratio and free/reduced lunch (FRL) rate.
  4. Match each school to the nearest city within 25 miles by haversine distance.
  5. Aggregate per city:
     - avg_pupil_teacher_ratio: mean pupil-teacher ratio (lower = better)
     - avg_frl_rate: mean free/reduced lunch rate (lower generally = more resources)
     - k12_school_count: number of schools matched
  6. Derive avg_school_rating (1-10 composite):
     - PTR component (50%): scaled inversely from ratio (10 = very low PTR)
     - FRL component (50%): scaled inversely from FRL rate (10 = very low FRL)

Output: data/school_ratings.parquet
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

API_BASE = "https://educationdata.urban.org/api/v1/schools/ccd/directory/2022/"
MATCH_RADIUS_MILES = 25

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def fetch_schools_for_state(state: str) -> pd.DataFrame:
    """Fetch all open, regular schools for a state from the Education Data API."""
    cache_path = CACHE_DIR / f"ccd_schools_{state}.parquet"
    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    url = API_BASE
    params = {
        "state_location": state,
        "school_status": 1,    # open schools only
        "school_type": 1,      # regular schools only
        "limit": 1000,
    }

    while url:
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning("API error for %s: %s", state, e)
            break

        results = data.get("results", [])
        all_rows.extend(results)

        url = data.get("next")
        params = {}  # next URL includes all params
        if url:
            time.sleep(0.3)  # rate limiting

    if not all_rows:
        log.warning("No schools returned for %s", state)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_parquet(cache_path)
    log.info("Fetched %d schools for %s", len(df), state)
    return df


def fetch_all_schools() -> pd.DataFrame:
    """Fetch school data for all states."""
    frames = []
    for state in STATES:
        df = fetch_schools_for_state(state)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No school data fetched from any state!")
        sys.exit(1)

    schools = pd.concat(frames, ignore_index=True)
    log.info("Total schools fetched: %d", len(schools))
    return schools


def clean_schools(schools: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean school data."""
    df = schools.copy()

    # Ensure numeric types
    for col in ["enrollment", "teachers_fte", "free_or_reduced_price_lunch",
                "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter: valid coordinates, positive enrollment and teacher FTE
    df = df.dropna(subset=["latitude", "longitude", "enrollment", "teachers_fte"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df[df["enrollment"] > 0]
    df = df[df["teachers_fte"] > 0]

    # Compute per-school metrics
    df["pupil_teacher_ratio"] = (df["enrollment"] / df["teachers_fte"]).round(1)

    # FRL rate (handle missing free_or_reduced_price_lunch gracefully)
    if "free_or_reduced_price_lunch" in df.columns:
        frl = df["free_or_reduced_price_lunch"].clip(lower=0)
        df["frl_rate"] = (frl / df["enrollment"]).clip(0, 1).round(3)
    else:
        df["frl_rate"] = np.nan

    # Drop extreme outliers (data entry errors)
    df = df[df["pupil_teacher_ratio"].between(3, 60)]

    log.info("Clean schools: %d (after filtering)", len(df))
    return df


def match_schools_to_cities(
    schools: pd.DataFrame,
    cities: pd.DataFrame,
    radius_miles: float = MATCH_RADIUS_MILES,
) -> pd.DataFrame:
    """Match each school to the nearest city within radius_miles."""
    log.info("Matching %d schools to %d cities (radius=%.0f mi)...",
             len(schools), len(cities), radius_miles)

    city_lats = cities["lat"].values
    city_lons = cities["lon"].values
    city_ids = cities["city_id"].values

    matched_city_ids = []

    for _, school in schools.iterrows():
        s_lat = school["latitude"]
        s_lon = school["longitude"]

        best_dist = float("inf")
        best_city_id = None

        for i in range(len(city_ids)):
            dist = haversine_miles(s_lat, s_lon, city_lats[i], city_lons[i])
            if dist < best_dist:
                best_dist = dist
                best_city_id = city_ids[i]

        if best_dist <= radius_miles:
            matched_city_ids.append(best_city_id)
        else:
            matched_city_ids.append(None)

    schools = schools.copy()
    schools["city_id"] = matched_city_ids

    matched = schools.dropna(subset=["city_id"])
    matched["city_id"] = matched["city_id"].astype(int)
    log.info("Matched %d/%d schools to cities", len(matched), len(schools))
    return matched


def compute_city_school_ratings(
    matched: pd.DataFrame,
    cities: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate school data per city into a composite rating."""

    grouped = matched.groupby("city_id").agg(
        avg_pupil_teacher_ratio=("pupil_teacher_ratio", "mean"),
        avg_frl_rate=("frl_rate", "mean"),
        k12_school_count=("enrollment", "count"),
        total_k12_enrollment=("enrollment", "sum"),
    ).reset_index()

    grouped["avg_pupil_teacher_ratio"] = grouped["avg_pupil_teacher_ratio"].round(1)
    grouped["avg_frl_rate"] = grouped["avg_frl_rate"].round(3)

    # Composite rating (1-10 scale)
    # PTR component: lower ratio = better. Typical range 10-25.
    #   10 → score 10, 25 → score 3, clipped to [1, 10]
    ptr = grouped["avg_pupil_teacher_ratio"]
    ptr_score = np.clip(10 - (ptr - 10) * (7 / 15), 1, 10)

    # FRL component: lower rate = more resources. Range 0-1.
    #   0.0 → score 10, 1.0 → score 1
    frl = grouped["avg_frl_rate"].fillna(0.5)  # assume middle if missing
    frl_score = np.clip(10 - frl * 9, 1, 10)

    # 50/50 weighted composite, rounded to 1 decimal
    grouped["avg_school_rating"] = ((ptr_score * 0.5 + frl_score * 0.5)).round(1)

    # Merge with full city list to include cities with no matched schools
    result = cities[["city_id", "name", "state"]].merge(
        grouped, on="city_id", how="left",
    )

    # Cities with no matched schools get null rating (not fabricated)
    log.info("Cities with school data: %d/%d",
             result["avg_school_rating"].notna().sum(), len(result))

    return result


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    # Fetch and clean school data
    schools = fetch_all_schools()
    schools = clean_schools(schools)

    # Match schools to cities
    matched = match_schools_to_cities(schools, cities)

    # Compute ratings
    result = compute_city_school_ratings(matched, cities)

    # Summary
    print(f"\n{'='*60}")
    print("K-12 SCHOOL RATING SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(result)}")
    print(f"Cities with school data: {result['avg_school_rating'].notna().sum()}")
    print(f"Avg pupil-teacher ratio: {result['avg_pupil_teacher_ratio'].mean():.1f}")
    print(f"Avg FRL rate: {result['avg_frl_rate'].mean():.3f}")
    print(f"Avg school rating: {result['avg_school_rating'].mean():.1f}")
    print(f"\nRating distribution:")
    for bucket in [(8, 10, "Excellent"), (6, 8, "Good"), (4, 6, "Average"), (1, 4, "Below avg")]:
        low, high, label = bucket
        count = ((result["avg_school_rating"] >= low) & (result["avg_school_rating"] < high)).sum()
        print(f"  {label} ({low}-{high}): {count}")
    print(f"\nTop 15 cities by school rating:")
    top = result.nlargest(15, "avg_school_rating")
    for _, row in top.iterrows():
        print(f"  {row['name']}, {row['state']}: "
              f"rating={row['avg_school_rating']:.1f}, "
              f"PTR={row['avg_pupil_teacher_ratio']:.1f}, "
              f"FRL={row['avg_frl_rate']:.1%}, "
              f"schools={row['k12_school_count']}")

    # Save
    output_cols = ["city_id", "name", "state", "avg_school_rating",
                   "avg_pupil_teacher_ratio", "avg_frl_rate",
                   "k12_school_count", "total_k12_enrollment"]
    output = result[output_cols]

    output_path = DATA_DIR / "school_ratings.parquet"
    write_parquet_with_metadata(
        output,
        output_path,
        source_name="NCES Common Core of Data — School Directory 2022-23",
        source_url="https://educationdata.urban.org/api/v1/schools/ccd/directory/2022/",
        date_collected="2022-2023",
        notes=(
            "K-12 school quality ratings derived from NCES CCD 2022-23 school-level data "
            "via the Urban Institute Education Data Portal API. Metrics: pupil-teacher ratio "
            "(enrollment / teachers_fte) and free/reduced lunch rate (FRL). Composite rating "
            "(1-10) is a 50/50 weighted average of inverse PTR score and inverse FRL score. "
            f"Schools matched to cities within {MATCH_RADIUS_MILES} miles by haversine distance. "
            "Only open, regular (non-charter, non-alternative) schools included."
        ),
    )
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
