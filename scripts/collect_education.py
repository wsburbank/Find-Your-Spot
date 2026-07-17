"""
Collect college/university data from NCES IPEDS.

Sources:
  - IPEDS Institutional Characteristics (HD2022):
    https://nces.ed.gov/ipeds/datacenter/data/HD2022.zip
    Contains institution name, location, sector, Carnegie classification, coordinates.

  - IPEDS 12-Month Enrollment (EFFY2022):
    https://nces.ed.gov/ipeds/datacenter/data/EFFY2022.zip
    Contains total 12-month unduplicated headcount enrollment.

Sector codes (SECTOR field):
  1 = Public, 4-year or above
  2 = Private not-for-profit, 4-year or above
  3 = Private for-profit, 4-year or above
  4 = Public, 2-year          (community colleges)
  5 = Private not-for-profit, 2-year
  6 = Private for-profit, 2-year
  7-9 = Less-than-2-year institutions

Carnegie 2018 Basic Classification (C18BASIC):
  15 = Doctoral Universities: Very High Research Activity (R1)
  16 = Doctoral Universities: High Research Activity (R2)
  17 = Doctoral/Professional Universities
  18-21 = Master's Colleges & Universities
  22-24 = Baccalaureate Colleges

Output: data/education.parquet
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
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

HD_URL = "https://nces.ed.gov/ipeds/datacenter/data/HD2022.zip"
EFFY_URL = "https://nces.ed.gov/ipeds/datacenter/data/EFFY2022.zip"

MATCH_RADIUS_MILES = 25  # institutions within this radius count for a city

# Carnegie codes for "major university"
R1_R2_CARNEGIE = {15, 16}
DOCTORAL_CARNEGIE = {15, 16, 17}
MAJOR_ENROLLMENT_THRESHOLD = 15_000  # also qualifies as "major" if enrollment >= this

# Carnegie 2018 Basic codes 1-14 are Associate's Colleges / institutions.
# Many historically 2-year colleges were reclassified as sector=1 (public 4-year)
# after adding bachelor's programs, but they still function primarily as community
# colleges. Using Carnegie classification catches these (e.g., Dallas College,
# Wenatchee Valley College, UAA Community & Technical College).
ASSOCIATES_CARNEGIE = set(range(1, 15))


def download_ipeds_csv(url: str, cache_name: str) -> pd.DataFrame:
    """Download an IPEDS ZIP file and extract the CSV as a DataFrame."""
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        return pd.read_csv(cache_path, encoding="latin-1", low_memory=False)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")
                     and "_rv" not in n.lower()]
        if not csv_names:
            raise FileNotFoundError(f"No CSV found in {url}")
        csv_name = csv_names[0]
        log.info("Extracting %s", csv_name)
        with zf.open(csv_name) as f:
            data = f.read()

    cache_path.write_bytes(data)
    return pd.read_csv(cache_path, encoding="latin-1", low_memory=False)


def load_institutions() -> pd.DataFrame:
    """Load and clean IPEDS institutional characteristics."""
    hd = download_ipeds_csv(HD_URL, "hd2022.csv")
    log.info("Raw HD rows: %d", len(hd))

    # Filter to active, US-based institutions with valid coordinates
    hd = hd[hd["CYACTIVE"] == 1].copy()
    hd = hd[hd["STABBR"].isin([
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
        "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
        "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
        "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
        "WY",
    ])]
    hd["LATITUDE"] = pd.to_numeric(hd["LATITUDE"], errors="coerce")
    hd["LONGITUD"] = pd.to_numeric(hd["LONGITUD"], errors="coerce")
    hd = hd.dropna(subset=["LATITUDE", "LONGITUD"])
    hd = hd[(hd["LATITUDE"] != 0) & (hd["LONGITUD"] != 0)]

    # Filter to degree-granting institutions (sectors 1-6)
    hd = hd[hd["SECTOR"].isin([1, 2, 3, 4, 5, 6])].copy()

    # Clean Carnegie classification
    hd["C18BASIC"] = pd.to_numeric(hd["C18BASIC"], errors="coerce")

    log.info("Active, degree-granting US institutions with coordinates: %d", len(hd))

    # Sector breakdown
    sector_labels = {
        1: "Public 4-year", 2: "Private nonprofit 4-year",
        3: "Private for-profit 4-year", 4: "Public 2-year",
        5: "Private nonprofit 2-year", 6: "Private for-profit 2-year",
    }
    for s, label in sector_labels.items():
        count = (hd["SECTOR"] == s).sum()
        log.info("  Sector %d (%s): %d", s, label, count)

    return hd[["UNITID", "INSTNM", "CITY", "STABBR", "SECTOR",
               "ICLEVEL", "CONTROL", "C18BASIC", "INSTSIZE",
               "LATITUDE", "LONGITUD"]].reset_index(drop=True)


def load_enrollment() -> pd.DataFrame:
    """Load 12-month enrollment from IPEDS EFFY."""
    effy = download_ipeds_csv(EFFY_URL, "effy2022.csv")
    log.info("Raw EFFY rows: %d", len(effy))

    # Filter to EFFYLEV=1 (all students, all levels)
    effy["EFFYLEV"] = pd.to_numeric(effy["EFFYLEV"], errors="coerce")
    effy = effy[effy["EFFYLEV"] == 1].copy()

    effy["EFYTOTLT"] = pd.to_numeric(effy["EFYTOTLT"], errors="coerce")
    effy = effy[effy["EFYTOTLT"] > 0]

    result = effy[["UNITID", "EFYTOTLT"]].copy()
    result = result.rename(columns={"EFYTOTLT": "enrollment"})

    # Some institutions have multiple rows for EFFYLEV=1 with different EFFYALEV;
    # take the max (EFFYALEV=1 is the overall total)
    result = result.groupby("UNITID", as_index=False)["enrollment"].max()

    log.info("Institutions with enrollment data: %d", len(result))
    return result


def match_institutions_to_cities(
    institutions: pd.DataFrame,
    cities: pd.DataFrame,
    radius_miles: float = MATCH_RADIUS_MILES,
) -> pd.DataFrame:
    """Match each institution to the nearest city within radius_miles.

    Returns a DataFrame of institutions with a city_id column added.
    Institutions not near any city are dropped.
    """
    log.info("Matching %d institutions to %d cities (radius=%.0f mi)...",
             len(institutions), len(cities), radius_miles)

    city_lats = cities["lat"].values
    city_lons = cities["lon"].values
    city_ids = cities["city_id"].values

    matched_city_ids = []
    matched_distances = []

    for _, inst in institutions.iterrows():
        inst_lat = inst["LATITUDE"]
        inst_lon = inst["LONGITUD"]

        best_dist = float("inf")
        best_city_id = None

        for i in range(len(city_ids)):
            dist = haversine_miles(inst_lat, inst_lon, city_lats[i], city_lons[i])
            if dist < best_dist:
                best_dist = dist
                best_city_id = city_ids[i]

        if best_dist <= radius_miles:
            matched_city_ids.append(best_city_id)
            matched_distances.append(round(best_dist, 1))
        else:
            matched_city_ids.append(None)
            matched_distances.append(None)

    institutions = institutions.copy()
    institutions["city_id"] = matched_city_ids
    institutions["distance_to_city"] = matched_distances

    matched = institutions.dropna(subset=["city_id"])
    log.info("Matched %d/%d institutions to cities",
             len(matched), len(institutions))
    return matched


def compute_city_education_metrics(
    matched: pd.DataFrame,
    cities: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate institution data per city into education metrics."""

    # Classify institutions
    matched = matched.copy()
    matched["is_4year"] = matched["SECTOR"].isin([1, 2, 3])
    # Community college = public 2-year (sector 4) OR public 4-year with
    # associate-focused Carnegie classification (catches reclassified colleges)
    matched["is_public_2year"] = (
        (matched["SECTOR"] == 4)
        | ((matched["SECTOR"] == 1) & matched["C18BASIC"].isin(ASSOCIATES_CARNEGIE))
    )
    matched["is_2year"] = matched["SECTOR"].isin([4, 5, 6])
    matched["is_r1_r2"] = matched["C18BASIC"].isin(R1_R2_CARNEGIE)
    matched["is_doctoral"] = matched["C18BASIC"].isin(DOCTORAL_CARNEGIE)
    matched["is_major"] = (
        matched["is_r1_r2"] |
        (matched["is_4year"] & (matched["enrollment"] >= MAJOR_ENROLLMENT_THRESHOLD))
    )

    # Aggregate per city
    grouped = matched.groupby("city_id").agg(
        university_count=("is_4year", "sum"),
        community_college_count=("is_public_2year", "sum"),
        total_institutions=("UNITID", "count"),
        has_r1_r2=("is_r1_r2", "any"),
        has_major_university=("is_major", "any"),
        has_community_college=("is_public_2year", "any"),
        largest_enrollment=("enrollment", "max"),
        total_student_population=("enrollment", "sum"),
    ).reset_index()

    # Convert bool columns
    for col in ["has_r1_r2", "has_major_university", "has_community_college"]:
        grouped[col] = grouped[col].astype(bool)

    # Merge with cities to get population for college_town_score
    result = cities[["city_id", "name", "state", "population"]].merge(
        grouped, on="city_id", how="left",
    )

    # Fill missing values (cities with no nearby institutions)
    fill_zero = ["university_count", "community_college_count", "total_institutions",
                 "largest_enrollment", "total_student_population"]
    for col in fill_zero:
        result[col] = result[col].fillna(0).astype(int)
    fill_false = ["has_r1_r2", "has_major_university", "has_community_college"]
    for col in fill_false:
        result[col] = result[col].fillna(False).infer_objects(copy=False).astype(bool)

    # College town score: student-to-population ratio (capped at 1.0)
    result["college_town_score"] = np.where(
        result["population"] > 0,
        (result["total_student_population"] / result["population"]).clip(0, 1.0).round(3),
        0.0,
    )

    log.info("Cities with at least 1 institution: %d/%d",
             (result["total_institutions"] > 0).sum(), len(result))
    log.info("Cities with a major university: %d",
             result["has_major_university"].sum())
    log.info("Cities with a community college: %d",
             result["has_community_college"].sum())

    return result


def main():
    # Load and merge IPEDS data
    institutions = load_institutions()
    enrollment = load_enrollment()

    institutions = institutions.merge(enrollment, on="UNITID", how="left")
    institutions["enrollment"] = institutions["enrollment"].fillna(0).astype(int)

    log.info("Institutions with enrollment > 0: %d",
             (institutions["enrollment"] > 0).sum())

    # Load city list
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    # Match institutions to cities
    matched = match_institutions_to_cities(institutions, cities)

    # Compute per-city metrics
    result = compute_city_education_metrics(matched, cities)

    # Summary
    print(f"\n{'='*60}")
    print("EDUCATION DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(result)}")
    print(f"Cities with institutions: {(result['total_institutions'] > 0).sum()}")
    print(f"Cities with 4-year university: {(result['university_count'] > 0).sum()}")
    print(f"Cities with community college: {result['has_community_college'].sum()}")
    print(f"Cities with major university (R1/R2 or 15K+ enrollment): "
          f"{result['has_major_university'].sum()}")
    print(f"Cities with R1/R2 research university: {result['has_r1_r2'].sum()}")
    print(f"\nTop 10 college towns (student/population ratio):")
    top_college = result.nlargest(10, "college_town_score")
    for _, row in top_college.iterrows():
        print(f"  {row['name']}, {row['state']}: "
              f"score={row['college_town_score']:.3f}, "
              f"students={row['total_student_population']:,}, "
              f"pop={row['population']:,}")

    # Drop helper columns before saving
    output_cols = ["city_id", "name", "state", "university_count",
                   "community_college_count", "total_institutions",
                   "has_major_university", "has_r1_r2",
                   "has_community_college", "largest_enrollment",
                   "total_student_population", "college_town_score"]
    output = result[output_cols]

    output_path = DATA_DIR / "education.parquet"
    write_parquet_with_metadata(
        output,
        output_path,
        source_name="NCES IPEDS Institutional Characteristics & Enrollment (2022)",
        source_url="https://nces.ed.gov/ipeds/datacenter/data/HD2022.zip",
        date_collected="2022-2023",
        notes=(
            "College/university data from NCES IPEDS 2022-23. Institutional characteristics "
            "(HD2022) provide location, sector, and Carnegie classification. 12-month enrollment "
            "(EFFY2022) provides student headcount. Institutions matched to cities within "
            f"{MATCH_RADIUS_MILES} miles by haversine distance. Major university = R1/R2 Carnegie "
            f"or enrollment >= {MAJOR_ENROLLMENT_THRESHOLD:,}. Community college = public 2-year "
            "sector. College town score = total student population / city population (capped at 1.0)."
        ),
    )
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
