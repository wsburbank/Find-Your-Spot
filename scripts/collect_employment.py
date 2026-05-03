"""
Collect unemployment rate and job growth data from Census ACS.

Source: US Census Bureau — American Community Survey 5-Year Estimates
  - 2022 ACS: B23025 (Employment Status) for unemployment rate
  - 2021 ACS: B23025 (Employment Status) for year-over-year job growth comparison
  - API: https://api.census.gov/data/{year}/acs/acs5

Strategy:
  1. Get unique county FIPS from master city list
  2. Fetch county-level employment data from ACS 2022 (unemployment, employed count)
  3. Fetch county-level employment data from ACS 2021 (employed count for growth calc)
  4. Compute unemployment rate = unemployed / civilian_labor_force * 100
  5. Compute job growth = (employed_2022 - employed_2021) / employed_2021 * 100
  6. Map county data to cities

Previous approach used BLS LAUS API which has a 25-request/day rate limit without
an API key. Census ACS provides the same county-level data with no rate limit.

Output: data/employment.parquet
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"

# ACS table B23025 — Employment Status for Population 16+
# B23025_003E = Civilian labor force
# B23025_004E = Civilian labor force: Employed
# B23025_005E = Civilian labor force: Unemployed
ACS_FIELDS = "B23025_003E,B23025_004E,B23025_005E"


def fetch_acs_employment(year: int) -> pd.DataFrame:
    """Fetch county-level employment data from Census ACS for all states.

    Args:
        year: ACS year (e.g., 2022)

    Returns:
        DataFrame with columns: fips5, labor_force, employed, unemployed
    """
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    all_rows = []

    # Fetch state by state to avoid timeout
    for state_fips in range(1, 57):
        if state_fips in (3, 7, 14, 43, 52):
            continue  # skip non-state FIPS codes
        state_str = str(state_fips).zfill(2)

        params = {
            "get": f"NAME,{ACS_FIELDS}",
            "for": "county:*",
            "in": f"state:{state_str}",
        }

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 204 or not r.text.strip():
                    break  # no data for this state FIPS
                r.raise_for_status()
                data = r.json()
                if len(data) > 1:
                    all_rows.extend(data[1:])  # skip header row
                break
            except (requests.RequestException, ValueError) as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    log.warning("Failed to fetch state %s for %d: %s",
                                state_str, year, e)

    if not all_rows:
        log.error("No ACS employment data returned for %d", year)
        return pd.DataFrame()

    # Header: NAME, B23025_003E, B23025_004E, B23025_005E, state, county
    df = pd.DataFrame(all_rows, columns=[
        "name", "labor_force", "employed", "unemployed", "state_fips", "county_fips"
    ])

    # Convert to numeric, treating Census sentinel values as null
    for col in ["labor_force", "employed", "unemployed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Census uses negative numbers as sentinel/suppressed values
        df.loc[df[col] < 0, col] = np.nan

    df["fips5"] = df["state_fips"].str.zfill(2) + df["county_fips"].str.zfill(3)

    log.info("ACS %d: %d counties fetched, %d with valid employment data",
             year, len(df), df["employed"].notna().sum())

    return df[["fips5", "name", "labor_force", "employed", "unemployed"]]


def compute_employment_metrics(
    acs_2022: pd.DataFrame,
    acs_2021: pd.DataFrame,
) -> pd.DataFrame:
    """Compute unemployment rate and job growth from ACS data.

    - Unemployment rate = (unemployed / labor_force) * 100 from 2022 data
    - Job growth = (employed_2022 - employed_2021) / employed_2021 * 100
    """
    # Unemployment rate from 2022
    metrics = acs_2022[["fips5"]].copy()
    lf = acs_2022["labor_force"]
    unemp = acs_2022["unemployed"]
    metrics["unemployment_rate"] = np.where(
        (lf > 0) & lf.notna() & unemp.notna(),
        (unemp / lf * 100).round(1),
        np.nan,
    )

    # Job growth: compare 2021 to 2022 employed counts
    merged = metrics.merge(
        acs_2021[["fips5", "employed"]].rename(columns={"employed": "emp_2021"}),
        on="fips5",
        how="left",
    )
    merged = merged.merge(
        acs_2022[["fips5", "employed"]].rename(columns={"employed": "emp_2022"}),
        on="fips5",
        how="left",
    )

    emp21 = merged["emp_2021"]
    emp22 = merged["emp_2022"]
    merged["job_growth_rate"] = np.where(
        (emp21 > 0) & emp21.notna() & emp22.notna(),
        ((emp22 - emp21) / emp21 * 100).round(2),
        np.nan,
    )

    return merged[["fips5", "unemployment_rate", "job_growth_rate"]]


def map_to_cities(county_data: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Map county-level employment data to cities."""
    cities_copy = cities[["city_id", "fips_state", "fips_county"]].copy()
    cities_copy["fips5"] = (
        cities_copy["fips_state"].astype(str).str.zfill(2) +
        cities_copy["fips_county"].astype(str).str.zfill(3)
    )

    result = cities_copy[["city_id", "fips5"]].merge(
        county_data[["fips5", "unemployment_rate", "job_growth_rate"]],
        on="fips5",
        how="left",
    )

    return result[["city_id", "unemployment_rate", "job_growth_rate"]]


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Fetch ACS employment data for 2022 and 2021
    log.info("Fetching ACS 2022 employment data...")
    acs_2022 = fetch_acs_employment(2022)

    log.info("Fetching ACS 2021 employment data...")
    acs_2021 = fetch_acs_employment(2021)

    if acs_2022.empty:
        log.error("No 2022 data collected")
        sys.exit(1)

    # Compute metrics
    county_data = compute_employment_metrics(acs_2022, acs_2021)
    log.info("County data: %d rows", len(county_data))

    has_unemp = county_data["unemployment_rate"].notna().sum()
    has_growth = county_data["job_growth_rate"].notna().sum()
    log.info("Counties with unemployment data: %d/%d (%.1f%%)",
             has_unemp, len(county_data), has_unemp / len(county_data) * 100)
    log.info("Counties with job growth data: %d/%d (%.1f%%)",
             has_growth, len(county_data), has_growth / len(county_data) * 100)

    # Map to cities
    result = map_to_cities(county_data, cities)

    # Report
    log.info("\nEmployment data coverage:")
    for col in ["unemployment_rate", "job_growth_rate"]:
        n = result[col].notna().sum()
        log.info("  %s: %d/%d cities (%.1f%%)", col, n, len(result),
                 n / len(result) * 100)

    unemp = result["unemployment_rate"].dropna()
    if not unemp.empty:
        log.info("\nUnemployment rate stats:")
        log.info("  min=%.1f, median=%.1f, mean=%.1f, max=%.1f",
                 unemp.min(), unemp.median(), unemp.mean(), unemp.max())

    growth = result["job_growth_rate"].dropna()
    if not growth.empty:
        log.info("\nJob growth rate stats:")
        log.info("  min=%.1f%%, median=%.1f%%, mean=%.1f%%, max=%.1f%%",
                 growth.min(), growth.median(), growth.mean(), growth.max())

        log.info("\nTop 10 highest unemployment:")
        top_unemp = result.nlargest(10, "unemployment_rate").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in top_unemp.iterrows():
            log.info("  %s, %s: %.1f%%", row["name"], row["state"],
                     row["unemployment_rate"])

        log.info("\nTop 10 fastest job growth:")
        top_growth = result.nlargest(10, "job_growth_rate").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in top_growth.iterrows():
            log.info("  %s, %s: %.1f%%", row["name"], row["state"],
                     row["job_growth_rate"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "employment.parquet",
        source_name="US Census Bureau — ACS 5-Year Estimates (2021, 2022), Table B23025",
        source_url="https://api.census.gov/data/2022/acs/acs5",
        date_collected="2022",
        notes=(
            "County-level employment data from ACS 5-Year Estimates. "
            "Unemployment rate from 2022 (B23025_005E / B23025_003E). "
            "Job growth rate computed as year-over-year change in employed "
            "population (B23025_004E) from 2021 to 2022."
        ),
    )

    log.info("\nSaved: data/employment.parquet (%d cities)", len(result))


if __name__ == "__main__":
    main()
