"""
Collect water quality data from EPA ECHO (SDWIS - Safe Drinking Water Act).

Source: EPA ECHO — Safe Drinking Water Information System
URL: https://echo.epa.gov/tools/web-services/facility-search-drinking-water
API: https://echodata.epa.gov/echo/sdw_rest_services

Collects Community Water System (CWS) violation data by state, aggregates to
county level, then maps to cities. Metrics include:
  - Total water systems per county
  - Systems with violations (past 12 quarters / 3 years)
  - Systems with health-based violations
  - Violation rate (% of systems with violations)
  - Population served by systems with violations

Output: data/water_quality.parquet
"""

import logging
import sys
import time
from io import StringIO
from pathlib import Path

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

# EPA ECHO SDWA endpoints (two-step: search then download)
ECHO_SEARCH_URL = "https://echodata.epa.gov/echo/sdw_rest_services.get_systems"
ECHO_DOWNLOAD_URL = "https://echodata.epa.gov/echo/sdw_rest_services.get_download"

# All 50 US states
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# Columns to request (by position in metadata):
# 1=PWS_NAME, 2=PWSID, 6=COUNTIES_SERVED, 14=POPULATION_SERVED_COUNT,
# 18=QTRS_WITH_VIO, 19=QTRS_WITH_SNC, 20=SERIOUS_VIOLATOR,
# 22=HEALTH_FLAG, 28=RULES_VIO, 48=FIPS_CODES
DOWNLOAD_COLUMNS = "1,2,6,14,18,19,20,22,28,48"


def fetch_state_water_systems(state: str, retries: int = 3) -> pd.DataFrame:
    """Fetch all community water systems for a state via ECHO two-step API.

    Step 1: Search to get a QueryID
    Step 2: Download CSV using the QueryID
    """
    for attempt in range(retries):
        try:
            # Step 1: Search
            search_params = {
                "output": "JSON",
                "p_st": state,
                "p_ptype": "CWS",  # Community Water Systems only
                "responseset": 5000,
            }
            resp = requests.get(ECHO_SEARCH_URL, params=search_params, timeout=45)
            resp.raise_for_status()
            data = resp.json().get("Results", {})

            if "Error" in data:
                log.warning("  %s: API error: %s", state, data["Error"])
                return pd.DataFrame()

            qid = data.get("QueryID")
            total_rows = int(data.get("QueryRows", 0))

            if not qid or total_rows == 0:
                log.info("  %s: no CWS systems found", state)
                return pd.DataFrame()

            # Step 2: Download CSV with selected columns
            dl_params = {
                "output": "CSV",
                "qid": qid,
                "responseset": 5000,
                "qcolumns": DOWNLOAD_COLUMNS,
            }
            resp2 = requests.get(ECHO_DOWNLOAD_URL, params=dl_params, timeout=60)
            resp2.raise_for_status()

            df = pd.read_csv(StringIO(resp2.text), low_memory=False)
            df["state_code"] = state
            return df

        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                log.warning("  %s attempt %d failed: %s — retrying in %ds",
                            state, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                log.error("  %s: all attempts failed", state)
                return pd.DataFrame()

    return pd.DataFrame()


def collect_all_states() -> pd.DataFrame:
    """Collect water system data for all 50 states."""
    cache_file = CACHE_DIR / "echo_sdwis_all.csv"

    if cache_file.exists():
        log.info("Using cached SDWIS data from %s", cache_file)
        return pd.read_csv(cache_file, low_memory=False)

    log.info("Fetching EPA ECHO SDWIS data for all 50 states...")
    all_dfs = []

    for i, state in enumerate(US_STATES, 1):
        df = fetch_state_water_systems(state)
        if not df.empty:
            all_dfs.append(df)
            log.info("  [%d/50] %s: %d CWS systems", i, state, len(df))
        else:
            log.info("  [%d/50] %s: no data", i, state)
        time.sleep(1.5)  # Rate limiting

    if not all_dfs:
        log.error("No SDWIS data collected from any state")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(cache_file, index=False)
    log.info("Combined: %d total CWS systems across %d states",
             len(combined), combined["state_code"].nunique())
    return combined


def aggregate_to_counties(systems_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate water system data to county-level metrics."""
    if systems_df.empty:
        return pd.DataFrame()

    # Normalize column names (ECHO returns CamelCase)
    col_map = {}
    for col in systems_df.columns:
        if "fips" in col.lower():
            col_map[col] = "fips_code"
        elif "qtrs" in col.lower() and "vio" in col.lower() and "snc" not in col.lower():
            col_map[col] = "qtrs_with_vio"
        elif "qtrs" in col.lower() and "snc" in col.lower():
            col_map[col] = "qtrs_with_snc"
        elif "population" in col.lower():
            col_map[col] = "pop_served"
        elif "health" in col.lower():
            col_map[col] = "health_flag"
        elif "serious" in col.lower():
            col_map[col] = "serious_violator"
        elif "rules" in col.lower() and "vio" in col.lower():
            col_map[col] = "rules_vio"
        elif "counties" in col.lower():
            col_map[col] = "county_name"

    systems_df = systems_df.rename(columns=col_map)

    # Parse FIPS codes — pad to 5 digits (state + county)
    if "fips_code" not in systems_df.columns:
        log.error("No FIPS column found after rename. Columns: %s", systems_df.columns.tolist())
        return pd.DataFrame()

    systems_df["fips_5"] = systems_df["fips_code"].astype(str).str.split(".").str[0].str.zfill(5)
    # Some entries have just state FIPS (e.g., '8' for CO) — skip those
    systems_df = systems_df[systems_df["fips_5"].str.len() == 5].copy()
    systems_df = systems_df[systems_df["fips_5"] != "00000"].copy()

    # Ensure numeric columns
    systems_df["qtrs_with_vio"] = pd.to_numeric(systems_df.get("qtrs_with_vio", 0), errors="coerce").fillna(0)
    systems_df["pop_served"] = pd.to_numeric(systems_df.get("pop_served", 0), errors="coerce").fillna(0)
    systems_df["rules_vio"] = pd.to_numeric(systems_df.get("rules_vio", 0), errors="coerce").fillna(0)

    # Boolean flags
    systems_df["has_violations"] = systems_df["qtrs_with_vio"] > 0
    if "health_flag" in systems_df.columns:
        systems_df["has_health_vio"] = systems_df["health_flag"].astype(str).str.lower() == "yes"
    else:
        systems_df["has_health_vio"] = False
    if "serious_violator" in systems_df.columns:
        systems_df["is_serious"] = systems_df["serious_violator"].astype(str).str.lower() == "yes"
    else:
        systems_df["is_serious"] = False

    # Aggregate to county
    county_agg = systems_df.groupby("fips_5").agg(
        water_systems_total=("fips_5", "count"),
        water_systems_with_violations=("has_violations", "sum"),
        water_systems_health_violations=("has_health_vio", "sum"),
        water_systems_serious_violators=("is_serious", "sum"),
        water_total_pop_served=("pop_served", "sum"),
        water_total_violations=("rules_vio", "sum"),
        water_pop_served_violators=("pop_served", lambda x: x[systems_df.loc[x.index, "has_violations"]].sum()),
    ).reset_index()

    # Violations per system per year (rules_vio covers 3 years / 12 quarters)
    county_agg["water_violation_rate"] = (
        county_agg["water_total_violations"] / county_agg["water_systems_total"] / 3
    ).round(2)

    # Calculate % of population served by systems with violations
    county_agg["water_pop_pct_affected"] = (
        county_agg["water_pop_served_violators"] /
        county_agg["water_total_pop_served"].clip(lower=1) * 100
    ).round(1)

    # Extract state/county FIPS
    county_agg["fips_state"] = county_agg["fips_5"].str[:2]
    county_agg["fips_county"] = county_agg["fips_5"].str[2:5]

    log.info("Aggregated to %d counties", len(county_agg))
    return county_agg


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Collect SDWIS data
    systems = collect_all_states()
    if systems.empty:
        log.error("No SDWIS data — aborting")
        sys.exit(1)

    # Aggregate to counties
    county_data = aggregate_to_counties(systems)
    if county_data.empty:
        log.error("County aggregation failed — aborting")
        sys.exit(1)

    # Match to cities via fips_state + fips_county
    cities_match = cities[["city_id", "fips_state", "fips_county"]].copy()
    cities_match["fips_state"] = cities_match["fips_state"].astype(str).str.zfill(2)
    cities_match["fips_county"] = cities_match["fips_county"].astype(str).str.zfill(3)

    # Select output columns
    output_cols = [
        "fips_state", "fips_county",
        "water_systems_total", "water_systems_with_violations",
        "water_systems_health_violations", "water_systems_serious_violators",
        "water_violation_rate", "water_pop_pct_affected",
    ]
    county_output = county_data[[c for c in output_cols if c in county_data.columns]]

    result = cities_match.merge(county_output, on=["fips_state", "fips_county"], how="left")
    result = result.drop(columns=["fips_state", "fips_county"])

    # Fill missing with 0 (counties with no water systems in ECHO)
    fill_cols = ["water_systems_total", "water_systems_with_violations",
                 "water_systems_health_violations", "water_systems_serious_violators"]
    for col in fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    # Report
    log.info("\nWater quality coverage:")
    for col in result.columns:
        if col == "city_id":
            continue
        valid = result[col].notna().sum()
        log.info("  %s: %d/%d (%.1f%%)", col, valid, len(result), valid / len(result) * 100)

    if "water_violation_rate" in result.columns:
        log.info("\nCities with highest water violation rates:")
        top = result.nlargest(10, "water_violation_rate").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in top.iterrows():
            log.info("  %s, %s: %.1f%% violation rate (%d/%d systems)",
                     row["name"], row["state"], row["water_violation_rate"],
                     row["water_systems_with_violations"], row["water_systems_total"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "water_quality.parquet",
        source_name="EPA ECHO — Safe Drinking Water Information System (SDWIS)",
        source_url="https://echo.epa.gov/tools/web-services/facility-search-drinking-water",
        date_collected="2025",
        notes=(
            "Community Water System violation data from EPA ECHO SDWIS API. "
            "Metrics: total systems, systems with violations (past 12 quarters), "
            "health-based violations, serious violators, violation rate (%), "
            "and population percentage affected. County-level data matched to cities via FIPS."
        ),
    )
    log.info("\nSaved: data/water_quality.parquet (%d rows)", len(result))


if __name__ == "__main__":
    main()
