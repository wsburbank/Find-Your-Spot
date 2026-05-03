"""
Collect commute / transportation-to-work data from Census ACS.

Sources:
  - Census ACS 5-Year (2022):
    B08013_001E = Aggregate travel time to work (minutes) for workers 16+
    B08301_001E = Total workers 16+ (all modes, including work-from-home)
    B08301_002E = Car, truck, or van
    B08301_010E = Public transportation
    B08301_019E = Walked
    B08301_021E = Worked from home

  Mean commute = B08013_001E / (B08301_001E - B08301_021E)
  Work-from-home pct = B08301_021E / B08301_001E * 100
  Public transit pct  = B08301_010E / B08301_001E * 100
  Drove pct           = B08301_002E / B08301_001E * 100
  Walked pct          = B08301_019E / B08301_001E * 100

Output: data/commute.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
VARIABLES = "NAME,B08013_001E,B08301_001E,B08301_002E,B08301_010E,B08301_019E,B08301_021E"

OUTPUT_PATH = PROJECT_ROOT / "data" / "commute.parquet"


def fetch_acs_commute_data() -> pd.DataFrame:
    """Fetch commute data from Census ACS for all places, state by state."""
    log.info("Fetching ACS commute data for all places...")

    all_frames = []
    # Fetch state-by-state to avoid timeouts on the all-states query
    # State FIPS codes: 01-56 (skipping gaps)
    states_url = f"{ACS_URL}?get=NAME&for=state:*"
    resp = requests.get(states_url, timeout=30)
    resp.raise_for_status()
    state_codes = [row[1] for row in resp.json()[1:]]
    log.info("Fetching commute data for %d states...", len(state_codes))

    for state_fips in state_codes:
        url = f"{ACS_URL}?get={VARIABLES}&for=place:*&in=state:{state_fips}"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if len(data) > 1:
                df_state = pd.DataFrame(data[1:], columns=data[0])
                all_frames.append(df_state)
        except Exception:
            log.warning("Failed to fetch state %s, skipping", state_fips)

    df = pd.concat(all_frames, ignore_index=True)
    log.info("Fetched raw commute data for %d places", len(df))

    df = df.rename(columns={
        "NAME": "raw_name",
        "B08013_001E": "agg_travel_time",
        "B08301_001E": "total_workers",
        "B08301_002E": "drove",
        "B08301_010E": "public_transit",
        "B08301_019E": "walked",
        "B08301_021E": "work_from_home",
        "state": "fips_state",
        "place": "fips_place",
    })

    # Convert to numeric
    numeric_cols = ["agg_travel_time", "total_workers", "drove",
                    "public_transit", "walked", "work_from_home"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace Census sentinel values (negative numbers) with NaN
    for col in numeric_cols:
        bad = df[col] < 0
        if bad.any():
            log.warning("Replacing %d sentinel values in %s with NaN", bad.sum(), col)
            df.loc[bad, col] = np.nan

    return df


def compute_commute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive mean commute time and mode-of-transport percentages."""
    # Commuters = total workers minus work-from-home
    commuters = df["total_workers"] - df["work_from_home"]
    commuters = commuters.replace(0, np.nan)  # avoid division by zero
    total = df["total_workers"].replace(0, np.nan)

    df["mean_commute_minutes"] = (df["agg_travel_time"] / commuters).round(1)
    df["pct_drove"] = (df["drove"] / total * 100).round(1)
    df["pct_public_transit"] = (df["public_transit"] / total * 100).round(1)
    df["pct_walked"] = (df["walked"] / total * 100).round(1)
    df["pct_work_from_home"] = (df["work_from_home"] / total * 100).round(1)

    # Sanity: cap mean commute at 90 min (ACS data can have outliers for tiny places)
    df.loc[df["mean_commute_minutes"] > 90, "mean_commute_minutes"] = np.nan

    return df


def match_to_city_list(df: pd.DataFrame) -> pd.DataFrame:
    """Match Census places to the master city list via FIPS codes."""
    cities = pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    merged = cities[["city_id", "name", "state", "fips_state", "fips_place"]].merge(
        df[["fips_state", "fips_place", "mean_commute_minutes",
            "pct_drove", "pct_public_transit", "pct_walked", "pct_work_from_home"]],
        on=["fips_state", "fips_place"],
        how="left",
    )

    matched = merged["mean_commute_minutes"].notna().sum()
    log.info("Matched %d/%d cities (%.1f%%)", matched, len(cities),
             matched / len(cities) * 100)

    return merged


def main():
    raw = fetch_acs_commute_data()
    with_metrics = compute_commute_metrics(raw)
    result = match_to_city_list(with_metrics)

    # Log summary stats
    log.info("Mean commute (matched cities): %.1f min",
             result["mean_commute_minutes"].mean())
    log.info("Median commute (matched cities): %.1f min",
             result["mean_commute_minutes"].median())
    log.info("Work-from-home avg: %.1f%%",
             result["pct_work_from_home"].mean())

    write_parquet_with_metadata(
        result,
        OUTPUT_PATH,
        source_name="US Census Bureau ACS 5-Year (2022)",
        source_url="https://api.census.gov/data/2022/acs/acs5",
        notes=(
            "Commute data from ACS table B08013 (aggregate travel time) and B08301 "
            "(means of transportation to work). Mean commute = aggregate time / "
            "(total workers - work from home). Mode percentages computed from total workers."
        ),
    )
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
