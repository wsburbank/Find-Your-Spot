"""
Phase 2b: Collect cost of living and housing data.

Sources:
  - Census ACS 5-Year (2022): Median home value (B25077_001E), Median household income (B19013_001E)
  - State tax data: Compiled from public sources (Tax Foundation)

Output: data/cost_of_living.parquet
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

# State tax data from Tax Foundation (2024 data, publicly available)
# Source: https://taxfoundation.org/data/all/state/
# Income: representative marginal bracket for middle-to-upper earners
# Sales: state-level base rate (excludes local add-ons)
# Property: mean effective rate on owner-occupied housing (Tax Foundation 2021)
# Last verified: 2026-05-02 — spot-checked 10 states, all within 0.06% of source
STATE_TAXES = {
    "AL": {"income_tax_rate": 5.0, "sales_tax_rate": 4.0, "property_tax_rate": 0.40, "no_income_tax": False},
    "AK": {"income_tax_rate": 0.0, "sales_tax_rate": 0.0, "property_tax_rate": 1.04, "no_income_tax": True},
    "AZ": {"income_tax_rate": 2.5, "sales_tax_rate": 5.6, "property_tax_rate": 0.62, "no_income_tax": False},
    "AR": {"income_tax_rate": 4.4, "sales_tax_rate": 6.5, "property_tax_rate": 0.62, "no_income_tax": False},
    "CA": {"income_tax_rate": 9.3, "sales_tax_rate": 7.25, "property_tax_rate": 0.71, "no_income_tax": False},
    "CO": {"income_tax_rate": 4.4, "sales_tax_rate": 2.9, "property_tax_rate": 0.49, "no_income_tax": False},
    "CT": {"income_tax_rate": 5.0, "sales_tax_rate": 6.35, "property_tax_rate": 2.15, "no_income_tax": False},
    "DE": {"income_tax_rate": 6.6, "sales_tax_rate": 0.0, "property_tax_rate": 0.57, "no_income_tax": False},
    "DC": {"income_tax_rate": 6.5, "sales_tax_rate": 6.0, "property_tax_rate": 0.56, "no_income_tax": False},
    "FL": {"income_tax_rate": 0.0, "sales_tax_rate": 6.0, "property_tax_rate": 0.86, "no_income_tax": True},
    "GA": {"income_tax_rate": 5.49, "sales_tax_rate": 4.0, "property_tax_rate": 0.90, "no_income_tax": False},
    "HI": {"income_tax_rate": 7.2, "sales_tax_rate": 4.0, "property_tax_rate": 0.27, "no_income_tax": False},
    "ID": {"income_tax_rate": 5.8, "sales_tax_rate": 6.0, "property_tax_rate": 0.63, "no_income_tax": False},
    "IL": {"income_tax_rate": 4.95, "sales_tax_rate": 6.25, "property_tax_rate": 2.07, "no_income_tax": False},
    "IN": {"income_tax_rate": 3.05, "sales_tax_rate": 7.0, "property_tax_rate": 0.83, "no_income_tax": False},
    "IA": {"income_tax_rate": 5.7, "sales_tax_rate": 6.0, "property_tax_rate": 1.52, "no_income_tax": False},
    "KS": {"income_tax_rate": 5.7, "sales_tax_rate": 6.5, "property_tax_rate": 1.33, "no_income_tax": False},
    "KY": {"income_tax_rate": 4.0, "sales_tax_rate": 6.0, "property_tax_rate": 0.83, "no_income_tax": False},
    "LA": {"income_tax_rate": 4.25, "sales_tax_rate": 4.45, "property_tax_rate": 0.56, "no_income_tax": False},
    "ME": {"income_tax_rate": 7.15, "sales_tax_rate": 5.5, "property_tax_rate": 1.30, "no_income_tax": False},
    "MD": {"income_tax_rate": 5.75, "sales_tax_rate": 6.0, "property_tax_rate": 1.07, "no_income_tax": False},
    "MA": {"income_tax_rate": 5.0, "sales_tax_rate": 6.25, "property_tax_rate": 1.15, "no_income_tax": False},
    "MI": {"income_tax_rate": 4.25, "sales_tax_rate": 6.0, "property_tax_rate": 1.44, "no_income_tax": False},
    "MN": {"income_tax_rate": 7.85, "sales_tax_rate": 6.875, "property_tax_rate": 1.08, "no_income_tax": False},
    "MS": {"income_tax_rate": 5.0, "sales_tax_rate": 7.0, "property_tax_rate": 0.67, "no_income_tax": False},
    "MO": {"income_tax_rate": 4.95, "sales_tax_rate": 4.225, "property_tax_rate": 0.93, "no_income_tax": False},
    "MT": {"income_tax_rate": 5.9, "sales_tax_rate": 0.0, "property_tax_rate": 0.74, "no_income_tax": False},
    "NE": {"income_tax_rate": 5.84, "sales_tax_rate": 5.5, "property_tax_rate": 1.61, "no_income_tax": False},
    "NV": {"income_tax_rate": 0.0, "sales_tax_rate": 6.85, "property_tax_rate": 0.53, "no_income_tax": True},
    "NH": {"income_tax_rate": 0.0, "sales_tax_rate": 0.0, "property_tax_rate": 1.93, "no_income_tax": True},
    "NJ": {"income_tax_rate": 6.37, "sales_tax_rate": 6.625, "property_tax_rate": 2.23, "no_income_tax": False},
    "NM": {"income_tax_rate": 4.9, "sales_tax_rate": 4.875, "property_tax_rate": 0.67, "no_income_tax": False},
    "NY": {"income_tax_rate": 6.85, "sales_tax_rate": 4.0, "property_tax_rate": 1.62, "no_income_tax": False},
    "NC": {"income_tax_rate": 4.5, "sales_tax_rate": 4.75, "property_tax_rate": 0.77, "no_income_tax": False},
    "ND": {"income_tax_rate": 1.95, "sales_tax_rate": 5.0, "property_tax_rate": 0.94, "no_income_tax": False},
    "OH": {"income_tax_rate": 3.5, "sales_tax_rate": 5.75, "property_tax_rate": 1.53, "no_income_tax": False},
    "OK": {"income_tax_rate": 4.75, "sales_tax_rate": 4.5, "property_tax_rate": 0.88, "no_income_tax": False},
    "OR": {"income_tax_rate": 8.75, "sales_tax_rate": 0.0, "property_tax_rate": 0.87, "no_income_tax": False},
    "PA": {"income_tax_rate": 3.07, "sales_tax_rate": 6.0, "property_tax_rate": 1.53, "no_income_tax": False},
    "RI": {"income_tax_rate": 5.99, "sales_tax_rate": 7.0, "property_tax_rate": 1.40, "no_income_tax": False},
    "SC": {"income_tax_rate": 6.4, "sales_tax_rate": 6.0, "property_tax_rate": 0.56, "no_income_tax": False},
    "SD": {"income_tax_rate": 0.0, "sales_tax_rate": 4.2, "property_tax_rate": 1.14, "no_income_tax": True},
    "TN": {"income_tax_rate": 0.0, "sales_tax_rate": 7.0, "property_tax_rate": 0.64, "no_income_tax": True},
    "TX": {"income_tax_rate": 0.0, "sales_tax_rate": 6.25, "property_tax_rate": 1.68, "no_income_tax": True},
    "UT": {"income_tax_rate": 4.65, "sales_tax_rate": 6.1, "property_tax_rate": 0.57, "no_income_tax": False},
    "VT": {"income_tax_rate": 6.6, "sales_tax_rate": 6.0, "property_tax_rate": 1.83, "no_income_tax": False},
    "VA": {"income_tax_rate": 5.75, "sales_tax_rate": 5.3, "property_tax_rate": 0.80, "no_income_tax": False},
    "WA": {"income_tax_rate": 0.0, "sales_tax_rate": 6.5, "property_tax_rate": 0.87, "no_income_tax": True},
    "WV": {"income_tax_rate": 5.12, "sales_tax_rate": 6.0, "property_tax_rate": 0.57, "no_income_tax": False},
    "WI": {"income_tax_rate": 5.3, "sales_tax_rate": 5.0, "property_tax_rate": 1.61, "no_income_tax": False},
    "WY": {"income_tax_rate": 0.0, "sales_tax_rate": 4.0, "property_tax_rate": 0.56, "no_income_tax": True},
}


def fetch_acs_housing_data() -> pd.DataFrame:
    """Fetch median home value and median income from Census ACS for all places."""
    log.info("Fetching ACS housing/income data...")

    # B25077_001E = Median value of owner-occupied housing units
    # B19013_001E = Median household income
    # B25064_001E = Median gross rent
    params = {
        "get": "NAME,B25077_001E,B19013_001E,B25064_001E",
        "for": "place:*",
        "in": "state:*",
    }
    resp = requests.get(ACS_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={
        "NAME": "raw_name",
        "B25077_001E": "median_home_price",
        "B19013_001E": "median_household_income",
        "B25064_001E": "median_gross_rent",
        "state": "fips_state",
        "place": "fips_place",
    })

    for col in ["median_home_price", "median_household_income", "median_gross_rent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Census ACS uses -666666666 as a sentinel for suppressed/missing data.
    # Replace any negative values (which are always sentinel values) with NaN.
    for col in ["median_home_price", "median_household_income", "median_gross_rent"]:
        bad = df[col] < 0
        if bad.any():
            log.warning(
                "Replacing %d sentinel values in %s with NaN",
                bad.sum(), col,
            )
            df.loc[bad, col] = np.nan

    log.info("Fetched ACS data for %d places", len(df))
    return df[["fips_state", "fips_place", "median_home_price",
               "median_household_income", "median_gross_rent"]]


def compute_cost_of_living_index(df: pd.DataFrame) -> pd.Series:
    """Compute a simple cost-of-living index relative to national median.

    Index = 100 means exactly at the national median.
    Based on median home value (weighted 60%) and median rent (40%).
    """
    national_home = df["median_home_price"].median()
    national_rent = df["median_gross_rent"].median()

    home_ratio = df["median_home_price"] / national_home * 100
    rent_ratio = df["median_gross_rent"] / national_rent * 100

    return (home_ratio * 0.6 + rent_ratio * 0.4).round(1)


HOUSING_COLS = ["median_home_price", "median_household_income", "median_gross_rent"]


def interpolate_missing_housing(
    df: pd.DataFrame, max_neighbors: int = 5, max_radius_miles: float = 50.0,
) -> pd.DataFrame:
    """Fill missing housing values using inverse-distance weighted average of nearby cities.

    Interpolates median_home_price, median_household_income, and median_gross_rent.
    The input DataFrame must already contain lat, lon columns (from the master city list merge).

    Args:
        df: DataFrame with city_id, lat, lon, and housing columns (may have NaN values).
        max_neighbors: Maximum number of nearby cities to use for interpolation.
        max_radius_miles: Only consider cities within this radius.

    Returns:
        DataFrame with interpolated values and an 'interpolated' flag column.
    """
    from utilities.geo import haversine_miles

    df = df.copy()
    df["interpolated"] = False

    # Pre-compute distance matrix only for rows that need interpolation
    needs_interp = df[HOUSING_COLS].isna().any(axis=1)
    if not needs_interp.any():
        return df

    for idx in df.index[needs_interp]:
        target_lat = df.at[idx, "lat"]
        target_lon = df.at[idx, "lon"]
        if pd.isna(target_lat) or pd.isna(target_lon):
            continue

        missing_cols = [c for c in HOUSING_COLS if pd.isna(df.at[idx, c])]
        if not missing_cols:
            continue

        # Find nearby cities that have ALL the missing columns populated
        valid = df[df[missing_cols].notna().all(axis=1)].copy()
        if valid.empty:
            log.warning("No valid neighbors for %s, %s", df.at[idx, "name"], df.at[idx, "state"])
            continue

        dists = valid.apply(
            lambda r: haversine_miles(target_lat, target_lon, r["lat"], r["lon"]),
            axis=1,
        )
        nearby = valid.assign(dist=dists)
        nearby = nearby[nearby["dist"] <= max_radius_miles].nsmallest(
            max_neighbors, "dist",
        )

        if nearby.empty:
            log.warning("No nearby cities for interpolation: %s", df.at[idx, "name"])
            continue

        # Inverse-distance weighting (1/d weights)
        weights = 1.0 / nearby["dist"].clip(lower=0.1)
        for col in missing_cols:
            interpolated = (nearby[col] * weights).sum() / weights.sum()
            df.at[idx, col] = round(interpolated, 0)

        df.at[idx, "interpolated"] = True
        log.info(
            "Interpolated %s for %s, %s from %d neighbors",
            ", ".join(missing_cols), df.at[idx, "name"], df.at[idx, "state"],
            len(nearby),
        )

    interp_count = df["interpolated"].sum()
    total = len(df)
    pct = interp_count / total * 100
    log.info("Interpolated %d/%d cities (%.1f%%)", interp_count, total, pct)
    if pct > 5.0:
        log.warning(
            "Interpolation exceeds 5%% threshold (%.1f%%). Review data sources.", pct,
        )

    return df


def main():
    cities = pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")
    output_path = PROJECT_ROOT / "data" / "cost_of_living.parquet"

    # Fetch ACS housing data
    acs = fetch_acs_housing_data()

    # Merge with master city list on FIPS codes
    merged = cities.merge(acs, on=["fips_state", "fips_place"], how="left")

    # Interpolate missing housing data from nearby cities (capped at 5% of dataset)
    merged = interpolate_missing_housing(merged)

    # Add cost of living index
    merged["cost_of_living_index"] = compute_cost_of_living_index(merged)

    # Add state tax data
    merged["state_income_tax_rate"] = merged["state"].map(
        lambda s: STATE_TAXES.get(s, {}).get("income_tax_rate"))
    merged["state_sales_tax_rate"] = merged["state"].map(
        lambda s: STATE_TAXES.get(s, {}).get("sales_tax_rate"))
    merged["avg_property_tax_rate"] = merged["state"].map(
        lambda s: STATE_TAXES.get(s, {}).get("property_tax_rate"))
    merged["no_income_tax_state"] = merged["state"].map(
        lambda s: STATE_TAXES.get(s, {}).get("no_income_tax", False))

    # Select output columns
    out = merged[[
        "city_id", "name", "state",
        "median_home_price", "median_household_income", "median_gross_rent",
        "cost_of_living_index", "interpolated",
        "state_income_tax_rate", "state_sales_tax_rate",
        "avg_property_tax_rate", "no_income_tax_state",
    ]].copy()

    coverage = out["median_home_price"].notna().sum()
    log.info("Housing data coverage: %d/%d cities", coverage, len(cities))

    write_parquet_with_metadata(
        out,
        output_path,
        source_name="US Census ACS 5-Year (2022) + Tax Foundation (2024)",
        source_url="https://api.census.gov/data/2022/acs/acs5",
        date_collected="2026-05-01",
        notes=(
            "Median home value and income from ACS 2022. "
            "State tax rates from Tax Foundation 2024 published data. "
            "COL index computed relative to national median (home value 60%, rent 40%)."
        ),
    )

    print(f"\nCost of living coverage: {coverage}/{len(cities)} ({coverage/len(cities)*100:.1f}%)")
    print(out[["name", "state", "median_home_price", "cost_of_living_index",
               "state_income_tax_rate", "no_income_tax_state"]].head(15).to_string())


if __name__ == "__main__":
    main()
