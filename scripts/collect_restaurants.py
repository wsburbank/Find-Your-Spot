"""
Collect restaurant and food scene data from Census County Business Patterns.

Source: US Census Bureau — County Business Patterns (CBP) 2022
URL: https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22co.zip

NAICS codes used:
  722511 - Full-Service Restaurants
  722513 - Limited-Service Restaurants (fast food, quick service)
  722515 - Snack and Nonalcoholic Beverage Bars (coffee shops, juice bars)
  722410 - Drinking Places (Alcoholic Beverages) — bars, pubs, taverns
  312120 - Breweries (craft and commercial)

The CBP bulk file provides establishment counts by county and NAICS code. We map
counties to our cities using FIPS codes from the master city list and apportion
by population share when multiple cities share a county.

Output: data/restaurants.parquet
"""

import logging
import sys
import zipfile
from io import BytesIO
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

# CBP bulk county file (all NAICS codes, all counties)
CBP_URL = "https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22co.zip"

# NAICS codes and groupings
NAICS_CODES = {
    "restaurants_fullservice": ["722511"],
    "restaurants_quickservice": ["722513"],
    "coffee_snack_bars": ["722515"],
    "bars": ["722410"],
    "breweries": ["312120"],
}

# Fallback parent codes for suppression recovery
FALLBACK_CODES = {
    "restaurants_fullservice": "7225",
    "restaurants_quickservice": "7225",
    "coffee_snack_bars": "7225",
    "bars": "7224",
    "breweries": "3121",
}


def download_cbp() -> pd.DataFrame:
    """Download CBP county-level bulk data file."""
    cache_file = CACHE_DIR / "cbp22co.txt"

    if cache_file.exists():
        log.info("Using cached CBP file: %s", cache_file)
        return pd.read_csv(cache_file, low_memory=False)

    log.info("Downloading CBP county file from %s ...", CBP_URL)
    resp = requests.get(CBP_URL, timeout=180)
    resp.raise_for_status()
    log.info("  Downloaded %d MB", len(resp.content) // (1024 * 1024))

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        txt_file = zf.namelist()[0]
        log.info("  Extracting: %s", txt_file)
        with zf.open(txt_file) as f:
            df = pd.read_csv(f, low_memory=False)

    # Cache for future runs
    df.to_csv(cache_file, index=False)
    log.info("  Cached to %s (%d rows)", cache_file, len(df))
    return df


def extract_restaurant_data(cbp: pd.DataFrame) -> pd.DataFrame:
    """Filter CBP data to restaurant/food NAICS codes and aggregate by county."""
    # Normalize FIPS
    cbp["fips_state"] = cbp["fipstate"].astype(str).str.zfill(2)
    cbp["fips_county"] = cbp["fipscty"].astype(str).str.zfill(3)
    cbp["establishments"] = pd.to_numeric(cbp["est"], errors="coerce").fillna(0).astype(int)

    results = []

    for group_name, codes in NAICS_CODES.items():
        # First try exact NAICS codes
        mask = cbp["naics"].isin(codes)
        group_data = cbp[mask].groupby(["fips_state", "fips_county"])["establishments"].sum().reset_index()
        group_data = group_data.rename(columns={"establishments": group_name})

        # Fallback recovery: for counties with 0 from detail codes, check parent code
        fallback_code = FALLBACK_CODES.get(group_name)
        if fallback_code:
            fb_mask = cbp["naics"] == fallback_code
            fb_data = cbp[fb_mask][["fips_state", "fips_county", "establishments"]].copy()
            fb_data = fb_data.rename(columns={"establishments": "parent_count"})

            if not fb_data.empty and not group_data.empty:
                merged = fb_data.merge(group_data, on=["fips_state", "fips_county"], how="left")
                merged[group_name] = merged[group_name].fillna(0)
                # Counties with parent data but no detail = suppressed
                suppressed = merged[(merged[group_name] == 0) & (merged["parent_count"] > 0)]
                if len(suppressed) > 0:
                    log.info("  %s: recovered %d suppressed counties via fallback %s",
                             group_name, len(suppressed), fallback_code)
                    recovered = suppressed[["fips_state", "fips_county", "parent_count"]].rename(
                        columns={"parent_count": group_name}
                    )
                    group_data = pd.concat([group_data, recovered], ignore_index=True)
            elif group_data.empty and not fb_data.empty:
                # No detail at all, use parent
                group_data = fb_data.rename(columns={"parent_count": group_name})

        log.info("  %s: %d counties with data, %d total establishments",
                 group_name, len(group_data), group_data[group_name].sum())
        results.append(group_data)

    # Merge all groups together
    combined = results[0]
    for df in results[1:]:
        combined = combined.merge(df, on=["fips_state", "fips_county"], how="outer")

    combined = combined.fillna(0)
    for col in NAICS_CODES:
        combined[col] = combined[col].astype(int)

    log.info("Combined: %d counties with food/drink data", len(combined))
    return combined


def aggregate_to_cities(county_data: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county-level restaurant data to cities by population share."""
    cities_copy = cities[["city_id", "fips_state", "fips_county", "population"]].copy()
    cities_copy["fips_state"] = cities_copy["fips_state"].astype(str).str.zfill(2)
    cities_copy["fips_county"] = cities_copy["fips_county"].astype(str).str.zfill(3)

    result = cities_copy.merge(county_data, on=["fips_state", "fips_county"], how="left")

    # Apportion by population share within each county
    county_pop = result.groupby(["fips_state", "fips_county"])["population"].transform("sum")
    pop_share = result["population"] / county_pop.clip(lower=1)

    for group_name in NAICS_CODES:
        if group_name in result.columns:
            result[group_name] = (result[group_name].fillna(0) * pop_share).round(0).astype(int)

    # Calculate totals and per-capita
    result["restaurants_total"] = (
        result["restaurants_fullservice"] +
        result["restaurants_quickservice"] +
        result["coffee_snack_bars"]
    )
    result["restaurants_per_10k"] = (
        result["restaurants_total"] / result["population"] * 10000
    ).round(1)
    result["bars_per_10k"] = (
        result["bars"] / result["population"] * 10000
    ).round(1)

    # Keep only metric columns
    metric_cols = [
        "restaurants_fullservice", "restaurants_quickservice", "coffee_snack_bars",
        "bars", "breweries", "restaurants_total", "restaurants_per_10k", "bars_per_10k",
    ]
    return result[["city_id"] + metric_cols]


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Download bulk CBP data
    cbp = download_cbp()
    log.info("CBP data: %d rows, NAICS codes: %d unique", len(cbp), cbp["naics"].nunique())

    # Extract restaurant/food data
    county_data = extract_restaurant_data(cbp)

    # Aggregate to cities
    result = aggregate_to_cities(county_data, cities)

    # Report coverage
    log.info("\nRestaurant data coverage:")
    for col in result.columns:
        if col == "city_id":
            continue
        has_data = (result[col] > 0).sum()
        log.info("  %s: %d/%d cities (total: %s)",
                 col, has_data, len(result), f"{result[col].sum():,.0f}")

    log.info("\nTop 10 cities by restaurants per 10K:")
    top = result.nlargest(10, "restaurants_per_10k").merge(
        cities[["city_id", "name", "state", "population"]], on="city_id"
    )
    for _, row in top.iterrows():
        log.info("  %s, %s (pop %dk): %.1f restaurants/10k, %d breweries",
                 row["name"], row["state"], row["population"] // 1000,
                 row["restaurants_per_10k"], row["breweries"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "restaurants.parquet",
        source_name="US Census Bureau — County Business Patterns 2022",
        source_url="https://www2.census.gov/programs-surveys/cbp/datasets/2022/cbp22co.zip",
        date_collected="2022",
        notes=(
            "Restaurant and food/drink establishment counts from Census CBP 2022 by NAICS code. "
            "Full-service: 722511. Quick-service: 722513. Coffee/snack: 722515. "
            "Bars: 722410. Breweries: 312120. County-level data apportioned to cities "
            "by population share. Per-capita rates calculated per 10K residents."
        ),
    )
    log.info("\nSaved: data/restaurants.parquet (%d rows)", len(result))


if __name__ == "__main__":
    main()
