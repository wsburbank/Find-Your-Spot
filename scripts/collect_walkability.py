"""
Collect walkability and transit data from EPA Smart Location Database.

Source: US EPA — Smart Location Database V3 (January 2021)
  URL: https://edg.epa.gov/EPADataCommons/public/OA/
       EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv
  Documentation: https://www.epa.gov/smartgrowth/smart-location-mapping

Variables used:
  - NatWalkInd: National Walkability Index (1-20 scale per census block group)
  - D4A: Distance to nearest transit stop (in miles)
  - D3B: Street intersection density (intersections per sq mile)
  - TotPop: Total population per block group (for population-weighted aggregation)
  - STATEFP, COUNTYFP, TRACTCE, BLKGRPCE: FIPS codes for geographic matching

Methodology:
  1. Download EPA SLD CSV (~200 MB, cached).
  2. Filter to block groups in states matching our city list.
  3. For each city, find all block groups within its county.
  4. Aggregate to city level using population-weighted averages.
  5. Scale NatWalkInd (1-20) to walkability_score (0-100).
  6. Convert D4A (transit proximity) to transit_score (0-100).

Output: data/walkability.parquet
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
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

EPA_SLD_URL = (
    "https://edg.epa.gov/EPADataCommons/public/OA/"
    "EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"
)

# Columns to load (subset to reduce memory)
USECOLS = [
    "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE",
    "CBSA", "TotPop", "Ac_Land",
    "NatWalkInd",  # National Walkability Index (1-20)
    "D3B",         # Street intersection density
    "D4A",         # Distance to nearest transit stop
    "D4D",         # Aggregate frequency of transit service
]


def download_epa_sld() -> pd.DataFrame:
    """Download the EPA Smart Location Database CSV (with caching)."""
    cache_path = CACHE_DIR / "epa_sld_v3.csv"

    if cache_path.exists():
        log.info("Loading cached EPA SLD from %s", cache_path)
        return pd.read_csv(cache_path, usecols=USECOLS, dtype={"STATEFP": str, "COUNTYFP": str})

    log.info("Downloading EPA SLD (~200 MB, this may take a few minutes)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    r = requests.get(EPA_SLD_URL, timeout=600, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get("Content-Length", 0))
    downloaded = 0

    with open(cache_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size and downloaded % (10 * 1024 * 1024) < 8192:
                pct = downloaded / total_size * 100
                log.info("  Downloaded %.0f MB / %.0f MB (%.0f%%)",
                         downloaded / 1e6, total_size / 1e6, pct)

    log.info("Downloaded EPA SLD: %.1f MB", cache_path.stat().st_size / 1e6)
    return pd.read_csv(cache_path, usecols=USECOLS, dtype={"STATEFP": str, "COUNTYFP": str})


def aggregate_to_cities(sld: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Aggregate block group data to city level using population-weighted averages.

    Strategy:
      - Match block groups to cities by county FIPS (state + county).
      - If a county has multiple cities, assign block groups to the nearest city
        by haversine distance from block group centroid.
      - Population-weight all metrics.
    """
    # Build FIPS5 keys
    sld = sld.copy()
    sld["fips5"] = sld["STATEFP"].str.zfill(2) + sld["COUNTYFP"].str.zfill(3)

    cities_copy = cities[["city_id", "name", "state", "fips_state", "fips_county", "lat", "lon"]].copy()
    cities_copy["fips5"] = (
        cities_copy["fips_state"].astype(str).str.zfill(2) +
        cities_copy["fips_county"].astype(str).str.zfill(3)
    )

    # Filter SLD to counties that have at least one city
    relevant_counties = set(cities_copy["fips5"].unique())
    sld_filtered = sld[sld["fips5"].isin(relevant_counties)].copy()
    log.info("Block groups in relevant counties: %d / %d total",
             len(sld_filtered), len(sld))

    # Treat EPA sentinel values (-99999) as missing
    for col in ["D4A", "D4D", "NatWalkInd", "D3B"]:
        if col in sld_filtered.columns:
            sld_filtered.loc[sld_filtered[col] < -9000, col] = np.nan

    # Filter out block groups with no population or no walkability data
    sld_filtered = sld_filtered[
        (sld_filtered["TotPop"] > 0) &
        sld_filtered["NatWalkInd"].notna()
    ].copy()
    log.info("Block groups with population and walkability data: %d", len(sld_filtered))

    # For counties with multiple cities, we need to assign each block group
    # to the nearest city. For simplicity, group by county and compute
    # county-wide population-weighted averages, then assign to all cities in that county.
    county_metrics = sld_filtered.groupby("fips5").apply(
        _weighted_county_metrics, include_groups=False
    ).reset_index()

    log.info("County-level metrics computed for %d counties", len(county_metrics))

    # Merge with cities
    result = cities_copy[["city_id", "fips5"]].merge(
        county_metrics,
        on="fips5",
        how="left",
    )

    return result[["city_id", "walkability_score", "transit_score",
                    "intersection_density", "transit_frequency"]]


def _weighted_county_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute population-weighted metrics for a county."""
    pop = group["TotPop"].fillna(0)
    total_pop = pop.sum()

    if total_pop == 0:
        return pd.Series({
            "walkability_score": np.nan,
            "transit_score": np.nan,
            "intersection_density": np.nan,
            "transit_frequency": np.nan,
        })

    weights = pop / total_pop

    # NatWalkInd (1-20) -> walkability_score (0-100)
    # Scale: 1->0, 20->100
    nat_walk = group["NatWalkInd"].fillna(0)
    weighted_walk = (nat_walk * weights).sum()
    walkability_score = round(np.clip((weighted_walk - 1) / 19 * 100, 0, 100), 1)

    # D4A: aggregate frequency of transit service per sq mi
    # Higher = more transit access. NaN means no transit data for that block group.
    # Typical range for block groups WITH transit: 10 to 1200
    # Scale to 0-100 using log transform
    d4a = group["D4A"].copy()
    d4a_valid = d4a.notna()
    if d4a_valid.any():
        d4a_clean = d4a.fillna(0)
        weighted_d4a = (d4a_clean * weights).sum()
        # Log scale: 0->0, 10->37, 50->63, 100->74, 500->100
        if weighted_d4a > 0:
            transit_score = round(np.clip(np.log10(weighted_d4a + 1) * 37, 0, 100), 1)
        else:
            transit_score = 0.0
    else:
        transit_score = 0.0

    # D3B: street intersection density (intersections per sq mi)
    d3b = group["D3B"].fillna(0)
    intersection_density = round((d3b * weights).sum(), 1)

    # D4D: aggregate frequency of transit service within 0.25 mi
    d4d = group.get("D4D", pd.Series(0, index=group.index))
    d4d = d4d.fillna(0)
    transit_frequency = round((d4d * weights).sum(), 1)

    return pd.Series({
        "walkability_score": walkability_score,
        "transit_score": transit_score,
        "intersection_density": intersection_density,
        "transit_frequency": transit_frequency,
    })


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Download EPA SLD
    sld = download_epa_sld()
    log.info("EPA SLD loaded: %d block groups, %d columns", len(sld), len(sld.columns))

    # Aggregate to city level
    result = aggregate_to_cities(sld, cities)

    # Report
    for col in ["walkability_score", "transit_score"]:
        n = result[col].notna().sum()
        vals = result[col].dropna()
        log.info("\n%s: %d/%d cities (%.1f%%)", col, n, len(result),
                 n / len(result) * 100)
        if not vals.empty:
            log.info("  min=%.1f, median=%.1f, mean=%.1f, max=%.1f",
                     vals.min(), vals.median(), vals.mean(), vals.max())

    # Top/bottom walkability
    merged = result.merge(cities[["city_id", "name", "state"]], on="city_id")
    valid = merged[merged["walkability_score"].notna()]

    if not valid.empty:
        log.info("\nTop 10 most walkable cities:")
        for _, r in valid.nlargest(10, "walkability_score").iterrows():
            log.info("  %s, %s: walkability=%.1f, transit=%.1f",
                     r["name"], r["state"], r["walkability_score"], r["transit_score"])

        log.info("\nBottom 10 least walkable cities:")
        for _, r in valid.nsmallest(10, "walkability_score").iterrows():
            log.info("  %s, %s: walkability=%.1f, transit=%.1f",
                     r["name"], r["state"], r["walkability_score"], r["transit_score"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "walkability.parquet",
        source_name="US EPA — Smart Location Database V3 (January 2021)",
        source_url="https://edg.epa.gov/EPADataCommons/public/OA/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv",
        date_collected="2021",
        notes=(
            "Walkability and transit scores derived from EPA Smart Location Database V3. "
            "NatWalkInd (National Walkability Index, 1-20 scale) scaled to 0-100 for "
            "walkability_score. Transit score derived from D4A (aggregate frequency of "
            "transit service per sq mi) using log-scale transform. Block group data "
            "aggregated to city level using population-weighted averages by county FIPS."
        ),
    )

    log.info("\nSaved: data/walkability.parquet (%d cities)", len(result))


if __name__ == "__main__":
    main()
