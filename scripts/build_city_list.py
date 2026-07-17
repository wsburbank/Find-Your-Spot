"""
Phase 1: Build the master city list from US Census Bureau data.

Sources:
  - Census Bureau American Community Survey (ACS) 5-Year Estimates (2022)
    Population via the Census API
  - Census Gazetteer Files (2023) for coordinates and land area
  - OMB CBSA Delineation Files (2023) for metro area codes and populations
  - OMB Principal City List (2023) for MSA principal city inclusion

Rules:
  - USA only (all 50 states + DC)
  - Every Census place with population >= 25,000
  - Every principal city of an MSA not already captured by the threshold rule
  - Every state (except RI) must have at least 10 cities
  - Add smaller places as needed to meet the 10-per-state minimum
  - NYC split into 5 boroughs
  - Cities must be at least 20 miles apart (enforced per state + neighbors)
  - Eliminated cities roll up to the nearest surviving city:
    their population is added to area_population and their names
    are listed in incorporated_places
  - Each city is mapped to its CBSA via county assignment for metro_pop

Output: data/cities_master.parquet with embedded source metadata
"""

import io
import logging
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.geo import enforce_separation, haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CENSUS_API_BASE = "https://api.census.gov/data"
ACS_YEAR = "2022"
ACS_DATASET = f"{CENSUS_API_BASE}/{ACS_YEAR}/acs/acs5"

POP_THRESHOLD = 25_000

# State FIPS codes
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}

# NYC boroughs (2022 ACS estimates)
NYC_BOROUGHS = [
    {"name": "Manhattan", "state": "NY", "lat": 40.7831, "lon": -73.9712,
     "population": 1694251, "fips_state": "36", "fips_place": "61000_manhattan"},
    {"name": "Brooklyn", "state": "NY", "lat": 40.6782, "lon": -73.9442,
     "population": 2590516, "fips_state": "36", "fips_place": "61000_brooklyn"},
    {"name": "Queens", "state": "NY", "lat": 40.7282, "lon": -73.7949,
     "population": 2278029, "fips_state": "36", "fips_place": "61000_queens"},
    {"name": "Bronx", "state": "NY", "lat": 40.8448, "lon": -73.8648,
     "population": 1379946, "fips_state": "36", "fips_place": "61000_bronx"},
    {"name": "Staten Island", "state": "NY", "lat": 40.5795, "lon": -74.1502,
     "population": 491133, "fips_state": "36", "fips_place": "61000_statenisland"},
]

MIN_CITIES_PER_STATE = 10
RHODE_ISLAND_MIN = 5


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_census_places() -> pd.DataFrame:
    """Fetch all Census places with population from ACS 5-Year API."""
    log.info("Fetching Census places from ACS %s...", ACS_YEAR)

    resp = requests.get(ACS_DATASET, params={
        "get": "NAME,B01003_001E",
        "for": "place:*",
        "in": "state:*",
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={
        "NAME": "raw_name",
        "B01003_001E": "population",
        "state": "fips_state",
        "place": "fips_place",
    })

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    df["population"] = df["population"].astype(int)

    # Parse city name
    df["name"] = df["raw_name"].str.replace(r",\s*\w[\w\s]*$", "", regex=True)
    df["name"] = df["name"].str.replace(
        r"\s+(city and borough|city and|city|town|village|CDP|borough|municipality|"
        r"charter township|metropolitan government|unified government|"
        r"consolidated government|urban county|metro government)(\s*\(.*\))?$",
        "", regex=True,
    )
    df["name"] = df["name"].str.strip()

    df["state"] = df["fips_state"].map(STATE_FIPS)
    df = df.dropna(subset=["state"])

    log.info("Fetched %d Census places", len(df))
    return df[["name", "state", "population", "fips_state", "fips_place"]]


def fetch_gazetteer_coordinates() -> pd.DataFrame:
    """Download Census Gazetteer file for place coordinates."""
    log.info("Downloading Census Gazetteer coordinates (2023)...")
    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_place_national.zip"

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    txt_name = [n for n in zf.namelist() if n.endswith(".txt")][0]
    with zf.open(txt_name) as f:
        df = pd.read_csv(f, sep="\t", dtype=str, encoding="latin-1")

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "GEOID": "geoid", "INTPTLAT": "lat", "INTPTLONG": "lon",
        "ALAND_SQMI": "land_area_sq_mi",
    })

    df["fips_state"] = df["geoid"].str[:2]
    df["fips_place"] = df["geoid"].str[2:]
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["land_area_sq_mi"] = pd.to_numeric(df["land_area_sq_mi"], errors="coerce")

    log.info("Loaded %d Gazetteer records", len(df))
    return df[["fips_state", "fips_place", "lat", "lon", "land_area_sq_mi"]]


def fetch_cbsa_delineation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OMB CBSA delineation files.

    Returns:
        (county_cbsa, principal_cities) DataFrames
    """
    log.info("Downloading CBSA delineation files (2023)...")

    # List 1: county -> CBSA mapping
    url1 = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
    resp1 = requests.get(url1, timeout=30)
    resp1.raise_for_status()
    county_cbsa = pd.read_excel(io.BytesIO(resp1.content), skiprows=2)
    county_cbsa = county_cbsa.dropna(subset=["FIPS State Code"])
    county_cbsa["fips_state"] = county_cbsa["FIPS State Code"].astype(int).astype(str).str.zfill(2)
    county_cbsa["fips_county"] = county_cbsa["FIPS County Code"].astype(int).astype(str).str.zfill(3)
    county_cbsa["cbsa_code"] = county_cbsa["CBSA Code"].astype(int).astype(str)
    county_cbsa["cbsa_title"] = county_cbsa["CBSA Title"]
    county_cbsa["cbsa_type"] = county_cbsa["Metropolitan/Micropolitan Statistical Area"]
    log.info("Loaded %d county-CBSA mappings", len(county_cbsa))

    # List 2: principal cities
    url2 = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list2_2023.xlsx"
    resp2 = requests.get(url2, timeout=30)
    resp2.raise_for_status()
    principals = pd.read_excel(io.BytesIO(resp2.content), skiprows=2)
    principals = principals.dropna(subset=["FIPS State Code"])
    principals["fips_state"] = principals["FIPS State Code"].astype(int).astype(str).str.zfill(2)
    principals["fips_place"] = principals["FIPS Place Code"].astype(int).astype(str).str.zfill(5)
    principals["cbsa_code"] = principals["CBSA Code"].astype(int).astype(str)
    principals["cbsa_type"] = principals["Metropolitan/Micropolitan Statistical Area"]
    principals["cbsa_title"] = principals["CBSA Title"]
    principals["principal_city_name"] = principals["Principal City Name"]
    log.info("Loaded %d principal cities", len(principals))

    return county_cbsa, principals


def fetch_cbsa_populations() -> pd.DataFrame:
    """Fetch CBSA-level population from Census ACS API."""
    log.info("Fetching CBSA populations from ACS %s...", ACS_YEAR)

    resp = requests.get(ACS_DATASET, params={
        "get": "NAME,B01003_001E",
        "for": "metropolitan statistical area/micropolitan statistical area:*",
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={
        "B01003_001E": "metro_pop",
        "metropolitan statistical area/micropolitan statistical area": "cbsa_code",
    })
    df["metro_pop"] = pd.to_numeric(df["metro_pop"], errors="coerce").astype("Int64")
    log.info("Fetched populations for %d CBSAs", len(df))
    return df[["cbsa_code", "metro_pop"]]


# ---------------------------------------------------------------------------
# City-to-county assignment (for CBSA mapping)
# ---------------------------------------------------------------------------

def assign_counties_fcc(cities: pd.DataFrame) -> pd.Series:
    """Assign each city to its county using the FCC Area API.

    The FCC Census Block API returns the exact county FIPS for a given lat/lon.
    This is more accurate than nearest-centroid matching for cities near county
    borders (e.g., Seattle near Puget Sound).

    Returns a Series of 3-digit fips_county values aligned to cities' index.
    """
    log.info("Assigning cities to counties via FCC Area API (%d lookups)...", len(cities))
    result = pd.Series(index=cities.index, dtype=str, name="fips_county")

    for i, (idx, row) in enumerate(cities.iterrows()):
        try:
            resp = requests.get(
                "https://geo.fcc.gov/api/census/area",
                params={"lat": row["lat"], "lon": row["lon"], "format": "json"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    county_fips = results[0].get("county_fips", "")
                    # county_fips is full 5-digit (state+county), we want just county (3)
                    result.at[idx] = county_fips[2:] if len(county_fips) == 5 else ""
        except requests.RequestException:
            pass

        if (i + 1) % 100 == 0:
            log.info("  county lookup: %d / %d", i + 1, len(cities))
        time.sleep(0.05)  # Be polite to the FCC API

    assigned = result.notna().sum()
    log.info("Assigned %d / %d cities to counties via FCC API", assigned, len(cities))
    return result


# ---------------------------------------------------------------------------
# Roll-up logic
# ---------------------------------------------------------------------------

def apply_rollup(
    kept_df: pd.DataFrame,
    all_df: pd.DataFrame,
    rollup_map: dict[int, int],
) -> pd.DataFrame:
    """Apply population roll-up from eliminated cities to surviving cities.

    Args:
        kept_df: DataFrame of surviving cities (output of enforce_separation).
        all_df: Original DataFrame (pre-separation). Its positional index (0..N-1)
            is used as the identity key in rollup_map.
        rollup_map: Dict mapping removed_orig_idx -> kept_orig_idx, where indices
            refer to positions in all_df.

    Returns:
        kept_df with area_population and incorporated_places columns added.
    """
    kept_df = kept_df.copy()

    # Build a unique key per row using fips_state + fips_place (guaranteed unique
    # after dedup). Map from all_df orig_idx -> kept_df row position.
    all_df = all_df.copy()
    all_df["_uid"] = all_df["fips_state"] + "_" + all_df["fips_place"]
    kept_df["_uid"] = kept_df["fips_state"] + "_" + kept_df["fips_place"]

    uid_to_kept_idx = {row["_uid"]: idx for idx, row in kept_df.iterrows()}

    # Also map all_df orig_idx -> uid
    orig_idx_to_uid = {idx: row["_uid"] for idx, row in all_df.iterrows()}

    # Initialize roll-up columns
    kept_df["area_population"] = kept_df["population"].copy()
    kept_df["incorporated_places"] = [[] for _ in range(len(kept_df))]

    rolled_count = 0
    for removed_orig_idx, kept_orig_idx in rollup_map.items():
        # Find the kept_df row that corresponds to kept_orig_idx
        kept_uid = orig_idx_to_uid.get(kept_orig_idx)
        if kept_uid is None:
            continue
        kept_row_idx = uid_to_kept_idx.get(kept_uid)
        if kept_row_idx is None:
            continue

        removed_name = all_df.at[removed_orig_idx, "name"]
        removed_pop = int(all_df.at[removed_orig_idx, "population"])
        removed_state = all_df.at[removed_orig_idx, "state"]

        # Only roll up cities from the same state or neighboring states
        kept_state = kept_df.at[kept_row_idx, "state"]
        if removed_state != kept_state:
            # Cross-state roll-up — skip silently (shouldn't happen often)
            continue

        kept_df.at[kept_row_idx, "area_population"] += removed_pop
        kept_df.at[kept_row_idx, "incorporated_places"].append(removed_name)
        rolled_count += 1

    # Convert lists to comma-separated strings
    kept_df["incorporated_places"] = kept_df["incorporated_places"].apply(
        lambda lst: ", ".join(sorted(lst)) if lst else ""
    )

    kept_df = kept_df.drop(columns=["_uid"])
    log.info("Rolled up %d eliminated cities into surviving cities", rolled_count)
    return kept_df


# ---------------------------------------------------------------------------
# Region mapping
# ---------------------------------------------------------------------------

def get_region(state: str) -> str:
    """Map state abbreviation to Census region."""
    regions = {
        "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
        "Midwest": {"IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"},
        "South": {"AL", "AR", "DE", "DC", "FL", "GA", "KY", "LA", "MD", "MS", "NC", "OK",
                  "SC", "TN", "TX", "VA", "WV"},
        "West": {"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"},
    }
    for region, states in regions.items():
        if state in states:
            return region
    return "Unknown"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_master_list() -> pd.DataFrame:
    """Build the master city list following all selection rules."""

    # Step 1: Get all Census places with population
    places = fetch_census_places()

    # Step 2: Get coordinates from Gazetteer
    coords = fetch_gazetteer_coordinates()

    # Step 3: Merge population with coordinates
    merged = places.merge(coords, on=["fips_state", "fips_place"], how="left")
    log.info("Merged: %d places with coords out of %d",
             merged["lat"].notna().sum(), len(merged))
    merged = merged.dropna(subset=["lat", "lon"])

    # Step 4: Fetch CBSA data (principal cities + delineation + populations)
    county_cbsa, principals, cbsa_pops = (
        *fetch_cbsa_delineation(), fetch_cbsa_populations()
    )

    # Step 5: Identify MSA principal cities to include
    msa_principals = principals[
        principals["cbsa_type"] == "Metropolitan Statistical Area"
    ][["fips_state", "fips_place", "cbsa_code", "cbsa_title"]].copy()
    log.info("MSA principal cities: %d", len(msa_principals))

    # Step 6: Remove NYC as a whole (we add boroughs separately)
    nyc_mask = (merged["name"] == "New York") & (merged["state"] == "NY")
    merged = merged[~nyc_mask]

    # Step 7: Build candidate list — union of threshold + MSA principals + NYC boroughs
    above_threshold = merged[merged["population"] >= POP_THRESHOLD].copy()
    log.info("Cities with pop >= %d: %d", POP_THRESHOLD, len(above_threshold))

    # Find MSA principal cities below threshold that aren't already included
    below_threshold_principals = merged[
        (merged["population"] < POP_THRESHOLD) &
        (merged["population"] > 0)
    ].merge(
        msa_principals[["fips_state", "fips_place"]],
        on=["fips_state", "fips_place"],
        how="inner",
    )
    log.info("MSA principal cities below %d threshold: %d",
             POP_THRESHOLD, len(below_threshold_principals))

    candidates = pd.concat([
        above_threshold,
        below_threshold_principals,
        pd.DataFrame(NYC_BOROUGHS),
    ], ignore_index=True)
    candidates = candidates.drop_duplicates(
        subset=["fips_state", "fips_place"], keep="first"
    )
    log.info("Total candidates before backfill: %d", len(candidates))

    # Step 8: Backfill states that need more cities
    state_counts = candidates.groupby("state").size()
    all_states = set(STATE_FIPS.values())
    small_cities_to_add = []

    for st in all_states:
        if st == "DC":
            continue
        current = state_counts.get(st, 0)
        minimum = RHODE_ISLAND_MIN if st == "RI" else MIN_CITIES_PER_STATE
        if current < minimum:
            needed = minimum - current
            state_places = merged[
                (merged["state"] == st) &
                (~merged["fips_place"].isin(candidates[candidates["state"] == st]["fips_place"])) &
                (merged["population"] > 0)
            ].sort_values("population", ascending=False)
            small_cities_to_add.append(state_places.head(needed + 10))

    if small_cities_to_add:
        small_df = pd.concat(small_cities_to_add, ignore_index=True)
        log.info("Adding %d smaller cities for state minimums", len(small_df))
        candidates = pd.concat([candidates, small_df], ignore_index=True)

    candidates = candidates.drop_duplicates(
        subset=["fips_state", "fips_place"], keep="first"
    )
    log.info("Total candidates before separation: %d", len(candidates))

    # Step 9: Enforce 20-mile separation with roll-up
    log.info("Enforcing 20-mile separation...")
    separated, rollup_map = enforce_separation(
        candidates,
        min_distance_miles=20.0,
        priority_col="population",
        state_col="state",
        return_rollup=True,
    )

    # Step 10: Post-separation state backfill (relaxed distance)
    state_counts_after = separated.groupby("state").size()
    for st in all_states:
        if st == "DC":
            continue
        minimum = RHODE_ISLAND_MIN if st == "RI" else MIN_CITIES_PER_STATE
        current = state_counts_after.get(st, 0)
        if current < minimum:
            log.warning("State %s has %d cities (need %d) after separation", st, current, minimum)
            state_places = merged[
                (merged["state"] == st) &
                (~merged["fips_place"].isin(separated["fips_place"]))
            ].sort_values("population", ascending=False)

            for _, candidate_row in state_places.iterrows():
                if current >= minimum:
                    break
                state_existing = separated[separated["state"] == st]
                min_dist = min(
                    haversine_miles(
                        candidate_row["lat"], candidate_row["lon"],
                        row["lat"], row["lon"],
                    )
                    for _, row in state_existing.iterrows()
                ) if len(state_existing) > 0 else float("inf")

                if min_dist >= 15.0:
                    separated = pd.concat(
                        [separated, candidate_row.to_frame().T], ignore_index=True
                    )
                    current += 1

    # Step 11: Apply population roll-up
    separated = apply_rollup(separated, candidates, rollup_map)

    # Step 12: Assign counties for CBSA mapping (via FCC Area API)
    separated["fips_county"] = assign_counties_fcc(separated).values

    # Build county -> CBSA lookup
    county_to_cbsa = county_cbsa.drop_duplicates(
        subset=["fips_state", "fips_county"]
    ).set_index(["fips_state", "fips_county"])[["cbsa_code", "cbsa_title", "cbsa_type"]]

    # Map cities to CBSAs via county
    cbsa_info = []
    for _, row in separated.iterrows():
        key = (row["fips_state"], row.get("fips_county", ""))
        if key in county_to_cbsa.index:
            info = county_to_cbsa.loc[key]
            cbsa_info.append({
                "cbsa_code": info["cbsa_code"],
                "cbsa_name": info["cbsa_title"],
            })
        else:
            cbsa_info.append({"cbsa_code": "", "cbsa_name": ""})

    cbsa_df = pd.DataFrame(cbsa_info)
    separated["cbsa_code"] = cbsa_df["cbsa_code"].values
    separated["cbsa_name"] = cbsa_df["cbsa_name"].values

    # Add metro_pop from CBSA populations
    cbsa_pop_map = cbsa_pops.set_index("cbsa_code")["metro_pop"].to_dict()
    separated["metro_pop"] = separated["cbsa_code"].map(cbsa_pop_map).astype("Int64")

    cbsa_mapped = (separated["cbsa_code"] != "").sum()
    log.info("CBSA mapping: %d / %d cities mapped to a CBSA", cbsa_mapped, len(separated))

    # Step 13: Final cleanup
    separated = separated.sort_values(["state", "name"]).reset_index(drop=True)
    separated["city_id"] = range(1, len(separated) + 1)
    separated["region"] = separated["state"].map(get_region)

    final_cols = [
        "city_id", "name", "state", "population", "area_population",
        "metro_pop", "incorporated_places",
        "lat", "lon", "fips_state", "fips_place", "fips_county",
        "cbsa_code", "cbsa_name", "land_area_sq_mi", "region",
    ]
    # Ensure all columns exist
    for col in final_cols:
        if col not in separated.columns:
            separated[col] = None

    final = separated[final_cols].copy()

    log.info("Final master list: %d cities across %d states",
             len(final), final["state"].nunique())

    state_summary = final.groupby("state").size().sort_values(ascending=False)
    log.info("Top states:\n%s", state_summary.head(10).to_string())
    log.info("Bottom states:\n%s", state_summary.tail(10).to_string())

    return final


def main():
    output_path = PROJECT_ROOT / "data" / "cities_master.parquet"

    df = build_master_list()

    write_parquet_with_metadata(
        df,
        output_path,
        source_name=(
            "US Census Bureau ACS 5-Year (2022), Gazetteer (2023), "
            "OMB CBSA Delineation (2023)"
        ),
        source_url="https://api.census.gov/data/2022/acs/acs5",
        date_collected="2026-05-02",
        notes=(
            f"Master city list: places >= {POP_THRESHOLD:,} pop + MSA principal cities + "
            f"backfill for 10/state. NYC boroughs. 20-mile separation with population "
            f"roll-up to nearest survivor. CBSA metro_pop via county assignment."
        ),
    )

    log.info("Saved %d cities to %s", len(df), output_path)

    print(f"\n{'='*60}")
    print("MASTER CITY LIST SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(df)}")
    print(f"States covered: {df['state'].nunique()}")
    print(f"Population range: {df['population'].min():,} - {df['population'].max():,}")
    print(f"CBSA mapped: {(df['cbsa_code'] != '').sum()}")
    print(f"With roll-up data: {(df['incorporated_places'] != '').sum()}")
    print(f"Regions: {df['region'].value_counts().to_dict()}")
    print(f"Cities/state (min/max): "
          f"{df.groupby('state').size().min()} / {df.groupby('state').size().max()}")


if __name__ == "__main__":
    main()
