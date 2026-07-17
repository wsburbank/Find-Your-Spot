"""
Collect performing arts and entertainment venue data from Census County Business Patterns.

Source: US Census Bureau — County Business Patterns (CBP) 2022
URL: https://www2.census.gov/programs-surveys/cbp/datasets/2022/
API: https://api.census.gov/data/2022/cbp

NAICS codes used:
  71111  - Theater companies and dinner theaters
  71112  - Dance companies
  71113  - Musical groups and artists
  71119  - Other performing arts companies
  71131  - Promoters with facilities (concert venues, arenas)
  71211  - Museums
  71212  - Historical sites
  71219  - Nature parks and similar institutions
  712110 - Museums (6-digit for more precise counts)

The CBP API provides establishment counts by county and NAICS code. We map
counties to our cities using FIPS codes from the master city list.

Output: data/entertainment.parquet
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"

CBP_API_BASE = "https://api.census.gov/data/2022/cbp"

# NAICS codes and what they represent
NAICS_GROUPS = {
    "performing_arts": {
        "codes": ["71111", "71112", "71113", "71119"],
        "fallback_code": "7111",
        "fallback_3digit": "711",
        "sibling_codes": ["7112", "7113", "7114", "7115"],
        "label": "Performing arts companies (theater, dance, music)",
    },
    "concert_venues": {
        "codes": ["71131"],
        "fallback_code": "71131",
        "label": "Promoters of performing arts with facilities (concert venues/arenas)",
    },
    "museums_cbp": {
        "codes": ["71211", "71212", "71219"],
        "fallback_code": "7121",
        "label": "Museums, historical sites, and nature parks",
    },
}


def query_cbp_by_naics(naics_code: str, retries: int = 3) -> pd.DataFrame:
    """Query Census CBP API for establishment counts by county for a NAICS code.

    Args:
        naics_code: 5 or 6 digit NAICS code.
        retries: Number of retries on failure.

    Returns:
        DataFrame with columns: fips_state, fips_county, naics_code, establishments.
    """
    params = {
        "get": "ESTAB,NAICS2017,NAME",
        "for": "county:*",
        "NAICS2017": naics_code,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(CBP_API_BASE, params=params, timeout=60)
            if resp.status_code == 204:
                log.info("  NAICS %s: no data (204)", naics_code)
                return pd.DataFrame()
            resp.raise_for_status()

            data = resp.json()
            if len(data) <= 1:
                log.info("  NAICS %s: no records", naics_code)
                return pd.DataFrame()

            header = data[0]
            rows = data[1:]
            # Deduplicate column names (API returns NAICS2017 twice)
            seen = {}
            clean_header = []
            for col in header:
                if col in seen:
                    seen[col] += 1
                    clean_header.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    clean_header.append(col)
            df = pd.DataFrame(rows, columns=clean_header)

            # Parse columns
            df["fips_state"] = df["state"].str.zfill(2)
            df["fips_county"] = df["county"].str.zfill(3)
            df["naics_code"] = naics_code
            df["establishments"] = pd.to_numeric(df["ESTAB"], errors="coerce").fillna(0).astype(int)

            log.info("  NAICS %s: %d counties, %d total establishments",
                     naics_code, len(df), df["establishments"].sum())

            return df[["fips_state", "fips_county", "naics_code", "establishments"]]

        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                log.warning("  NAICS %s attempt %d failed: %s — retrying in %ds",
                            naics_code, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                log.error("  NAICS %s: all %d attempts failed", naics_code, retries)
                return pd.DataFrame()


def collect_all_naics() -> pd.DataFrame:
    """Collect CBP data for all NAICS codes of interest.

    Uses a fallback strategy: queries 5-digit codes first, then queries the
    broader parent code (4-digit) for each group. For counties where the sum
    of 5-digit codes is zero but the parent code has data, we use the parent
    code count instead. This recovers data suppressed by Census disclosure
    avoidance rules in small counties.
    """
    all_records = []
    group_5digit: dict[str, pd.DataFrame] = {}

    for group_name, group_info in NAICS_GROUPS.items():
        log.info("Collecting %s (%s)...", group_name, group_info["label"])
        group_dfs = []
        for code in group_info["codes"]:
            df = query_cbp_by_naics(code)
            if not df.empty:
                df["group"] = group_name
                group_dfs.append(df)
            time.sleep(1)

        if group_dfs:
            combined = pd.concat(group_dfs, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["fips_state", "fips_county", "naics_code",
                                             "establishments", "group"])
        group_5digit[group_name] = combined

        # Now query the fallback (parent) code
        fallback = group_info.get("fallback_code")
        if fallback and fallback not in group_info["codes"]:
            log.info("  Querying fallback %s for suppression recovery...", fallback)
            fb_df = query_cbp_by_naics(fallback)
            time.sleep(1)

            if not fb_df.empty:
                # Sum 5-digit counts per county
                if not combined.empty:
                    sum_5d = (combined.groupby(["fips_state", "fips_county"])
                              ["establishments"].sum().reset_index()
                              .rename(columns={"establishments": "sum_5digit"}))
                    fb_merged = fb_df.merge(sum_5d, on=["fips_state", "fips_county"], how="left")
                    fb_merged["sum_5digit"] = fb_merged["sum_5digit"].fillna(0)
                else:
                    fb_merged = fb_df.copy()
                    fb_merged["sum_5digit"] = 0

                # Counties where 5-digit data is zero but parent has data = suppressed
                suppressed = fb_merged[
                    (fb_merged["sum_5digit"] == 0) & (fb_merged["establishments"] > 0)
                ].copy()

                if len(suppressed) > 0:
                    log.info("  Recovered %d counties with %d suppressed establishments via %s",
                             len(suppressed), suppressed["establishments"].sum(), fallback)
                    recovered = pd.DataFrame({
                        "fips_state": suppressed["fips_state"].values,
                        "fips_county": suppressed["fips_county"].values,
                        "naics_code": fallback,
                        "establishments": suppressed["establishments"].values,
                        "group": group_name,
                    })
                    group_dfs.append(recovered)

        # 3-digit fallback: for counties where even the 4-digit parent is suppressed,
        # use the 3-digit code minus known sibling groups to estimate.
        fb3 = group_info.get("fallback_3digit")
        sibling_codes = group_info.get("sibling_codes", [])
        if fb3 and sibling_codes:
            log.info("  Querying 3-digit fallback %s for deeper suppression recovery...", fb3)
            fb3_df = query_cbp_by_naics(fb3)
            time.sleep(1)

            if not fb3_df.empty:
                # Sum all currently known data for this group per county
                all_current = pd.concat(group_dfs, ignore_index=True) if group_dfs else pd.DataFrame()
                if not all_current.empty:
                    known_sum = (all_current.groupby(["fips_state", "fips_county"])
                                 ["establishments"].sum().reset_index()
                                 .rename(columns={"establishments": "known"}))
                else:
                    known_sum = pd.DataFrame(columns=["fips_state", "fips_county", "known"])

                # Query sibling codes to subtract from the 3-digit total
                sibling_total = []
                for sib in sibling_codes:
                    sib_df = query_cbp_by_naics(sib)
                    if not sib_df.empty:
                        sibling_total.append(sib_df)
                    time.sleep(1)

                if sibling_total:
                    sib_sum = (pd.concat(sibling_total, ignore_index=True)
                               .groupby(["fips_state", "fips_county"])
                               ["establishments"].sum().reset_index()
                               .rename(columns={"establishments": "sibling"}))
                else:
                    sib_sum = pd.DataFrame(columns=["fips_state", "fips_county", "sibling"])

                fb3_merged = fb3_df.merge(known_sum, on=["fips_state", "fips_county"], how="left")
                fb3_merged = fb3_merged.merge(sib_sum, on=["fips_state", "fips_county"], how="left")
                fb3_merged["known"] = fb3_merged["known"].fillna(0)
                fb3_merged["sibling"] = fb3_merged["sibling"].fillna(0)

                # Estimate = 3digit_total - siblings - already_known
                fb3_merged["estimated"] = (fb3_merged["establishments"]
                                           - fb3_merged["sibling"]
                                           - fb3_merged["known"]).clip(lower=0).astype(int)

                deeply_suppressed = fb3_merged[
                    (fb3_merged["known"] == 0) & (fb3_merged["estimated"] > 0)
                ].copy()

                if len(deeply_suppressed) > 0:
                    log.info("  Recovered %d more counties with %d establishments via 3-digit %s",
                             len(deeply_suppressed), deeply_suppressed["estimated"].sum(), fb3)
                    recovered = pd.DataFrame({
                        "fips_state": deeply_suppressed["fips_state"].values,
                        "fips_county": deeply_suppressed["fips_county"].values,
                        "naics_code": fb3,
                        "establishments": deeply_suppressed["estimated"].values,
                        "group": group_name,
                    })
                    group_dfs.append(recovered)

        if group_dfs:
            all_records.extend(group_dfs)

    if not all_records:
        log.error("No data collected from CBP API")
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


def aggregate_to_cities(cbp_data: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CBP establishment counts from counties to cities.

    Each city has a fips_county. We sum the establishments in that county across
    relevant NAICS groups. Note: if multiple cities share the same county, they
    share the county's establishment count (this is a limitation of county-level data,
    but it's the best granularity CBP provides).
    """
    # Sum establishments per county per group
    county_agg = (cbp_data.groupby(["fips_state", "fips_county", "group"])
                  ["establishments"].sum()
                  .reset_index())

    # Pivot so each group is a column
    county_pivot = county_agg.pivot_table(
        index=["fips_state", "fips_county"],
        columns="group",
        values="establishments",
        fill_value=0,
    ).reset_index()

    # Flatten column names
    county_pivot.columns = [
        c[1] if c[1] else c[0] for c in county_pivot.columns
    ] if isinstance(county_pivot.columns, pd.MultiIndex) else county_pivot.columns

    # Ensure all expected columns exist
    for group_name in NAICS_GROUPS:
        if group_name not in county_pivot.columns:
            county_pivot[group_name] = 0

    # Zero-pad FIPS codes in cities for matching
    cities_copy = cities[["city_id", "fips_state", "fips_county", "population"]].copy()
    cities_copy["fips_state"] = cities_copy["fips_state"].astype(str).str.zfill(2)
    cities_copy["fips_county"] = cities_copy["fips_county"].astype(str).str.zfill(3)

    # Merge county data to cities
    result = cities_copy.merge(
        county_pivot,
        on=["fips_state", "fips_county"],
        how="left",
    )

    # For cities sharing a county, apportion by population share
    county_pop = result.groupby(["fips_state", "fips_county"])["population"].transform("sum")
    pop_share = result["population"] / county_pop.clip(lower=1)

    for group_name in NAICS_GROUPS:
        col = group_name
        if col in result.columns:
            # Apportion by population share, round to nearest int
            result[col] = (result[col].fillna(0) * pop_share).round(0).astype(int)

    # Rename columns to final names
    result = result.rename(columns={
        "performing_arts": "performing_arts_venues",
        "concert_venues": "concert_venue_count",
        "museums_cbp": "museums_cbp_count",
    })

    # Keep only city_id and the metric columns
    metric_cols = ["performing_arts_venues", "concert_venue_count", "museums_cbp_count"]
    return result[["city_id"] + [c for c in metric_cols if c in result.columns]]


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Collect from Census CBP API
    cbp_data = collect_all_naics()
    if cbp_data.empty:
        log.error("No CBP data collected — aborting")
        sys.exit(1)

    log.info("\nTotal CBP records: %d across %d county-NAICS combinations",
             len(cbp_data), cbp_data.groupby(["fips_state", "fips_county"]).ngroups)

    # Aggregate to cities
    result = aggregate_to_cities(cbp_data, cities)

    # Report
    log.info("\nEntertainment data coverage:")
    for col in result.columns:
        if col == "city_id":
            continue
        has_data = (result[col] > 0).sum()
        total = result[col].sum()
        log.info("  %s: %d/%d cities have data (total: %d)",
                 col, has_data, len(result), total)

    log.info("\nTop 10 cities by performing arts venues:")
    top = result.nlargest(10, "performing_arts_venues").merge(
        cities[["city_id", "name", "state"]], on="city_id"
    )
    for _, row in top.iterrows():
        log.info("  %s, %s: %d performing arts, %d concert venues",
                 row["name"], row["state"],
                 row["performing_arts_venues"], row["concert_venue_count"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "entertainment.parquet",
        source_name="US Census Bureau — County Business Patterns 2022",
        source_url="https://api.census.gov/data/2022/cbp",
        date_collected="2022",
        notes=(
            "Entertainment establishment counts from Census CBP 2022 by NAICS code. "
            "Performing arts: NAICS 71111-71119. Concert venues: NAICS 71131. "
            "Museums/historical: NAICS 71211-71219. County-level data apportioned "
            "to cities by population share within each county."
        ),
    )
    print(f"\nSaved: data/entertainment.parquet ({len(result)} rows)")


if __name__ == "__main__":
    main()
