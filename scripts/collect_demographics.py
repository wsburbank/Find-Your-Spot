"""
Phase 2g: Collect demographic data from US Census ACS 5-Year (2022).

Sources:
  - B03002: Hispanic/Latino Origin by Race → diversity index
  - S0101:  Age and Sex → median age, % under 18, % over 65
  - S1501:  Educational Attainment → % bachelor's+, % graduate+
  - S2701:  Health Insurance Coverage → % uninsured
  - DP03:   Economic Characteristics → unemployment, poverty, industry breakdown

Output: data/demographics.parquet
"""

import logging
import math
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

# Census ACS 5-Year (2022)
ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"
ACS_SUBJECT = f"{ACS_BASE}/subject"
ACS_PROFILE = f"{ACS_BASE}/profile"

# ---------------------------------------------------------------------------
# Race/Ethnicity (B03002) — for diversity index
# ---------------------------------------------------------------------------
# B03002_001E = Total
# B03002_003E = White alone (not Hispanic/Latino)
# B03002_004E = Black/African American alone (not Hispanic)
# B03002_005E = American Indian/Alaska Native alone (not Hispanic)
# B03002_006E = Asian alone (not Hispanic)
# B03002_007E = Native Hawaiian/Pacific Islander alone (not Hispanic)
# B03002_008E = Some other race alone (not Hispanic)
# B03002_009E = Two or more races (not Hispanic)
# B03002_012E = Hispanic or Latino (any race)
RACE_VARS = (
    "B03002_001E,B03002_003E,B03002_004E,B03002_005E,"
    "B03002_006E,B03002_007E,B03002_008E,B03002_009E,B03002_012E"
)
RACE_GROUPS = [
    "B03002_003E",  # White
    "B03002_004E",  # Black
    "B03002_005E",  # Native American
    "B03002_006E",  # Asian
    "B03002_007E",  # Pacific Islander
    "B03002_008E",  # Other
    "B03002_009E",  # Two+
    "B03002_012E",  # Hispanic/Latino
]

# ---------------------------------------------------------------------------
# Age (S0101) — subject table
# ---------------------------------------------------------------------------
# S0101_C01_032E = Median age (total)
# S0101_C02_022E = % under 18
# S0101_C02_030E = % 65 and over
AGE_VARS = "S0101_C01_032E,S0101_C02_022E,S0101_C02_030E"

# ---------------------------------------------------------------------------
# Education (S1501) — subject table
# ---------------------------------------------------------------------------
# S1501_C02_015E = % bachelor's degree or higher (pop 25+)
# S1501_C02_014E = % high school graduate or higher (pop 25+)
EDUCATION_VARS = "S1501_C02_015E,S1501_C02_014E"

# ---------------------------------------------------------------------------
# Health Insurance (S2701) — subject table
# ---------------------------------------------------------------------------
# S2701_C05_001E = % uninsured (civilian noninstitutionalized)
HEALTH_VARS = "S2701_C05_001E"

# ---------------------------------------------------------------------------
# Economic (DP03) — profile table
# ---------------------------------------------------------------------------
# DP03_0005PE  = % unemployed (of civilian labor force)
# DP03_0128PE  = % below poverty level
# Industry breakdown (% of employed civilians 16+):
# DP03_0033PE  = Agriculture, forestry, fishing, hunting, mining
# DP03_0034PE  = Construction
# DP03_0035PE  = Manufacturing
# DP03_0036PE  = Wholesale trade
# DP03_0037PE  = Retail trade
# DP03_0038PE  = Transportation, warehousing, utilities
# DP03_0039PE  = Information
# DP03_0040PE  = Finance, insurance, real estate
# DP03_0041PE  = Professional, scientific, management, admin
# DP03_0042PE  = Educational services, health care, social assistance
# DP03_0043PE  = Arts, entertainment, recreation, accommodation, food
# DP03_0044PE  = Other services
# DP03_0045PE  = Public administration
ECON_VARS = (
    "DP03_0005PE,DP03_0128PE,"
    "DP03_0033PE,DP03_0034PE,DP03_0035PE,DP03_0036PE,DP03_0037PE,"
    "DP03_0038PE,DP03_0039PE,DP03_0040PE,DP03_0041PE,DP03_0042PE,"
    "DP03_0043PE,DP03_0044PE,DP03_0045PE"
)

INDUSTRY_FIELDS = {
    "DP03_0033PE": "Agriculture/Mining",
    "DP03_0034PE": "Construction",
    "DP03_0035PE": "Manufacturing",
    "DP03_0036PE": "Wholesale Trade",
    "DP03_0037PE": "Retail Trade",
    "DP03_0038PE": "Transportation/Utilities",
    "DP03_0039PE": "Information/Tech",
    "DP03_0040PE": "Finance/Real Estate",
    "DP03_0041PE": "Professional/Scientific",
    "DP03_0042PE": "Education/Healthcare",
    "DP03_0043PE": "Arts/Entertainment/Food",
    "DP03_0044PE": "Other Services",
    "DP03_0045PE": "Public Administration",
}


def _fetch_census_by_state(
    base_url: str,
    variables: str,
    state_fips_list: list[str],
) -> pd.DataFrame:
    """Fetch Census data state-by-state to avoid timeouts.

    Returns a DataFrame with columns for each variable plus 'state' and 'place'.
    """
    all_rows = []
    headers = None

    for fips in state_fips_list:
        params = {
            "get": f"NAME,{variables}",
            "for": "place:*",
            "in": f"state:{fips}",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("Failed to fetch state %s from %s, skipping",
                        fips, base_url.split("/")[-1])
            continue

        if headers is None:
            headers = data[0]
        all_rows.extend(data[1:])
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=headers)
    return df


def _get_state_fips() -> list[str]:
    """Get list of all state FIPS codes from Census API."""
    resp = requests.get(f"{ACS_BASE}?get=NAME&for=state:*", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [row[-1] for row in data[1:]]


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """Convert columns to numeric, replacing Census sentinel values with NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Census uses negative values as sentinels for suppressed data
            df.loc[df[col] < 0, col] = np.nan


def compute_diversity_index(df: pd.DataFrame) -> pd.Series:
    """Compute Simpson's Diversity Index from race/ethnicity counts.

    Returns values 0-1 where 0 = no diversity (100% one group)
    and 1 = maximum diversity (even split across all groups).
    The index is: 1 - sum(p_i^2) for each group proportion p_i.
    """
    total = df["B03002_001E"].values.astype(float)
    total = np.where(total == 0, 1, total)  # avoid division by zero

    diversity = np.ones(len(df))
    for col in RACE_GROUPS:
        p = df[col].values.astype(float) / total
        diversity -= p ** 2

    return pd.Series(np.clip(diversity, 0, 1), index=df.index)


def _top_industries(row: pd.Series) -> str:
    """Return comma-separated top 3 industries for a city."""
    industry_vals = {}
    for var, name in INDUSTRY_FIELDS.items():
        val = row.get(var)
        if pd.notna(val) and val > 0:
            industry_vals[name] = val

    if not industry_vals:
        return ""

    sorted_industries = sorted(industry_vals.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(name for name, _ in sorted_industries[:3])


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    state_fips = _get_state_fips()
    log.info("Fetching data for %d states", len(state_fips))

    # --- Fetch race/ethnicity data (B03002, detailed table) ---
    log.info("Fetching race/ethnicity data (B03002)...")
    race_df = _fetch_census_by_state(ACS_BASE, RACE_VARS, state_fips)
    log.info("Race/ethnicity: %d places", len(race_df))

    # --- Fetch age data (S0101, subject table) ---
    log.info("Fetching age data (S0101)...")
    age_df = _fetch_census_by_state(ACS_SUBJECT, AGE_VARS, state_fips)
    log.info("Age: %d places", len(age_df))

    # --- Fetch education data (S1501, subject table) ---
    log.info("Fetching education data (S1501)...")
    edu_df = _fetch_census_by_state(ACS_SUBJECT, EDUCATION_VARS, state_fips)
    log.info("Education: %d places", len(edu_df))

    # --- Fetch health insurance data (S2701, subject table) ---
    log.info("Fetching health insurance data (S2701)...")
    health_df = _fetch_census_by_state(ACS_SUBJECT, HEALTH_VARS, state_fips)
    log.info("Health insurance: %d places", len(health_df))

    # --- Fetch economic data (DP03, profile table) ---
    log.info("Fetching economic data (DP03)...")
    econ_df = _fetch_census_by_state(ACS_PROFILE, ECON_VARS, state_fips)
    log.info("Economic: %d places", len(econ_df))

    # --- Merge all Census data ---
    # Start with race data and merge others on (state, place)
    _to_numeric(race_df, ["B03002_001E"] + RACE_GROUPS)
    race_df["diversity_index"] = compute_diversity_index(race_df)

    # Compute % for each major race/ethnicity group
    total = race_df["B03002_001E"].astype(float).replace(0, np.nan)
    race_df["pct_white"] = (race_df["B03002_003E"].astype(float) / total * 100).round(1)
    race_df["pct_black"] = (race_df["B03002_004E"].astype(float) / total * 100).round(1)
    race_df["pct_asian"] = (race_df["B03002_006E"].astype(float) / total * 100).round(1)
    race_df["pct_hispanic"] = (race_df["B03002_012E"].astype(float) / total * 100).round(1)

    # Merge on (state, place) FIPS
    merge_key = ["state", "place"]
    census = race_df[merge_key + [
        "diversity_index", "pct_white", "pct_black", "pct_asian", "pct_hispanic",
    ]].copy()

    if len(age_df):
        _to_numeric(age_df, ["S0101_C01_032E", "S0101_C02_022E", "S0101_C02_030E"])
        age_df = age_df.rename(columns={
            "S0101_C01_032E": "median_age",
            "S0101_C02_022E": "pct_under_18",
            "S0101_C02_030E": "pct_over_65",
        })
        census = census.merge(age_df[merge_key + ["median_age", "pct_under_18", "pct_over_65"]],
                              on=merge_key, how="left")

    if len(edu_df):
        _to_numeric(edu_df, ["S1501_C02_015E", "S1501_C02_014E"])
        edu_df = edu_df.rename(columns={
            "S1501_C02_015E": "pct_bachelors_plus",
            "S1501_C02_014E": "pct_high_school_plus",
        })
        census = census.merge(edu_df[merge_key + ["pct_bachelors_plus", "pct_high_school_plus"]],
                              on=merge_key, how="left")

    if len(health_df):
        _to_numeric(health_df, ["S2701_C05_001E"])
        health_df = health_df.rename(columns={"S2701_C05_001E": "pct_uninsured"})
        census = census.merge(health_df[merge_key + ["pct_uninsured"]],
                              on=merge_key, how="left")

    if len(econ_df):
        econ_cols = ["DP03_0005PE", "DP03_0128PE"] + list(INDUSTRY_FIELDS.keys())
        _to_numeric(econ_df, econ_cols)
        econ_df = econ_df.rename(columns={
            "DP03_0005PE": "unemployment_rate",
            "DP03_0128PE": "poverty_rate",
        })

        # Compute top 3 industries per place
        econ_df["major_industries"] = econ_df.apply(_top_industries, axis=1)

        keep_cols = merge_key + ["unemployment_rate", "poverty_rate", "major_industries"]
        census = census.merge(econ_df[keep_cols], on=merge_key, how="left")

    # --- Match to our city list ---
    # Pad FIPS codes to match cities_master format
    census["state"] = census["state"].astype(str).str.zfill(2)
    census["place"] = census["place"].astype(str).str.zfill(5)

    cities["fips_state"] = cities["fips_state"].astype(str).str.zfill(2)
    cities["fips_place"] = cities["fips_place"].astype(str).str.zfill(5)

    result = cities[["city_id", "name", "state", "fips_state", "fips_place"]].merge(
        census,
        left_on=["fips_state", "fips_place"],
        right_on=["state", "place"],
        how="left",
        suffixes=("", "_census"),
    )

    # Drop merge keys, keep our columns
    result = result.drop(columns=["state_census", "place", "fips_state", "fips_place"],
                         errors="ignore")

    matched = result["diversity_index"].notna().sum()
    log.info("Matched %d/%d cities (%.1f%%)", matched, len(cities), matched / len(cities) * 100)

    # Save
    output_path = DATA_DIR / "demographics.parquet"
    write_parquet_with_metadata(
        result,
        output_path,
        source_name="US Census Bureau ACS 5-Year Estimates (2022)",
        source_url="https://api.census.gov/data/2022/acs/acs5",
        notes=(
            "Demographics from Census ACS 2022: diversity index (Simpson's, from B03002), "
            "race/ethnicity percentages, median age (S0101), educational attainment (S1501), "
            "health insurance coverage (S2701), unemployment/poverty/industry (DP03). "
            f"Matched {matched}/{len(cities)} cities by FIPS codes."
        ),
    )

    # Print summary
    print(f"\n{'='*60}")
    print("DEMOGRAPHICS DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Cities matched: {matched}/{len(cities)}")

    num_cols = result.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if col == "city_id":
            continue
        vals = result[col].dropna()
        if len(vals):
            print(f"  {col:25s}: min={vals.min():7.1f}  median={vals.median():7.1f}  "
                  f"max={vals.max():7.1f}  missing={result[col].isna().sum()}")

    # Sample cities
    print(f"\nSample cities:")
    samples = ["New York", "Los Angeles", "Houston", "Boulder", "Boise City", "Asheville"]
    for name in samples:
        rows = result[result["name"].str.contains(name, case=False)]
        if len(rows):
            r = rows.iloc[0]
            print(f"  {r['name']:25s} {r['state']}: diversity={r.get('diversity_index',0):.2f}  "
                  f"age={r.get('median_age',0):.0f}  bachelors={r.get('pct_bachelors_plus',0):.0f}%  "
                  f"unemp={r.get('unemployment_rate',0):.1f}%  uninsured={r.get('pct_uninsured',0):.1f}%  "
                  f"industries={r.get('major_industries','')}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
