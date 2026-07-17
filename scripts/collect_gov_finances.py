"""
Collect US Census of Governments - Municipal Finance Data (FY2023).

Source: US Census Bureau, Annual Survey of State and Local Government Finances
  - https://www2.census.gov/programs-surveys/gov-finances/tables/2023/
  - Data year: Fiscal Year 2023 (released July 2025)

Supplementary source: Lincoln Institute of Land Policy, Fiscally Standardized Cities (FiSC)
  - https://www.lincolninst.edu/data/fiscally-standardized-cities/
  - Provides "fiscally standardized" debt that aggregates city + county + school district + special districts
  - Covers 218 large US cities (2023 data)

Metrics collected (per capita):
  Spending:
    - police_spending_pc: Police protection expenditure per capita
    - fire_spending_pc: Fire protection expenditure per capita
    - parks_spending_pc: Parks and recreation expenditure per capita
    - roads_spending_pc: Highways/roads expenditure per capita
    - sewerage_spending_pc: Sewerage expenditure per capita
    - health_spending_pc: Health + hospitals expenditure per capita
    - total_revenue_pc: Total general revenue per capita
  Debt:
    - debt_outstanding_pc: Long-term debt outstanding (end of FY) per capita
    - short_term_debt_pc: Short-term debt outstanding (end of FY) per capita
    - debt_interest_pc: Annual interest payments on general debt per capita
  FiSC (supplementary, ~150 cities):
    - fisc_debt_outstanding_pc: Fiscally standardized total debt per capita
      (includes overlapping county, school district, and special district debt)

Census of Governments item codes (2023 format):
  GOV_ID = 12 chars [0:12] (state 2 + type 1 + county 3 + unit 6)
  Item code = 3 chars [12:15]
  Amount = variable width [15:] in $1,000s, followed by year "2023" and flag

  Expenditure codes:
    E24 = Police protection
    E25 = Fire protection
    E61 = Parks and recreation
    E62 = Highways/roads
    E32 = Sewerage
    E44 = Hospitals
    E29 = Health (public health activities)

  Aggregate codes:
    T01 = Total revenue

  Debt codes:
    49U = Long-term debt outstanding (end of fiscal year)
    64V = Short-term debt outstanding (end of fiscal year)
    I89 = Interest on general debt

Output: data/gov_finances.parquet
"""

import io
import logging
import re
import sys
import zipfile
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
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Census of Governments 2023 Individual Unit File
GOV_FINANCE_URL = "https://www2.census.gov/programs-surveys/gov-finances/tables/2023/2023_Individual_Unit_Files.zip"

# FiSC Full Dataset
FISC_URL = "https://www.lincolninst.edu/app/uploads/2026/01/FiSC-Full-Dataset-2023-Update.xlsx"

# Expenditure item codes (amounts in $1,000s)
EXPENDITURE_ITEMS = {
    "E24": "police_spending",
    "E25": "fire_spending",
    "E61": "parks_spending",
    "E62": "roads_spending",
    "E32": "sewerage_spending",
    "E29": "health_spending",
    "E44": "hospital_spending",
}

# Aggregate items
AGGREGATE_ITEMS = {
    "T01": "total_revenue",
}

# Debt items
DEBT_ITEMS = {
    "49U": "debt_outstanding",       # Long-term debt outstanding (end of FY)
    "64V": "short_term_debt",        # Short-term debt outstanding (end of FY)
    "I89": "debt_interest",          # Interest on general debt
}


def download_census_finance() -> tuple[list[str], list[str]]:
    """Download and extract the 2023 Census of Governments individual unit file.

    Returns:
        Tuple of (finance_data_lines, pid_lines) as lists of strings.
    """
    cache_path = CACHE_DIR / "gov_finance_2023.zip"

    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        data = cache_path.read_bytes()
    else:
        log.info("Downloading Census of Governments 2023 data...")
        resp = requests.get(GOV_FINANCE_URL, timeout=120)
        if resp.status_code != 200:
            log.error("Download failed (HTTP %d)", resp.status_code)
            sys.exit(1)
        data = resp.content
        cache_path.write_bytes(data)
        log.info("Cached to %s", cache_path)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        fin_file = next(f for f in zf.namelist() if "FinEst" in f and f.endswith(".txt"))
        pid_file = next(f for f in zf.namelist() if "PID" in f and f.endswith(".txt"))

        with zf.open(fin_file) as f:
            fin_lines = [line.decode("latin-1").strip() for line in f.readlines()]
        with zf.open(pid_file) as f:
            pid_lines = [line.decode("latin-1").strip() for line in f.readlines()]

    log.info("Finance records: %d, PID records: %d", len(fin_lines), len(pid_lines))
    return fin_lines, pid_lines


def parse_pid(pid_lines: list[str]) -> pd.DataFrame:
    """Parse the PID (Population ID) file to identify municipal governments.

    2023 PID file layout (146 chars per line):
      [0:2]    = State FIPS code
      [2]      = Government type (2 = municipal)
      [3:6]    = County code
      [0:12]   = Full government ID
      [12:80]  = Government name (68 chars)
      [80:111] = County name (31 chars)
      [111:116] = FIPS place code (5 chars)
      [116:]   = Population + '23' + spaces + fiscal year end (MMDDYY)
    """
    records = []
    for line in pid_lines:
        if len(line) < 120:
            continue
        gov_type = line[2] if len(line) > 2 else ""

        # Only municipal governments (type 2)
        if gov_type != "2":
            continue

        gov_id = line[:12]
        fips_state = line[0:2]
        name = line[12:80].strip()
        fips_place = line[111:116].strip()

        # Parse population from position 116+
        # Format: spaces + digits + '23' + spaces + MMDDYY
        pop_section = line[116:]
        pop_match = re.match(r"\s*(\d+?)23\s", pop_section)
        pop = int(pop_match.group(1)) if pop_match else 0

        records.append({
            "gov_id": gov_id,
            "gov_name": name,
            "fips_state": fips_state,
            "fips_place": fips_place,
            "population_2023": pop,
        })

    df = pd.DataFrame(records)
    log.info("Parsed %d municipal governments", len(df))
    return df


def parse_finance_data(fin_lines: list[str], gov_ids: set[str]) -> pd.DataFrame:
    """Parse finance data lines for the given government IDs.

    2023 finance file format (fixed-width, ~32 chars):
      [0:12]  = Government ID
      [12:15] = Item code (3 chars)
      [15:]   = Amount (in $1,000s) + "2023" + flag char
    """
    all_items = {**EXPENDITURE_ITEMS, **AGGREGATE_ITEMS, **DEBT_ITEMS}
    target_codes = set(all_items.keys())

    records = []
    for line in fin_lines:
        if len(line) < 17:
            continue
        gov_id = line[:12]
        if gov_id not in gov_ids:
            continue
        item_code = line[12:15]
        if item_code not in target_codes:
            continue

        # Extract amount: digits between item code and "2023"
        rest = line[15:]
        amount_str = rest.split("2023")[0].strip()
        try:
            amount = int(amount_str)  # In $1,000s
        except ValueError:
            continue

        records.append({
            "gov_id": gov_id,
            "item_code": item_code,
            "amount_thousands": amount,
        })

    df = pd.DataFrame(records)
    log.info(
        "Parsed %d finance records for %d item codes",
        len(df), df["item_code"].nunique() if len(df) > 0 else 0,
    )
    return df


def load_fisc_data() -> pd.DataFrame:
    """Load FiSC (Fiscally Standardized Cities) debt data.

    Downloads the full dataset Excel file and extracts the most recent year's
    fiscally-standardized debt per capita for each city.

    Returns:
        DataFrame with columns: state_abbr, city_name, fisc_debt_outstanding_pc
    """
    cache_path = CACHE_DIR / "fisc_full_dataset.xlsx"

    if not cache_path.exists():
        log.info("Downloading FiSC dataset...")
        resp = requests.get(FISC_URL, timeout=120)
        if resp.status_code != 200:
            log.warning("FiSC download failed (HTTP %d), skipping enrichment", resp.status_code)
            return pd.DataFrame()
        cache_path.write_bytes(resp.content)
        log.info("Cached FiSC to %s", cache_path)

    import openpyxl

    log.info("Loading FiSC Excel file...")
    wb = openpyxl.load_workbook(str(cache_path), read_only=True)
    ws = wb["Data"]

    # Header row to find column indices
    headers = list(next(ws.iter_rows(max_row=1, values_only=True)))
    year_idx = headers.index("year")
    city_name_idx = headers.index("city_name")
    pop_idx = headers.index("city_population")
    # Fiscally standardized (all govts combined) debt outstanding
    debt_idx = headers.index("debt_outstanding")

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        year = row[year_idx]
        city = row[city_name_idx]
        debt = row[debt_idx]
        if year == 2023 and city and debt is not None:
            rows.append({
                "fisc_city_name": city,
                "fisc_debt_outstanding_pc": debt,
            })

    wb.close()
    df = pd.DataFrame(rows)

    # Filter out aggregate/median rows
    df = df[~df["fisc_city_name"].str.contains("Median|Mean|Total", case=False, na=False)]

    # Parse state abbreviation and city name from "ST: CityName" format
    df["state_abbr"] = df["fisc_city_name"].str.extract(r"^([A-Z]{2}):", expand=False)
    df["city_name_clean"] = df["fisc_city_name"].str.replace(r"^[A-Z]{2}:\s*", "", regex=True)

    # Normalize city names for matching
    df["city_name_clean"] = (
        df["city_name_clean"]
        .str.replace(r"\bSt\.\s", "Saint ", regex=True)
        .str.replace(r"\bFt\.\s", "Fort ", regex=True)
        .str.strip()
    )

    log.info("Loaded FiSC data for %d cities (2023)", len(df))
    return df


def match_fisc_to_cities(fisc_df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Match FiSC cities to our master city list by state + name.

    Returns:
        DataFrame with city_id and fisc_debt_outstanding_pc columns.
    """
    if fisc_df.empty:
        return pd.DataFrame(columns=["city_id", "fisc_debt_outstanding_pc"])

    # State abbreviation mapping for our master list
    from rapidfuzz import fuzz, process

    # Prepare master city list for matching
    cities_match = cities[["city_id", "name", "state"]].copy()
    cities_match["match_key"] = cities_match["state"] + ": " + cities_match["name"]

    results = []
    for _, fisc_row in fisc_df.iterrows():
        state = fisc_row["state_abbr"]
        fisc_name = fisc_row["city_name_clean"]

        # Filter to same state
        state_cities = cities_match[cities_match["state"] == state]
        if state_cities.empty:
            continue

        # Try exact match first
        exact = state_cities[state_cities["name"].str.lower() == fisc_name.lower()]
        if len(exact) == 1:
            results.append({
                "city_id": exact.iloc[0]["city_id"],
                "fisc_debt_outstanding_pc": fisc_row["fisc_debt_outstanding_pc"],
            })
            continue

        # Fuzzy match
        names = state_cities["name"].tolist()
        match = process.extractOne(fisc_name, names, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            matched_name = match[0]
            matched_city = state_cities[state_cities["name"] == matched_name].iloc[0]
            results.append({
                "city_id": matched_city["city_id"],
                "fisc_debt_outstanding_pc": fisc_row["fisc_debt_outstanding_pc"],
            })

    result_df = pd.DataFrame(results)
    log.info("FiSC matched: %d/%d cities", len(result_df), len(fisc_df))
    return result_df


def main():
    # --- Census of Governments ---
    fin_lines, pid_lines = download_census_finance()

    # Parse government IDs (municipal only)
    pid_df = parse_pid(pid_lines)

    # Parse finance data for municipal governments
    gov_id_set = set(pid_df["gov_id"].tolist())
    finance_df = parse_finance_data(fin_lines, gov_id_set)

    if finance_df.empty:
        log.error("No finance data parsed!")
        sys.exit(1)

    # Pivot: one row per government, columns for each metric
    all_items = {**EXPENDITURE_ITEMS, **AGGREGATE_ITEMS, **DEBT_ITEMS}
    finance_df["metric"] = finance_df["item_code"].map(all_items)
    pivoted = finance_df.pivot_table(
        index="gov_id",
        columns="metric",
        values="amount_thousands",
        aggfunc="sum",
    ).reset_index()

    # Merge with PID info
    merged = pid_df.merge(pivoted, on="gov_id", how="inner")
    log.info("Governments with finance data: %d", len(merged))

    # Combine health + hospital spending
    if "health_spending" in merged.columns and "hospital_spending" in merged.columns:
        merged["health_hospital_spending"] = (
            merged["health_spending"].fillna(0) + merged["hospital_spending"].fillna(0)
        )
    elif "health_spending" in merged.columns:
        merged["health_hospital_spending"] = merged["health_spending"]
    elif "hospital_spending" in merged.columns:
        merged["health_hospital_spending"] = merged["hospital_spending"]

    # Filter to those with population > 0
    merged = merged[merged["population_2023"] > 0].copy()
    log.info("Governments with population > 0: %d", len(merged))

    # Calculate per-capita values (amounts are in $1,000s, convert to dollars)
    spending_cols = [
        "police_spending", "fire_spending", "parks_spending",
        "roads_spending", "sewerage_spending", "health_hospital_spending",
        "total_revenue",
        "debt_outstanding", "short_term_debt", "debt_interest",
    ]

    for col in spending_cols:
        if col in merged.columns:
            pc_col = f"{col}_pc"
            merged[pc_col] = (merged[col] * 1000 / merged["population_2023"]).round(0)

    # Match to master city list via FIPS state + place code
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    cities["fips_state"] = cities["fips_state"].astype(str).str.zfill(2)
    cities["fips_place"] = cities["fips_place"].astype(str).str.zfill(5)
    merged["fips_state"] = merged["fips_state"].astype(str).str.zfill(2)
    merged["fips_place"] = merged["fips_place"].astype(str).str.zfill(5)

    # Join on FIPS state + place
    pc_cols = [c for c in merged.columns if c.endswith("_pc")]
    join_cols = ["fips_state", "fips_place"] + pc_cols

    result = cities[["city_id", "name", "state", "fips_state", "fips_place"]].merge(
        merged[join_cols].drop_duplicates(subset=["fips_state", "fips_place"], keep="first"),
        on=["fips_state", "fips_place"],
        how="left",
    )

    matched = result["total_revenue_pc"].notna().sum()
    log.info("Census cities matched: %d/%d (%.1f%%)",
             matched, len(result), matched / len(result) * 100)

    debt_matched = result["debt_outstanding_pc"].notna().sum()
    log.info("Census debt data matched: %d/%d (%.1f%%)",
             debt_matched, len(result), debt_matched / len(result) * 100)

    # --- FiSC Enrichment ---
    fisc_df = load_fisc_data()
    fisc_matched = match_fisc_to_cities(fisc_df, cities)

    if not fisc_matched.empty:
        result = result.merge(fisc_matched, on="city_id", how="left")
        fisc_count = result["fisc_debt_outstanding_pc"].notna().sum()
        log.info("FiSC enriched: %d/%d cities (%.1f%%)",
                 fisc_count, len(result), fisc_count / len(result) * 100)
    else:
        result["fisc_debt_outstanding_pc"] = np.nan

    # Output columns
    output_pc_cols = [c for c in result.columns if c.endswith("_pc")]
    output_cols = ["city_id"] + output_pc_cols
    output = result[output_cols].copy()

    # Write output
    write_parquet_with_metadata(
        output,
        DATA_DIR / "gov_finances.parquet",
        source_name="US Census Bureau - Annual Survey of State and Local Government Finances FY2023; Lincoln Institute FiSC Database 2023",
        source_url="https://www2.census.gov/programs-surveys/gov-finances/tables/2023/",
        notes=(
            "Municipal government finance data (FY2023). Per-capita spending on police, "
            "fire, parks, roads, sewerage, health/hospitals, plus debt outstanding, "
            "short-term debt, and interest on debt. Supplemented with FiSC fiscally-"
            "standardized debt (includes overlapping jurisdictions) for ~150-180 cities. "
            f"Census coverage: {matched}/{len(result)} cities. "
            f"Debt coverage: {debt_matched}/{len(result)} cities. "
            "Amounts in dollars per capita per year."
        ),
    )

    # Summary
    print(f"\nGovernment Finance Data Summary (FY2023):")
    print(f"  Census cities matched: {matched}/{len(result)} ({matched/len(result)*100:.1f}%)")
    print(f"  Debt data matched: {debt_matched}/{len(result)} ({debt_matched/len(result)*100:.1f}%)")
    if not fisc_matched.empty:
        fisc_count = result["fisc_debt_outstanding_pc"].notna().sum()
        print(f"  FiSC enriched: {fisc_count}/{len(result)} ({fisc_count/len(result)*100:.1f}%)")

    if matched > 0:
        print(f"\n  Per-capita metrics (median, range):")
        for col in output_pc_cols:
            valid = output[col].dropna()
            if len(valid) > 0:
                print(f"    {col:35s}: median ${valid.median():>10,.0f}, "
                      f"range ${valid.min():>10,.0f} - ${valid.max():>12,.0f}")


if __name__ == "__main__":
    main()
