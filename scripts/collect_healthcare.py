"""
Collect healthcare access data from HRSA and County Health Rankings.

Sources:
  1. County Health Rankings 2024 — health outcomes, physicians per capita
     URL: https://www.countyhealthrankings.org/
  2. CDC PLACES — city-level health outcomes via Socrata API
     URL: https://chronicdata.cdc.gov/

Output: data/healthcare.parquet
"""

import logging
import sys
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

# County Health Rankings 2024 national data download
# This is an Excel file with county-level health metrics
CHR_URL = (
    "https://www.countyhealthrankings.org/sites/default/files/media/document/"
    "analytic_data2024.csv"
)

# CDC PLACES dataset ID on Socrata (county-level data, 2023 release)
CDC_PLACES_COUNTY_URL = (
    "https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD"
)


def download_county_health_rankings() -> pd.DataFrame:
    """Download County Health Rankings analytic data."""
    cache_file = CACHE_DIR / "chr_analytic_2024.csv"

    if cache_file.exists():
        log.info("Using cached CHR data from %s", cache_file)
        return pd.read_csv(cache_file, low_memory=False)

    log.info("Downloading County Health Rankings 2024...")
    log.info("  URL: %s", CHR_URL)

    resp = requests.get(CHR_URL, timeout=120)

    if resp.status_code != 200:
        # Try alternative URL format
        alt_url = (
            "https://www.countyhealthrankings.org/sites/default/files/media/document/"
            "analytic_data2024_0.csv"
        )
        log.info("  Primary URL failed (%d), trying alternative...", resp.status_code)
        resp = requests.get(alt_url, timeout=120)

    if resp.status_code != 200:
        log.warning("  CHR download failed with status %d", resp.status_code)
        return pd.DataFrame()

    df = pd.read_csv(BytesIO(resp.content), low_memory=False, encoding="latin-1")
    df.to_csv(cache_file, index=False)
    log.info("  Downloaded %d rows, cached to %s", len(df), cache_file)
    return df


def download_cdc_places() -> pd.DataFrame:
    """Download CDC PLACES county-level health outcomes."""
    cache_file = CACHE_DIR / "cdc_places_county.csv"

    if cache_file.exists():
        log.info("Using cached CDC PLACES data from %s", cache_file)
        return pd.read_csv(cache_file, low_memory=False)

    log.info("Downloading CDC PLACES county data...")
    log.info("  URL: %s", CDC_PLACES_COUNTY_URL)

    resp = requests.get(CDC_PLACES_COUNTY_URL, timeout=300)
    if resp.status_code != 200:
        log.warning("  CDC PLACES download failed: %d", resp.status_code)
        return pd.DataFrame()

    df = pd.read_csv(BytesIO(resp.content), low_memory=False)
    log.info("  Downloaded %d rows", len(df))

    # Cache (can be large)
    df.to_csv(cache_file, index=False)
    return df


def process_chr(chr_df: pd.DataFrame) -> pd.DataFrame:
    """Extract healthcare metrics from County Health Rankings."""
    if chr_df.empty:
        return pd.DataFrame()

    # CHR has variable column names depending on year; try common ones
    # Look for FIPS, and key health metrics
    chr_df.columns = [c.strip() for c in chr_df.columns]

    # Find the FIPS column
    fips_col = None
    for candidate in ["5-digit FIPS Code", "fipscode", "FIPS", "fips", "county_fips"]:
        if candidate in chr_df.columns:
            fips_col = candidate
            break

    if fips_col is None:
        # Search for any column containing 'fips'
        fips_candidates = [c for c in chr_df.columns if "fips" in c.lower()]
        if fips_candidates:
            fips_col = fips_candidates[0]
        else:
            log.error("Cannot find FIPS column. Available: %s", chr_df.columns[:20].tolist())
            return pd.DataFrame()

    log.info("  Using FIPS column: '%s'", fips_col)

    # Look for health metric columns (CHR uses various naming conventions)
    # Common metrics and their possible column names
    metric_map = {}
    for col in chr_df.columns:
        col_lower = col.lower()
        if "primary care" in col_lower and "rate" in col_lower:
            metric_map["primary_care_physicians_rate"] = col
        elif "mental health" in col_lower and "provider" in col_lower and "rate" in col_lower:
            metric_map["mental_health_providers_rate"] = col
        elif "dentist" in col_lower and "rate" in col_lower:
            metric_map["dentists_rate"] = col
        elif "uninsured" in col_lower and ("raw" in col_lower or "value" in col_lower):
            if "uninsured_rate" not in metric_map:
                metric_map["uninsured_rate"] = col
        elif "preventable" in col_lower and "hospital" in col_lower:
            metric_map["preventable_hospital_stays"] = col

    log.info("  Found CHR metrics: %s", list(metric_map.keys()))

    if not metric_map:
        # Fall back to numeric columns that might be useful
        log.warning("  No recognized metric columns found. Columns sample: %s",
                    chr_df.columns[:30].tolist())
        return pd.DataFrame()

    # Build result
    result = pd.DataFrame()
    fips_values = chr_df[fips_col].astype(str).str.zfill(5)
    result["fips_state"] = fips_values.str[:2]
    result["fips_county"] = fips_values.str[2:5]

    for out_name, src_col in metric_map.items():
        result[out_name] = pd.to_numeric(chr_df[src_col], errors="coerce").values

    # Remove header/state summary rows (county FIPS '000' = state-level)
    result = result[result["fips_county"] != "000"].copy()
    result = result.dropna(subset=["fips_state", "fips_county"])

    log.info("  Processed CHR: %d counties", len(result))
    return result


def process_cdc_places(places_df: pd.DataFrame) -> pd.DataFrame:
    """Extract health outcomes from CDC PLACES county data."""
    if places_df.empty:
        return pd.DataFrame()

    # CDC PLACES has columns like: StateAbbr, CountyName, CountyFIPS, MeasureId, Data_Value
    # It's in long format — pivot to wide

    # Check structure
    if "CountyFIPS" not in places_df.columns and "LocationID" not in places_df.columns:
        log.warning("  CDC PLACES: unexpected columns: %s", places_df.columns[:10].tolist())
        return pd.DataFrame()

    fips_col = "CountyFIPS" if "CountyFIPS" in places_df.columns else "LocationID"
    measure_col = "MeasureId" if "MeasureId" in places_df.columns else "Measure"
    value_col = "Data_Value" if "Data_Value" in places_df.columns else "data_value"

    if measure_col not in places_df.columns or value_col not in places_df.columns:
        log.warning("  CDC PLACES: missing measure/value columns")
        return pd.DataFrame()

    # Filter to measures of interest
    measures_of_interest = {
        "ACCESS2": "lack_health_insurance_pct",
        "CHECKUP": "annual_checkup_pct",
        "DENTAL": "dental_visit_pct",
        "MHLTH": "poor_mental_health_days_pct",
        "PHLTH": "poor_physical_health_days_pct",
        "OBESITY": "obesity_pct",
        "DIABETES": "diabetes_pct",
    }

    available_measures = places_df[measure_col].unique()
    log.info("  CDC PLACES measures available: %d", len(available_measures))

    filtered = places_df[places_df[measure_col].isin(measures_of_interest.keys())].copy()
    if filtered.empty:
        log.warning("  No matching measures found")
        return pd.DataFrame()

    # Pivot to wide format
    filtered["metric_name"] = filtered[measure_col].map(measures_of_interest)
    filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce")

    pivoted = filtered.pivot_table(
        index=fips_col,
        columns="metric_name",
        values=value_col,
        aggfunc="mean",
    ).reset_index()

    # Extract FIPS
    fips = pivoted[fips_col].astype(str).str.zfill(5)
    result = pd.DataFrame()
    result["fips_state"] = fips.str[:2]
    result["fips_county"] = fips.str[2:5]

    for col in measures_of_interest.values():
        if col in pivoted.columns:
            result[col] = pivoted[col].values

    log.info("  Processed CDC PLACES: %d counties, %d metrics",
             len(result), len([c for c in result.columns if c not in ("fips_state", "fips_county")]))
    return result


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Download data sources
    chr_raw = download_county_health_rankings()
    chr_data = process_chr(chr_raw)

    cdc_raw = download_cdc_places()
    cdc_data = process_cdc_places(cdc_raw)

    if chr_data.empty and cdc_data.empty:
        log.error("No healthcare data collected — aborting")
        sys.exit(1)

    # Match to cities via fips_state + fips_county
    cities_match = cities[["city_id", "fips_state", "fips_county"]].copy()
    cities_match["fips_state"] = cities_match["fips_state"].astype(str).str.zfill(2)
    cities_match["fips_county"] = cities_match["fips_county"].astype(str).str.zfill(3)

    result = cities_match.copy()

    if not chr_data.empty:
        result = result.merge(chr_data, on=["fips_state", "fips_county"], how="left")
        matched = result.drop(columns=["city_id", "fips_state", "fips_county"]).notna().any(axis=1).sum()
        log.info("CHR matched: %d/%d cities", matched, len(result))

    if not cdc_data.empty:
        result = result.merge(cdc_data, on=["fips_state", "fips_county"], how="left")
        cdc_cols = [c for c in cdc_data.columns if c not in ("fips_state", "fips_county")]
        if cdc_cols:
            matched = result[cdc_cols[0]].notna().sum()
            log.info("CDC PLACES matched: %d/%d cities", matched, len(result))

    # Drop merge keys
    result = result.drop(columns=["fips_state", "fips_county"])

    # Report coverage
    log.info("\nHealthcare coverage:")
    for col in result.columns:
        if col == "city_id":
            continue
        valid = result[col].notna().sum()
        log.info("  %s: %d/%d (%.1f%%)", col, valid, len(result), valid / len(result) * 100)

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "healthcare.parquet",
        source_name="County Health Rankings 2024 + CDC PLACES 2023",
        source_url="https://www.countyhealthrankings.org/",
        date_collected="2024",
        notes=(
            "Healthcare access and outcomes from County Health Rankings (physician/dentist/mental "
            "health provider rates per capita) and CDC PLACES (health behaviors and outcomes "
            "by county). County-level data matched to cities via FIPS codes."
        ),
    )
    log.info("\nSaved: data/healthcare.parquet (%d rows)", len(result))


if __name__ == "__main__":
    main()
