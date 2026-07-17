"""
Collect natural disaster risk data from FEMA National Risk Index (NRI).

Source: FEMA National Risk Index — County-level composite risk scores
URL: https://hazards.fema.gov/nri/data-resources
Data: Risk scores for 18 natural hazard types at county level

The NRI provides composite Risk Index scores (0-100) and individual hazard
risk scores. We match counties to our cities via FIPS codes.

Output: data/disaster_risk.parquet
"""

import logging
import sys
import time
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

# FEMA NRI data download — county-level CSV
# The NRI provides a direct download link for county-level data
NRI_DOWNLOAD_URL = "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties/NRI_Table_Counties.zip"

# Fallback: FEMA OpenFEMA API for disaster declarations
FEMA_API_BASE = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"


def download_nri() -> pd.DataFrame:
    """Download FEMA NRI county-level data."""
    cache_file = CACHE_DIR / "NRI_Table_Counties.csv"

    if cache_file.exists():
        log.info("Using cached NRI data from %s", cache_file)
        return pd.read_csv(cache_file, low_memory=False)

    log.info("Downloading FEMA NRI county data...")
    log.info("  URL: %s", NRI_DOWNLOAD_URL)

    resp = requests.get(NRI_DOWNLOAD_URL, timeout=300)
    resp.raise_for_status()
    log.info("  Downloaded %d MB", len(resp.content) // (1024 * 1024))

    # Extract CSV from ZIP
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV found in NRI ZIP file")
        log.info("  Extracting: %s", csv_files[0])
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    # Cache for future runs
    df.to_csv(cache_file, index=False)
    log.info("  Cached to %s (%d rows)", cache_file, len(df))
    return df


def fetch_disaster_declarations() -> pd.DataFrame:
    """Fetch historical disaster declaration counts per county from FEMA API.

    Counts major disaster declarations per county over past 20 years.
    """
    log.info("Fetching FEMA disaster declarations (past 20 years)...")
    all_records = []
    skip = 0
    batch_size = 1000
    cutoff_year = 2004

    while True:
        params = {
            "$filter": f"fyDeclared ge {cutoff_year}",
            "$select": "fipsStateCode,fipsCountyCode,incidentType,fyDeclared,declarationType",
            "$skip": skip,
            "$top": batch_size,
            "$orderby": "id",
        }
        try:
            resp = requests.get(FEMA_API_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("DisasterDeclarationsSummaries", [])
            if not records:
                break
            all_records.extend(records)
            skip += batch_size
            if skip % 5000 == 0:
                log.info("  Fetched %d records so far...", len(all_records))
            time.sleep(0.5)
        except requests.RequestException as e:
            log.warning("  FEMA API error at skip=%d: %s", skip, e)
            break

    if not all_records:
        log.warning("No disaster declarations retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    log.info("  Total declarations fetched: %d", len(df))
    return df


def process_nri(nri_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract key risk scores from NRI data at county level."""
    # NRI columns of interest (all are RISK_SCORE fields, 0-100 scale)
    # Column naming: {HAZARD}_RISKS = Risk Score for that hazard
    risk_cols = {
        "RISK_SCORE": "overall_risk_score",
        "RISK_RATNG": "overall_risk_rating",
        # Individual hazards — Risk Index scores
        "ERQK_RISKS": "earthquake_risk",
        "HRCN_RISKS": "hurricane_risk",
        "TRND_RISKS": "tornado_risk",
        "RFLD_RISKS": "riverine_flood_risk",
        "CFLD_RISKS": "coastal_flood_risk",
        "WFIR_RISKS": "wildfire_risk",
        "HWAV_RISKS": "heat_wave_risk",
        "CWAV_RISKS": "cold_wave_risk",
        "DRGT_RISKS": "drought_risk",
        "HAIL_RISKS": "hail_risk",
        "SWND_RISKS": "strong_wind_risk",
        "WNTW_RISKS": "winter_weather_risk",
        "LTNG_RISKS": "lightning_risk",
        "VLCN_RISKS": "volcano_risk",
        "TSUN_RISKS": "tsunami_risk",
    }

    # Check which columns exist
    available = {k: v for k, v in risk_cols.items() if k in nri_raw.columns}
    missing = set(risk_cols.keys()) - set(available.keys())
    if missing:
        log.warning("NRI columns not found: %s", missing)

    # Extract FIPS — NRI uses STCOFIPS (5-digit state+county FIPS)
    fips_col = None
    for candidate in ["STCOFIPS", "STATE_FIPS", "STATEFIPS"]:
        if candidate in nri_raw.columns:
            fips_col = candidate
            break

    if fips_col is None:
        # Try STATEFIPS + COUNTYFIPS combo
        if "STATEFIPS" in nri_raw.columns and "COUNTYFIPS" in nri_raw.columns:
            nri_raw["_stcofips"] = (
                nri_raw["STATEFIPS"].astype(str).str.zfill(2) +
                nri_raw["COUNTYFIPS"].astype(str).str.zfill(3)
            )
            fips_col = "_stcofips"
        else:
            log.error("Cannot find FIPS columns in NRI data. Available: %s",
                      [c for c in nri_raw.columns if "fips" in c.lower() or "state" in c.lower()])
            return pd.DataFrame()

    result = pd.DataFrame()
    stcofips = nri_raw[fips_col].astype(str).str.zfill(5)
    result["fips_state"] = stcofips.str[:2]
    result["fips_county"] = stcofips.str[2:5]

    for nri_col, out_col in available.items():
        if nri_col == "RISK_RATNG":
            result[out_col] = nri_raw[nri_col].values
        else:
            result[out_col] = pd.to_numeric(nri_raw[nri_col], errors="coerce").values

    log.info("Processed NRI: %d counties, %d risk metrics", len(result), len(available))
    return result


def process_declarations(decl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate disaster declarations to county-level counts."""
    if decl_df.empty:
        return pd.DataFrame()

    # Clean FIPS
    decl_df = decl_df.dropna(subset=["fipsStateCode", "fipsCountyCode"])
    decl_df["fips_state"] = decl_df["fipsStateCode"].astype(str).str.zfill(2)
    decl_df["fips_county"] = decl_df["fipsCountyCode"].astype(str).str.zfill(3)

    # Filter to major disasters only
    major = decl_df[decl_df["declarationType"] == "DR"]

    # Count declarations per county
    county_counts = (major.groupby(["fips_state", "fips_county"])
                     .size().reset_index(name="disaster_declarations_20yr"))

    # Count by type
    type_counts = (major.groupby(["fips_state", "fips_county", "incidentType"])
                   .size().reset_index(name="count"))

    # Pivot top incident types
    type_pivot = type_counts.pivot_table(
        index=["fips_state", "fips_county"],
        columns="incidentType",
        values="count",
        fill_value=0,
    ).reset_index()

    if isinstance(type_pivot.columns, pd.MultiIndex):
        type_pivot.columns = [c[1] if c[1] else c[0] for c in type_pivot.columns]

    result = county_counts.merge(type_pivot, on=["fips_state", "fips_county"], how="left")
    log.info("Declaration counts: %d counties", len(result))
    return result


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Download NRI data
    try:
        nri_raw = download_nri()
        nri = process_nri(nri_raw)
    except Exception as e:
        log.error("Failed to download/process NRI: %s", e)
        nri = pd.DataFrame()

    # Fetch disaster declarations
    try:
        decl_raw = fetch_disaster_declarations()
        declarations = process_declarations(decl_raw)
    except Exception as e:
        log.error("Failed to fetch disaster declarations: %s", e)
        declarations = pd.DataFrame()

    if nri.empty and declarations.empty:
        log.error("No disaster risk data collected — aborting")
        sys.exit(1)

    # Match to cities via fips_state + fips_county
    cities_match = cities[["city_id", "fips_state", "fips_county", "name", "state"]].copy()
    cities_match["fips_state"] = cities_match["fips_state"].astype(str).str.zfill(2)
    cities_match["fips_county"] = cities_match["fips_county"].astype(str).str.zfill(3)

    result = cities_match[["city_id", "fips_state", "fips_county"]].copy()

    # Merge NRI scores
    if not nri.empty:
        result = result.merge(nri, on=["fips_state", "fips_county"], how="left")
        matched = result["overall_risk_score"].notna().sum()
        log.info("NRI matched: %d/%d cities (%.1f%%)",
                 matched, len(result), matched / len(result) * 100)

    # Merge declaration counts
    if not declarations.empty:
        result = result.merge(
            declarations[["fips_state", "fips_county", "disaster_declarations_20yr"]],
            on=["fips_state", "fips_county"],
            how="left",
        )
        result["disaster_declarations_20yr"] = result["disaster_declarations_20yr"].fillna(0).astype(int)

    # Drop merge keys from output
    result = result.drop(columns=["fips_state", "fips_county"])

    # Report coverage
    log.info("\nDisaster risk coverage:")
    for col in result.columns:
        if col == "city_id":
            continue
        if result[col].dtype == "object":
            valid = result[col].notna().sum()
        else:
            valid = result[col].notna().sum()
        log.info("  %s: %d/%d cities (%.1f%%)", col, valid, len(result), valid / len(result) * 100)

    if "overall_risk_score" in result.columns:
        log.info("\nHighest risk cities:")
        top = result.nlargest(10, "overall_risk_score").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in top.iterrows():
            log.info("  %s, %s: risk=%.1f, rating=%s",
                     row["name"], row["state"],
                     row.get("overall_risk_score", 0),
                     row.get("overall_risk_rating", "N/A"))

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "disaster_risk.parquet",
        source_name="FEMA National Risk Index + OpenFEMA Disaster Declarations",
        source_url="https://hazards.fema.gov/nri/data-resources",
        date_collected="2024",
        notes=(
            "Natural disaster risk scores from FEMA NRI (county-level, 0-100 scale) "
            "covering 18 hazard types. Supplemented with FEMA disaster declaration "
            "counts per county over past 20 years. County data matched to cities via FIPS."
        ),
    )
    log.info("\nSaved: data/disaster_risk.parquet (%d rows)", len(result))


if __name__ == "__main__":
    main()
