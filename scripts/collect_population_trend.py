"""
Collect population trend data from Census Population Estimates Program (PEP).

Source: US Census Bureau — Population Estimates Program (PEP)
URL: https://www2.census.gov/programs-surveys/popest/datasets/
Data years: 2010-2023 annual estimates at city/place level

Calculates 5-year and 10-year population growth rates for each city.

Output: data/population_trend.parquet
"""

import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"

# Census PEP files — sub-county population estimates
# 2020-2024 vintage (post-2020 Census base, most recent)
PEP_2024_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
    "cities/totals/sub-est2024.csv"
)

# 2010-2020 vintage (uses 2010 Census base, includes annual estimates through 2020)
PEP_2020_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/"
    "cities/SUB-EST2020_ALL.csv"
)


def download_pep(url: str, label: str) -> pd.DataFrame:
    """Download a PEP CSV file from Census Bureau."""
    log.info("Downloading %s from %s ...", label, url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # Try different encodings — Census files often use latin-1
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(StringIO(resp.content.decode(encoding)))
            log.info("  Parsed %d rows with %s encoding", len(df), encoding)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(f"Could not parse {url} with any encoding")


def extract_place_populations(df: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Extract place-level population estimates from a PEP DataFrame.

    Args:
        df: Raw PEP DataFrame.
        vintage: '2020-2023' or '2010-2020' to determine column naming.

    Returns:
        DataFrame with fips_state, fips_place, and population columns by year.
    """
    # Filter to places only (SUMLEV 162 = incorporated places + CDPs)
    if "SUMLEV" in df.columns:
        df = df[df["SUMLEV"] == 162].copy()
    elif "sumlev" in df.columns:
        df = df[df["sumlev"] == 162].copy()
    else:
        log.warning("No SUMLEV column found — using all rows")

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Extract state and place FIPS
    if "state" in df.columns and "place" in df.columns:
        df["fips_state"] = df["state"].astype(str).str.split(".").str[0].str.zfill(2)
        df["fips_place"] = df["place"].astype(str).str.split(".").str[0].str.zfill(5)
    else:
        log.error("Cannot find state/place FIPS columns in %s", list(df.columns))
        return pd.DataFrame()

    # Find population estimate columns (named like popestimate2023)
    pop_cols = {}
    for col in df.columns:
        if "popestimate" in col:
            # Extract year from column name (handle popestimate042020 -> skip it)
            year_str = col.replace("popestimate", "")
            if len(year_str) == 4:
                try:
                    year = int(year_str)
                    pop_cols[year] = col
                except ValueError:
                    continue

    # Also check for census2010pop, census2020pop
    for col in df.columns:
        if col == "census2010pop":
            pop_cols[2010] = col
        elif col == "census2020pop":
            pop_cols[2020] = col

    if not pop_cols:
        log.error("No population estimate columns found. Columns: %s", list(df.columns))
        return pd.DataFrame()

    log.info("  Found population data for years: %s", sorted(pop_cols.keys()))

    # Build clean output
    result = df[["fips_state", "fips_place"]].copy()
    for year, col in sorted(pop_cols.items()):
        result[f"pop_{year}"] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no valid FIPS
    result = result.dropna(subset=["fips_state", "fips_place"])
    return result


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities from master list", len(cities))

    # Download both PEP vintages
    pep_2024 = download_pep(PEP_2024_URL, "PEP 2020-2024")
    pep_2020 = download_pep(PEP_2020_URL, "PEP 2010-2020")

    # Extract place-level populations
    places_2024 = extract_place_populations(pep_2024, "2020-2024")
    places_2020 = extract_place_populations(pep_2020, "2010-2020")

    if places_2024.empty and places_2020.empty:
        log.error("No population data extracted — aborting")
        sys.exit(1)

    # Merge the two vintages on fips_state + fips_place
    if not places_2020.empty and not places_2024.empty:
        # From 2010-2020 vintage, keep 2010-2019 columns (2020 from new vintage is more accurate)
        cols_2020 = ["fips_state", "fips_place"] + [
            c for c in places_2020.columns if c.startswith("pop_") and int(c.split("_")[1]) < 2020
        ]
        places = places_2024.merge(
            places_2020[cols_2020],
            on=["fips_state", "fips_place"],
            how="outer",
        )
    elif not places_2024.empty:
        places = places_2024
    else:
        places = places_2020

    log.info("Combined population data: %d places, years in data: %s",
             len(places),
             sorted([c for c in places.columns if c.startswith("pop_")]))

    # Match to our cities via fips_state + fips_place
    cities_match = cities[["city_id", "fips_state", "fips_place", "name", "state"]].copy()
    cities_match["fips_state"] = cities_match["fips_state"].astype(str).str.zfill(2)
    cities_match["fips_place"] = cities_match["fips_place"].astype(str).str.zfill(5)

    result = cities_match.merge(places, on=["fips_state", "fips_place"], how="left")

    # Calculate growth rates
    # Use most recent year available and compare to 5/10 years prior
    pop_years = sorted([int(c.split("_")[1]) for c in result.columns if c.startswith("pop_")])
    most_recent = max(pop_years)
    log.info("Most recent population year: %d", most_recent)

    recent_col = f"pop_{most_recent}"

    # 5-year growth
    year_5 = most_recent - 5
    if year_5 in pop_years:
        col_5 = f"pop_{year_5}"
        result["pop_growth_5yr"] = (
            (result[recent_col] - result[col_5]) / result[col_5] * 100
        ).round(2)
        valid_5 = result["pop_growth_5yr"].notna().sum()
        log.info("5-year growth (%d→%d): %d/%d cities",
                 year_5, most_recent, valid_5, len(result))
    else:
        log.warning("Cannot calculate 5-year growth: no data for %d", year_5)

    # 10-year growth
    year_10 = most_recent - 10
    if year_10 in pop_years:
        col_10 = f"pop_{year_10}"
        result["pop_growth_10yr"] = (
            (result[recent_col] - result[col_10]) / result[col_10] * 100
        ).round(2)
        valid_10 = result["pop_growth_10yr"].notna().sum()
        log.info("10-year growth (%d→%d): %d/%d cities",
                 year_10, most_recent, valid_10, len(result))
    else:
        log.warning("Cannot calculate 10-year growth: no data for %d", year_10)

    # Keep only the output columns we need
    output_cols = ["city_id"]
    if "pop_growth_5yr" in result.columns:
        output_cols.append("pop_growth_5yr")
    if "pop_growth_10yr" in result.columns:
        output_cols.append("pop_growth_10yr")

    # Also keep most recent population as a cross-check
    if recent_col in result.columns:
        result["pop_estimate_latest"] = result[recent_col].astype("Int64")
        output_cols.append("pop_estimate_latest")

    output = result[output_cols].copy()

    # Report
    log.info("\nPopulation trend coverage:")
    for col in output.columns:
        if col == "city_id":
            continue
        valid = output[col].notna().sum()
        log.info("  %s: %d/%d cities (%.1f%%)", col, valid, len(output), valid / len(output) * 100)

    if "pop_growth_5yr" in output.columns:
        log.info("\nFastest growing (5yr):")
        top = output.nlargest(10, "pop_growth_5yr").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in top.iterrows():
            log.info("  %s, %s: +%.1f%%", row["name"], row["state"], row["pop_growth_5yr"])

        log.info("\nFastest declining (5yr):")
        bottom = output.nsmallest(5, "pop_growth_5yr").merge(
            cities[["city_id", "name", "state"]], on="city_id"
        )
        for _, row in bottom.iterrows():
            log.info("  %s, %s: %.1f%%", row["name"], row["state"], row["pop_growth_5yr"])

    # Save
    write_parquet_with_metadata(
        output,
        DATA_DIR / "population_trend.parquet",
        source_name="US Census Bureau — Population Estimates Program (PEP)",
        source_url="https://www2.census.gov/programs-surveys/popest/datasets/",
        date_collected="2023",
        notes=(
            "Population growth rates calculated from Census PEP annual place-level estimates. "
            "5-year and 10-year growth rates as percentages. "
            "2020-2023 vintage uses 2020 Census base; 2010-2020 vintage uses 2010 Census base."
        ),
    )
    log.info("\nSaved: data/population_trend.parquet (%d rows)", len(output))


if __name__ == "__main__":
    main()
