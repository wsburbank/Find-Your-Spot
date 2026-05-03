"""
Collect museum data from IMLS (Institute of Museum and Library Services) Museum Data Files.

Source: IMLS Museum Data Files (2018 — final release)
URL: https://www.imls.gov/research-evaluation/data/museum-data-files
Download: https://www.imls.gov/sites/default/files/2018_csv_museum_data_files.zip

The dataset contains ~33,000 museums across the US with geocodes, discipline
classification, and NAICS codes. We match each museum to the nearest city in
our master list and produce per-city counts by museum type.

Output: data/museums.parquet
"""

import io
import logging
import sys
import zipfile
from pathlib import Path

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

IMLS_URL = "https://www.imls.gov/sites/default/files/2018_csv_museum_data_files.zip"


def download_imls_data() -> pd.DataFrame:
    """Download and extract the IMLS museum CSV from the zip archive."""
    cache_path = CACHE_DIR / "imls_museums.csv"
    if cache_path.exists():
        log.info("Using cached IMLS data: %s", cache_path)
        return pd.read_csv(cache_path, encoding="latin-1", low_memory=False)

    log.info("Downloading IMLS Museum Data Files from %s", IMLS_URL)
    resp = requests.get(IMLS_URL, timeout=120)
    resp.raise_for_status()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found in IMLS zip")

        # Pick the main data file (usually the largest CSV)
        csv_name = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
        log.info("Extracting %s from zip (%d files total)", csv_name, len(csv_names))

        with zf.open(csv_name) as f:
            raw = f.read()
            # Save to cache
            cache_path.write_bytes(raw)
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", low_memory=False)

    log.info("IMLS data: %d rows, columns: %s", len(df), df.columns.tolist())
    return df


def clean_imls_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the IMLS dataset to usable museum records with coordinates."""
    # Identify coordinate columns (IMLS uses LONGITUDE, LATITUDE or similar)
    col_map = {}
    for col in df.columns:
        cl = col.upper().strip()
        if "LAT" in cl:
            col_map["lat"] = col
        elif "LON" in cl:
            col_map["lon"] = col
        elif cl in ("COMMONNAME", "MUSEUM_NAME", "NAME"):
            col_map["museum_name"] = col
        elif cl in ("DISCIPL", "DISCIPLINE", "MUSEUM_TYPE"):
            col_map["discipline"] = col
        elif cl in ("STATE", "PHYSSTATE", "PHYS_STATE"):
            col_map["state"] = col
        elif cl in ("CITY", "PHYSCITY", "PHYS_CITY"):
            col_map["city"] = col

    log.info("Column mapping: %s", col_map)

    if "lat" not in col_map or "lon" not in col_map:
        # Try to find them by examining column values
        log.warning("Could not auto-detect lat/lon columns, listing all: %s", df.columns.tolist())
        raise RuntimeError("Cannot find lat/lon columns in IMLS data")

    out = pd.DataFrame()
    out["museum_name"] = df.get(col_map.get("museum_name", ""), "Unknown")
    out["discipline"] = df.get(col_map.get("discipline", ""), "Unknown")
    out["city"] = df.get(col_map.get("city", ""), "")
    out["state"] = df.get(col_map.get("state", ""), "")
    out["lat"] = pd.to_numeric(df[col_map["lat"]], errors="coerce")
    out["lon"] = pd.to_numeric(df[col_map["lon"]], errors="coerce")

    # Drop records without coordinates
    before = len(out)
    out = out.dropna(subset=["lat", "lon"])
    # Filter to valid US coordinates
    out = out[(out["lat"] > 17) & (out["lat"] < 72) &
              (out["lon"] > -180) & (out["lon"] < -60)]
    log.info("Filtered %d -> %d museums with valid US coordinates", before, len(out))

    return out.reset_index(drop=True)


def match_museums_to_cities(museums: pd.DataFrame, cities: pd.DataFrame,
                            max_distance_mi: float = 30.0) -> pd.DataFrame:
    """Match each museum to the nearest city within max_distance_mi."""
    city_lats = cities["lat"].values
    city_lons = cities["lon"].values
    city_ids = cities["city_id"].values

    matches = []
    for _, museum in museums.iterrows():
        mlat, mlon = museum["lat"], museum["lon"]

        # Vectorized haversine to all cities
        distances = [
            haversine_miles(mlat, mlon, city_lats[i], city_lons[i])
            for i in range(len(city_lats))
        ]
        min_idx = min(range(len(distances)), key=lambda i: distances[i])
        min_dist = distances[min_idx]

        if min_dist <= max_distance_mi:
            matches.append({
                "city_id": city_ids[min_idx],
                "discipline": museum["discipline"],
                "distance_mi": round(min_dist, 1),
            })

    return pd.DataFrame(matches)


def aggregate_by_city(matched: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Aggregate museum counts per city."""
    if matched.empty:
        result = cities[["city_id"]].copy()
        result["museums_count"] = 0
        return result

    # Total museum count per city
    counts = matched.groupby("city_id").size().reset_index(name="museums_count")

    # Merge back to full city list (cities with 0 museums get 0)
    result = cities[["city_id"]].merge(counts, on="city_id", how="left")
    result["museums_count"] = result["museums_count"].fillna(0).astype(int)

    return result


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Download and clean IMLS data
    raw = download_imls_data()
    museums = clean_imls_data(raw)
    log.info("Clean museums: %d records", len(museums))

    # Match to cities
    log.info("Matching museums to nearest cities (this may take a minute)...")
    matched = match_museums_to_cities(museums, cities, max_distance_mi=30.0)
    log.info("Matched %d museums to %d unique cities",
             len(matched), matched["city_id"].nunique())

    # Aggregate
    result = aggregate_by_city(matched, cities)

    # Report
    has_museums = (result["museums_count"] > 0).sum()
    total_matched = result["museums_count"].sum()
    log.info("Cities with museums: %d/%d (%.1f%%)",
             has_museums, len(result), has_museums / len(result) * 100)
    log.info("Total museums matched: %d", total_matched)
    log.info("Top 10 cities by museum count:")
    top = result.nlargest(10, "museums_count").merge(
        cities[["city_id", "name", "state"]], on="city_id"
    )
    for _, row in top.iterrows():
        log.info("  %s, %s: %d museums", row["name"], row["state"],
                 row["museums_count"])

    # Save
    write_parquet_with_metadata(
        result,
        DATA_DIR / "museums.parquet",
        source_name="IMLS Museum Data Files",
        source_url="https://www.imls.gov/research-evaluation/data/museum-data-files",
        date_collected="2018",
        notes=(
            "Museum counts per city derived from IMLS 2018 Museum Data Files "
            "(final release). Each museum matched to nearest city within 30 miles "
            "by Haversine distance. Includes all museum disciplines."
        ),
    )
    print(f"\nSaved: data/museums.parquet ({len(result)} rows)")


if __name__ == "__main__":
    main()
