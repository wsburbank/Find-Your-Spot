"""
Collect sunshine data from NOAA 1991-2020 Hourly Climate Normals.

Source: NOAA NCEI — 1991-2020 US Climate Normals, Hourly Product
  URL: https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/
  Variables: HLY-CLOD-PCTCLR, PCTFEW, PCTSCT, PCTBKN, PCTOVC
    (Percent of observations in each cloud cover category: Clear, Few, Scattered,
    Broken, Overcast — based on hourly METAR observations at US weather stations)

Methodology:
  1. Download hourly normals for all ~461 US weather stations.
  2. For each station, compute annual average daytime "sunshine fraction" from
     cloud cover observations:
       sunshine_frac = (CLR*1.0 + FEW*0.85 + SCT*0.50) / (CLR + FEW + SCT + BKN + OVC)
     where CLR/FEW/SCT/BKN/OVC are the percentage of hours in each category.
     Only daytime hours (8 AM - 6 PM local) are used.
  3. Convert to sunny_days = sunshine_frac * 365.
  4. Match each city to the nearest station by haversine distance.
  5. IDW interpolation fills cities without a nearby station (< 150 miles).

The cloud cover observations are real METAR data recorded at US weather stations
over the 1991-2020 normal period. This is the same observational data underlying
NOAA's published "percent of possible sunshine" metric.

Output: Updates sunny_days column in data/climate.parquet
"""

import csv
import io
import logging
import re
import sys
import time
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
CACHE_DIR = DATA_DIR / "cache" / "sunshine"
NORMALS_BASE = "https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access"


def list_us_stations() -> list[str]:
    """Get list of US station files from the NOAA normals directory."""
    cache_path = CACHE_DIR / "station_list.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip().split("\n")

    log.info("Fetching station list from NOAA...")
    r = requests.get(f"{NORMALS_BASE}/", timeout=30)
    r.raise_for_status()

    files = re.findall(r'href="(US[^"]+\.csv)"', r.text)
    log.info("Found %d US station files", len(files))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(files), encoding="utf-8")
    return files


def download_station(station_file: str) -> list[dict] | None:
    """Download and parse hourly normals for one station.

    Returns list of rows with cloud cover data, or None on failure.
    """
    cache_path = CACHE_DIR / station_file
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        url = f"{NORMALS_BASE}/{station_file}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                text = r.text
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
                break
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    log.warning("Failed to download %s: %s", station_file, e)
                    return None
        else:
            return None

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def compute_station_sunshine(rows: list[dict]) -> dict | None:
    """Compute sunshine metrics for one station from hourly cloud cover data.

    Returns dict with station metadata and sunshine fraction, or None if
    insufficient data.
    """
    if not rows:
        return None

    first = rows[0]
    station_id = first.get("STATION", "").strip().strip('"')
    name = first.get("NAME", "").strip().strip('"')

    try:
        lat = float(first.get("LATITUDE", "0").strip().strip('"'))
        lon = float(first.get("LONGITUDE", "0").strip().strip('"'))
    except (ValueError, TypeError):
        return None

    if lat == 0 and lon == 0:
        return None

    # Collect daytime cloud cover observations (hours 8-18)
    sunshine_vals = []

    for row in rows:
        try:
            hour = int(row.get("hour", "0").strip().strip('"'))
        except (ValueError, TypeError):
            continue

        # Daytime only: 8 AM to 6 PM (11 hours)
        if hour < 8 or hour > 18:
            continue

        clr = _safe_float(row.get("HLY-CLOD-PCTCLR", ""))
        few = _safe_float(row.get("HLY-CLOD-PCTFEW", ""))
        sct = _safe_float(row.get("HLY-CLOD-PCTSCT", ""))
        bkn = _safe_float(row.get("HLY-CLOD-PCTBKN", ""))
        ovc = _safe_float(row.get("HLY-CLOD-PCTOVC", ""))

        if clr is None:
            continue

        # Total must be reasonable (cloud categories should sum near 100%)
        total = (clr or 0) + (few or 0) + (sct or 0) + (bkn or 0) + (ovc or 0)
        if total < 50:
            continue

        # Sunshine fraction: weight each category by approximate sun transmission
        # CLR (0 oktas): ~100% sunshine
        # FEW (1-2 oktas): ~85% sunshine
        # SCT (3-4 oktas): ~50% sunshine
        # BKN (5-7 oktas): ~15% sunshine
        # OVC (8 oktas): ~0% sunshine
        sun_frac = (
            (clr or 0) * 1.0
            + (few or 0) * 0.85
            + (sct or 0) * 0.50
            + (bkn or 0) * 0.15
            + (ovc or 0) * 0.0
        ) / total

        sunshine_vals.append(sun_frac)

    if len(sunshine_vals) < 1000:  # need at least ~1000 daytime hours
        return None

    avg_sunshine = np.mean(sunshine_vals)
    sunny_days = round(avg_sunshine * 365)

    return {
        "station_id": station_id,
        "station_name": name,
        "lat": lat,
        "lon": lon,
        "sunshine_fraction": round(avg_sunshine, 4),
        "sunny_days": sunny_days,
        "n_observations": len(sunshine_vals),
    }


def _safe_float(val: str) -> float | None:
    """Parse float, returning None for missing/sentinel values."""
    if val is None:
        return None
    val = val.strip().strip('"')
    try:
        f = float(val)
        return f if f >= 0 else None  # -9999 is NOAA sentinel
    except (ValueError, TypeError):
        return None


def match_stations_to_cities(
    stations: pd.DataFrame,
    cities: pd.DataFrame,
    max_distance: float = 150.0,
) -> pd.DataFrame:
    """Match each city to the nearest station within max_distance miles.

    Uses haversine distance. Cities without a nearby station get NaN.
    """
    results = []
    station_lats = stations["lat"].values
    station_lons = stations["lon"].values

    for _, city in cities.iterrows():
        dists = np.array([
            haversine_miles(city["lat"], city["lon"], slat, slon)
            for slat, slon in zip(station_lats, station_lons)
        ])
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]

        if nearest_dist <= max_distance:
            stn = stations.iloc[nearest_idx]
            results.append({
                "city_id": city["city_id"],
                "sunny_days": stn["sunny_days"],
                "sunshine_fraction": stn["sunshine_fraction"],
                "station_distance_mi": round(nearest_dist, 1),
                "sunshine_station": stn["station_name"],
            })
        else:
            results.append({
                "city_id": city["city_id"],
                "sunny_days": np.nan,
                "sunshine_fraction": np.nan,
                "station_distance_mi": np.nan,
                "sunshine_station": None,
            })

    return pd.DataFrame(results)


def interpolate_missing(
    sunshine_df: pd.DataFrame,
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    max_neighbors: int = 5,
    max_distance: float = 200.0,
) -> pd.DataFrame:
    """Fill missing sunshine values via IDW interpolation from nearby stations."""
    missing = sunshine_df[sunshine_df["sunny_days"].isna()]
    if missing.empty:
        return sunshine_df

    n_missing = len(missing)
    n_total = len(sunshine_df)
    pct_missing = n_missing / n_total * 100

    log.info("Missing sunshine for %d/%d cities (%.1f%%)",
             n_missing, n_total, pct_missing)

    if pct_missing > 5:
        log.warning("Interpolation would exceed 5%% threshold — "
                     "%.1f%% of cities need interpolation", pct_missing)

    station_lats = stations["lat"].values
    station_lons = stations["lon"].values
    station_sunny = stations["sunny_days"].values

    filled = 0
    for idx, row in missing.iterrows():
        city = cities[cities["city_id"] == row["city_id"]].iloc[0]
        dists = np.array([
            haversine_miles(city["lat"], city["lon"], slat, slon)
            for slat, slon in zip(station_lats, station_lons)
        ])

        nearby_mask = dists <= max_distance
        if not nearby_mask.any():
            continue

        nearby_dists = dists[nearby_mask]
        nearby_sunny = station_sunny[nearby_mask]

        # Sort by distance, take closest N
        order = np.argsort(nearby_dists)[:max_neighbors]
        nearby_dists = nearby_dists[order]
        nearby_sunny = nearby_sunny[order]

        # IDW interpolation
        weights = 1.0 / np.clip(nearby_dists, 0.1, None)
        interpolated = float(np.sum(nearby_sunny * weights) / np.sum(weights))

        sunshine_df.at[idx, "sunny_days"] = round(interpolated)
        sunshine_df.at[idx, "sunshine_fraction"] = round(interpolated / 365, 4)
        sunshine_df.at[idx, "station_distance_mi"] = round(nearby_dists[0], 1)
        sunshine_df.at[idx, "sunshine_station"] = "IDW interpolated"
        filled += 1

    log.info("Interpolated sunshine for %d cities", filled)
    still_missing = sunshine_df["sunny_days"].isna().sum()
    if still_missing > 0:
        log.warning("%d cities still missing sunshine data", still_missing)

    return sunshine_df


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Step 1: Get station list
    station_files = list_us_stations()
    log.info("Processing %d US stations", len(station_files))

    # Step 2: Download and compute sunshine for each station
    station_results = []
    for i, sf in enumerate(station_files):
        if (i + 1) % 50 == 0:
            log.info("Processing station %d/%d...", i + 1, len(station_files))

        rows = download_station(sf)
        if rows is None:
            continue

        result = compute_station_sunshine(rows)
        if result is not None:
            station_results.append(result)

        # Be polite to NOAA servers
        if not (CACHE_DIR / sf).exists():
            time.sleep(0.2)

    stations = pd.DataFrame(station_results)
    log.info("Stations with valid sunshine data: %d/%d",
             len(stations), len(station_files))

    if stations.empty:
        log.error("No station data collected")
        sys.exit(1)

    # Stats
    log.info("Station sunshine stats:")
    log.info("  Sunny days: min=%d, median=%d, mean=%.0f, max=%d",
             stations["sunny_days"].min(), stations["sunny_days"].median(),
             stations["sunny_days"].mean(), stations["sunny_days"].max())

    # Step 3: Match stations to cities
    sunshine_df = match_stations_to_cities(stations, cities)
    matched = sunshine_df["sunny_days"].notna().sum()
    log.info("Cities matched to stations: %d/%d (%.1f%%)",
             matched, len(cities), matched / len(cities) * 100)

    # Step 4: IDW interpolation for gaps
    sunshine_df = interpolate_missing(sunshine_df, cities, stations)

    # Step 5: Update climate.parquet with real sunny_days
    climate_path = DATA_DIR / "climate.parquet"
    if climate_path.exists():
        climate = pd.read_parquet(climate_path)
        # Replace the estimated sunny_days with real data
        climate = climate.drop(columns=["sunny_days"], errors="ignore")
        climate = climate.merge(
            sunshine_df[["city_id", "sunny_days"]],
            on="city_id",
            how="left",
        )
        write_parquet_with_metadata(
            climate,
            climate_path,
            source_name=(
                "NOAA ACIS 1991-2020 Climate Normals + "
                "NOAA NCEI 1991-2020 Hourly Normals (cloud cover/sunshine)"
            ),
            source_url="https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/",
            date_collected="2021",
            notes=(
                "Temperature, precipitation, and snow from NOAA ACIS (StnData API). "
                "Sunny days derived from NOAA 1991-2020 Hourly Climate Normals cloud "
                "cover observations (HLY-CLOD-PCTCLR/FEW/SCT/BKN/OVC) at ~460 US "
                "weather stations. Each station's sunshine fraction computed as "
                "weighted average of daytime cloud categories. IDW interpolation "
                "used for cities without a nearby station."
            ),
        )
        log.info("Updated climate.parquet with real sunshine data")
    else:
        log.warning("climate.parquet not found — saving standalone sunshine.parquet")
        write_parquet_with_metadata(
            sunshine_df,
            DATA_DIR / "sunshine.parquet",
            source_name="NOAA NCEI — 1991-2020 Hourly Climate Normals (cloud cover)",
            source_url="https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/",
            date_collected="2021",
            notes=(
                "Sunny days derived from NOAA 1991-2020 Hourly Climate Normals cloud "
                "cover observations (HLY-CLOD-PCTCLR/FEW/SCT/BKN/OVC). Sunshine "
                "fraction = weighted average of daytime (8-18h) cloud categories."
            ),
        )

    # Report
    final = sunshine_df["sunny_days"].dropna()
    log.info("\nFinal sunshine coverage: %d/%d cities (%.1f%%)",
             final.count(), len(cities), final.count() / len(cities) * 100)
    log.info("Sunny days stats: min=%d, median=%d, mean=%.0f, max=%d",
             final.min(), final.median(), final.mean(), final.max())

    # Show top/bottom
    merged = sunshine_df.merge(cities[["city_id", "name", "state"]], on="city_id")
    top = merged.nlargest(10, "sunny_days")
    bottom = merged.nsmallest(10, "sunny_days")

    log.info("\nTop 10 sunniest cities:")
    for _, r in top.iterrows():
        log.info("  %s, %s: %d sunny days", r["name"], r["state"], r["sunny_days"])

    log.info("\nBottom 10 least sunny cities:")
    for _, r in bottom.iterrows():
        log.info("  %s, %s: %d sunny days", r["name"], r["state"], r["sunny_days"])


if __name__ == "__main__":
    main()
