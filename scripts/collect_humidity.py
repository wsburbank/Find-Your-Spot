"""
Collect humidity (dew point) data from NOAA 1991-2020 Hourly Climate Normals.

Source: NOAA NCEI — 1991-2020 US Climate Normals, Hourly Product
  URL: https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/
  Variable: HLY-DEWP-NORMAL (hourly average dew point in degrees Fahrenheit)

Methodology:
  1. Reuse the same station CSV files cached by collect_sunshine.py.
  2. For each station, compute average summer dew point (June/July/August)
     from HLY-DEWP-NORMAL values.
  3. Match each city to the nearest station by haversine distance.
  4. IDW interpolation fills cities without a nearby station.

Dew point is a better indicator of humidity comfort than relative humidity
because it doesn't fluctuate with temperature throughout the day.
  - Below 55F: comfortable/dry
  - 55-60F: noticeable but comfortable
  - 60-65F: humid, slightly uncomfortable
  - 65-70F: very humid, uncomfortable
  - Above 70F: oppressive

Output: data/humidity.parquet (avg_summer_dewpoint column)
"""

import csv
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "sunshine"  # Reuse sunshine cache


def _safe_float(val: str) -> float | None:
    """Parse float, returning None for missing/sentinel values."""
    if val is None:
        return None
    val = val.strip().strip('"')
    try:
        f = float(val)
        return f if f > -100 else None  # NOAA sentinel is typically -9999
    except (ValueError, TypeError):
        return None


def compute_station_dewpoint(station_file: Path) -> dict | None:
    """Compute average summer dew point for one station from cached CSV.

    Returns dict with station metadata and avg_summer_dewpoint, or None.
    """
    if not station_file.exists():
        return None

    text = station_file.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

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

    # Collect summer dew point values (June=6, July=7, August=8)
    summer_dewpoints = []
    for row in rows:
        try:
            month = int(row.get("month", "0").strip().strip('"'))
        except (ValueError, TypeError):
            continue

        if month not in (6, 7, 8):
            continue

        dewp = _safe_float(row.get("HLY-DEWP-NORMAL", ""))
        if dewp is not None:
            summer_dewpoints.append(dewp)

    if len(summer_dewpoints) < 500:  # Need reasonable coverage of summer hours
        return None

    avg_dewpoint = np.mean(summer_dewpoints)

    return {
        "station_id": station_id,
        "station_name": name,
        "lat": lat,
        "lon": lon,
        "avg_summer_dewpoint": round(avg_dewpoint, 1),
        "n_observations": len(summer_dewpoints),
    }


def match_stations_to_cities(
    stations: pd.DataFrame,
    cities: pd.DataFrame,
    max_distance: float = 150.0,
) -> pd.DataFrame:
    """Match each city to the nearest station within max_distance miles."""
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
                "avg_summer_dewpoint": stn["avg_summer_dewpoint"],
                "humidity_station_dist_mi": round(nearest_dist, 1),
            })
        else:
            results.append({
                "city_id": city["city_id"],
                "avg_summer_dewpoint": np.nan,
                "humidity_station_dist_mi": np.nan,
            })

    return pd.DataFrame(results)


def interpolate_missing(
    humidity_df: pd.DataFrame,
    cities: pd.DataFrame,
    stations: pd.DataFrame,
    max_neighbors: int = 5,
    max_distance: float = 200.0,
) -> pd.DataFrame:
    """Fill missing dew point values via IDW interpolation from nearby stations."""
    missing = humidity_df[humidity_df["avg_summer_dewpoint"].isna()]
    if missing.empty:
        return humidity_df

    n_missing = len(missing)
    n_total = len(humidity_df)
    pct_missing = n_missing / n_total * 100

    log.info("Missing humidity for %d/%d cities (%.1f%%)",
             n_missing, n_total, pct_missing)

    if pct_missing > 5:
        log.warning("Interpolation would exceed 5%% threshold — "
                     "%.1f%% of cities need interpolation", pct_missing)

    station_lats = stations["lat"].values
    station_lons = stations["lon"].values
    station_dewp = stations["avg_summer_dewpoint"].values

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
        nearby_dewp = station_dewp[nearby_mask]

        order = np.argsort(nearby_dists)[:max_neighbors]
        nearby_dists = nearby_dists[order]
        nearby_dewp = nearby_dewp[order]

        weights = 1.0 / np.clip(nearby_dists, 0.1, None)
        interpolated = float(np.sum(nearby_dewp * weights) / np.sum(weights))

        humidity_df.at[idx, "avg_summer_dewpoint"] = round(interpolated, 1)
        humidity_df.at[idx, "humidity_station_dist_mi"] = round(nearby_dists[0], 1)
        filled += 1

    log.info("Interpolated humidity for %d cities", filled)
    still_missing = humidity_df["avg_summer_dewpoint"].isna().sum()
    if still_missing > 0:
        log.warning("%d cities still missing humidity data", still_missing)

    return humidity_df


def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities", len(cities))

    # Get station files from cache (already downloaded by collect_sunshine.py)
    if not CACHE_DIR.exists():
        log.error("No cached station files found. Run collect_sunshine.py first.")
        sys.exit(1)

    station_files = sorted(CACHE_DIR.glob("US*.csv"))
    log.info("Found %d cached station files", len(station_files))

    # Compute summer dew point for each station
    station_results = []
    for i, sf in enumerate(station_files):
        if (i + 1) % 100 == 0:
            log.info("Processing station %d/%d...", i + 1, len(station_files))
        result = compute_station_dewpoint(sf)
        if result:
            station_results.append(result)

    stations_df = pd.DataFrame(station_results)
    log.info("Computed dew point for %d stations", len(stations_df))

    # Stats
    log.info("Dew point range: %.1fF to %.1fF (median %.1fF)",
             stations_df["avg_summer_dewpoint"].min(),
             stations_df["avg_summer_dewpoint"].max(),
             stations_df["avg_summer_dewpoint"].median())

    # Match stations to cities
    humidity_df = match_stations_to_cities(stations_df, cities)
    matched = humidity_df["avg_summer_dewpoint"].notna().sum()
    log.info("Direct matches: %d/%d cities", matched, len(cities))

    # Interpolate missing
    humidity_df = interpolate_missing(humidity_df, cities, stations_df)

    final_coverage = humidity_df["avg_summer_dewpoint"].notna().sum()
    log.info("Final coverage: %d/%d cities (%.1f%%)",
             final_coverage, len(cities), final_coverage / len(cities) * 100)

    # Write output
    output = humidity_df[["city_id", "avg_summer_dewpoint"]].copy()
    write_parquet_with_metadata(
        output,
        DATA_DIR / "humidity.parquet",
        source_name="NOAA NCEI 1991-2020 US Climate Normals, Hourly Product",
        source_url="https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/",
        notes=(
            "Average summer (June/July/August) dew point in degrees Fahrenheit, "
            "computed from HLY-DEWP-NORMAL variable. Dew point is a direct measure "
            "of atmospheric moisture content. Below 55F = dry/comfortable, "
            "55-60F = noticeable, 60-65F = humid, 65-70F = very humid, "
            f"above 70F = oppressive. Coverage: {final_coverage}/{len(cities)} cities."
        ),
    )

    # Summary
    print(f"\nHumidity Data Summary:")
    print(f"  Stations processed: {len(stations_df)}")
    print(f"  Cities with data: {final_coverage}/{len(cities)} ({final_coverage/len(cities)*100:.1f}%)")
    if final_coverage > 0:
        valid = output["avg_summer_dewpoint"].dropna()
        print(f"  Avg summer dew point range: {valid.min():.1f}F - {valid.max():.1f}F")
        print(f"  Median: {valid.median():.1f}F")
        # Comfort distribution
        print(f"\n  Comfort distribution:")
        print(f"    Dry (<55F):           {(valid < 55).sum()} cities")
        print(f"    Comfortable (55-60F): {((valid >= 55) & (valid < 60)).sum()} cities")
        print(f"    Humid (60-65F):       {((valid >= 60) & (valid < 65)).sum()} cities")
        print(f"    Very humid (65-70F):  {((valid >= 65) & (valid < 70)).sum()} cities")
        print(f"    Oppressive (70F+):    {(valid >= 70).sum()} cities")


if __name__ == "__main__":
    main()
