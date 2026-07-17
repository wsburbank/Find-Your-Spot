"""
Phase 2a: Collect climate/weather data from NOAA via ACIS.

Source: NOAA ACIS (Applied Climate Information System) - 1991-2020 Climate Normals
URL: https://data.rcc-acis.org/

Strategy:
  1. For each city, use ACIS StnMeta (bbox search) to find stations ACIS knows
  2. Try each station until we get normals data
  3. Compute annual summaries

This approach queries ACIS for stations it actually has data for, rather than
using the GHCN station list which includes many stations without normals.

Output: data/climate.parquet
"""

import logging
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

ACIS_URL = "https://data.rcc-acis.org"


def load_master_cities() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")


def _safe_float(val) -> float | None:
    if val is None or val == "M" or val == "T" or val == "" or val == "S":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def find_acis_stations(lat: float, lon: float, radius_deg: float = 0.75) -> list[dict]:
    """Find ACIS stations near a point using bbox search."""
    bbox = (f"{lon - radius_deg:.4f},{lat - radius_deg:.4f},"
            f"{lon + radius_deg:.4f},{lat + radius_deg:.4f}")
    payload = {
        "bbox": bbox,
        "meta": "name,state,ll,sids",
        "elems": "maxt",
    }
    try:
        resp = requests.post(f"{ACIS_URL}/StnMeta", json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        stations = data.get("meta", [])
        for s in stations:
            ll = s.get("ll", [0, 0])
            s["distance"] = haversine_miles(lat, lon, ll[1], ll[0])
        stations.sort(key=lambda s: s["distance"])
        return stations
    except Exception:
        return []


def get_normals(sid: str) -> list | None:
    """Query ACIS StnData for monthly normals."""
    payload = {
        "sid": sid,
        "sdate": "2020-01",
        "edate": "2020-12",
        "meta": "name",
        "elems": [
            {"name": "maxt", "interval": "mly", "duration": "mly",
             "reduce": "mean", "normal": "1"},
            {"name": "mint", "interval": "mly", "duration": "mly",
             "reduce": "mean", "normal": "1"},
            {"name": "pcpn", "interval": "mly", "duration": "mly",
             "reduce": "sum", "normal": "1"},
            {"name": "snow", "interval": "mly", "duration": "mly",
             "reduce": "sum", "normal": "1"},
        ],
    }
    try:
        resp = requests.post(f"{ACIS_URL}/StnData", json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        if len(rows) >= 12:
            # Check that we have actual data, not all M's
            maxt_vals = [_safe_float(r[1]) for r in rows]
            if sum(1 for v in maxt_vals if v is not None) >= 6:
                return rows
    except Exception:
        pass
    return None


def fetch_city_climate(city: pd.Series) -> dict | None:
    """Fetch climate normals for a single city.

    Tries a 0.75-degree bbox first, then expands to 1.5 degrees for a retry.
    """
    for radius in (0.75, 1.5):
        stations = find_acis_stations(city["lat"], city["lon"], radius_deg=radius)
        for station in stations[:8]:
            sids = station.get("sids", [])
            for sid_entry in sids[:3]:
                sid = sid_entry.split()[0] if isinstance(sid_entry, str) else str(sid_entry)
                normals = get_normals(sid)
                if normals:
                    return _process_normals(normals, city, station)

    return None


def _process_normals(normals: list, city: pd.Series, station: dict) -> dict:
    """Process raw normals into climate metrics."""
    mt = [_safe_float(r[1]) for r in normals]
    mn = [_safe_float(r[2]) for r in normals]
    mp = [_safe_float(r[3]) for r in normals]
    ms = [_safe_float(r[4]) for r in normals]

    summer_high = [mt[i] for i in [5, 6, 7] if mt[i] is not None]
    winter_low = [mn[i] for i in [11, 0, 1] if mn[i] is not None]
    valid_pcpn = [v for v in mp if v is not None]
    valid_snow = [v for v in ms if v is not None]

    annual_rain = sum(valid_pcpn) if valid_pcpn else None

    # sunny_days is now sourced from NOAA Hourly Climate Normals cloud cover data
    # (see scripts/collect_sunshine.py). This column is kept here as None and
    # overwritten by collect_sunshine.py when it updates climate.parquet.

    return {
        "city_id": city["city_id"],
        "name": city["name"],
        "state": city["state"],
        "station_name": station.get("name", ""),
        "station_distance_mi": round(station.get("distance", 0), 1),
        "avg_temp_summer": round(np.mean(summer_high), 1) if summer_high else None,
        "avg_temp_winter": round(np.mean(winter_low), 1) if winter_low else None,
        "avg_high_july": mt[6],
        "avg_low_january": mn[0],
        "annual_rainfall": round(annual_rain, 1) if annual_rain else None,
        "annual_snow": round(sum(valid_snow), 1) if valid_snow else None,
        "sunny_days": None,  # Populated by collect_sunshine.py
    }


CLIMATE_NUMERIC_COLS = [
    "avg_temp_summer", "avg_temp_winter", "avg_high_july", "avg_low_january",
    "annual_rainfall", "annual_snow", "sunny_days",
]


def interpolate_missing_climate(
    climate_df: pd.DataFrame,
    cities_master: pd.DataFrame,
    failed_names: list[str],
    max_neighbors: int = 5,
    max_radius_miles: float = 100.0,
) -> pd.DataFrame:
    """Fill missing climate data using inverse-distance weighted interpolation.

    Args:
        climate_df: DataFrame of cities that have climate data.
        cities_master: Full master city list with lat/lon.
        failed_names: List of "Name, State" strings for cities missing climate data.
        max_neighbors: Max nearby cities to use.
        max_radius_miles: Search radius for neighbors.

    Returns:
        Updated climate_df with interpolated rows appended.
    """
    # Build lookup of cities that already have data
    have_data = set(climate_df["city_id"].tolist())
    valid = climate_df.merge(
        cities_master[["city_id", "lat", "lon"]], on="city_id", how="left",
    )

    new_rows = []
    for _, city in cities_master.iterrows():
        if city["city_id"] in have_data:
            continue

        target_lat, target_lon = city["lat"], city["lon"]
        if pd.isna(target_lat) or pd.isna(target_lon):
            continue

        # Compute distances to cities with climate data
        dists = valid.apply(
            lambda r: haversine_miles(target_lat, target_lon, r["lat"], r["lon"]),
            axis=1,
        )
        nearby = valid.assign(dist=dists)
        nearby = nearby[nearby["dist"] <= max_radius_miles].nsmallest(
            max_neighbors, "dist",
        )

        if nearby.empty:
            log.warning("No nearby climate data for %s, %s", city["name"], city["state"])
            continue

        weights = 1.0 / nearby["dist"].clip(lower=0.1)
        row = {
            "city_id": city["city_id"],
            "name": city["name"],
            "state": city["state"],
            "station_name": "interpolated",
            "station_distance_mi": 0.0,
        }
        for col in CLIMATE_NUMERIC_COLS:
            col_valid = nearby[nearby[col].notna()]
            if col_valid.empty:
                row[col] = None
            else:
                w = 1.0 / col_valid["dist"].clip(lower=0.1)
                val = (col_valid[col] * w).sum() / w.sum()
                row[col] = round(val, 1) if col != "sunny_days" else int(round(val))

        new_rows.append(row)
        log.info(
            "Interpolated climate for %s, %s from %d neighbors (nearest %.0f mi)",
            city["name"], city["state"], len(nearby), nearby["dist"].min(),
        )

    if new_rows:
        climate_df = pd.concat([climate_df, pd.DataFrame(new_rows)], ignore_index=True)
        log.info("Interpolated climate for %d cities", len(new_rows))

    return climate_df


def fill_missing_columns(
    climate_df: pd.DataFrame,
    cities_master: pd.DataFrame,
    columns: list[str] | None = None,
    max_neighbors: int = 5,
    max_radius_miles: float = 100.0,
) -> pd.DataFrame:
    """Fill per-column NaN values via IDW interpolation from nearby cities.

    Unlike interpolate_missing_climate (which handles fully-missing rows), this
    fills individual null cells in rows that already have partial data — e.g. a
    city with temperature data but missing snowfall.

    Args:
        climate_df: Climate DataFrame (must include city_id).
        cities_master: Master city list with lat/lon.
        columns: Which numeric columns to fill.  Defaults to CLIMATE_NUMERIC_COLS.
        max_neighbors: Max nearby cities to use for interpolation.
        max_radius_miles: Search radius for neighbors.

    Returns:
        Updated climate_df with nulls filled where possible.
    """
    if columns is None:
        columns = CLIMATE_NUMERIC_COLS

    merged = climate_df.merge(
        cities_master[["city_id", "lat", "lon"]], on="city_id", how="left",
    )

    filled_total = 0
    for col in columns:
        if col not in merged.columns:
            continue

        missing_mask = merged[col].isna()
        n_missing = missing_mask.sum()
        if n_missing == 0:
            continue

        have_mask = merged[col].notna()
        donors = merged[have_mask]
        if donors.empty:
            continue

        donor_lats = donors["lat"].values
        donor_lons = donors["lon"].values
        donor_vals = donors[col].values

        filled = 0
        for idx in merged.index[missing_mask]:
            tlat, tlon = merged.at[idx, "lat"], merged.at[idx, "lon"]
            if pd.isna(tlat) or pd.isna(tlon):
                continue

            # Vectorized haversine to all donors
            dists = _vectorized_haversine(tlat, tlon, donor_lats, donor_lons)
            within = dists <= max_radius_miles
            if not within.any():
                continue

            nearby_dists = dists[within]
            nearby_vals = donor_vals[within]
            # Keep closest max_neighbors
            if len(nearby_dists) > max_neighbors:
                order = np.argsort(nearby_dists)[:max_neighbors]
                nearby_dists = nearby_dists[order]
                nearby_vals = nearby_vals[order]

            weights = 1.0 / np.clip(nearby_dists, 0.1, None)
            val = float(np.sum(nearby_vals * weights) / np.sum(weights))
            climate_df.at[idx, col] = int(round(val)) if col == "sunny_days" else round(val, 1)
            filled += 1

        filled_total += filled
        log.info("Filled %s: %d/%d missing values interpolated", col, filled, n_missing)

    log.info("Total cells filled by column interpolation: %d", filled_total)
    return climate_df


def _vectorized_haversine(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Haversine distance from one point to an array of points (miles)."""
    R = 3958.8
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cities = load_master_cities()
    output_path = PROJECT_ROOT / "data" / "climate.parquet"

    log.info("Collecting climate data for %d cities (parallel)...", len(cities))

    results = []
    failed = []
    total = len(cities)
    done_count = 0

    city_rows = [row for _, row in cities.iterrows()]

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_city = {
            executor.submit(fetch_city_climate, city): city
            for city in city_rows
        }
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            done_count += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed.append(f"{city['name']}, {city['state']}")
            except Exception:
                failed.append(f"{city['name']}, {city['state']}")

            if done_count % 50 == 0:
                log.info("Progress: %d/%d cities (%d found, %.0f%%)",
                         done_count, total, len(results),
                         len(results) / done_count * 100 if done_count else 0)

    climate_df = pd.DataFrame(results)

    if climate_df.empty:
        log.error("No climate data retrieved!")
        sys.exit(1)

    # Interpolate climate for cities that ACIS couldn't cover
    if failed:
        climate_df = interpolate_missing_climate(climate_df, cities, failed)

    # Fill per-column gaps (e.g. station had temp but no snowfall)
    climate_df = fill_missing_columns(climate_df, cities)

    write_parquet_with_metadata(
        climate_df,
        output_path,
        source_name="NOAA ACIS 1991-2020 Climate Normals",
        source_url="https://data.rcc-acis.org/",
        date_collected="2026-05-02",
        notes=(
            "30-year climate normals (1991-2020) from NOAA ACIS. "
            "Nearest ACIS station with normals used per city. "
            "Sunny days estimated from precipitation + latitude."
        ),
    )

    coverage = len(climate_df) / total * 100
    log.info("Saved climate data: %d/%d cities (%.1f%%)",
             len(climate_df), total, coverage)

    if failed:
        log.warning("Missing %d cities", len(failed))

    print(f"\nClimate coverage: {len(climate_df)}/{total} ({coverage:.1f}%)")
    print(climate_df[["name", "state", "avg_temp_summer", "avg_temp_winter",
                       "annual_rainfall", "annual_snow", "sunny_days"]].head(15).to_string())


def backfill():
    """Fill missing per-column values in existing climate.parquet without re-fetching."""
    cities = load_master_cities()
    output_path = PROJECT_ROOT / "data" / "climate.parquet"

    climate_df = pd.read_parquet(output_path)
    log.info("Loaded existing climate data: %d rows", len(climate_df))

    before = {col: climate_df[col].isna().sum() for col in CLIMATE_NUMERIC_COLS
              if col in climate_df.columns}
    log.info("Missing before: %s", {k: v for k, v in before.items() if v > 0})

    climate_df = fill_missing_columns(climate_df, cities)

    after = {col: climate_df[col].isna().sum() for col in CLIMATE_NUMERIC_COLS
             if col in climate_df.columns}
    log.info("Missing after:  %s", {k: v for k, v in after.items() if v > 0})

    write_parquet_with_metadata(
        climate_df,
        output_path,
        source_name="NOAA ACIS 1991-2020 Climate Normals",
        source_url="https://data.rcc-acis.org/",
        date_collected="2026-05-02",
        notes=(
            "30-year climate normals (1991-2020) from NOAA ACIS. "
            "Nearest ACIS station with normals used per city. "
            "Sunny days estimated from precipitation + latitude. "
            "Missing snow/rain values filled by IDW interpolation from nearby cities."
        ),
    )
    log.info("Saved updated climate data to %s", output_path)

    # Summary
    for col in CLIMATE_NUMERIC_COLS:
        if col in before and before[col] > 0:
            print(f"  {col}: {before[col]} missing -> {after.get(col, 0)} missing "
                  f"({before[col] - after.get(col, 0)} filled)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        backfill()
    else:
        main()
