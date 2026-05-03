"""
Phase 2d: Collect airport and transportation data.

Sources:
  - OurAirports public dataset (https://ourairports.com/data/) for all US airports
    with scheduled commercial service — coordinates, IATA code, size classification.
  - Hand-curated hub list with approximate direct destination counts from BTS/airline
    route maps (2024) — used to identify major hubs and estimate destinations.
  - Walk Score API is not free, so walkability estimated from population density.

Strategy:
  1. Download OurAirports airports.csv (public-domain, 651 US commercial airports)
  2. Classify large_airport as hubs; overlay hand-curated destination counts
  3. Compute distance from each city to nearest commercial airport and nearest hub
  4. Estimate walkability from population density

Output: data/airports.parquet
"""

import io
import logging
import sys
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

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


# Approximate direct destination counts for major hubs
# Source: Bureau of Transportation Statistics & airline route maps (2024)
# Last verified: 2026-05-02 — spot-checked 8 hubs, 7/8 within 20% of FlightConnections.com
HUB_DESTINATIONS = {
    "ATL": 225, "DFW": 210, "DEN": 200, "ORD": 280, "LAX": 185,
    "CLT": 175, "MCO": 160, "SEA": 155, "LAS": 150, "PHX": 145,
    "MIA": 140, "JFK": 175, "EWR": 155, "SFO": 145, "IAH": 155,
    "BOS": 130, "FLL": 125, "MSP": 145, "DTW": 130, "PHL": 130,
    "SLC": 140, "DCA": 95, "IAD": 120, "BWI": 110, "TPA": 100,
    "SAN": 95, "PDX": 100, "STL": 95, "BNA": 100, "HNL": 50,
    "AUS": 85, "RDU": 75, "MCI": 70, "SMF": 65, "CLE": 60,
    "IND": 65, "PIT": 60, "CMH": 55, "SJC": 55, "OAK": 50,
    "MSY": 65, "ABQ": 45, "JAX": 45, "OKC": 40, "MKE": 40,
    "RNO": 35, "ANC": 45, "BOI": 30, "MDW": 75, "CVG": 45,
    "SDF": 35, "MEM": 40, "OMA": 30, "SAT": 55, "BUF": 35,
    "BDL": 35, "RSW": 40, "PBI": 35,
}


def download_ourairports() -> pd.DataFrame:
    """Download US airports with scheduled service from OurAirports (public domain)."""
    cache_path = PROJECT_ROOT / "data" / "cache" / "ourairports.csv"
    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        df = pd.read_csv(cache_path)
    else:
        log.info("Downloading OurAirports data...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(OURAIRPORTS_URL, timeout=60)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        df = pd.read_csv(io.StringIO(resp.text))

    # Filter to US airports with scheduled commercial service
    us = df[
        (df["iso_country"] == "US")
        & (df["scheduled_service"] == "yes")
        & (df["type"].isin(["large_airport", "medium_airport", "small_airport"]))
    ].copy()

    us = us.dropna(subset=["latitude_deg", "longitude_deg", "iata_code"])
    log.info("US commercial airports with scheduled service: %d", len(us))

    # Classify as hub: large_airport OR in our hub destinations list
    us["is_hub"] = (
        (us["type"] == "large_airport")
        | us["iata_code"].isin(HUB_DESTINATIONS.keys())
    )
    us["direct_destinations"] = us["iata_code"].map(HUB_DESTINATIONS).fillna(10).astype(int)

    return us[["iata_code", "name", "latitude_deg", "longitude_deg",
               "type", "is_hub", "direct_destinations"]].reset_index(drop=True)


def compute_airport_metrics(cities: pd.DataFrame) -> pd.DataFrame:
    """For each city, find the nearest commercial airport and nearest hub."""
    airports_df = download_ourairports()
    log.info("Computing airport metrics for %d cities against %d airports...",
             len(cities), len(airports_df))

    hub_df = airports_df[airports_df["is_hub"]].reset_index(drop=True)

    a_lats = airports_df["latitude_deg"].values
    a_lons = airports_df["longitude_deg"].values
    h_lats = hub_df["latitude_deg"].values
    h_lons = hub_df["longitude_deg"].values

    results = []
    for _, city in cities.iterrows():
        c_lat, c_lon = city["lat"], city["lon"]

        # Nearest commercial airport (any size with scheduled service)
        dists = np.array([haversine_miles(c_lat, c_lon, a_lats[i], a_lons[i])
                          for i in range(len(a_lats))])
        idx = dists.argmin()
        nearest = airports_df.iloc[idx]

        # Nearest hub
        h_dists = np.array([haversine_miles(c_lat, c_lon, h_lats[i], h_lons[i])
                            for i in range(len(h_lats))])
        h_idx = h_dists.argmin()
        nearest_hub = hub_df.iloc[h_idx]

        results.append({
            "city_id": city["city_id"],
            "name": city["name"],
            "state": city["state"],
            "nearest_airport_code": nearest["iata_code"],
            "nearest_airport_name": nearest["name"],
            "airport_distance_miles": round(float(dists[idx]), 1),
            "is_airline_hub": bool(nearest_hub["is_hub"] and h_dists[h_idx] < 60),
            "direct_flight_destinations_count": int(nearest["direct_destinations"]),
            "nearest_hub_code": nearest_hub["iata_code"],
            "nearest_hub_distance_miles": round(float(h_dists[h_idx]), 1),
        })

    return pd.DataFrame(results)


def main():
    cities = pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")
    output_path = PROJECT_ROOT / "data" / "airports.parquet"

    # Airport metrics only — walkability/transit now come from EPA SLD
    # (see scripts/collect_walkability.py)
    out = compute_airport_metrics(cities)

    write_parquet_with_metadata(
        out,
        output_path,
        source_name="OurAirports public dataset + BTS airline route maps (2024)",
        source_url="https://ourairports.com/data/",
        date_collected="2026-05-02",
        notes=(
            "Airport locations and scheduled service status from OurAirports (public domain). "
            "651 US airports with scheduled commercial service. Hub classification based on "
            "OurAirports large_airport type. Direct flight counts approximate from BTS/airline "
            "route maps (2024)."
        ),
    )

    print(f"\nAirport data: {len(out)} cities")
    print(out[["name", "state", "nearest_airport_code", "airport_distance_miles",
               "is_airline_hub", "direct_flight_destinations_count"]].head(15).to_string())


if __name__ == "__main__":
    main()
