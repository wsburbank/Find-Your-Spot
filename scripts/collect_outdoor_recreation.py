"""
Phase 2f: Collect real outdoor recreation data from public sources.

Replaces estimated outdoor recreation fields in geography.parquet with real
data from authoritative public sources.

Sources:
  - State parks: PAD-US 4.1 via ArcGIS REST API (USGS)
  - Campgrounds: USFS EDW recreation sites + NPS API campgrounds
  - Hiking trails: USFS EDW National Forest trails + NPS Public Trails
  - Mountain biking trails: USFS EDW trails (bicycle-managed) + NPS (bike-allowed)
  - Rock climbing areas: OpenBeta GraphQL API

Output: data/outdoor_recreation.parquet
  Columns: city_id, state_parks_nearby, camping_areas_count,
           hiking_trails_count, mountain_biking_trails, rock_climbing_areas_nearby
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
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "outdoor_recreation"

# Radius in miles for proximity counts
RADIUS_STATE_PARKS = 50
RADIUS_CAMPGROUNDS = 60
RADIUS_TRAILS = 50
RADIUS_CLIMBING = 75

# ArcGIS REST pagination limit
ARCGIS_PAGE_SIZE = 2000


# ---------------------------------------------------------------------------
# 1. PAD-US State Parks
# ---------------------------------------------------------------------------

PADUS_URL = (
    "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/"
    "Fee_Managers_PADUS/FeatureServer/0/query"
)


def fetch_state_parks() -> pd.DataFrame:
    """Fetch US state park locations from PAD-US ArcGIS REST API.

    Returns DataFrame with columns: name, state, lat, lon, acres.
    Parks are deduplicated (multi-parcel parks merged by name+state,
    keeping the centroid of the largest parcel).
    """
    cache_path = RAW_DIR / "padus_state_parks.parquet"
    if cache_path.exists():
        log.info("Loading cached state parks from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Fetching state parks from PAD-US ArcGIS REST API...")
    all_features = []
    offset = 0

    while True:
        params = {
            "where": "Des_Tp='SP'",
            "outFields": "Unit_Nm,State_Nm,GIS_Acres",
            "returnGeometry": "false",
            "returnCentroid": "true",
            "outSR": "4326",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": ARCGIS_PAGE_SIZE,
        }
        resp = requests.get(PADUS_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        log.info("  Fetched %d features (offset %d)", len(features), offset)
        offset += ARCGIS_PAGE_SIZE

        if len(features) < ARCGIS_PAGE_SIZE:
            break
        time.sleep(0.5)

    log.info("Total state park parcels fetched: %d", len(all_features))

    rows = []
    for f in all_features:
        attrs = f.get("attributes", {})
        centroid = f.get("centroid", {})
        if not centroid or centroid.get("x") is None:
            continue
        rows.append({
            "name": attrs.get("Unit_Nm", ""),
            "state": attrs.get("State_Nm", ""),
            "lat": centroid["y"],
            "lon": centroid["x"],
            "acres": attrs.get("GIS_Acres", 0) or 0,
        })

    df = pd.DataFrame(rows)

    # Deduplicate: keep largest parcel per park name + state
    df = (
        df.sort_values("acres", ascending=False)
        .drop_duplicates(subset=["name", "state"], keep="first")
        .reset_index(drop=True)
    )
    log.info("Unique state parks after dedup: %d", len(df))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. USFS EDW Recreation Sites (campgrounds & trailheads)
# ---------------------------------------------------------------------------

USFS_REC_SITES_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/"
    "EDW_RecInfraRecreationSites_02/MapServer/0/query"
)

CAMPGROUND_TYPES = ("CAMPGROUND", "CAMPING AREA", "GROUP CAMPGROUND", "HORSE CAMP")


def fetch_usfs_campgrounds() -> pd.DataFrame:
    """Fetch campground locations from USFS EDW recreation sites.

    Returns DataFrame with columns: name, lat, lon, site_type.
    """
    cache_path = RAW_DIR / "usfs_campgrounds.parquet"
    if cache_path.exists():
        log.info("Loading cached USFS campgrounds from %s", cache_path)
        return pd.read_parquet(cache_path)

    type_list = ",".join(f"'{t}'" for t in CAMPGROUND_TYPES)
    where = f"SITE_TYPE IN ({type_list})"

    log.info("Fetching USFS campgrounds from EDW...")
    all_features = _paginate_arcgis(USFS_REC_SITES_URL, where,
                                     "SITE_NAME,SITE_TYPE", return_geometry=True)

    rows = []
    for f in all_features:
        attrs = f.get("attributes", {})
        geom = f.get("geometry", {})
        if geom.get("x") is None:
            continue
        rows.append({
            "name": attrs.get("SITE_NAME", ""),
            "lat": geom["y"],
            "lon": geom["x"],
            "site_type": attrs.get("SITE_TYPE", ""),
        })

    df = pd.DataFrame(rows)
    log.info("USFS campgrounds: %d", len(df))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. NPS API Campgrounds
# ---------------------------------------------------------------------------

NPS_API_URL = "https://developer.nps.gov/api/v1/campgrounds"
NPS_API_KEY = "DEMO_KEY"


def fetch_nps_campgrounds() -> pd.DataFrame:
    """Fetch campground locations from NPS API.

    Returns DataFrame with columns: name, lat, lon, park_code.
    """
    cache_path = RAW_DIR / "nps_campgrounds.parquet"
    if cache_path.exists():
        log.info("Loading cached NPS campgrounds from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Fetching campgrounds from NPS API...")
    all_results = []
    start = 0
    limit = 50

    while True:
        params = {
            "api_key": NPS_API_KEY,
            "limit": limit,
            "start": start,
        }
        for attempt in range(3):
            resp = requests.get(NPS_API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt * 10
                log.warning("  Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            log.warning("  NPS API rate limit exceeded — returning %d campgrounds collected so far", len(all_results))
            break

        data = resp.json()
        results = data.get("data", [])
        if not results:
            break

        all_results.extend(results)
        total = int(data.get("total", 0))
        start += limit
        log.info("  Fetched %d/%d campgrounds", len(all_results), total)

        if start >= total:
            break
        time.sleep(1.5)

    rows = []
    for cg in all_results:
        lat = cg.get("latitude")
        lon = cg.get("longitude")
        if not lat or not lon:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (ValueError, TypeError):
            continue
        if lat == 0 and lon == 0:
            continue
        rows.append({
            "name": cg.get("name", ""),
            "lat": lat,
            "lon": lon,
            "park_code": cg.get("parkCode", ""),
        })

    df = pd.DataFrame(rows)
    log.info("NPS campgrounds with coordinates: %d", len(df))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 4. USFS EDW Trails (hiking & mountain biking)
# ---------------------------------------------------------------------------

USFS_TRAILS_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/"
    "EDW_TrailNFSPublish_01/MapServer/0/query"
)


def fetch_usfs_trails() -> pd.DataFrame:
    """Fetch trail midpoint locations from USFS EDW.

    Returns DataFrame with columns: name, lat, lon, is_hiking, is_biking.

    The USFS trail dataset uses date ranges (e.g. '01/01-12/31') in the
    hiker_pedestrian_managed and bicycle_managed fields.  A non-null,
    non-'N/A' value means that activity is allowed on the trail.
    """
    cache_path = RAW_DIR / "usfs_trails.parquet"
    if cache_path.exists():
        log.info("Loading cached USFS trails from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Fetching USFS trails from EDW (this may take a few minutes)...")
    # Only land trails (TERRA) — skip water and snow trails
    where = "trail_type='TERRA'"
    fields = "trail_name,hiker_pedestrian_managed,bicycle_managed"

    all_features = _paginate_arcgis(
        USFS_TRAILS_URL, where, fields,
        return_geometry=True, out_sr="4326", delay=0.5,
    )

    rows = []
    for f in all_features:
        attrs = f.get("attributes", {})
        geom = f.get("geometry", {})

        # Extract midpoint of trail polyline
        paths = geom.get("paths", [])
        if not paths or not paths[0]:
            continue

        # Use midpoint of first path segment
        coords = paths[0]
        mid_idx = len(coords) // 2
        lon, lat = coords[mid_idx][0], coords[mid_idx][1]

        hiker = attrs.get("hiker_pedestrian_managed")
        bike = attrs.get("bicycle_managed")

        # Non-null, non-'N/A' means the activity is managed (allowed)
        is_hiking = bool(hiker and hiker != "N/A")
        is_biking = bool(bike and bike != "N/A")

        rows.append({
            "name": attrs.get("trail_name", ""),
            "lat": lat,
            "lon": lon,
            "is_hiking": is_hiking,
            "is_biking": is_biking,
        })

    df = pd.DataFrame(rows)
    log.info("USFS trails: %d total, %d hiking, %d biking",
             len(df), df["is_hiking"].sum(), df["is_biking"].sum())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 4b. NPS Public Trails (national park trails)
# ---------------------------------------------------------------------------

NPS_TRAILS_URL = (
    "https://mapservices.nps.gov/arcgis/rest/services/NationalDatasets/"
    "NPS_Public_Trails/FeatureServer/0/query"
)

# TRLUSE values that indicate hiking or biking
_HIKING_KEYWORDS = ("Hiker", "Pedestrian", "Hike", "Pack")
_BIKING_KEYWORDS = ("Bicycle",)


def fetch_nps_trails() -> pd.DataFrame:
    """Fetch trail locations from NPS Public Trails ArcGIS FeatureServer.

    Returns DataFrame with columns: name, lat, lon, is_hiking, is_biking.
    Covers ~31K trail segments across all NPS units (national parks,
    monuments, recreation areas, seashores, etc.).
    """
    cache_path = RAW_DIR / "nps_trails.parquet"
    if cache_path.exists():
        log.info("Loading cached NPS trails from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Fetching NPS Public Trails (this may take a few minutes)...")
    all_features = _paginate_arcgis(
        NPS_TRAILS_URL,
        where="1=1",
        out_fields="TRLNAME,TRLUSE",
        return_geometry=True,
        out_sr="4326",
        delay=0.5,
    )

    rows = []
    for f in all_features:
        attrs = f.get("attributes", {})
        geom = f.get("geometry", {})

        paths = geom.get("paths", [])
        if not paths or not paths[0]:
            continue

        coords = paths[0]
        mid_idx = len(coords) // 2
        lon, lat = coords[mid_idx][0], coords[mid_idx][1]

        trluse = attrs.get("TRLUSE") or ""
        is_hiking = any(kw in trluse for kw in _HIKING_KEYWORDS)
        is_biking = any(kw in trluse for kw in _BIKING_KEYWORDS)

        # If TRLUSE is Unknown or empty, count as hiking (most NPS trails are)
        if not is_hiking and not is_biking:
            is_hiking = True

        rows.append({
            "name": attrs.get("TRLNAME", ""),
            "lat": lat,
            "lon": lon,
            "is_hiking": is_hiking,
            "is_biking": is_biking,
        })

    df = pd.DataFrame(rows)
    log.info("NPS trails: %d total, %d hiking, %d biking",
             len(df), df["is_hiking"].sum(), df["is_biking"].sum())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 5. OpenBeta Climbing Areas
# ---------------------------------------------------------------------------

OPENBETA_URL = "https://api.openbeta.io"


def fetch_climbing_areas() -> pd.DataFrame:
    """Fetch rock climbing area locations from OpenBeta GraphQL API.

    Queries state-level sub-areas under USA to get ~900 climbing regions
    with coordinates and route counts.

    Returns DataFrame with columns: name, state_name, lat, lon, total_climbs.
    """
    cache_path = RAW_DIR / "openbeta_climbing.parquet"
    if cache_path.exists():
        log.info("Loading cached climbing areas from %s", cache_path)
        return pd.read_parquet(cache_path)

    log.info("Fetching climbing areas from OpenBeta...")

    # Fetch USA -> states -> sub-areas in a single query.
    # This avoids per-state queries that are unreliable (502s).
    query = """
    {
      areas(filter: {leaf_status: {isLeaf: false}, area_name: {match: "USA"}}) {
        areaName
        children {
          areaName
          totalClimbs
          metadata { lat lng }
          children {
            areaName
            totalClimbs
            metadata { lat lng }
          }
        }
      }
    }
    """
    resp = requests.post(OPENBETA_URL, json={"query": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    usa_areas = [a for a in data["data"]["areas"] if a["areaName"] == "USA"]
    if not usa_areas:
        log.error("Could not find USA in OpenBeta")
        return pd.DataFrame()

    states = usa_areas[0]["children"]
    log.info("Found %d state-level climbing areas", len(states))

    all_areas = []
    for state in states:
        state_name = state["areaName"]
        if state_name.startswith("Test"):
            continue

        children = state.get("children") or []

        if not children:
            # Use state-level entry as a single area
            if state["metadata"]["lat"] and state["metadata"]["lng"]:
                all_areas.append({
                    "name": state_name,
                    "state_name": state_name,
                    "lat": state["metadata"]["lat"],
                    "lon": state["metadata"]["lng"],
                    "total_climbs": state["totalClimbs"],
                })
        else:
            for child in children:
                lat = child["metadata"]["lat"]
                lng = child["metadata"]["lng"]
                if not lat or not lng:
                    continue
                all_areas.append({
                    "name": child["areaName"],
                    "state_name": state_name,
                    "lat": lat,
                    "lon": lng,
                    "total_climbs": child["totalClimbs"],
                })

    df = pd.DataFrame(all_areas)
    log.info("Climbing areas: %d across %d states",
             len(df), df["state_name"].nunique())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Shared ArcGIS pagination helper
# ---------------------------------------------------------------------------

def _paginate_arcgis(
    url: str,
    where: str,
    out_fields: str,
    return_geometry: bool = False,
    out_sr: str | None = None,
    delay: float = 0.3,
) -> list[dict]:
    """Page through an ArcGIS REST FeatureServer/MapServer query."""
    all_features = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": str(return_geometry).lower(),
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": ARCGIS_PAGE_SIZE,
        }
        if out_sr:
            params["outSR"] = out_sr

        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            log.error("ArcGIS error at offset %d: %s", offset, data["error"])
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        log.info("  Fetched %d features (offset %d, total so far: %d)",
                 len(features), offset, len(all_features))
        offset += ARCGIS_PAGE_SIZE

        if len(features) < ARCGIS_PAGE_SIZE:
            break
        time.sleep(delay)

    return all_features


# ---------------------------------------------------------------------------
# 6. Compute per-city proximity counts
# ---------------------------------------------------------------------------

def _vectorized_haversine(lat1: float, lon1: float,
                          lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Haversine from one point to arrays of points. Returns miles."""
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)

    dlat = lats_r - lat1_r
    dlon = lons_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1_r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def compute_city_counts(
    cities: pd.DataFrame,
    state_parks: pd.DataFrame,
    campgrounds: pd.DataFrame,
    trails: pd.DataFrame,
    climbing: pd.DataFrame,
) -> pd.DataFrame:
    """For each city, count nearby recreation features using haversine distance.

    Returns DataFrame with city_id and the five outdoor recreation columns.
    """
    log.info("Computing per-city proximity counts for %d cities...", len(cities))

    # Pre-convert to numpy for vectorized haversine
    sp_lats = state_parks["lat"].values if len(state_parks) else np.array([])
    sp_lons = state_parks["lon"].values if len(state_parks) else np.array([])

    cg_lats = campgrounds["lat"].values if len(campgrounds) else np.array([])
    cg_lons = campgrounds["lon"].values if len(campgrounds) else np.array([])

    # Split trails into hiking and biking
    hiking_trails = trails[trails["is_hiking"]] if len(trails) else pd.DataFrame()
    biking_trails = trails[trails["is_biking"]] if len(trails) else pd.DataFrame()
    ht_lats = hiking_trails["lat"].values if len(hiking_trails) else np.array([])
    ht_lons = hiking_trails["lon"].values if len(hiking_trails) else np.array([])
    bt_lats = biking_trails["lat"].values if len(biking_trails) else np.array([])
    bt_lons = biking_trails["lon"].values if len(biking_trails) else np.array([])

    cl_lats = climbing["lat"].values if len(climbing) else np.array([])
    cl_lons = climbing["lon"].values if len(climbing) else np.array([])
    cl_routes = climbing["total_climbs"].values if len(climbing) else np.array([])

    results = []
    for i, (_, city) in enumerate(cities.iterrows()):
        lat, lon = city["lat"], city["lon"]

        # State parks within radius
        if len(sp_lats):
            sp_dists = _vectorized_haversine(lat, lon, sp_lats, sp_lons)
            n_state_parks = int(np.sum(sp_dists <= RADIUS_STATE_PARKS))
        else:
            n_state_parks = 0

        # Campgrounds within radius
        if len(cg_lats):
            cg_dists = _vectorized_haversine(lat, lon, cg_lats, cg_lons)
            n_campgrounds = int(np.sum(cg_dists <= RADIUS_CAMPGROUNDS))
        else:
            n_campgrounds = 0

        # Hiking trails within radius
        if len(ht_lats):
            ht_dists = _vectorized_haversine(lat, lon, ht_lats, ht_lons)
            n_hiking = int(np.sum(ht_dists <= RADIUS_TRAILS))
        else:
            n_hiking = 0

        # MTB trails within radius
        if len(bt_lats):
            bt_dists = _vectorized_haversine(lat, lon, bt_lats, bt_lons)
            n_biking = int(np.sum(bt_dists <= RADIUS_TRAILS))
        else:
            n_biking = 0

        # Climbing routes within radius (sum of routes from all areas in range)
        if len(cl_lats):
            cl_dists = _vectorized_haversine(lat, lon, cl_lats, cl_lons)
            cl_mask = cl_dists <= RADIUS_CLIMBING
            n_climbing_areas = int(np.sum(cl_mask))
            n_climbing_routes = int(np.sum(cl_routes[cl_mask]))
        else:
            n_climbing_areas = 0
            n_climbing_routes = 0

        results.append({
            "city_id": city["city_id"],
            "state_parks_nearby": n_state_parks,
            "camping_areas_count": n_campgrounds,
            "hiking_trails_count": n_hiking,
            "mountain_biking_trails": n_biking,
            "rock_climbing_areas_nearby": n_climbing_areas,
            "climbing_routes_nearby": n_climbing_routes,
        })

        if (i + 1) % 100 == 0:
            log.info("  Processed %d/%d cities", i + 1, len(cities))

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Loaded %d cities from master list", len(cities))

    # Collect from all sources
    state_parks = fetch_state_parks()
    usfs_camps = fetch_usfs_campgrounds()
    nps_camps = fetch_nps_campgrounds()
    usfs_trails = fetch_usfs_trails()
    nps_trails = fetch_nps_trails()
    climbing = fetch_climbing_areas()

    # Combine USFS + NPS campgrounds
    camp_parts = [usfs_camps[["name", "lat", "lon"]]]
    if len(nps_camps) and "lat" in nps_camps.columns:
        camp_parts.append(nps_camps[["name", "lat", "lon"]])
    campgrounds = pd.concat(camp_parts, ignore_index=True)
    log.info("Combined campgrounds: %d (USFS: %d, NPS: %d)",
             len(campgrounds), len(usfs_camps), len(nps_camps))

    # Combine USFS + NPS trails
    trail_cols = ["name", "lat", "lon", "is_hiking", "is_biking"]
    trail_parts = [usfs_trails[trail_cols]]
    if len(nps_trails):
        trail_parts.append(nps_trails[trail_cols])
    trails = pd.concat(trail_parts, ignore_index=True)
    log.info("Combined trails: %d (USFS: %d, NPS: %d)",
             len(trails), len(usfs_trails), len(nps_trails))

    # Compute per-city counts
    rec_df = compute_city_counts(cities, state_parks, campgrounds, trails, climbing)

    # Save outdoor recreation parquet
    output_path = DATA_DIR / "outdoor_recreation.parquet"
    write_parquet_with_metadata(
        rec_df,
        output_path,
        source_name="PAD-US/USFS EDW/NPS/OpenBeta outdoor recreation data",
        source_url="https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/Fee_Managers_PADUS/FeatureServer",
        notes=(
            "State parks: PAD-US 4.1 via ArcGIS REST (USGS, 2024). "
            f"Campgrounds: USFS EDW recreation sites ({len(usfs_camps)}) + "
            f"NPS API ({len(nps_camps)}). "
            f"Trails: USFS EDW ({len(usfs_trails)}) + NPS Public Trails ({len(nps_trails)}), "
            f"combined {len(trails)} segments "
            f"({trails['is_hiking'].sum()} hiking, {trails['is_biking'].sum()} biking). "
            f"Rock climbing: OpenBeta ({len(climbing)} areas). "
            f"Proximity radii: state parks {RADIUS_STATE_PARKS}mi, "
            f"campgrounds {RADIUS_CAMPGROUNDS}mi, trails {RADIUS_TRAILS}mi, "
            f"climbing {RADIUS_CLIMBING}mi."
        ),
    )

    # Print summary
    print(f"\n{'='*60}")
    print("OUTDOOR RECREATION DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Cities: {len(rec_df)}")
    print(f"\nRaw data collected:")
    print(f"  State parks (PAD-US):     {len(state_parks):,}")
    print(f"  USFS campgrounds:         {len(usfs_camps):,}")
    print(f"  NPS campgrounds:          {len(nps_camps):,}")
    print(f"  USFS trail segments:      {len(usfs_trails):,}")
    print(f"  NPS trail segments:       {len(nps_trails):,}")
    print(f"  Combined trails:          {len(trails):,}")
    print(f"    Hiking trails:          {trails['is_hiking'].sum():,}")
    print(f"    Biking trails:          {trails['is_biking'].sum():,}")
    print(f"  Climbing areas:           {len(climbing):,}")
    print(f"\nPer-city counts (radius-based):")
    for col in ["state_parks_nearby", "camping_areas_count", "hiking_trails_count",
                "mountain_biking_trails", "rock_climbing_areas_nearby"]:
        vals = rec_df[col]
        print(f"  {col:30s}: min={vals.min():4d}  median={vals.median():6.0f}  "
              f"max={vals.max():5d}  mean={vals.mean():6.1f}")

    # Show top 10 cities by each metric
    print(f"\nTop 10 cities by hiking trails:")
    top = rec_df.merge(cities[["city_id", "name", "state"]], on="city_id")
    for _, r in top.nlargest(10, "hiking_trails_count").iterrows():
        print(f"  {r['name']}, {r['state']}: {r['hiking_trails_count']} trails")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
