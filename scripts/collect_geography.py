"""
Phase 2e: Collect geography and outdoor recreation data.

Sources:
  - USGS GNIS: Named summits/peaks (bulk download, ~70K features)
    https://www.usgs.gov/us-board-on-geographic-names/download-gnis-data
  - Natural Earth: 1:10m coastline shapefile for ocean distance
    https://www.naturalearthdata.com/downloads/10m-physical-vectors/
  - NPS: National Park locations (hardcoded from NPS.gov)
  - NSAA: Ski resort locations (hardcoded from public data)

Strategy:
  1. Download GNIS summit data -> count real peaks within 50mi of each city
  2. Download Natural Earth coastline -> compute ocean distance from actual shoreline
  3. Desert classification from known desert region polygons
  4. Ski resort and national park proximity from curated lists

Output: data/geography.parquet
"""

import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.download import download_file
from utilities.geo import haversine_miles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "geography"

# --- Data download URLs ---
GNIS_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/"
    "DomesticNames/DomesticNames_National_Text.zip"
)
COASTLINE_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"

# ---------------------------------------------------------------------------
# Hardcoded reference data (desert, ski resorts, national parks)
# Mountains and ocean are now sourced from real downloaded data (see above).
# ---------------------------------------------------------------------------

# Desert regions — approximate bounding circles for major US deserts
DESERT_AREAS = [
    {"name": "Sonoran", "lat": 32.5, "lon": -112.0, "radius": 200},
    {"name": "Mojave", "lat": 35.5, "lon": -116.0, "radius": 150},
    {"name": "Chihuahuan", "lat": 32.0, "lon": -106.5, "radius": 150},
    {"name": "Great Basin", "lat": 40.0, "lon": -117.0, "radius": 200},
    {"name": "Colorado Plateau", "lat": 37.5, "lon": -110.0, "radius": 120},
]

# Major ski resorts with locations
# Source: National Ski Areas Association / public resort data
# Last verified: 2026-05-02 — spot-checked 10 resorts, all within 3.4 mi of actual
SKI_RESORTS = [
    # Colorado
    ("Vail", 39.6403, -106.3742), ("Breckenridge", 39.4817, -106.0384),
    ("Aspen", 39.1911, -106.8175), ("Steamboat", 40.4572, -106.8045),
    ("Telluride", 37.9375, -107.8123), ("Winter Park", 39.8868, -105.7625),
    ("Copper Mountain", 39.5022, -106.1497), ("Keystone", 39.6069, -105.9497),
    ("Arapahoe Basin", 39.6426, -105.8719), ("Crested Butte", 38.8986, -106.9650),
    ("Loveland", 39.6800, -105.8978), ("Eldora", 39.9372, -105.5828),
    ("Wolf Creek", 37.4733, -106.7937), ("Monarch", 38.5128, -106.3322),
    ("Purgatory", 37.6303, -107.8145),
    # Utah
    ("Park City", 40.6461, -111.4980), ("Snowbird", 40.5830, -111.6538),
    ("Deer Valley", 40.6374, -111.4783), ("Alta", 40.5884, -111.6386),
    ("Brighton", 40.5980, -111.5832), ("Solitude", 40.6199, -111.5927),
    ("Snowbasin", 41.2160, -111.8569), ("Powder Mountain", 41.3797, -111.7808),
    ("Brian Head", 37.6932, -112.8498),
    # California
    ("Mammoth Mountain", 37.6308, -119.0326), ("Palisades Tahoe", 39.1968, -120.2354),
    ("Heavenly", 38.9353, -119.9401), ("Northstar", 39.2746, -120.1210),
    ("Kirkwood", 38.6850, -120.0653), ("Bear Mountain", 34.2272, -116.8600),
    ("Mountain High", 34.3722, -117.6931), ("Sugar Bowl", 39.3048, -120.3341),
    ("Big Bear", 34.2364, -116.8909), ("June Mountain", 37.7674, -119.0895),
    ("Mt Shasta", 41.3546, -122.2041),
    # Wyoming
    ("Jackson Hole", 43.5877, -110.8279), ("Grand Targhee", 43.7893, -110.9579),
    ("Snow King", 43.4771, -110.7646),
    # Montana
    ("Big Sky", 45.2837, -111.4013), ("Whitefish", 48.4930, -114.3532),
    ("Red Lodge", 45.1864, -109.3351), ("Bridger Bowl", 45.8172, -110.8973),
    ("Discovery", 46.2489, -113.2358),
    # Vermont
    ("Stowe", 44.5303, -72.7814), ("Killington", 43.6045, -72.7968),
    ("Sugarbush", 44.1362, -72.8978), ("Jay Peak", 44.9264, -72.5286),
    ("Okemo", 43.4017, -72.7172), ("Stratton", 43.1133, -72.9081),
    ("Mount Snow", 42.9600, -72.9206), ("Bolton Valley", 44.4172, -72.8500),
    ("Mad River Glen", 44.2036, -72.9158),
    # New Hampshire
    ("Loon Mountain", 44.0362, -71.6218), ("Cannon Mountain", 44.1567, -71.6981),
    ("Bretton Woods", 44.2567, -71.4628), ("Waterville Valley", 43.9653, -71.5281),
    ("Wildcat Mountain", 44.2639, -71.2378), ("Attitash", 44.0825, -71.2297),
    ("Cranmore", 44.0547, -71.1117), ("Gunstock", 43.5517, -71.3636),
    ("Sunapee", 43.3264, -72.0811), ("Ragged Mountain", 43.4744, -71.8339),
    # New York
    ("Whiteface", 44.3657, -73.9026), ("Gore Mountain", 43.6742, -74.0069),
    ("Hunter Mountain", 42.2000, -74.2317), ("Windham Mountain", 42.2969, -74.2578),
    ("Holiday Valley", 42.2667, -78.6697), ("Bristol Mountain", 42.7358, -77.4139),
    ("Greek Peak", 42.5092, -76.1478),
    # Idaho
    ("Sun Valley", 43.6974, -114.3514), ("Schweitzer", 48.3675, -116.6228),
    ("Bogus Basin", 43.7631, -116.1022), ("Brundage", 44.8567, -116.1547),
    ("Tamarack", 44.6756, -116.1064),
    # Oregon
    ("Mt Bachelor", 43.9792, -121.6886), ("Mt Hood Meadows", 45.3308, -121.6669),
    ("Timberline", 45.3311, -121.7108), ("Mt Hood Skibowl", 45.3025, -121.7711),
    ("Mt Ashland", 42.0794, -122.7181), ("Anthony Lakes", 44.9608, -118.2308),
    # Washington
    ("Crystal Mountain", 46.9347, -121.5047), ("Stevens Pass", 47.7445, -121.0890),
    ("Mission Ridge", 47.2917, -120.4003), ("Mt Baker", 48.8574, -121.6658),
    ("White Pass", 46.6375, -121.3908), ("Snoqualmie Pass", 47.4205, -121.4134),
    ("49 Degrees North", 48.3022, -117.5656),
    # New Mexico
    ("Taos", 36.5960, -105.4545), ("Santa Fe", 35.7956, -105.7927),
    ("Ski Apache", 33.3972, -105.7939), ("Angel Fire", 36.3906, -105.2842),
    ("Sandia Peak", 35.2100, -106.4372),
    # Michigan
    ("Boyne Mountain", 45.1649, -84.9332), ("Nubs Nob", 45.4683, -84.9194),
    ("Crystal Mountain MI", 44.5228, -85.9775), ("Caberfae Peaks", 44.2389, -85.7656),
    # Wisconsin
    ("Granite Peak", 44.9361, -89.6827), ("Devils Head", 43.4228, -89.7119),
    # North Carolina
    ("Sugar Mountain", 36.1166, -81.8693), ("Beech Mountain", 36.1825, -81.8786),
    ("Cataloochee", 35.5728, -83.0939), ("Appalachian Ski Mtn", 36.0925, -81.6278),
    # Pennsylvania
    ("Camelback", 41.0522, -75.3569), ("Blue Mountain PA", 40.8200, -75.9269),
    ("Seven Springs", 40.0231, -79.2975), ("Elk Mountain", 41.6894, -75.5686),
    # West Virginia
    ("Snowshoe", 38.4118, -79.9961), ("Canaan Valley", 38.9953, -79.4472),
    ("Winterplace", 37.5947, -81.0675),
    # Maine
    ("Sugarloaf", 45.0331, -70.3131), ("Sunday River", 44.4733, -70.8570),
    ("Saddleback", 44.9375, -70.5044),
    # Massachusetts
    ("Wachusett", 42.5003, -71.8856), ("Jiminy Peak", 42.5536, -73.2708),
    ("Berkshire East", 42.6247, -72.8956),
    # Connecticut
    ("Mohawk Mountain", 41.8383, -73.2831),
    # Alaska
    ("Alyeska", 60.9664, -149.0981), ("Eaglecrest", 58.3047, -134.5142),
    # Minnesota
    ("Lutsen Mountains", 47.6603, -90.7018), ("Spirit Mountain", 46.7153, -92.2211),
    # South Dakota
    ("Terry Peak", 44.3731, -103.7583),
    # Arizona
    ("Snowbowl", 35.3308, -111.7078), ("Sunrise Park", 33.9700, -109.5550),
    # Nevada
    ("Mt Rose", 39.3153, -119.8808), ("Diamond Peak", 39.2533, -119.9156),
    # Tennessee / Virginia
    ("Ober Gatlinburg", 35.7069, -83.5200), ("Massanutten", 38.4097, -78.7389),
    ("Wintergreen", 37.9314, -78.9406), ("Bryce Resort", 38.8322, -78.7531),
]

# National Parks with locations
# Source: NPS.gov
# Last verified: 2026-05-02 — spot-checked 10 parks, all within park boundaries
NATIONAL_PARKS = [
    ("Yellowstone", 44.4280, -110.5885), ("Grand Teton", 43.7904, -110.6818),
    ("Glacier", 48.7596, -113.7870), ("Yosemite", 37.8651, -119.5383),
    ("Grand Canyon", 36.1069, -112.1129), ("Zion", 37.2982, -113.0263),
    ("Rocky Mountain", 40.3428, -105.6836), ("Acadia", 44.3386, -68.2733),
    ("Great Smoky Mountains", 35.6131, -83.5532), ("Olympic", 47.8021, -123.6044),
    ("Shenandoah", 38.2928, -78.6796), ("Mount Rainier", 46.8523, -121.7603),
    ("Joshua Tree", 33.8734, -115.9010), ("Death Valley", 36.5054, -117.0794),
    ("Arches", 38.7331, -109.5925), ("Canyonlands", 38.2136, -109.9025),
    ("Bryce Canyon", 37.5930, -112.1871), ("Capitol Reef", 38.3670, -111.2615),
    ("Everglades", 25.2866, -80.8987), ("Badlands", 43.8554, -102.3397),
    ("Crater Lake", 42.8684, -122.1685), ("Redwood", 41.2132, -124.0046),
    ("Sequoia", 36.4864, -118.5658), ("Kings Canyon", 36.8879, -118.5551),
    ("Denali", 63.3333, -150.5000), ("Haleakala", 20.7204, -156.1552),
    ("Hawaii Volcanoes", 19.4194, -155.2885),
    ("Mesa Verde", 37.1838, -108.4887), ("Petrified Forest", 34.8200, -109.8892),
    ("Saguaro", 32.2967, -111.1666), ("Big Bend", 29.2498, -103.2502),
    ("Guadalupe Mountains", 31.8913, -104.8606), ("Carlsbad Caverns", 32.1479, -104.5567),
    ("Theodore Roosevelt", 46.9790, -103.4526), ("Wind Cave", 43.5724, -103.4837),
    ("Voyageurs", 48.4839, -92.8280), ("Cuyahoga Valley", 41.2808, -81.5678),
    ("Indiana Dunes", 41.6533, -87.0524), ("Mammoth Cave", 37.1862, -86.0998),
    ("Hot Springs", 34.5217, -93.0424), ("Congaree", 33.7948, -80.7821),
    ("Biscayne", 25.4824, -80.2101), ("Dry Tortugas", 24.6285, -82.8732),
    ("Virgin Islands", 18.3358, -64.7307),
    ("Pinnacles", 36.4906, -121.1825), ("Channel Islands", 34.0069, -119.7785),
    ("Lassen Volcanic", 40.4977, -121.4207),
    ("North Cascades", 48.7718, -121.2985), ("Kenai Fjords", 60.0432, -149.8161),
    ("Gates of the Arctic", 67.7833, -153.3000), ("Wrangell-St Elias", 61.0000, -142.0000),
    ("Katmai", 58.5000, -155.0000), ("Lake Clark", 60.9650, -153.4167),
    ("Kobuk Valley", 67.5500, -159.1500),
    ("Black Canyon of the Gunnison", 38.5754, -107.7416),
    ("Great Sand Dunes", 37.7916, -105.5943),
    ("White Sands", 32.7872, -106.3257),
    ("Gateway Arch", 38.6247, -90.1848),
    ("New River Gorge", 38.0660, -81.0785),
]


# ---------------------------------------------------------------------------
# Download & parse real geographic reference data
# ---------------------------------------------------------------------------

def _download_and_extract(url: str, dest_zip: Path, extract_dir: Path) -> Path:
    """Download a zip file (with caching) and extract it."""
    download_file(url, dest_zip, max_age_days=90)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(extract_dir)
        log.info("Extracted %s -> %s", dest_zip.name, extract_dir)
    return extract_dir


def load_mountain_coords() -> np.ndarray:
    """Download GNIS and return an (N, 2) array of [lat, lon] for mountain features.

    Source: USGS Geographic Names Information System (GNIS)
    Filters to features that indicate real mountains/peaks:
      - Summit features whose names contain Mountain, Mount, Mt, Peak, or Butte
      - Range features (all names)
    This excludes hills, mounds, knobs, and other minor topographic features
    that GNIS classifies as "Summit" but aren't mountains (~38K features).
    """
    zip_path = CACHE_DIR / "gnis_domestic.zip"
    extract_dir = CACHE_DIR / "gnis_domestic"
    _download_and_extract(GNIS_URL, zip_path, extract_dir)

    txt_path = extract_dir / "Text" / "DomesticNames_National.txt"
    log.info("Loading GNIS features from %s ...", txt_path.name)
    df = pd.read_csv(txt_path, sep="|", dtype=str, low_memory=False)

    # Summit features with mountain-indicating names
    summits = df[df["feature_class"] == "Summit"]
    name_lower = summits["feature_name"].str.lower()
    mountain_summits = summits[
        name_lower.str.contains("mountain")
        | name_lower.str.contains(r"^mount\b")
        | name_lower.str.contains(r"\bmt\b")
        | name_lower.str.contains("peak")
        | name_lower.str.contains("butte")
    ]

    # All Range features
    ranges = df[df["feature_class"] == "Range"]

    combined = pd.concat([mountain_summits, ranges])
    coords = combined[["prim_lat_dec", "prim_long_dec"]].astype(float).dropna().values
    log.info(
        "Loaded %d mountain features from GNIS "
        "(%d mountain/peak/butte summits + %d ranges)",
        len(coords), len(mountain_summits), len(ranges),
    )
    return coords


def load_coastline_coords() -> np.ndarray:
    """Download Natural Earth 1:10m coastline and return an (N, 2) array of [lat, lon].

    Vertices are filtered to the US bounding box (including Alaska and Hawaii).
    ~85,000 coastline vertices in the US region.
    """
    zip_path = CACHE_DIR / "ne_coastline.zip"
    extract_dir = CACHE_DIR / "ne_coastline"
    _download_and_extract(COASTLINE_URL, zip_path, extract_dir)

    shp_path = extract_dir / "ne_10m_coastline.shp"
    log.info("Loading Natural Earth coastline from %s ...", shp_path.name)
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)

    # US bounding box: lat 17-72, lon -180 to -65 (plus western Aleutians > 170)
    points = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        for lon, lat in geom.coords:
            if 17 <= lat <= 72 and (-180 <= lon <= -65 or lon > 170):
                points.append((lat, lon))

    coords = np.array(points)
    log.info("Loaded %d coastline vertices in US region", len(coords))
    return coords


# ---------------------------------------------------------------------------
# Vectorized distance helpers (pure numpy, no extra dependencies)
# ---------------------------------------------------------------------------

def _haversine_vec(lat1: float, lon1: float,
                   lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Haversine distance from one point to an array of points (miles)."""
    R = 3958.8
    rlat1 = np.radians(lat1)
    rlats = np.radians(lats)
    dlat = rlats - rlat1
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlats) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def _count_within(lat: float, lon: float,
                  ref_lats: np.ndarray, ref_lons: np.ndarray,
                  radius_miles: float) -> int:
    """Count reference points within radius_miles of (lat, lon)."""
    dists = _haversine_vec(lat, lon, ref_lats, ref_lons)
    return int(np.sum(dists < radius_miles))


def _min_distance(lat: float, lon: float,
                  ref_lats: np.ndarray, ref_lons: np.ndarray) -> float:
    """Minimum distance from (lat, lon) to any reference point (miles)."""
    dists = _haversine_vec(lat, lon, ref_lats, ref_lons)
    return float(np.min(dists))


# ---------------------------------------------------------------------------
# Main classification
# ---------------------------------------------------------------------------

# Minimum mountain features (Mountain/Mount/Peak/Butte summits + Ranges)
# within 50 miles to classify as "has_mountains".
# Threshold of 15 catches real mountain cities (Asheville=1017, Denver=378,
# Wenatchee=219) while excluding flat cities (Chicago=5, Houston=1, Miami=0).
MOUNTAIN_RADIUS_MI = 50
MOUNTAIN_THRESHOLD = 15


def load_lake_coords() -> tuple[np.ndarray, np.ndarray]:
    """Load boatable lake reference points from data/lakes.parquet.

    Returns:
        (coords, areas) where coords is an (N, 2) array of [lat, lon] and
        areas is an (N,) array of lake area in acres.  If the lakes file
        doesn't exist, logs a warning and returns empty arrays.
    """
    lakes_path = PROJECT_ROOT / "data" / "lakes.parquet"
    if not lakes_path.exists():
        log.warning("lakes.parquet not found — run scripts/collect_lakes.py first")
        return np.empty((0, 2)), np.empty(0)

    lakes = pd.read_parquet(lakes_path)
    coords = lakes[["lat", "lon"]].values
    areas = lakes["area_acres"].values
    log.info("Loaded %d lake reference points (%d unique lakes)",
             len(coords), (~lakes["is_boundary_point"]).sum())
    return coords, areas


def _nearest_lake(lat: float, lon: float,
                  lake_lats: np.ndarray, lake_lons: np.ndarray,
                  lake_areas: np.ndarray) -> tuple[float, float, int]:
    """Find nearest boatable lake distance, its area, and count within 50mi.

    Returns:
        (nearest_miles, nearest_area_acres, count_within_50mi)
    """
    if len(lake_lats) == 0:
        return 999.0, 0.0, 0

    dists = _haversine_vec(lat, lon, lake_lats, lake_lons)
    min_idx = int(np.argmin(dists))
    nearest_dist = float(dists[min_idx])
    nearest_area = float(lake_areas[min_idx])
    count_50 = int(np.sum(dists < 50))
    return nearest_dist, nearest_area, count_50


def classify_geography(cities: pd.DataFrame) -> pd.DataFrame:
    """Classify geographic features for each city using real reference data."""
    log.info("Classifying geography for %d cities...", len(cities))

    # Load real data
    mountain_coords = load_mountain_coords()
    mtn_lats, mtn_lons = mountain_coords[:, 0], mountain_coords[:, 1]

    coastline_coords = load_coastline_coords()
    coast_lats, coast_lons = coastline_coords[:, 0], coastline_coords[:, 1]

    lake_coords, lake_areas = load_lake_coords()
    lake_lats = lake_coords[:, 0] if len(lake_coords) > 0 else np.empty(0)
    lake_lons = lake_coords[:, 1] if len(lake_coords) > 0 else np.empty(0)

    # Pre-compute arrays for ski resorts and national parks
    ski_arr = np.array([(r[1], r[2]) for r in SKI_RESORTS])
    ski_lats, ski_lons = ski_arr[:, 0], ski_arr[:, 1]
    park_arr = np.array([(p[1], p[2]) for p in NATIONAL_PARKS])
    park_lats, park_lons = park_arr[:, 0], park_arr[:, 1]

    results = []
    for i, (_, city) in enumerate(cities.iterrows()):
        lat, lon = city["lat"], city["lon"]

        # --- Ocean proximity (Natural Earth coastline) ---
        ocean_dist = _min_distance(lat, lon, coast_lats, coast_lons)

        # --- Mountain proximity (GNIS mountain/peak/butte summits + ranges) ---
        # Alaska and Hawaii: always mountainous. Alaska has the Alaska Range,
        # Chugach, Brooks Range; Hawaii has Mauna Kea, Mauna Loa, Haleakala.
        # Hawaiian peaks use native names that don't match English keywords,
        # so GNIS name filtering undercounts them.
        if city["state"] in ("AK", "HI"):
            near_mountain = True
            mountain_count = _count_within(lat, lon, mtn_lats, mtn_lons,
                                           MOUNTAIN_RADIUS_MI)
            mountain_dist = 0.0
        else:
            mountain_count = _count_within(lat, lon, mtn_lats, mtn_lons,
                                           MOUNTAIN_RADIUS_MI)
            near_mountain = mountain_count >= MOUNTAIN_THRESHOLD
            if near_mountain:
                mountain_dist = 0.0
            else:
                mountain_dist = _min_distance(lat, lon, mtn_lats, mtn_lons)

        # --- Desert proximity ---
        in_desert = False
        for desert in DESERT_AREAS:
            d = haversine_miles(lat, lon, desert["lat"], desert["lon"])
            if d < desert["radius"]:
                in_desert = True
                break

        # --- Ski resort distance ---
        ski_dist = _min_distance(lat, lon, ski_lats, ski_lons)

        # --- National parks within 100mi ---
        parks_100 = _count_within(lat, lon, park_lats, park_lons, 100)

        # --- Boatable lakes (NHD: lakes/reservoirs >= 300 acres) ---
        lake_dist, lake_area, lakes_50 = _nearest_lake(
            lat, lon, lake_lats, lake_lons, lake_areas,
        )

        results.append({
            "city_id": city["city_id"],
            "name": city["name"],
            "state": city["state"],
            "has_ocean": ocean_dist < 50,
            "ocean_distance_miles": round(ocean_dist, 1),
            "has_mountains": near_mountain,
            "mountain_distance_miles": round(min(mountain_dist, 999), 1),
            "mountain_features_within_50mi": mountain_count,
            "has_desert": in_desert,
            "has_lakes": lake_dist < 50,
            "nearest_boatable_lake_miles": round(lake_dist, 1),
            "nearest_lake_area_acres": round(lake_area, 0),
            "boatable_lakes_within_50mi": lakes_50,
            "ski_resort_distance_miles": round(ski_dist, 1),
            "national_parks_within_100mi": parks_100,
        })

        if (i + 1) % 200 == 0:
            log.info("  Processed %d / %d cities", i + 1, len(cities))

    return pd.DataFrame(results)


def main():
    cities = pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")
    output_path = PROJECT_ROOT / "data" / "geography.parquet"

    geo_df = classify_geography(cities)

    write_parquet_with_metadata(
        geo_df,
        output_path,
        source_name=(
            "USGS GNIS summits + Natural Earth coastline + NHD lakes + NPS/NSAA"
        ),
        source_url="https://www.usgs.gov/us-board-on-geographic-names/download-gnis-data",
        date_collected="2026-05-02",
        notes=(
            "Mountains: USGS GNIS features named Mountain/Mount/Peak/Butte + Ranges (~38K). "
            "has_mountains = True when >= 15 mountain features within 50mi (AK always True). "
            "Ocean: Natural Earth 1:10m coastline (~85K vertices in US region). "
            "Lakes: USGS NHD waterbodies >= 300 acres (boatable size). "
            "nearest_boatable_lake_miles = distance to nearest lake/reservoir >= 300 acres. "
            "Desert: approximate bounding circles for major US deserts. "
            "Ski resorts from NSAA public data. National parks from NPS.gov. "
            "Outdoor recreation (trails, parks, campgrounds) in outdoor_recreation.parquet."
        ),
    )

    print(f"\nGeography data: {len(geo_df)} cities")
    print(geo_df[["name", "state", "has_ocean", "ocean_distance_miles",
                   "has_mountains", "mountain_features_within_50mi",
                   "nearest_boatable_lake_miles", "boatable_lakes_within_50mi",
                   "ski_resort_distance_miles", "national_parks_within_100mi"
                   ]].head(20).to_string())

    # Summary stats
    print(f"\nOcean cities (<50mi): {geo_df['has_ocean'].sum()}")
    print(f"Mountain cities (>={MOUNTAIN_THRESHOLD} features in {MOUNTAIN_RADIUS_MI}mi): "
          f"{geo_df['has_mountains'].sum()}")
    print(f"Desert cities: {geo_df['has_desert'].sum()}")
    print(f"Lake cities (<50mi to boatable lake): {geo_df['has_lakes'].sum()}")
    print(f"Lake distance stats:")
    print(f"  Median: {geo_df['nearest_boatable_lake_miles'].median():.1f} mi")
    print(f"  Mean:   {geo_df['nearest_boatable_lake_miles'].mean():.1f} mi")
    print(f"  Max:    {geo_df['nearest_boatable_lake_miles'].max():.1f} mi")

    # Spot-check key cities (especially lake-relevant ones)
    for name in ["Anchorage", "Nashville", "Knoxville", "Sacramento",
                 "Salt Lake", "Lake Havasu", "Reno", "Austin",
                 "Chicago", "Milwaukee", "Denver", "Miami", "Honolulu"]:
        row = geo_df[geo_df["name"].str.contains(name, case=False)]
        if len(row):
            r = row.iloc[0]
            print(f"\n  {r['name']}, {r['state']}: "
                  f"ocean={r['ocean_distance_miles']}mi, "
                  f"mtns={r['has_mountains']}, "
                  f"lake={r['nearest_boatable_lake_miles']}mi "
                  f"({r['nearest_lake_area_acres']:.0f}ac, "
                  f"{r['boatable_lakes_within_50mi']} within 50mi)")


if __name__ == "__main__":
    main()
