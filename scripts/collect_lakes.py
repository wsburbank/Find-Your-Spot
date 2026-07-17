"""
Collect boatable lake data from the USGS National Hydrography Dataset (NHD).

Source: USGS NHD Waterbody - Large Scale (MapServer layer 12)
  https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12

Strategy:
  1. Query NHD for all waterbodies >= 300 acres (1.214 km²) of type
     LakePond (390) or Reservoir (436) — these are boatable water bodies.
  2. For medium lakes (< 100 km²): fetch simplified polygons, compute centroids.
  3. For large lakes (>= 100 km²): fetch full polygons, sample boundary points
     every ~5 miles for accurate shoreline distance calculation.
  4. Combine into a single reference point array and save as data/lakes.parquet.

The resulting dataset supports computing distance from any city to the nearest
lake large enough for ski boating, wakeboarding, etc.

300 acres (1.214 km²) is a reasonable minimum for motorboat recreation —
roughly 0.5 mi across, enough room for towing and maneuvering.

Output: data/lakes.parquet
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NHD_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"
)

# 300 acres in km²
MIN_AREA_SQKM = 1.214

# Large lake threshold — these get boundary sampling instead of just centroids
LARGE_LAKE_SQKM = 100.0

# FTYPE codes for boatable water bodies
FTYPE_LAKEPOND = 390
FTYPE_RESERVOIR = 436

# Boundary point sampling interval for large lakes (degrees, ~5 miles at mid-latitudes)
SAMPLE_INTERVAL_DEG = 0.07

# Page size for API queries (NHD max is 2000)
PAGE_SIZE = 2000

# US bounding box for filtering (same as used in collect_geography.py)
US_BBOX = "-180,17,-65,72"

CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "lakes"


def _query_nhd_paged(
    where: str,
    out_fields: str,
    return_geometry: bool = True,
    max_offset: float | None = None,
) -> list[dict]:
    """Query NHD waterbody layer with pagination.

    Args:
        where: SQL WHERE clause.
        out_fields: Comma-separated field names.
        return_geometry: Whether to include polygon geometry.
        max_offset: Geometry simplification tolerance (degrees).
            Higher values = more simplified = less data.

    Returns:
        List of feature dicts from the API.
    """
    all_features = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": str(return_geometry).lower(),
            "outSR": "4326",
            "resultRecordCount": str(PAGE_SIZE),
            "resultOffset": str(offset),
            "geometryType": "esriGeometryEnvelope",
            "geometry": US_BBOX,
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "f": "json",
        }
        if max_offset is not None:
            params["maxAllowableOffset"] = str(max_offset)

        for attempt in range(3):
            try:
                resp = requests.get(NHD_URL, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 2:
                    log.warning("Retry %d for offset %d: %s", attempt + 1, offset, e)
                    time.sleep(2 ** attempt)
                else:
                    raise

        if "error" in data:
            raise RuntimeError(f"NHD API error: {data['error']}")

        features = data.get("features", [])
        all_features.extend(features)
        log.info("  Fetched %d features (offset=%d, total so far=%d)",
                 len(features), offset, len(all_features))

        if len(features) < PAGE_SIZE:
            break  # Last page

        offset += PAGE_SIZE
        time.sleep(0.5)  # Be polite to the API

    return all_features


def _polygon_centroid(rings: list[list[list[float]]]) -> tuple[float, float]:
    """Compute the centroid of a polygon from its rings.

    Uses the simple average of exterior ring vertices (adequate for our
    distance calculations on medium-sized lakes).

    Returns:
        (lat, lon) tuple.
    """
    if not rings or not rings[0]:
        return (0.0, 0.0)

    # Use only the exterior ring (first ring)
    pts = np.array(rings[0])  # shape (N, 2), columns are [lon, lat]
    return float(pts[:, 1].mean()), float(pts[:, 0].mean())


def _sample_boundary_points(
    rings: list[list[list[float]]],
    interval_deg: float = SAMPLE_INTERVAL_DEG,
) -> list[tuple[float, float]]:
    """Sample points along polygon boundary at regular intervals.

    For large lakes, we need shoreline points rather than just the centroid
    so cities near the shore get accurate short distances.

    Returns:
        List of (lat, lon) tuples.
    """
    points = []
    if not rings:
        return points

    # Process exterior ring only
    exterior = rings[0]
    if len(exterior) < 2:
        return points

    pts = np.array(exterior)  # [lon, lat]
    # Compute cumulative distance in degrees (approximate)
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

    accumulated = 0.0
    points.append((pts[0, 1], pts[0, 0]))  # First point

    for i, seg_len in enumerate(segment_lengths):
        accumulated += seg_len
        if accumulated >= interval_deg:
            points.append((pts[i + 1, 1], pts[i + 1, 0]))
            accumulated = 0.0

    return points


def collect_lakes() -> pd.DataFrame:
    """Collect boatable lake reference points from NHD.

    Returns:
        DataFrame with columns: name, lat, lon, area_acres, area_sqkm,
        ftype, is_boundary_point.
    """
    log.info("Collecting boatable lakes from NHD (>= 300 acres)...")

    # --- Step 1: Medium lakes (300 acres to 100 km²) — simplified geometry ---
    log.info("Step 1: Querying medium lakes (%.1f - %.0f km²)...",
             MIN_AREA_SQKM, LARGE_LAKE_SQKM)
    medium_where = (
        f"AREASQKM >= {MIN_AREA_SQKM} AND AREASQKM < {LARGE_LAKE_SQKM} "
        f"AND FTYPE IN ({FTYPE_LAKEPOND}, {FTYPE_RESERVOIR})"
    )
    medium_features = _query_nhd_paged(
        where=medium_where,
        out_fields="GNIS_NAME,AREASQKM,FTYPE",
        return_geometry=True,
        max_offset=0.01,  # Heavy simplification — we only need centroids
    )
    log.info("Got %d medium lakes", len(medium_features))

    # Extract centroids for medium lakes
    rows = []
    for feat in medium_features:
        attrs = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        if not rings:
            continue

        lat, lon = _polygon_centroid(rings)
        if lat == 0 and lon == 0:
            continue

        area_sqkm = attrs["AREASQKM"]
        rows.append({
            "name": attrs.get("GNIS_NAME") or "",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "area_acres": round(area_sqkm * 247.105, 0),
            "area_sqkm": round(area_sqkm, 3),
            "ftype": attrs["FTYPE"],
            "is_boundary_point": False,
        })

    # --- Step 2: Large lakes (>= 100 km²) — full geometry, sample boundary ---
    log.info("Step 2: Querying large lakes (>= %.0f km²)...", LARGE_LAKE_SQKM)
    large_where = (
        f"AREASQKM >= {LARGE_LAKE_SQKM} "
        f"AND FTYPE IN ({FTYPE_LAKEPOND}, {FTYPE_RESERVOIR})"
    )
    large_features = _query_nhd_paged(
        where=large_where,
        out_fields="GNIS_NAME,AREASQKM,FTYPE",
        return_geometry=True,
        max_offset=0.005,  # Less simplification for accurate boundary
    )
    log.info("Got %d large lakes", len(large_features))

    for feat in large_features:
        attrs = feat["attributes"]
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])
        if not rings:
            continue

        area_sqkm = attrs["AREASQKM"]
        name = attrs.get("GNIS_NAME") or ""

        # Add centroid
        lat, lon = _polygon_centroid(rings)
        if lat != 0 or lon != 0:
            rows.append({
                "name": name,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "area_acres": round(area_sqkm * 247.105, 0),
                "area_sqkm": round(area_sqkm, 3),
                "ftype": attrs["FTYPE"],
                "is_boundary_point": False,
            })

        # Sample boundary points for accurate shoreline distance
        boundary_pts = _sample_boundary_points(rings)
        for blat, blon in boundary_pts:
            rows.append({
                "name": name,
                "lat": round(blat, 5),
                "lon": round(blon, 5),
                "area_acres": round(area_sqkm * 247.105, 0),
                "area_sqkm": round(area_sqkm, 3),
                "ftype": attrs["FTYPE"],
                "is_boundary_point": True,
            })

    df = pd.DataFrame(rows)
    log.info("Total reference points: %d (%d centroids, %d boundary)",
             len(df),
             (~df["is_boundary_point"]).sum(),
             df["is_boundary_point"].sum())

    # Deduplicate very close points (within ~0.01 degrees ≈ 0.7 miles)
    df = df.round({"lat": 3, "lon": 3}).drop_duplicates(subset=["lat", "lon"])
    log.info("After dedup: %d reference points", len(df))

    return df


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_lakes()

    output_path = PROJECT_ROOT / "data" / "lakes.parquet"
    write_parquet_with_metadata(
        df,
        output_path,
        source_name="USGS National Hydrography Dataset (NHD) - Waterbody Large Scale",
        source_url="https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12",
        date_collected="2026-05-02",
        notes=(
            "Boatable lakes and reservoirs >= 300 acres (1.214 km²) from NHD. "
            "FTYPE 390 (LakePond) + 436 (Reservoir). "
            "Medium lakes represented by centroids. "
            "Large lakes (>= 100 km²) also have sampled boundary points for "
            "accurate shoreline distance. Covers all 50 US states."
        ),
    )

    # Summary
    centroids = df[~df["is_boundary_point"]]
    print(f"\nLake data: {len(df)} reference points "
          f"({len(centroids)} lake centroids, "
          f"{len(df) - len(centroids)} boundary points)")
    print(f"Unique lakes: {len(centroids)}")
    print(f"  LakePond (390): {(centroids['ftype'] == 390).sum()}")
    print(f"  Reservoir (436): {(centroids['ftype'] == 436).sum()}")
    print(f"\nSize distribution (centroids):")
    for label, lo, hi in [
        ("300-1000 acres", 300, 1000),
        ("1000-5000 acres", 1000, 5000),
        ("5000-25000 acres", 5000, 25000),
        ("25000+ acres", 25000, float("inf")),
    ]:
        count = ((centroids["area_acres"] >= lo) & (centroids["area_acres"] < hi)).sum()
        print(f"  {label}: {count}")

    # Show largest lakes
    print(f"\nTop 20 largest lakes:")
    top = centroids.nlargest(20, "area_acres")
    for _, r in top.iterrows():
        print(f"  {r['name'] or '(unnamed)':30s}  {r['area_acres']:>10,.0f} acres  "
              f"({r['lat']:.2f}, {r['lon']:.2f})")


if __name__ == "__main__":
    main()
