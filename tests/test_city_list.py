"""Tests for Phase 1: Master city list."""

import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def cities():
    return pd.read_parquet(DATA_DIR / "cities_master.parquet")


def test_city_list_exists():
    assert (DATA_DIR / "cities_master.parquet").exists()


def test_city_count(cities):
    assert len(cities) >= 700, f"Expected 700+ cities, got {len(cities)}"


def test_all_50_states_plus_dc(cities):
    states = cities["state"].unique()
    assert len(states) >= 51, f"Expected 51 states+DC, got {len(states)}"
    assert "CA" in states
    assert "TX" in states
    assert "NY" in states
    assert "DC" in states


def test_required_columns(cities):
    required = [
        "city_id", "name", "state", "population", "lat", "lon", "region",
        "area_population", "metro_pop", "incorporated_places",
        "cbsa_code", "cbsa_name",
    ]
    for col in required:
        assert col in cities.columns, f"Missing column: {col}"


def test_no_duplicate_city_ids(cities):
    assert cities["city_id"].is_unique


def test_population_range(cities):
    assert cities["population"].min() > 0
    assert cities["population"].max() > 1_000_000  # At least one big city


def test_coordinates_valid(cities):
    # US latitude range: ~18 to ~72 (Hawaii to Alaska)
    assert cities["lat"].min() > 17
    assert cities["lat"].max() < 72
    # US longitude range: ~-180 to ~-65 (Alaska to Maine)
    assert cities["lon"].min() > -180
    assert cities["lon"].max() < -65


def test_regions_assigned(cities):
    assert cities["region"].notna().all()
    regions = set(cities["region"].unique())
    assert "South" in regions
    assert "West" in regions
    assert "Midwest" in regions
    assert "Northeast" in regions


def test_nyc_boroughs(cities):
    """At least one NYC borough should survive the 20-mile separation filter.

    The boroughs are all within 20 miles of each other, so the separation
    algorithm keeps only the highest-population one (Brooklyn, ~2.6M).
    """
    nyc_names = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}
    ny_cities = set(cities[cities["state"] == "NY"]["name"])
    found = nyc_names & ny_cities
    assert len(found) >= 1, f"Expected at least one NYC borough, found none in {ny_cities}"
    assert "Brooklyn" in found, f"Expected Brooklyn (largest borough) to survive, got: {found}"


def test_state_minimum_coverage(cities):
    """Most states should have at least 5 cities."""
    state_counts = cities.groupby("state").size()
    # Allow DC to have just 1
    non_dc = state_counts[state_counts.index != "DC"]
    assert non_dc.min() >= 5, f"Min state count: {non_dc.min()}"


def test_area_population_gte_population(cities):
    """Area population should always be >= city population (includes roll-ups)."""
    valid = cities.dropna(subset=["area_population"])
    assert (valid["area_population"] >= valid["population"]).all()


def test_cbsa_coverage(cities):
    """Most cities should be mapped to a CBSA."""
    mapped = (cities["cbsa_code"] != "").sum()
    pct = mapped / len(cities)
    assert pct > 0.85, f"CBSA coverage {pct:.1%} < 85%"


def test_metro_pop_reasonable(cities):
    """Metro populations should be larger than city populations where present."""
    has_metro = cities[cities["metro_pop"].notna() & (cities["metro_pop"] > 0)]
    assert len(has_metro) > 0
    # Metro pop should generally be >= city population
    larger = (has_metro["metro_pop"] >= has_metro["population"]).mean()
    assert larger > 0.80, f"Only {larger:.1%} of metro pops >= city pop"


def test_incorporated_places_format(cities):
    """Incorporated places should be comma-separated strings or empty."""
    for val in cities["incorporated_places"]:
        assert isinstance(val, str), f"Expected string, got {type(val)}"


def test_rollup_cities_exist(cities):
    """Cities with roll-up data should exist."""
    has_rollup = cities[cities["incorporated_places"] != ""]
    assert len(has_rollup) > 50, f"Expected 50+ cities with roll-ups, got {len(has_rollup)}"
