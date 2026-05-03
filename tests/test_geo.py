"""Tests for utilities/geo.py — distance calculations and city separation."""
import pandas as pd
import pytest

from utilities.geo import enforce_separation, find_nearby_cities, haversine_miles


class TestHaversine:
    def test_same_point(self):
        assert haversine_miles(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_nyc_to_newark(self):
        # NYC (40.7128, -74.0060) to Newark (40.7357, -74.1724)
        dist = haversine_miles(40.7128, -74.0060, 40.7357, -74.1724)
        assert 8.0 < dist < 10.0, f"Expected ~8.9 miles, got {dist}"

    def test_houston_to_sugar_land(self):
        # Houston (29.7604, -95.3698) to Sugar Land (29.6197, -95.6349)
        dist = haversine_miles(29.7604, -95.3698, 29.6197, -95.6349)
        assert 16.0 < dist < 20.0, f"Expected ~18 miles, got {dist}"

    def test_la_to_nyc(self):
        # Los Angeles to New York — should be roughly 2450 miles.
        dist = haversine_miles(34.0522, -118.2437, 40.7128, -74.0060)
        assert 2400 < dist < 2500, f"Expected ~2450 miles, got {dist}"

    def test_symmetry(self):
        d1 = haversine_miles(40.0, -74.0, 34.0, -118.0)
        d2 = haversine_miles(34.0, -118.0, 40.0, -74.0)
        assert abs(d1 - d2) < 0.001


class TestFindNearbyCities:
    @pytest.fixture
    def cities_df(self):
        return pd.DataFrame({
            "name": ["NYC", "Newark", "Philadelphia", "Boston"],
            "state": ["NY", "NJ", "PA", "MA"],
            "lat": [40.7128, 40.7357, 39.9526, 42.3601],
            "lon": [-74.0060, -74.1724, -75.1652, -71.0589],
        })

    def test_finds_newark_near_nyc(self, cities_df):
        result = find_nearby_cities(40.7128, -74.0060, cities_df, radius_miles=20)
        assert "Newark" in result["name"].values
        assert "Boston" not in result["name"].values

    def test_includes_self(self, cities_df):
        result = find_nearby_cities(40.7128, -74.0060, cities_df, radius_miles=20)
        assert "NYC" in result["name"].values

    def test_sorted_by_distance(self, cities_df):
        result = find_nearby_cities(40.7128, -74.0060, cities_df, radius_miles=100)
        distances = result["distance_miles"].tolist()
        assert distances == sorted(distances)


class TestEnforceSeparation:
    @pytest.fixture
    def close_cities(self):
        return pd.DataFrame({
            "name": ["Big City", "Small Neighbor", "Far Away"],
            "state": ["NY", "NJ", "CA"],
            "lat": [40.7128, 40.7357, 34.0522],
            "lon": [-74.0060, -74.1724, -118.2437],
            "population": [8000000, 300000, 4000000],
        })

    def test_removes_smaller_neighbor(self, close_cities):
        result = enforce_separation(close_cities, min_distance_miles=20.0)
        assert "Big City" in result["name"].values
        assert "Small Neighbor" not in result["name"].values
        assert "Far Away" in result["name"].values

    def test_keeps_all_when_far_enough(self, close_cities):
        result = enforce_separation(close_cities, min_distance_miles=1.0)
        assert len(result) == 3

    def test_state_partitioning_works(self):
        # Two cities in different non-neighboring states should both be kept
        # even if they hypothetically had the same coordinates (edge case test).
        df = pd.DataFrame({
            "name": ["City A", "City B"],
            "state": ["CA", "NY"],
            "lat": [40.0, 40.0],
            "lon": [-74.0, -74.0],
            "population": [100000, 200000],
        })
        result = enforce_separation(df, min_distance_miles=20.0)
        assert len(result) == 2

    def test_rollup_returns_mapping(self, close_cities):
        result, rollup = enforce_separation(
            close_cities, min_distance_miles=20.0, return_rollup=True,
        )
        assert isinstance(rollup, dict)
        assert len(rollup) == 1  # Small Neighbor was removed
        # The removed city should map to a kept city's original index
        removed_orig_idx = list(rollup.keys())[0]
        kept_orig_idx = rollup[removed_orig_idx]
        # Big City (pop 8M) is index 0 in the original DF after sort by pop desc
        # Small Neighbor maps to Big City (nearest kept)
        assert kept_orig_idx != removed_orig_idx

    def test_rollup_empty_when_no_removals(self):
        df = pd.DataFrame({
            "name": ["A", "B"],
            "state": ["CA", "NY"],
            "lat": [34.0, 40.7],
            "lon": [-118.0, -74.0],
            "population": [100000, 200000],
        })
        result, rollup = enforce_separation(
            df, min_distance_miles=20.0, return_rollup=True,
        )
        assert len(rollup) == 0
        assert len(result) == 2
