"""Tests for Phase 3: Integrated city dataset."""

import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def cities():
    return pd.read_parquet(DATA_DIR / "cities.parquet")


def test_unified_dataset_exists():
    assert (DATA_DIR / "cities.parquet").exists()


def test_city_count(cities):
    assert len(cities) >= 700


def test_climate_columns(cities):
    for col in ["avg_temp_summer", "avg_temp_winter", "annual_rainfall", "sunny_days"]:
        assert col in cities.columns
        # At least 90% coverage
        coverage = cities[col].notna().sum() / len(cities)
        assert coverage > 0.90, f"{col} coverage {coverage:.1%} < 90%"


def test_cost_columns(cities):
    for col in ["median_home_price", "cost_of_living_index",
                "state_income_tax_rate", "avg_property_tax_rate"]:
        assert col in cities.columns
        coverage = cities[col].notna().sum() / len(cities)
        assert coverage > 0.95, f"{col} coverage {coverage:.1%} < 95%"


def test_crime_columns(cities):
    assert "crime_rate_per_1000" in cities.columns
    assert cities["crime_rate_per_1000"].notna().all()


def test_airport_columns(cities):
    for col in ["airport_distance_miles", "is_airline_hub",
                "direct_flight_destinations_count", "walkability_score"]:
        assert col in cities.columns
        assert cities[col].notna().all(), f"{col} has nulls"


def test_geography_columns(cities):
    for col in ["has_ocean", "has_mountains", "has_desert",
                "ski_resort_distance_miles", "national_parks_within_100mi"]:
        assert col in cities.columns
        assert cities[col].notna().all()


def test_derived_columns(cities):
    """Columns needed by scoring engine."""
    derived = ["swimming_access", "university_count", "has_major_university",
               "has_community_college", "college_town_score"]
    for col in derived:
        assert col in cities.columns, f"Missing derived column: {col}"


def test_temperature_sanity(cities):
    """Summer should be warmer than winter."""
    both = cities.dropna(subset=["avg_temp_summer", "avg_temp_winter"])
    assert (both["avg_temp_summer"] > both["avg_temp_winter"]).all()


def test_home_prices_positive(cities):
    valid = cities["median_home_price"].dropna()
    assert (valid > 0).all()
    assert valid.median() > 100_000  # Reasonable national median


def test_crime_rates_reasonable(cities):
    rates = cities["crime_rate_per_1000"]
    assert rates.min() > 0
    assert rates.max() < 100  # Per 1,000 pop


def test_no_fake_data_marker(cities):
    """Verify no placeholder values that suggest fabricated data."""
    # Population shouldn't all be round numbers
    pop = cities["population"]
    non_round = (pop % 1000 != 0).sum()
    assert non_round > len(cities) * 0.5, "Too many round population numbers"
