"""Tests for scoring engine with real data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.scoring import (
    calculate_city_scores,
    load_cities,
    score_climate,
    score_city_size,
    score_cost_taxes,
    score_outdoor_recreation,
    score_lifestyle,
    score_education,
    score_practical,
)


@pytest.fixture
def cities_df():
    return load_cities()


@pytest.fixture
def sample_city(cities_df):
    """Return a city with good data coverage (Denver, CO or similar)."""
    co = cities_df[cities_df["state"] == "CO"]
    if not co.empty:
        return co.iloc[0]
    return cities_df.iloc[0]


def test_load_cities(cities_df):
    assert len(cities_df) >= 700
    assert "name" in cities_df.columns
    assert "state" in cities_df.columns


def test_score_climate_returns_0_to_100(sample_city):
    prefs = {"temp_preference": "four_seasons", "humidity_preference": "moderate_humidity"}
    score = score_climate(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_climate_empty_prefs(sample_city):
    score = score_climate(sample_city, {})
    assert score == 0


def test_score_city_size(sample_city):
    prefs = {"city_size": ["mid_size"], "density_preference": "suburban"}
    score = score_city_size(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_cost_taxes(sample_city):
    prefs = {"max_home_price": 400000, "cost_of_living": "moderate_cost"}
    score = score_cost_taxes(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_outdoor_recreation(sample_city):
    prefs = {"winter_sports": "ski_daytrip", "summer_activities": ["hiking"]}
    score = score_outdoor_recreation(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_lifestyle(sample_city):
    prefs = {"nightlife": "restaurants_bars", "food_scene": "good_variety"}
    score = score_lifestyle(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_education(sample_city):
    prefs = {"school_quality": "good_schools_nice"}
    score = score_education(sample_city, prefs)
    assert 0 <= score <= 100


def test_score_practical(sample_city):
    prefs = {"airport_access": "regional_airport", "safety_priority": "important"}
    score = score_practical(sample_city, prefs)
    assert 0 <= score <= 100


def test_calculate_city_scores_returns_top_n():
    prefs = {
        "temp_preference": "mild_year_round",
        "city_size": ["mid_size"],
        "max_home_price": 350000,
        "safety_priority": "important",
    }
    results = calculate_city_scores(prefs, top_n=10)
    assert len(results) == 10


def test_results_sorted_descending():
    prefs = {"temp_preference": "four_seasons", "city_size": ["big_metro"]}
    results = calculate_city_scores(prefs, top_n=5)
    scores = [r["total_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_result_structure():
    prefs = {"temp_preference": "hot_mild"}
    results = calculate_city_scores(prefs, top_n=1)
    r = results[0]
    assert "name" in r
    assert "state" in r
    assert "total_score" in r
    assert "scores" in r
    assert "stats" in r
    assert "lat" in r
    assert "lon" in r


def test_hot_climate_preference_favors_warm_cities():
    """Cities with hot preference should favor warm-weather cities."""
    prefs = {"temp_preference": "hot_mild", "snow_preference": "no_snow"}
    results = calculate_city_scores(prefs, top_n=50)
    # With many ties, check top 50 — warm states should be well represented
    warm_states = {"FL", "TX", "AZ", "CA", "HI", "LA", "GA", "SC", "AL", "NV"}
    result_states = {r["state"] for r in results}
    overlap = warm_states & result_states
    assert len(overlap) >= 2, f"Expected warm states in results, got {result_states}"


def test_mountain_preference_favors_mountain_cities():
    """Geography preference for mountains should favor mountain cities."""
    prefs = {
        "geography": ["mountains"],
        "winter_sports": "ski_1hr",
    }
    results = calculate_city_scores(prefs, top_n=50)
    # With many ties, check top 50 — mountain states should appear
    mountain_states = {"CO", "UT", "MT", "WY", "ID", "VT", "NH", "WV", "OR", "WA"}
    result_states = {r["state"] for r in results}
    overlap = mountain_states & result_states
    assert len(overlap) >= 1, f"Expected mountain states, got {result_states}"
