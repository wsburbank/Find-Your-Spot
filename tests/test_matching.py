"""Tests for utilities/matching.py — city name normalization and fuzzy matching."""
import pandas as pd
import pytest

from utilities.matching import (
    batch_match_cities,
    fuzzy_match_city,
    normalize_city_name,
    normalize_state,
)


class TestNormalizeCityName:
    @pytest.mark.parametrize("input_name,expected", [
        ("New York city", "new york"),
        ("New York City", "new york"),
        ("NEW YORK", "new york"),
        ("New York, NY", "new york"),
        ("Nashville-Davidson metropolitan government (balance)", "nashville-davidson"),
        ("St. Louis", "saint louis"),
        ("St. Petersburg", "saint petersburg"),
        ("Ft. Worth", "fort worth"),
        ("Mt. Vernon", "mount vernon"),
        ("Urban Honolulu CDP", "honolulu"),
        ("Louisville/Jefferson County metro government (balance)", "louisville/jefferson county"),
        ("  Denver  ", "denver"),
        ("Portland", "portland"),
    ])
    def test_normalization(self, input_name, expected):
        assert normalize_city_name(input_name) == expected

    def test_removes_borough_suffix(self):
        assert normalize_city_name("Manhattan borough") == "manhattan"

    def test_removes_village_suffix(self):
        assert normalize_city_name("Scarsdale village") == "scarsdale"


class TestNormalizeState:
    def test_abbreviation(self):
        assert normalize_state("CO") == "CO"

    def test_full_name(self):
        assert normalize_state("Colorado") == "CO"

    def test_case_insensitive(self):
        assert normalize_state("california") == "CA"
        assert normalize_state("TEXAS") == "TX"

    def test_whitespace(self):
        assert normalize_state("  NY  ") == "NY"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            normalize_state("Narnia")


class TestFuzzyMatchCity:
    @pytest.fixture
    def master_cities(self):
        return pd.DataFrame({
            "name": [
                "Portland", "Salem", "Eugene",
                "Portland", "Augusta", "Bangor",
                "Springfield", "Chicago", "Peoria",
            ],
            "state": [
                "OR", "OR", "OR",
                "ME", "ME", "ME",
                "IL", "IL", "IL",
            ],
        })

    def test_exact_match(self, master_cities):
        result = fuzzy_match_city("Portland", "OR", master_cities)
        assert result is not None
        assert result["name"] == "Portland"
        assert result["state"] == "OR"

    def test_state_filtering_prevents_cross_state(self, master_cities):
        result = fuzzy_match_city("Portland", "ME", master_cities)
        assert result is not None
        assert result["state"] == "ME"

    def test_fuzzy_match(self, master_cities):
        result = fuzzy_match_city("Spingfield", "IL", master_cities)
        assert result is not None
        assert result["name"] == "Springfield"

    def test_no_match_returns_none(self, master_cities):
        result = fuzzy_match_city("Nonexistent City", "OR", master_cities)
        assert result is None

    def test_no_state_match_returns_none(self, master_cities):
        result = fuzzy_match_city("Portland", "TX", master_cities)
        assert result is None


class TestBatchMatchCities:
    def test_batch_matching(self):
        source = pd.DataFrame({
            "name": ["New York city", "St. Louis", "Unknown Place"],
            "state": ["NY", "MO", "ZZ"],
        })
        master = pd.DataFrame({
            "name": ["New York", "Saint Louis", "Kansas City"],
            "state": ["NY", "MO", "MO"],
        })
        result = batch_match_cities(source, master)
        assert "matched_name" in result.columns
        assert "match_method" in result.columns
        assert result.iloc[0]["matched_name"] == "New York"
        assert result.iloc[1]["matched_name"] == "Saint Louis"
        assert result.iloc[2]["match_method"] == "unmatched"
