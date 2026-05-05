"""
Scoring Algorithm for Find Your Spot
Matches user preferences to city attributes and returns ranked results.

Range-based scoring
-------------------
Questions marked ``range=True`` in quiz_questions carry a ``range_map`` that
maps each option value to a ``(low, high)`` interval in the city metric's
units.  When the user selects multiple options the engine unions the intervals
to form a single ``(min, max)`` window and then scores the city:

* metric inside window            → full points
* metric within 20 % of a boundary → partial (linearly tapered)
* metric outside that buffer       → minimum points
"""

import pandas as pd
import numpy as np
from pathlib import Path

from components.quiz_questions import get_range_questions
from utilities.tax_estimate import estimate_taxes


def load_cities():
    """Load the city database."""
    data_path = Path(__file__).parent.parent / "data" / "cities.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"City database not found at {data_path}. "
            "Please run scripts/build_city_database.py first."
        )
    return pd.read_parquet(data_path)


# Cache range question definitions once at import time.
_RANGE_QUESTIONS: dict = {}


def _get_range_questions() -> dict:
    """Lazy-load and cache range question definitions."""
    global _RANGE_QUESTIONS
    if not _RANGE_QUESTIONS:
        _RANGE_QUESTIONS = get_range_questions()
    return _RANGE_QUESTIONS


def _range_score(city_value: float, selections: list, range_map: dict,
                 full: float = 25, partial: float = 15, miss: float = 5) -> float:
    """Score a city metric against a user's range selections.

    Args:
        city_value: The city's actual value for the metric.
        selections: List of selected option keys (e.g. ["keep_dry", "occasional_rain"]).
        range_map: {option_key: (low, high)} from the question definition.
        full: Points awarded when the value is inside the range.
        partial: Points awarded when the value is near the range boundary.
        miss: Points awarded when the value is outside the range.

    Returns:
        Score between *miss* and *full*.
    """
    if not selections:
        return 0

    # Union all selected intervals into one (min, max).
    lows, highs = [], []
    for sel in selections:
        if sel in range_map:
            lo, hi = range_map[sel]
            lows.append(lo)
            highs.append(hi)

    if not lows:
        return 0

    range_min = min(lows)
    range_max = max(highs)

    if range_min <= city_value <= range_max:
        return full

    # 20% buffer for partial credit
    span = max(range_max - range_min, 1)
    buffer = span * 0.20
    if (range_min - buffer) <= city_value <= (range_max + buffer):
        return partial

    return miss


# -----------------------------------------------------------------------
# Individual category scoring functions
# -----------------------------------------------------------------------

def score_climate(city, preferences):
    """Score city based on climate preferences (0-100)."""
    score = 0
    max_score = 0

    # Temperature preference (independent multi-select)
    temp_prefs = preferences.get("temp_preference", [])
    if isinstance(temp_prefs, str):
        temp_prefs = [temp_prefs]
    if temp_prefs:
        max_score += 25
        avg_summer = city["avg_temp_summer"]
        avg_winter = city["avg_temp_winter"]
        matches = False
        if "hot_mild" in temp_prefs and avg_summer >= 80 and avg_winter >= 35:
            matches = True
        if "four_seasons" in temp_prefs:
            temp_range = avg_summer - avg_winter
            if 30 <= temp_range <= 70 and avg_winter < 40:
                matches = True
        if "mild_year_round" in temp_prefs and avg_summer <= 85 and avg_winter >= 30:
            matches = True
        if "love_cold" in temp_prefs and avg_winter < 35:
            matches = True
        score += 25 if matches else 8

    # Humidity preference (range on avg_summer_dewpoint)
    humidity_prefs = preferences.get("humidity_preference", [])
    if isinstance(humidity_prefs, str):
        humidity_prefs = [humidity_prefs]
    if humidity_prefs:
        max_score += 20
        dewpoint = city.get("avg_summer_dewpoint")
        if pd.notna(dewpoint):
            rq = _get_range_questions().get("humidity_preference", {})
            score += _range_score(dewpoint, humidity_prefs,
                                  rq.get("range_map", {}), full=20, partial=12, miss=3)
        else:
            score += 10  # neutral when data missing

    # Rain preference (range)
    rain_prefs = preferences.get("rain_preference", [])
    if isinstance(rain_prefs, str):
        rain_prefs = [rain_prefs]
    if rain_prefs:
        max_score += 20
        rq = _get_range_questions().get("rain_preference", {})
        score += _range_score(city["annual_rainfall"], rain_prefs,
                              rq.get("range_map", {}), full=20, partial=12, miss=3)

    # Sunshine preference (range)
    sun_prefs = preferences.get("sunshine_preference", [])
    if isinstance(sun_prefs, str):
        sun_prefs = [sun_prefs]
    if sun_prefs:
        max_score += 20
        rq = _get_range_questions().get("sunshine_preference", {})
        score += _range_score(city["sunny_days"], sun_prefs,
                              rq.get("range_map", {}), full=20, partial=12, miss=3)

    # Snow preference (range)
    snow_prefs = preferences.get("snow_preference", [])
    if isinstance(snow_prefs, str):
        snow_prefs = [snow_prefs]
    if snow_prefs:
        max_score += 15
        rq = _get_range_questions().get("snow_preference", {})
        score += _range_score(city["annual_snow"], snow_prefs,
                              rq.get("range_map", {}), full=15, partial=9, miss=2)

    return (score / max_score * 100) if max_score > 0 else 0


def score_city_size(city, preferences):
    """Score city based on size/density preferences (0-100)."""
    score = 0
    max_score = 0

    # City size preference (independent multi-select)
    size_prefs = preferences.get("city_size", [])
    if isinstance(size_prefs, str):
        size_prefs = [size_prefs]
    if size_prefs:
        max_score += 40
        pop = city["population"]
        matches = False
        if "big_metro" in size_prefs and pop >= 1000000:
            matches = True
        if "large_city" in size_prefs and 100000 <= pop < 1000000:
            matches = True
        if "mid_size" in size_prefs and 30000 <= pop < 100000:
            matches = True
        if "small_city" in size_prefs and 15000 <= pop < 30000:
            matches = True
        if "small_town" in size_prefs and pop < 15000:
            matches = True
        score += 40 if matches else 10

    # Density preference (range)
    density_prefs = preferences.get("density_preference", [])
    if isinstance(density_prefs, str):
        density_prefs = [density_prefs]
    if density_prefs:
        max_score += 30
        rq = _get_range_questions().get("density_preference", {})
        score += _range_score(city["walkability_score"], density_prefs,
                              rq.get("range_map", {}), full=30, partial=18, miss=8)

    # Commute preferences (slider-based — unchanged)
    commute_prefs = preferences.get("commute_preferences", {})
    if commute_prefs:
        max_score += 30
        walk = city["walkability_score"]
        transit = city["transit_score"]
        walk_pref = commute_prefs.get("walk_bike_pref", 5)
        transit_pref = commute_prefs.get("transit_pref", 5)
        driving_pref = commute_prefs.get("driving_pref", 5)

        commute_score = 0
        if walk_pref >= 7:
            commute_score += (walk / 100) * 10 * (walk_pref / 10)
        if transit_pref >= 7:
            commute_score += (transit / 100) * 10 * (transit_pref / 10)
        if driving_pref >= 7:
            drive_score = 100 - min(100, city["population"] / 50000)
            commute_score += (drive_score / 100) * 10 * (driving_pref / 10)

        total_weight = max(1, (walk_pref + transit_pref + driving_pref) / 10)
        score += min(30, commute_score / total_weight * 3)

    # Max commute time (slider — scored against Census ACS mean_commute_minutes)
    max_commute = preferences.get("max_commute_time")
    if max_commute:
        max_score += 25
        mean_commute = city.get("mean_commute_minutes")
        if pd.isna(mean_commute):
            score += 12  # neutral when data is missing
        elif mean_commute <= max_commute:
            # Under budget — closer to the limit gets slightly less
            ratio = mean_commute / max_commute
            score += 25 if ratio <= 0.8 else 20
        else:
            # Over the user's max — penalize proportionally
            over_ratio = mean_commute / max_commute
            if over_ratio <= 1.15:
                score += 15
            elif over_ratio <= 1.30:
                score += 10
            else:
                score += 5

    return (score / max_score * 100) if max_score > 0 else 0


def score_cost_taxes(city, preferences):
    """Score city based on cost of living and tax preferences (0-100)."""
    score = 0
    max_score = 0

    # Max home price (slider)
    max_price = preferences.get("max_home_price")
    if max_price:
        max_score += 30
        home_price = city["median_home_price"]
        if home_price <= max_price:
            budget_ratio = home_price / max_price
            if budget_ratio <= 0.7:
                score += 30
            elif budget_ratio <= 0.85:
                score += 25
            else:
                score += 20
        else:
            over_ratio = home_price / max_price
            if over_ratio <= 1.1:
                score += 15
            elif over_ratio <= 1.25:
                score += 10
            else:
                score += 5

    # Cost of living (range)
    col_prefs = preferences.get("cost_of_living", [])
    if isinstance(col_prefs, str):
        col_prefs = [col_prefs]
    if col_prefs:
        max_score += 20
        rq = _get_range_questions().get("cost_of_living", {})
        score += _range_score(city["cost_of_living_index"], col_prefs,
                              rq.get("range_map", {}), full=20, partial=12, miss=4)

    # Tax preference (independent multi-select)
    tax_prefs = preferences.get("tax_preference", [])
    if isinstance(tax_prefs, str):
        tax_prefs = [tax_prefs]
    if tax_prefs:
        max_score += 25
        no_income = city["no_income_tax_state"]
        income_tax = city["state_income_tax_rate"]
        property_tax = city["avg_property_tax_rate"]
        sales_tax = city["state_sales_tax_rate"]

        best = 5  # minimum
        if "dont_care_tax" in tax_prefs:
            best = 25
        if "no_income_tax" in tax_prefs and no_income:
            best = max(best, 25)
        elif "no_income_tax" in tax_prefs and income_tax < 4:
            best = max(best, 15)
        if "low_property_tax" in tax_prefs and property_tax < 1:
            best = max(best, 25)
        elif "low_property_tax" in tax_prefs and property_tax < 1.5:
            best = max(best, 18)
        if "sales_tax_friendly" in tax_prefs and sales_tax < 5:
            best = max(best, 25)
        elif "sales_tax_friendly" in tax_prefs and sales_tax < 7:
            best = max(best, 18)
        if "balanced_tax" in tax_prefs and income_tax < 7 and property_tax < 2 and sales_tax < 8:
            best = max(best, 25)
        score += best

    # Rent budget (slider)
    max_rent = preferences.get("rent_budget")
    if max_rent:
        max_score += 25
        median_rent = city["median_gross_rent"]
        if median_rent <= max_rent:
            budget_ratio = median_rent / max_rent
            if budget_ratio <= 0.7:
                score += 25
            elif budget_ratio <= 0.85:
                score += 20
            else:
                score += 15
        else:
            over_ratio = median_rent / max_rent
            if over_ratio <= 1.15:
                score += 12
            elif over_ratio <= 1.3:
                score += 8
            else:
                score += 3

    return (score / max_score * 100) if max_score > 0 else 0


def score_outdoor_recreation(city, preferences):
    """Score city based on outdoor recreation preferences (0-100)."""
    score = 0
    max_score = 0

    # Winter sports (range on ski distance)
    winter_prefs = preferences.get("winter_sports", [])
    if isinstance(winter_prefs, str):
        winter_prefs = [winter_prefs]
    if winter_prefs:
        max_score += 30
        rq = _get_range_questions().get("winter_sports", {})
        score += _range_score(city["ski_resort_distance_miles"], winter_prefs,
                              rq.get("range_map", {}), full=30, partial=18, miss=8)

    # Summer activities (independent multi-select)
    summer_prefs = preferences.get("summer_activities", [])
    if summer_prefs:
        max_score += 25
        activity_score = 0
        activity_count = len(summer_prefs)

        if "hiking" in summer_prefs:
            trails = city["hiking_trails_count"]
            activity_score += 25 if trails >= 100 else (18 if trails >= 30 else 10)
        if "mtb_trails" in summer_prefs:
            mtb = city["mountain_biking_trails"]
            activity_score += 25 if mtb >= 50 else (18 if mtb >= 20 else 10)
        if "rock_climbing" in summer_prefs:
            rc = city["rock_climbing_areas_nearby"]
            activity_score += 25 if rc >= 10 else (18 if rc >= 3 else 8)
        if "swimming" in summer_prefs:
            if city["has_ocean"]:
                activity_score += 25
            else:
                lake_mi = city["nearest_boatable_lake_miles"]
                activity_score += 25 if lake_mi < 30 else (18 if lake_mi < 60 else 12)
        if "golf" in summer_prefs:
            activity_score += 20
        if "fishing" in summer_prefs:
            if city["has_ocean"]:
                activity_score += 25
            else:
                lake_mi = city["nearest_boatable_lake_miles"]
                activity_score += 25 if lake_mi < 30 else (18 if lake_mi < 60 else 10)

        if activity_count > 0:
            score += activity_score / activity_count

    # Camping & nature (range on total parks)
    camping_prefs = preferences.get("camping_nature", [])
    if isinstance(camping_prefs, str):
        camping_prefs = [camping_prefs]
    if camping_prefs:
        max_score += 20
        parks_total = city["national_parks_within_100mi"] + city["state_parks_nearby"]
        rq = _get_range_questions().get("camping_nature", {})
        score += _range_score(parks_total, camping_prefs,
                              rq.get("range_map", {}), full=20, partial=14, miss=8)

    # Water activities (independent multi-select)
    water_prefs = preferences.get("water_activities", [])
    if water_prefs:
        max_score += 25
        water_score = 0
        water_count = len(water_prefs)

        if "ocean_beach" in water_prefs:
            water_score += 25 if city["has_ocean"] else 5
        if "lake_recreation" in water_prefs:
            lake_mi = city["nearest_boatable_lake_miles"]
            if lake_mi < 20:
                water_score += 25
            elif lake_mi < 40:
                water_score += 20
            elif lake_mi < 60:
                water_score += 15
            else:
                water_score += 8
        if "river_activities" in water_prefs:
            water_score += 20
        if "water_not_priority" in water_prefs:
            water_score += 25

        if water_count > 0:
            score += water_score / water_count

    return (score / max_score * 100) if max_score > 0 else 0


def score_lifestyle(city, preferences):
    """Score city based on lifestyle and entertainment preferences (0-100)."""
    score = 0
    max_score = 0

    # Museums (range on museums_count — IMLS real data)
    museum_prefs = preferences.get("museums", [])
    if isinstance(museum_prefs, str):
        museum_prefs = [museum_prefs]
    if museum_prefs:
        max_score += 20
        rq = _get_range_questions().get("museums", {})
        score += _range_score(city["museums_count"], museum_prefs,
                              rq.get("range_map", {}), full=20, partial=14, miss=8)

    # Performing arts (range on performing_arts_venues — Census CBP)
    arts_prefs = preferences.get("performing_arts", [])
    if isinstance(arts_prefs, str):
        arts_prefs = [arts_prefs]
    if arts_prefs:
        max_score += 20
        rq = _get_range_questions().get("performing_arts", {})
        score += _range_score(city["performing_arts_venues"], arts_prefs,
                              rq.get("range_map", {}), full=20, partial=14, miss=8)

    # Concert venues (range on concert_venue_count — Census CBP)
    concert_prefs = preferences.get("concert_venues", [])
    if isinstance(concert_prefs, str):
        concert_prefs = [concert_prefs]
    if concert_prefs:
        max_score += 20
        rq = _get_range_questions().get("concert_venues", {})
        score += _range_score(city["concert_venue_count"], concert_prefs,
                              rq.get("range_map", {}), full=20, partial=14, miss=8)

    # Sports scene (range on major_pro_teams — Wikipedia real data)
    sports_prefs = preferences.get("sports_scene", [])
    if isinstance(sports_prefs, str):
        sports_prefs = [sports_prefs]
    if sports_prefs:
        max_score += 20
        rq = _get_range_questions().get("sports_scene", {})
        base = _range_score(city["major_pro_teams"], sports_prefs,
                            rq.get("range_map", {}), full=20, partial=14, miss=8)
        # Credit minor league presence if user selected that option
        if "minor_league" in sports_prefs and city["minor_pro_teams"] > 0:
            base = max(base, 18)
        score += base

    return (score / max_score * 100) if max_score > 0 else 0


def score_education(city, preferences):
    """Score city based on education and community preferences (0-100)."""
    score = 0
    max_score = 0

    # School quality (range on avg_school_rating — requires school_ratings data)
    school_prefs = preferences.get("school_quality", [])
    if isinstance(school_prefs, str):
        school_prefs = [school_prefs]
    if school_prefs:
        max_score += 25
        school_rating = city.get("avg_school_rating")
        if pd.notna(school_rating):
            rq = _get_range_questions().get("school_quality", {})
            score += _range_score(school_rating, school_prefs,
                                  rq.get("range_map", {}), full=25, partial=18, miss=10)
        else:
            score += 12  # neutral score when data missing

    # College proximity (independent multi-select — real IPEDS data)
    college_prefs = preferences.get("college_proximity", [])
    if isinstance(college_prefs, str):
        college_prefs = [college_prefs]
    if college_prefs:
        max_score += 20
        best = 8
        if "college_doesnt_matter" in college_prefs or "no_college_fine" in college_prefs:
            best = 20
        if "major_university" in college_prefs and city["has_major_university"]:
            best = max(best, 20)
        elif "major_university" in college_prefs and city["university_count"] >= 1:
            best = max(best, 15)
        if "community_college" in college_prefs and city["has_community_college"]:
            best = max(best, 20)
        if "college_town_vibe" in college_prefs and city["college_town_score"] >= 0.15:
            best = max(best, 20)
        elif "college_town_vibe" in college_prefs and city["college_town_score"] >= 0.05:
            best = max(best, 15)
        score += best

    # Remote work culture (range on pct_work_from_home)
    remote_prefs = preferences.get("remote_work", [])
    if isinstance(remote_prefs, str):
        remote_prefs = [remote_prefs]
    if remote_prefs:
        max_score += 20
        wfh = city.get("pct_work_from_home")
        if pd.notna(wfh):
            rq = _get_range_questions().get("remote_work", {})
            score += _range_score(wfh, remote_prefs,
                                  rq.get("range_map", {}), full=20, partial=12, miss=5)
        else:
            score += 10

    # Diversity (range on diversity_index)
    diversity_prefs = preferences.get("diversity", [])
    if isinstance(diversity_prefs, str):
        diversity_prefs = [diversity_prefs]
    if diversity_prefs:
        max_score += 20
        div_index = city.get("diversity_index")
        if pd.notna(div_index):
            rq = _get_range_questions().get("diversity", {})
            score += _range_score(div_index, diversity_prefs,
                                  rq.get("range_map", {}), full=20, partial=12, miss=5)
        else:
            score += 10

    return (score / max_score * 100) if max_score > 0 else 0


def score_practical(city, preferences):
    """Score city based on practical considerations (0-100)."""
    score = 0
    max_score = 0

    # Airport access (distance-based, independent multi-select)
    airport_prefs = preferences.get("airport_access", [])
    if isinstance(airport_prefs, str):
        airport_prefs = [airport_prefs]
    if airport_prefs:
        max_score += 25
        hub_dist = city["nearest_hub_distance_miles"]
        airport_dist = city["airport_distance_miles"]
        best = 5
        if "airport_not_factor" in airport_prefs:
            best = 25
        if "hub_1hr" in airport_prefs and hub_dist < 60:
            best = max(best, 25)
        elif "hub_1hr" in airport_prefs and hub_dist < 120:
            best = max(best, 15)
        if "hub_2hr" in airport_prefs and hub_dist < 120:
            best = max(best, 25)
        elif "hub_2hr" in airport_prefs and hub_dist < 180:
            best = max(best, 15)
        if "regional_1hr" in airport_prefs and airport_dist < 60:
            best = max(best, 25)
        elif "regional_1hr" in airport_prefs and airport_dist < 120:
            best = max(best, 15)
        if "regional_2hr" in airport_prefs and airport_dist < 120:
            best = max(best, 25)
        elif "regional_2hr" in airport_prefs and airport_dist < 180:
            best = max(best, 15)
        score += best

    # Air quality (range on median_aqi)
    aq_prefs = preferences.get("air_quality", [])
    if isinstance(aq_prefs, str):
        aq_prefs = [aq_prefs]
    if aq_prefs:
        max_score += 20
        aqi = city.get("median_aqi")
        if pd.notna(aqi):
            rq = _get_range_questions().get("air_quality", {})
            score += _range_score(aqi, aq_prefs,
                                  rq.get("range_map", {}), full=20, partial=14, miss=5)
        else:
            score += 10

    # City fiscal health (range on debt_outstanding_pc)
    fiscal_prefs = preferences.get("city_fiscal_health", [])
    if isinstance(fiscal_prefs, str):
        fiscal_prefs = [fiscal_prefs]
    if fiscal_prefs:
        max_score += 20
        debt = city.get("debt_outstanding_pc")
        if pd.notna(debt):
            rq = _get_range_questions().get("city_fiscal_health", {})
            score += _range_score(debt, fiscal_prefs,
                                  rq.get("range_map", {}), full=20, partial=14, miss=5)
        else:
            score += 10

    # Safety priority (range on crime rate)
    safety_prefs = preferences.get("safety_priority", [])
    if isinstance(safety_prefs, str):
        safety_prefs = [safety_prefs]
    if safety_prefs:
        max_score += 20
        rq = _get_range_questions().get("safety_priority", {})
        score += _range_score(city["crime_rate_per_1000"], safety_prefs,
                              rq.get("range_map", {}), full=20, partial=14, miss=5)

    # Geography (independent multi-select)
    geo_prefs = preferences.get("geography", [])
    if geo_prefs:
        max_score += 15
        geo_score = 0
        geo_count = len(geo_prefs)

        if "mountains" in geo_prefs:
            geo_score += 15 if city["has_mountains"] else 5
        if "ocean_coast" in geo_prefs:
            geo_score += 15 if city["has_ocean"] else 5
        if "lakes_rivers" in geo_prefs:
            lake_mi = city["nearest_boatable_lake_miles"]
            geo_score += 15 if lake_mi < 30 else (12 if lake_mi < 60 else 8)
        if "plains_prairies" in geo_prefs:
            geo_score += 15 if city["region"] == "Midwest" else 10
        if "desert" in geo_prefs:
            geo_score += 15 if city["has_desert"] else 5

        if geo_count > 0:
            score += geo_score / geo_count

    return (score / max_score * 100) if max_score > 0 else 0


def calculate_city_scores(preferences, top_n=10):
    """
    Calculate scores for all cities based on user preferences.

    Args:
        preferences: Dict of user preferences from quiz
        top_n: Number of top cities to return

    Returns:
        List of dicts with city info and scores, sorted by total score
    """
    cities_df = load_cities()

    # Extract user financials for tax estimation (if provided)
    financials = preferences.get("my_financials", {})
    my_income = financials.get("my_income", 0)
    my_home_value = financials.get("my_home_value", 0)
    my_annual_expenses = financials.get("my_annual_expenses", 0)
    has_financials = my_income > 0 or my_home_value > 0 or my_annual_expenses > 0

    results = []
    for _, city in cities_df.iterrows():
        climate_score = score_climate(city, preferences)
        size_score = score_city_size(city, preferences)
        cost_score = score_cost_taxes(city, preferences)
        outdoor_score = score_outdoor_recreation(city, preferences)
        lifestyle_score = score_lifestyle(city, preferences)
        education_score = score_education(city, preferences)
        practical_score = score_practical(city, preferences)

        category_scores = [
            climate_score, size_score, cost_score, outdoor_score,
            lifestyle_score, education_score, practical_score
        ]
        total_score = np.mean([s for s in category_scores if s > 0])

        result = {
            "city_id": city["city_id"],
            "name": city["name"],
            "state": city["state"],
            "lat": city["lat"],
            "lon": city["lon"],
            "population": city["population"],
            "region": city["region"],
            "total_score": round(total_score, 1),
            "scores": {
                "climate": round(climate_score, 1),
                "city_size": round(size_score, 1),
                "cost_taxes": round(cost_score, 1),
                "outdoor": round(outdoor_score, 1),
                "lifestyle": round(lifestyle_score, 1),
                "education": round(education_score, 1),
                "practical": round(practical_score, 1),
            },
            "stats": {
                "avg_temp_summer": round(city["avg_temp_summer"], 0),
                "avg_temp_winter": round(city["avg_temp_winter"], 0),
                "sunny_days": city["sunny_days"],
                "cost_of_living_index": city["cost_of_living_index"],
                "median_home_price": city["median_home_price"],
                "no_income_tax": city["no_income_tax_state"],
                "walkability_score": city["walkability_score"],
                "crime_rate": city["crime_rate_per_1000"],
            }
        }

        if has_financials:
            taxes = estimate_taxes(
                my_income, my_home_value, my_annual_expenses,
                city["state_income_tax_rate"],
                city["avg_property_tax_rate"],
                city["state_sales_tax_rate"],
                city.get("goods_rpp", 100.0),
                city.get("cost_of_living_index", 100.0),
            )
            result["estimated_taxes"] = taxes

        results.append(result)

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results[:top_n]


def get_city_details(city_id):
    """Get full details for a specific city."""
    cities_df = load_cities()
    city = cities_df[cities_df["city_id"] == city_id]
    if city.empty:
        return None
    return city.iloc[0].to_dict()
