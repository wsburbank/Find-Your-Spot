"""
Scoring Algorithm for Find Your Spot
Matches user preferences to city attributes and returns ranked results.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_cities():
    """Load the city database."""
    data_path = Path(__file__).parent.parent / "data" / "cities.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"City database not found at {data_path}. "
            "Please run scripts/build_city_database.py first."
        )
    return pd.read_parquet(data_path)


def score_climate(city, preferences):
    """Score city based on climate preferences (0-100)."""
    score = 0
    max_score = 0

    # Temperature preference
    temp_pref = preferences.get("temp_preference")
    if temp_pref:
        max_score += 25
        avg_summer = city["avg_temp_summer"]
        avg_winter = city["avg_temp_winter"]

        if temp_pref == "hot_mild":
            if avg_summer >= 85 and avg_winter >= 45:
                score += 25
            elif avg_summer >= 80 and avg_winter >= 35:
                score += 18
            else:
                score += 8
        elif temp_pref == "four_seasons":
            temp_range = avg_summer - avg_winter
            if 40 <= temp_range <= 60 and avg_winter < 40:
                score += 25
            elif 30 <= temp_range <= 70:
                score += 18
            else:
                score += 8
        elif temp_pref == "mild_year_round":
            if 60 <= avg_summer <= 80 and avg_winter >= 40:
                score += 25
            elif avg_summer <= 85 and avg_winter >= 30:
                score += 18
            else:
                score += 8
        elif temp_pref == "love_cold":
            if avg_winter < 25:
                score += 25
            elif avg_winter < 35:
                score += 18
            else:
                score += 8

    # Humidity preference (using rainfall as proxy - higher rainfall = higher humidity generally)
    humidity_pref = preferences.get("humidity_preference")
    if humidity_pref:
        max_score += 20
        rainfall = city["annual_rainfall"]
        # Approximating humidity from rainfall and region
        is_humid = rainfall > 40 or city["region"] in ["Southeast", "Midwest"]
        is_dry = rainfall < 20 or city["region"] in ["Southwest", "Mountain"]

        if humidity_pref == "low_humidity":
            if is_dry:
                score += 20
            elif rainfall < 30:
                score += 15
            else:
                score += 5
        elif humidity_pref == "moderate_humidity":
            if 20 <= rainfall <= 45:
                score += 20
            else:
                score += 12
        elif humidity_pref == "high_humidity":
            if is_humid:
                score += 20
            else:
                score += 10
        elif humidity_pref == "humidity_not_factor":
            score += 20

    # Rain preference
    rain_pref = preferences.get("rain_preference")
    if rain_pref:
        max_score += 15
        rainfall = city["annual_rainfall"]

        if rain_pref == "love_rain":
            if rainfall >= 45:
                score += 15
            elif rainfall >= 35:
                score += 10
            else:
                score += 5
        elif rain_pref == "occasional_rain":
            if 25 <= rainfall <= 45:
                score += 15
            else:
                score += 8
        elif rain_pref == "keep_dry":
            if rainfall < 20:
                score += 15
            elif rainfall < 30:
                score += 10
            else:
                score += 3
        elif rain_pref == "monsoon":
            if rainfall >= 50:
                score += 15
            elif rainfall >= 40:
                score += 10
            else:
                score += 8

    # Sunshine preference
    sun_pref = preferences.get("sunshine_preference")
    if sun_pref:
        max_score += 20
        sunny_days = city["sunny_days"]

        if sun_pref == "max_sunshine":
            if sunny_days >= 300:
                score += 20
            elif sunny_days >= 260:
                score += 15
            elif sunny_days >= 220:
                score += 8
            else:
                score += 3
        elif sun_pref == "moderate_sunshine":
            if 200 <= sunny_days <= 280:
                score += 20
            else:
                score += 12
        elif sun_pref == "cloudy_cozy":
            if sunny_days < 200:
                score += 20
            elif sunny_days < 240:
                score += 15
            else:
                score += 8
        elif sun_pref == "dont_care_sun":
            score += 20

    # Snow preference
    snow_pref = preferences.get("snow_preference")
    if snow_pref:
        max_score += 20
        snow = city["annual_snow"]

        if snow_pref == "ski_essential":
            if snow >= 50:
                score += 20
            elif snow >= 30:
                score += 15
            elif snow >= 10:
                score += 8
            else:
                score += 3
        elif snow_pref == "light_snow":
            if 5 <= snow <= 30:
                score += 20
            elif snow < 50:
                score += 15
            else:
                score += 8
        elif snow_pref == "no_snow":
            if snow < 5:
                score += 20
            elif snow < 15:
                score += 12
            else:
                score += 3
        elif snow_pref == "occasional_snow":
            if 10 <= snow <= 40:
                score += 20
            else:
                score += 12

    return (score / max_score * 100) if max_score > 0 else 0


def score_city_size(city, preferences):
    """Score city based on size/density preferences (0-100)."""
    score = 0
    max_score = 0

    # City size preference (now multi-select)
    size_prefs = preferences.get("city_size", [])
    if isinstance(size_prefs, str):
        size_prefs = [size_prefs]

    if size_prefs:
        max_score += 40
        pop = city["population"]

        # Check if city matches ANY of the selected size preferences
        matches = False
        if "big_metro" in size_prefs and pop >= 1000000:
            matches = True
        if "mid_size" in size_prefs and 100000 <= pop < 1000000:
            matches = True
        if "small_city" in size_prefs and 25000 <= pop < 100000:
            matches = True
        if "small_town" in size_prefs and pop < 25000:
            matches = True

        if matches:
            score += 40
        else:
            # Partial score for close matches
            if "mid_size" in size_prefs and (50000 <= pop < 100000 or 1000000 <= pop < 1500000):
                score += 25
            elif "small_city" in size_prefs and (10000 <= pop < 25000 or 100000 <= pop < 150000):
                score += 25
            else:
                score += 10

    # Density preference
    density_pref = preferences.get("density_preference")
    if density_pref:
        max_score += 30
        walkability = city["walkability_score"]

        if density_pref == "urban_jungle":
            if walkability >= 80:
                score += 30
            elif walkability >= 60:
                score += 20
            else:
                score += 10
        elif density_pref == "suburban":
            if 40 <= walkability <= 70:
                score += 30
            else:
                score += 15
        elif density_pref == "rural_access":
            if walkability < 50:
                score += 30
            elif walkability < 70:
                score += 20
            else:
                score += 10
        elif density_pref == "off_grid":
            if walkability < 30:
                score += 30
            elif walkability < 50:
                score += 20
            else:
                score += 10

    # Commute preferences (slider-based)
    commute_prefs = preferences.get("commute_preferences", {})
    if commute_prefs:
        max_score += 30
        walk = city["walkability_score"]
        transit = city["transit_score"]

        walk_pref = commute_prefs.get("walk_bike_pref", 5)
        transit_pref = commute_prefs.get("transit_pref", 5)
        driving_pref = commute_prefs.get("driving_pref", 5)

        commute_score = 0

        # Score based on how well city matches commute preferences
        if walk_pref >= 7:
            commute_score += (walk / 100) * 10 * (walk_pref / 10)
        if transit_pref >= 7:
            commute_score += (transit / 100) * 10 * (transit_pref / 10)
        if driving_pref >= 7:
            # Smaller cities are better for driving
            drive_score = 100 - min(100, city["population"] / 50000)
            commute_score += (drive_score / 100) * 10 * (driving_pref / 10)

        # Normalize
        total_weight = max(1, (walk_pref + transit_pref + driving_pref) / 10)
        score += min(30, commute_score / total_weight * 3)

    return (score / max_score * 100) if max_score > 0 else 0


def score_cost_taxes(city, preferences):
    """Score city based on cost of living and tax preferences (0-100)."""
    score = 0
    max_score = 0

    # Max home price (slider-based)
    max_price = preferences.get("max_home_price")
    if max_price:
        max_score += 30
        home_price = city["median_home_price"]

        if home_price <= max_price:
            # Full score if under budget, bonus for being well under
            budget_ratio = home_price / max_price
            if budget_ratio <= 0.7:
                score += 30
            elif budget_ratio <= 0.85:
                score += 25
            else:
                score += 20
        else:
            # Partial score if over budget
            over_ratio = home_price / max_price
            if over_ratio <= 1.1:
                score += 15
            elif over_ratio <= 1.25:
                score += 10
            else:
                score += 5

    # Overall cost of living
    col_pref = preferences.get("cost_of_living")
    if col_pref:
        max_score += 20
        col_index = city["cost_of_living_index"]

        if col_pref == "worth_paying":
            score += 20
        elif col_pref == "moderate_cost":
            if col_index <= 115:
                score += 20
            elif col_index <= 130:
                score += 15
            else:
                score += 8
        elif col_pref == "keep_affordable":
            if col_index <= 100:
                score += 20
            elif col_index <= 110:
                score += 15
            else:
                score += 6
        elif col_pref == "cheapest":
            if col_index <= 90:
                score += 20
            elif col_index <= 100:
                score += 15
            else:
                score += 4

    # Tax preference
    tax_pref = preferences.get("tax_preference")
    if tax_pref:
        max_score += 25
        no_income = city["no_income_tax_state"]
        income_tax = city["state_income_tax_rate"]
        property_tax = city["avg_property_tax_rate"]
        sales_tax = city["state_sales_tax_rate"]

        if tax_pref == "no_income_tax":
            if no_income:
                score += 25
            elif income_tax < 4:
                score += 15
            else:
                score += 5
        elif tax_pref == "low_property_tax":
            if property_tax < 1:
                score += 25
            elif property_tax < 1.5:
                score += 18
            else:
                score += 8
        elif tax_pref == "sales_tax_friendly":
            if sales_tax < 5:
                score += 25
            elif sales_tax < 7:
                score += 18
            else:
                score += 10
        elif tax_pref == "balanced_tax":
            if income_tax < 7 and property_tax < 2 and sales_tax < 8:
                score += 25
            else:
                score += 15
        elif tax_pref == "dont_care_tax":
            score += 25

    # Property tax tolerance
    prop_tax_pref = preferences.get("property_tax_tolerance")
    if prop_tax_pref:
        max_score += 25
        property_tax = city["avg_property_tax_rate"]

        if prop_tax_pref == "low_tax_essential":
            if property_tax < 1:
                score += 25
            elif property_tax < 1.5:
                score += 15
            else:
                score += 5
        elif prop_tax_pref == "moderate_tax":
            if 1 <= property_tax <= 2:
                score += 25
            else:
                score += 15
        elif prop_tax_pref == "pay_for_services":
            score += 25
        elif prop_tax_pref == "not_factor":
            score += 25

    return (score / max_score * 100) if max_score > 0 else 0


def score_outdoor_recreation(city, preferences):
    """Score city based on outdoor recreation preferences (0-100)."""
    score = 0
    max_score = 0

    # Winter sports
    winter_pref = preferences.get("winter_sports")
    if winter_pref:
        max_score += 30
        ski_dist = city["ski_resort_distance_miles"]

        if winter_pref == "ski_1hr":
            if ski_dist <= 60:
                score += 30
            elif ski_dist <= 100:
                score += 20
            else:
                score += 5
        elif winter_pref == "ski_daytrip":
            if ski_dist <= 180:
                score += 30
            elif ski_dist <= 300:
                score += 20
            else:
                score += 10
        elif winter_pref == "no_skiing":
            score += 30
        elif winter_pref == "hate_cold_sports":
            if city["annual_snow"] < 10:
                score += 30
            else:
                score += 15

    # Summer activities (multi-select, no "all of the above")
    summer_prefs = preferences.get("summer_activities", [])
    if summer_prefs:
        max_score += 25
        activity_score = 0
        activity_count = len(summer_prefs)

        if "mtb_trails" in summer_prefs:
            if city["mountain_biking_trails"] >= 50:
                activity_score += 25
            elif city["mountain_biking_trails"] >= 20:
                activity_score += 18
            else:
                activity_score += 10

        if "rock_climbing" in summer_prefs:
            if city["rock_climbing_areas_nearby"] >= 10:
                activity_score += 25
            elif city["rock_climbing_areas_nearby"] >= 3:
                activity_score += 18
            else:
                activity_score += 8

        if "swimming" in summer_prefs:
            if city["swimming_access"] in ["ocean", "lake"]:
                activity_score += 25
            else:
                activity_score += 15

        if "hiking" in summer_prefs:
            if city["hiking_trails_count"] >= 100:
                activity_score += 25
            elif city["hiking_trails_count"] >= 30:
                activity_score += 18
            else:
                activity_score += 10

        if "golf" in summer_prefs:
            # Most places have golf courses
            activity_score += 20

        if "fishing" in summer_prefs:
            if city["has_lakes"] or city["has_ocean"]:
                activity_score += 25
            else:
                activity_score += 15

        if activity_count > 0:
            score += activity_score / activity_count

    # Camping & nature
    camping_pref = preferences.get("camping_nature")
    if camping_pref:
        max_score += 20
        parks = city["national_parks_within_100mi"] + city["state_parks_nearby"]

        if camping_pref == "parks_essential":
            if parks >= 5:
                score += 20
            elif parks >= 2:
                score += 15
            else:
                score += 8
        elif camping_pref == "some_campgrounds":
            if city["camping_areas_count"] >= 20:
                score += 20
            elif city["camping_areas_count"] >= 10:
                score += 15
            else:
                score += 10
        elif camping_pref == "car_camping":
            score += 20
        elif camping_pref == "not_into_camping":
            score += 20

    # Water activities (multi-select)
    water_prefs = preferences.get("water_activities", [])
    if water_prefs:
        max_score += 25
        water_score = 0
        water_count = len(water_prefs)

        if "ocean_beach" in water_prefs:
            if city["has_ocean"]:
                water_score += 25
            else:
                water_score += 5

        if "lake_recreation" in water_prefs:
            if city["has_lakes"]:
                water_score += 25
            else:
                water_score += 10

        if "river_activities" in water_prefs:
            water_score += 20

        if "pool_enough" in water_prefs:
            water_score += 25

        if "water_not_priority" in water_prefs:
            water_score += 25

        if water_count > 0:
            score += water_score / water_count

    return (score / max_score * 100) if max_score > 0 else 0


def score_lifestyle(city, preferences):
    """Score city based on lifestyle and entertainment preferences (0-100)."""
    score = 0
    max_score = 0

    # Nightlife
    nightlife_pref = preferences.get("nightlife")
    if nightlife_pref:
        max_score += 20
        pop = city["population"]
        venues = city["performing_arts_venues"]

        if nightlife_pref == "vibrant_clubs":
            if pop >= 500000 and venues >= 30:
                score += 20
            elif pop >= 200000:
                score += 15
            else:
                score += 8
        elif nightlife_pref == "restaurants_bars":
            if pop >= 100000:
                score += 20
            elif pop >= 50000:
                score += 15
            else:
                score += 10
        elif nightlife_pref == "occasional_night":
            score += 20
        elif nightlife_pref == "quiet_evenings":
            if pop < 200000:
                score += 20
            else:
                score += 15

    # Arts & culture
    arts_pref = preferences.get("arts_culture")
    if arts_pref:
        max_score += 20
        museums = city["museums_count"]

        if arts_pref == "museums_essential":
            if museums >= 50:
                score += 20
            elif museums >= 20:
                score += 15
            else:
                score += 8
        elif arts_pref == "nice_to_have":
            if museums >= 10:
                score += 20
            else:
                score += 15
        elif arts_pref == "not_priority":
            score += 20

    # Live performances
    live_pref = preferences.get("live_performance")
    if live_pref:
        max_score += 20
        broadway = city["broadway_tour_stop"]
        capacity = city["concert_venue_capacity"]

        if live_pref == "broadway_essential":
            if broadway and capacity >= 10000:
                score += 20
            elif broadway or capacity >= 5000:
                score += 12
            else:
                score += 5
        elif live_pref == "local_venues":
            if city["performing_arts_venues"] >= 10:
                score += 20
            else:
                score += 15
        elif live_pref == "occasional_shows":
            score += 20
        elif live_pref == "not_important":
            score += 20

    # Sports scene
    sports_pref = preferences.get("sports_scene")
    if sports_pref:
        max_score += 20
        pro_teams = city["pro_sports_teams"]

        if sports_pref == "pro_teams":
            if pro_teams >= 3:
                score += 20
            elif pro_teams >= 1:
                score += 15
            else:
                score += 5
        elif sports_pref == "college_sports":
            if city["has_major_university"]:
                score += 20
            elif city["university_count"] >= 1:
                score += 15
            else:
                score += 10
        elif sports_pref == "recreation_leagues":
            score += 20
        elif sports_pref == "not_into_sports":
            score += 20

    # Food scene
    food_pref = preferences.get("food_scene")
    if food_pref:
        max_score += 20
        pop = city["population"]

        if food_pref == "foodie_paradise":
            if pop >= 500000:
                score += 20
            elif pop >= 200000:
                score += 15
            else:
                score += 10
        elif food_pref == "good_variety":
            if pop >= 50000:
                score += 20
            else:
                score += 15
        elif food_pref == "basics_fine":
            score += 20

    return (score / max_score * 100) if max_score > 0 else 0


def score_education(city, preferences):
    """Score city based on education and community preferences (0-100)."""
    score = 0
    max_score = 0

    # School quality
    school_pref = preferences.get("school_quality")
    if school_pref:
        max_score += 25
        school_rating = city["avg_school_rating"]

        if school_pref == "top_schools_essential":
            if school_rating >= 8:
                score += 25
            elif school_rating >= 7:
                score += 18
            else:
                score += 10
        elif school_pref == "good_schools_nice":
            if school_rating >= 6:
                score += 25
            else:
                score += 18
        elif school_pref == "schools_not_factor":
            score += 25

    # College proximity
    college_pref = preferences.get("college_proximity")
    if college_pref:
        max_score += 20
        has_uni = city["has_major_university"]
        uni_count = city["university_count"]
        cc = city["community_college_nearby"]

        if college_pref == "major_university":
            if has_uni:
                score += 20
            elif uni_count >= 1:
                score += 15
            else:
                score += 8
        elif college_pref == "community_college":
            if cc:
                score += 20
            else:
                score += 12
        elif college_pref == "college_town_vibe":
            if has_uni and city["population"] < 300000:
                score += 20
            elif uni_count >= 1:
                score += 15
            else:
                score += 10
        elif college_pref == "college_doesnt_matter":
            score += 20

    # College town vibes
    vibes_pref = preferences.get("college_town_vibes")
    if vibes_pref:
        max_score += 15
        has_uni = city["has_major_university"]
        pop = city["population"]

        if vibes_pref == "love_university_energy":
            if has_uni and pop < 500000:
                score += 15
            elif has_uni:
                score += 12
            else:
                score += 8
        elif vibes_pref == "vibes_dont_matter":
            score += 15
        elif vibes_pref == "prefer_established":
            if not has_uni or pop >= 500000:
                score += 15
            else:
                score += 10

    # Family-friendliness
    family_pref = preferences.get("family_friendliness")
    if family_pref:
        max_score += 20
        school_rating = city["avg_school_rating"]
        crime = city["crime_rate_per_1000"]

        if family_pref == "great_schools":
            if school_rating >= 7.5 and crime < 30:
                score += 20
            elif school_rating >= 6.5:
                score += 15
            else:
                score += 10
        elif family_pref == "family_activities":
            if city["state_parks_nearby"] >= 5:
                score += 20
            else:
                score += 15
        elif family_pref == "adult_focused":
            if city["population"] >= 200000:
                score += 20
            else:
                score += 15
        elif family_pref == "no_family_preference":
            score += 20

    # Diversity
    diversity_pref = preferences.get("diversity")
    if diversity_pref:
        max_score += 20
        pop = city["population"]
        is_diverse = pop >= 200000

        if diversity_pref == "very_diverse":
            if is_diverse:
                score += 20
            else:
                score += 12
        elif diversity_pref == "moderate_diversity":
            score += 20
        elif diversity_pref == "diversity_not_factor":
            score += 20

    return (score / max_score * 100) if max_score > 0 else 0


def score_practical(city, preferences):
    """Score city based on practical considerations (0-100)."""
    score = 0
    max_score = 0

    # Job market
    job_pref = preferences.get("job_market")
    if job_pref:
        max_score += 20
        industries = city["major_industries"]
        pop = city["population"]

        if job_pref == "tech_hub":
            if "Technology" in industries and pop >= 200000:
                score += 20
            elif pop >= 500000:
                score += 15
            else:
                score += 10
        elif job_pref == "healthcare_education":
            if "Healthcare" in industries or "Education" in industries:
                score += 20
            else:
                score += 15
        elif job_pref == "manufacturing_trade":
            if "Manufacturing" in industries:
                score += 20
            else:
                score += 15
        elif job_pref == "remote_work":
            score += 20
        elif job_pref == "retired_flexible":
            score += 20

    # Airport access
    airport_pref = preferences.get("airport_access")
    if airport_pref:
        max_score += 25
        is_hub = city["is_airline_hub"]
        destinations = city["direct_flight_destinations_count"]
        distance = city["airport_distance_miles"]

        if airport_pref == "major_hub":
            if is_hub and destinations >= 100:
                score += 25
            elif destinations >= 50:
                score += 18
            else:
                score += 10
        elif airport_pref == "regional_airport":
            if distance <= 60 and destinations >= 20:
                score += 25
            elif distance <= 100:
                score += 18
            else:
                score += 10
        elif airport_pref == "small_airport":
            score += 25
        elif airport_pref == "dont_fly":
            score += 25

    # Community financial health
    health_pref = preferences.get("community_health")
    if health_pref:
        max_score += 20
        unemployment = city["unemployment_rate"]
        job_growth = city["job_growth_rate"]

        if health_pref == "thriving_essential":
            if unemployment < 4 and job_growth > 2:
                score += 20
            elif unemployment < 5 and job_growth > 0:
                score += 15
            else:
                score += 8
        elif health_pref == "stable_economy":
            if unemployment < 6:
                score += 20
            else:
                score += 15
        elif health_pref == "up_and_coming":
            if job_growth > 3:
                score += 20
            else:
                score += 15
        elif health_pref == "not_concern":
            score += 20

    # Safety
    safety_pref = preferences.get("safety_priority")
    if safety_pref:
        max_score += 20
        crime = city["crime_rate_per_1000"]

        if safety_pref == "top_priority":
            if crime < 20:
                score += 20
            elif crime < 30:
                score += 15
            else:
                score += 8
        elif safety_pref == "important":
            if crime < 35:
                score += 20
            else:
                score += 12
        elif safety_pref == "moderate_concern":
            if crime < 45:
                score += 20
            else:
                score += 15
        elif safety_pref == "willing_tradeoff":
            score += 20

    # Geography (multi-select)
    geo_prefs = preferences.get("geography", [])
    if geo_prefs:
        max_score += 15
        geo_score = 0
        geo_count = len(geo_prefs)

        if "mountains" in geo_prefs:
            if city["has_mountains"]:
                geo_score += 15
            else:
                geo_score += 5

        if "ocean_coast" in geo_prefs:
            if city["has_ocean"]:
                geo_score += 15
            else:
                geo_score += 5

        if "lakes_rivers" in geo_prefs:
            if city["has_lakes"]:
                geo_score += 15
            else:
                geo_score += 10

        if "plains_prairies" in geo_prefs:
            if city["region"] == "Midwest":
                geo_score += 15
            else:
                geo_score += 10

        if "desert" in geo_prefs:
            if city["has_desert"]:
                geo_score += 15
            else:
                geo_score += 5

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

    results = []
    for _, city in cities_df.iterrows():
        # Calculate category scores
        climate_score = score_climate(city, preferences)
        size_score = score_city_size(city, preferences)
        cost_score = score_cost_taxes(city, preferences)
        outdoor_score = score_outdoor_recreation(city, preferences)
        lifestyle_score = score_lifestyle(city, preferences)
        education_score = score_education(city, preferences)
        practical_score = score_practical(city, preferences)

        # Calculate total score (equal weighting)
        category_scores = [
            climate_score, size_score, cost_score, outdoor_score,
            lifestyle_score, education_score, practical_score
        ]
        total_score = np.mean([s for s in category_scores if s > 0])

        results.append({
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
        })

    # Sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)

    return results[:top_n]


def get_city_details(city_id):
    """Get full details for a specific city."""
    cities_df = load_cities()
    city = cities_df[cities_df["city_id"] == city_id]
    if city.empty:
        return None
    return city.iloc[0].to_dict()
