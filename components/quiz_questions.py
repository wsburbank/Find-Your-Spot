"""
Quiz Questions Module for Find Your Spot
Defines questions organized by category to match users with cities.

Question types
--------------
- **range**: Options are ordered most-to-least.  User selects all acceptable
  levels; the scoring engine converts selections to a numerical min/max via
  ``range_map`` and scores cities on how well their metric fits that window.
- **multi_select**: Independent options (pick any that apply).  No ordering.
- **slider / sliders**: Numeric input, already handled separately.
- Single-select (radio): ``multi_select`` is False and ``range`` is absent.

Range map format
----------------
Each key in ``range_map`` corresponds to an option value.  The tuple is
``(low, high)`` in the units of the city metric named in ``range_metric``.
When the user selects several contiguous options the engine unions the
intervals: ``(min(all lows), max(all highs))``.
"""

QUIZ_QUESTIONS = {
    # ------------------------------------------------------------------
    # CLIMATE & WEATHER
    # ------------------------------------------------------------------
    "Climate & Weather": [
        {
            "id": "temp_preference",
            "question": "What's your ideal temperature vibe? (select all acceptable)",
            "options": [
                ("hot_mild", "Hot summers / mild winters"),
                ("four_seasons", "Four distinct seasons"),
                ("mild_year_round", "Mild year-round"),
                ("love_cold", "I love the cold"),
            ],
            "multi_select": True,
        },
        {
            "id": "humidity_preference",
            "question": "How do you feel about humidity? (select all acceptable)",
            "options": [
                ("high_humidity", "Don't mind humid summers"),
                ("moderate_humidity", "Moderate humidity"),
                ("low_humidity", "Dry climate (desert-like)"),
                ("humidity_not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
        {
            "id": "rain_preference",
            "question": "How much rain are you comfortable with? (select all acceptable)",
            "options": [
                ("monsoon", "Monsoon season (50+ in/yr)"),
                ("love_rain", "Rainy / Pacific NW vibes (35-50 in/yr)"),
                ("occasional_rain", "Occasional rain (15-35 in/yr)"),
                ("keep_dry", "Keep it dry (under 15 in/yr)"),
                ("rain_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "annual_rainfall",
            "range_map": {
                "monsoon": (50, 999),
                "love_rain": (35, 50),
                "occasional_rain": (15, 35),
                "keep_dry": (0, 15),
                "rain_not_factor": (0, 999),
            },
        },
        {
            "id": "sunshine_preference",
            "question": "How much sunshine do you want? (select all acceptable)",
            "options": [
                ("max_sunshine", "Maximum sunshine (300+ days)"),
                ("moderate_sunshine", "Moderate sunshine (200-300 days)"),
                ("cloudy_cozy", "Cloudy days are cozy (under 200 days)"),
                ("dont_care_sun", "Don't care"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "sunny_days",
            "range_map": {
                "max_sunshine": (300, 365),
                "moderate_sunshine": (200, 300),
                "cloudy_cozy": (0, 200),
                "dont_care_sun": (0, 365),
            },
        },
        {
            "id": "snow_preference",
            "question": "Snow preferences? (select all acceptable)",
            "options": [
                ("ski_essential", "Ski season essential (50+ in/yr)"),
                ("light_snow", "Light dustings are nice (15-50 in/yr)"),
                ("occasional_snow", "Occasional snow days (3-15 in/yr)"),
                ("no_snow", "No snow please (under 3 in/yr)"),
                ("snow_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "annual_snow",
            "range_map": {
                "ski_essential": (50, 999),
                "light_snow": (15, 50),
                "occasional_snow": (3, 15),
                "no_snow": (0, 3),
                "snow_not_factor": (0, 999),
            },
        },
    ],
    # ------------------------------------------------------------------
    # CITY SIZE & DENSITY
    # ------------------------------------------------------------------
    "City Size & Density": [
        {
            "id": "city_size",
            "question": "Acceptable city sizes? (select all that apply)",
            "options": [
                ("big_metro", "Big metro (1M+)"),
                ("large_city", "Large city (100K-1M)"),
                ("mid_size", "Mid-size city (30K-100K)"),
                ("small_city", "Small city (15K-30K)"),
                ("small_town", "Town (under 15K)"),
            ],
            "multi_select": True,
        },
        {
            "id": "density_preference",
            "question": "Population density preference? (select all acceptable)",
            "options": [
                ("urban_jungle", "Urban jungle (walkability 80+)"),
                ("suburban", "Suburban feel (walkability 40-80)"),
                ("rural_access", "Rural with town access (walkability 20-40)"),
                ("off_grid", "Off the grid (walkability under 20)"),
                ("density_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "walkability_score",
            "range_map": {
                "urban_jungle": (80, 100),
                "suburban": (40, 80),
                "rural_access": (20, 40),
                "off_grid": (0, 20),
                "density_not_factor": (0, 100),
            },
        },
        {
            "id": "commute_preferences",
            "question": "Rate your commute preferences",
            "type": "sliders",
            "sliders": [
                {"id": "walk_bike_pref", "label": "Walking/Biking", "min": 0, "max": 10, "default": 5},
                {"id": "transit_pref", "label": "Public Transit", "min": 0, "max": 10, "default": 5},
                {"id": "driving_pref", "label": "Driving", "min": 0, "max": 10, "default": 5},
            ],
        },
        {
            "id": "max_commute_time",
            "question": "Maximum acceptable commute time (minutes)?",
            "type": "slider",
            "min": 10,
            "max": 90,
            "default": 30,
            "step": 5,
        },
    ],
    # ------------------------------------------------------------------
    # COST OF LIVING & HOUSING
    # ------------------------------------------------------------------
    "Cost of Living & Housing": [
        {
            "id": "max_home_price",
            "question": "Maximum home purchase price?",
            "type": "slider",
            "min": 100000,
            "max": 2000000,
            "default": 400000,
            "step": 25000,
            "format": "$%d",
        },
        {
            "id": "cost_of_living",
            "question": "Overall cost of living? (select all acceptable)",
            "options": [
                ("worth_paying", "Worth paying for quality (130+)"),
                ("moderate_cost", "Moderate (100-130)"),
                ("keep_affordable", "Keep it affordable (80-100)"),
                ("cheapest", "Cheapest possible (under 80)"),
                ("col_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "cost_of_living_index",
            "range_map": {
                "worth_paying": (130, 999),
                "moderate_cost": (100, 130),
                "keep_affordable": (80, 100),
                "cheapest": (0, 80),
                "col_not_factor": (0, 999),
            },
        },
        {
            "id": "tax_preference",
            "question": "Tax preference? (select all that matter)",
            "options": [
                ("no_income_tax", "No state income tax please"),
                ("low_property_tax", "Low property taxes matter most"),
                ("sales_tax_friendly", "Sales tax friendly"),
                ("balanced_tax", "Balanced tax burden"),
                ("dont_care_tax", "Don't care about taxes"),
            ],
            "multi_select": True,
        },
        {
            "id": "property_tax_tolerance",
            "question": "Property tax tolerance? (select all acceptable)",
            "options": [
                ("pay_for_services", "Will pay for good services (2%+)"),
                ("moderate_tax", "Moderate is fine (1-2%)"),
                ("low_tax_essential", "Low taxes essential (under 1%)"),
                ("not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "avg_property_tax_rate",
            "range_map": {
                "pay_for_services": (2.0, 10.0),
                "moderate_tax": (1.0, 2.0),
                "low_tax_essential": (0.0, 1.0),
                "not_factor": (0.0, 10.0),
            },
        },
    ],
    # ------------------------------------------------------------------
    # OUTDOOR RECREATION
    # ------------------------------------------------------------------
    "Outdoor Recreation": [
        {
            "id": "winter_sports",
            "question": "Winter sports access? (select all acceptable)",
            "options": [
                ("ski_1hr", "Ski resort within 1 hour (under 60 mi)"),
                ("ski_daytrip", "Day-trip distance OK (60-180 mi)"),
                ("ski_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "ski_resort_distance_miles",
            "range_map": {
                "ski_1hr": (0, 60),
                "ski_daytrip": (60, 180),
                "ski_not_factor": (0, 9999),
            },
        },
        {
            "id": "summer_activities",
            "question": "Summer outdoor activities? (select all that interest you)",
            "options": [
                ("hiking", "Hiking trails"),
                ("mtb_trails", "Mountain biking trails"),
                ("rock_climbing", "Rock climbing areas"),
                ("swimming", "Swimming (lakes/pools/ocean)"),
                ("golf", "Golf courses"),
                ("fishing", "Fishing"),
            ],
            "multi_select": True,
        },
        {
            "id": "camping_nature",
            "question": "Camping & nature access? (select all acceptable)",
            "options": [
                ("parks_essential", "National/state parks nearby essential (5+)"),
                ("some_campgrounds", "Some campgrounds within reach (2+)"),
                ("camping_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "_parks_total",
            "range_map": {
                "parks_essential": (5, 999),
                "some_campgrounds": (2, 999),
                "camping_not_factor": (0, 999),
            },
        },
        {
            "id": "water_activities",
            "question": "Outdoor water activities? (select all that interest you)",
            "options": [
                ("ocean_beach", "Ocean/beach access"),
                ("lake_recreation", "Lake recreation"),
                ("river_activities", "River activities (rafting, fishing)"),
                ("water_not_priority", "Not a priority"),
            ],
            "multi_select": True,
        },
    ],
    # ------------------------------------------------------------------
    # LIFESTYLE & ENTERTAINMENT
    # ------------------------------------------------------------------
    "Lifestyle & Entertainment": [
        {
            "id": "museums",
            "question": "Museums nearby? (select all acceptable)",
            "options": [
                ("museums_rich", "Museum-rich city (30+ museums)"),
                ("museums_some", "Some museums available (10+)"),
                ("museums_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "museums_count",
            "range_map": {
                "museums_rich": (30, 9999),
                "museums_some": (10, 9999),
                "museums_not_factor": (0, 9999),
            },
        },
        {
            "id": "performing_arts",
            "question": "Performing arts scene? (select all acceptable)",
            "options": [
                ("thriving_arts", "Thriving arts scene (15+ companies)"),
                ("some_theater", "Some local theater/music (5+)"),
                ("arts_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "performing_arts_venues",
            "range_map": {
                "thriving_arts": (15, 9999),
                "some_theater": (5, 9999),
                "arts_not_factor": (0, 9999),
            },
        },
        {
            "id": "concert_venues",
            "question": "Concert and live event venues? (select all acceptable)",
            "options": [
                ("major_venues", "Major concert venues (8+ venues)"),
                ("some_venues", "Some live music options (3+)"),
                ("concerts_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "concert_venue_count",
            "range_map": {
                "major_venues": (8, 9999),
                "some_venues": (3, 9999),
                "concerts_not_factor": (0, 9999),
            },
        },
        {
            "id": "sports_scene",
            "question": "Sports scene? (select all that interest you)",
            "options": [
                ("multiple_pro", "Multiple pro teams (3+)"),
                ("some_pro", "At least one pro team"),
                ("minor_league", "Minor league is fine"),
                ("sports_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "pro_sports_teams",
            "range_map": {
                "multiple_pro": (3, 99),
                "some_pro": (1, 99),
                "minor_league": (0, 99),
                "sports_not_factor": (0, 99),
            },
        },
    ],
    # ------------------------------------------------------------------
    # EDUCATION & COMMUNITY
    # ------------------------------------------------------------------
    "Education & Community": [
        {
            "id": "school_quality",
            "question": "School quality importance? (select all acceptable)",
            "options": [
                ("top_schools_essential", "Top-rated schools essential (8+/10)"),
                ("good_schools_nice", "Good schools nice to have (6+/10)"),
                ("schools_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "avg_school_rating",
            "range_map": {
                "top_schools_essential": (8, 10),
                "good_schools_nice": (6, 10),
                "schools_not_factor": (0, 10),
            },
        },
        {
            "id": "college_proximity",
            "question": "College proximity? (select all that apply)",
            "options": [
                ("college_town_vibe", "Love college town energy/vibes"),
                ("major_university", "Want a major university nearby"),
                ("community_college", "Community college access is fine"),
                ("no_college_fine", "No college nearby is fine"),
                ("college_doesnt_matter", "Doesn't matter"),
            ],
            "multi_select": True,
        },
        {
            "id": "family_friendliness",
            "question": "Family-friendliness? (select all that apply)",
            "options": [
                ("great_schools", "Great schools & safe neighborhoods"),
                ("family_activities", "Family activities important"),
                ("adult_focused", "Adult-focused"),
                ("no_family_preference", "No preference"),
            ],
            "multi_select": True,
        },
        {
            "id": "diversity",
            "question": "Diversity & inclusion? (select all acceptable)",
            "options": [
                ("very_diverse", "Very diverse community"),
                ("moderate_diversity", "Moderate diversity"),
                ("diversity_not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
    ],
    # ------------------------------------------------------------------
    # PRACTICAL CONSIDERATIONS
    # ------------------------------------------------------------------
    "Practical Considerations": [
        {
            "id": "airport_access",
            "question": "Commercial airport access? (select all acceptable)",
            "options": [
                ("hub_1hr", "Major hub within 1 hour (under 60 mi)"),
                ("hub_2hr", "Major hub within 2 hours (under 120 mi)"),
                ("regional_1hr", "Regional airport within 1 hour (under 60 mi)"),
                ("regional_2hr", "Regional airport within 2 hours (under 120 mi)"),
                ("airport_not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
        {
            "id": "community_health",
            "question": "Community financial health? (select all acceptable)",
            "options": [
                ("thriving_essential", "Economically thriving essential"),
                ("stable_economy", "Stable economy"),
                ("up_and_coming", "Up-and-coming/revitalizing"),
                ("not_concern", "Not a concern"),
            ],
            "multi_select": True,
        },
        {
            "id": "safety_priority",
            "question": "Safety priority? (select all acceptable)",
            "options": [
                ("top_priority", "Top priority (crime under 20/1000)"),
                ("important", "Important (crime under 35/1000)"),
                ("moderate_concern", "Moderate concern (crime under 45/1000)"),
                ("safety_not_factor", "Not a factor"),
            ],
            "multi_select": True,
            "range": True,
            "range_metric": "crime_rate_per_1000",
            "range_map": {
                "top_priority": (0, 20),
                "important": (0, 35),
                "moderate_concern": (0, 45),
                "safety_not_factor": (0, 999),
            },
        },
        {
            "id": "geography",
            "question": "Geography preference? (select all that apply)",
            "options": [
                ("mountains", "Mountains nearby"),
                ("ocean_coast", "Ocean/coast"),
                ("lakes_rivers", "Lakes/rivers"),
                ("plains_prairies", "Plains/prairies"),
                ("desert", "Desert landscape"),
            ],
            "multi_select": True,
        },
    ],
}


# Special question types that need custom rendering
SLIDER_QUESTIONS = ["commute_preferences", "max_commute_time", "max_home_price"]


def get_all_questions():
    """Return a flat list of all questions with their category."""
    questions = []
    for category, category_questions in QUIZ_QUESTIONS.items():
        for q in category_questions:
            questions.append({**q, "category": category})
    return questions


def get_categories():
    """Return list of category names."""
    return list(QUIZ_QUESTIONS.keys())


def get_questions_by_category(category):
    """Return questions for a specific category."""
    return QUIZ_QUESTIONS.get(category, [])


def get_total_question_count():
    """Return total number of questions (counting slider groups as 1)."""
    count = 0
    for category_questions in QUIZ_QUESTIONS.values():
        count += len(category_questions)
    return count


def is_slider_question(question_id):
    """Check if a question uses slider input."""
    return question_id in SLIDER_QUESTIONS


def get_range_questions() -> dict:
    """Return a dict of {question_id: question_def} for all range questions."""
    result = {}
    for questions in QUIZ_QUESTIONS.values():
        for q in questions:
            if q.get("range"):
                result[q["id"]] = q
    return result
