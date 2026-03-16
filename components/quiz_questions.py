"""
Quiz Questions Module for Find Your Spot
Defines questions organized by category to match users with cities.
"""

QUIZ_QUESTIONS = {
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
                ("low_humidity", "Dry climate (desert-like)"),
                ("moderate_humidity", "Moderate humidity"),
                ("high_humidity", "Don't mind humid summers"),
                ("humidity_not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
        {
            "id": "rain_preference",
            "question": "How do you feel about rain?",
            "options": [
                ("love_rain", "Love it (Pacific NW vibes)"),
                ("occasional_rain", "Occasional is fine"),
                ("keep_dry", "Keep it dry"),
                ("monsoon", "Monsoon season sounds fun"),
            ],
            "multi_select": False,
        },
        {
            "id": "sunshine_preference",
            "question": "Sunshine matters to me...",
            "options": [
                ("max_sunshine", "Need 300+ sunny days"),
                ("moderate_sunshine", "Moderate sunshine"),
                ("cloudy_cozy", "Cloudy days are cozy"),
                ("dont_care_sun", "Don't care"),
            ],
            "multi_select": False,
        },
        {
            "id": "snow_preference",
            "question": "Snow preferences?",
            "options": [
                ("ski_essential", "Ski season essential"),
                ("light_snow", "Light dustings are nice"),
                ("no_snow", "No snow please"),
                ("occasional_snow", "Occasional snow days"),
            ],
            "multi_select": False,
        },
    ],
    "City Size & Density": [
        {
            "id": "city_size",
            "question": "Acceptable city sizes? (select all that apply)",
            "options": [
                ("big_metro", "Big metro (1M+)"),
                ("mid_size", "Mid-size (100K-1M)"),
                ("small_city", "Small city (25K-100K)"),
                ("small_town", "Small town (<25K)"),
            ],
            "multi_select": True,
        },
        {
            "id": "density_preference",
            "question": "Population density preference?",
            "options": [
                ("urban_jungle", "Urban jungle"),
                ("suburban", "Suburban feel"),
                ("rural_access", "Rural with town access"),
                ("off_grid", "Off the grid"),
            ],
            "multi_select": False,
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
            "id": "bedrooms_needed",
            "question": "Minimum bedrooms needed?",
            "type": "slider",
            "min": 1,
            "max": 6,
            "default": 3,
            "step": 1,
        },
        {
            "id": "housing_type",
            "question": "Preferred housing type?",
            "options": [
                ("single_family", "Single family home"),
                ("townhouse", "Townhouse/condo"),
                ("apartment", "Apartment/rental"),
                ("any_housing", "Any type is fine"),
            ],
            "multi_select": False,
        },
        {
            "id": "cost_of_living",
            "question": "Overall cost of living priority? (select all acceptable)",
            "options": [
                ("worth_paying", "Worth paying for quality"),
                ("moderate_cost", "Moderate"),
                ("keep_affordable", "Keep it affordable"),
                ("cheapest", "Cheapest possible"),
            ],
            "multi_select": True,
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
                ("low_tax_essential", "Low taxes essential (<1%)"),
                ("moderate_tax", "Moderate is fine (1-2%)"),
                ("pay_for_services", "Will pay for good services (2%+)"),
                ("not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
    ],
    "Outdoor Recreation": [
        {
            "id": "winter_sports",
            "question": "Winter sports access?",
            "options": [
                ("ski_1hr", "Ski resort within 1 hour essential"),
                ("ski_daytrip", "Day-trip distance OK (2-3 hrs)"),
                ("no_skiing", "Don't need skiing"),
                ("hate_cold_sports", "Hate cold weather sports"),
            ],
            "multi_select": False,
        },
        {
            "id": "summer_activities",
            "question": "Summer outdoor activities? (select all that interest you)",
            "options": [
                ("mtb_trails", "Mountain biking trails"),
                ("rock_climbing", "Rock climbing areas"),
                ("swimming", "Swimming (lakes/pools/ocean)"),
                ("hiking", "Hiking trails"),
                ("golf", "Golf courses"),
                ("fishing", "Fishing"),
            ],
            "multi_select": True,
        },
        {
            "id": "camping_nature",
            "question": "Camping & nature access?",
            "options": [
                ("parks_essential", "National/state parks nearby essential"),
                ("some_campgrounds", "Some campgrounds within reach"),
                ("car_camping", "Car camping is fine"),
                ("not_into_camping", "Not into camping"),
            ],
            "multi_select": False,
        },
        {
            "id": "water_activities",
            "question": "Water activities?",
            "options": [
                ("ocean_beach", "Ocean/beach access"),
                ("lake_recreation", "Lake recreation"),
                ("river_activities", "River activities (rafting, fishing)"),
                ("pool_enough", "Pool access is enough"),
                ("water_not_priority", "Not a priority"),
            ],
            "multi_select": True,
        },
    ],
    "Lifestyle & Entertainment": [
        {
            "id": "nightlife",
            "question": "Nightlife & entertainment?",
            "options": [
                ("vibrant_clubs", "Vibrant club scene"),
                ("restaurants_bars", "Good restaurants/bars"),
                ("occasional_night", "Occasional night out"),
                ("quiet_evenings", "Quiet evenings"),
            ],
            "multi_select": False,
        },
        {
            "id": "arts_culture",
            "question": "Arts & culture importance?",
            "options": [
                ("museums_essential", "Museums/theater essential"),
                ("nice_to_have", "Nice to have"),
                ("not_priority", "Not a priority"),
            ],
            "multi_select": False,
        },
        {
            "id": "live_performance",
            "question": "Live performance access?",
            "options": [
                ("broadway_essential", "Broadway tours & major concerts essential"),
                ("local_venues", "Local theater & music venues"),
                ("occasional_shows", "Occasional touring shows"),
                ("not_important", "Not important"),
            ],
            "multi_select": False,
        },
        {
            "id": "sports_scene",
            "question": "Sports scene?",
            "options": [
                ("pro_teams", "Pro teams required"),
                ("college_sports", "College sports"),
                ("recreation_leagues", "Recreation leagues"),
                ("not_into_sports", "Not into sports"),
            ],
            "multi_select": False,
        },
        {
            "id": "food_scene",
            "question": "Food scene priorities?",
            "options": [
                ("foodie_paradise", "Foodie paradise"),
                ("good_variety", "Good variety"),
                ("basics_fine", "Basics are fine"),
            ],
            "multi_select": False,
        },
    ],
    "Education & Community": [
        {
            "id": "school_quality",
            "question": "School quality importance? (select all that apply)",
            "options": [
                ("top_schools_essential", "Top-rated schools essential (for kids)"),
                ("good_schools_nice", "Good schools nice to have"),
                ("schools_not_factor", "Not a factor"),
            ],
            "multi_select": True,
        },
        {
            "id": "college_proximity",
            "question": "College proximity?",
            "options": [
                ("major_university", "Want a major university nearby"),
                ("community_college", "Community college access is fine"),
                ("college_town_vibe", "Love college town energy/vibes"),
                ("college_doesnt_matter", "Doesn't matter"),
            ],
            "multi_select": False,
        },
        {
            "id": "family_friendliness",
            "question": "Family-friendliness?",
            "options": [
                ("family_activities", "Family activities important"),
                ("kid_friendly", "Kid-friendly neighborhoods"),
                ("adult_focused", "Adult-focused"),
                ("no_family_preference", "No preference"),
            ],
            "multi_select": False,
        },
        {
            "id": "diversity",
            "question": "Diversity & inclusion?",
            "options": [
                ("very_diverse", "Very diverse community"),
                ("moderate_diversity", "Moderate diversity"),
                ("diversity_not_factor", "Not a factor"),
            ],
            "multi_select": False,
        },
    ],
    "Practical Considerations": [
        {
            "id": "job_market",
            "question": "Job market focus?",
            "options": [
                ("tech_hub", "Tech hub"),
                ("healthcare_education", "Healthcare/education"),
                ("manufacturing_trade", "Manufacturing/trade"),
                ("remote_work", "Remote work (anywhere)"),
                ("retired_flexible", "Retired/flexible"),
            ],
            "multi_select": False,
        },
        {
            "id": "airport_access",
            "question": "Airport access?",
            "options": [
                ("major_hub", "Major hub airport essential (direct flights)"),
                ("regional_airport", "Regional airport within 1 hour"),
                ("small_airport", "Small airport OK (connections fine)"),
                ("dont_fly", "Don't fly much"),
            ],
            "multi_select": False,
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
            "question": "Safety priority?",
            "options": [
                ("top_priority", "Top priority"),
                ("important", "Important"),
                ("moderate_concern", "Moderate concern"),
                ("willing_tradeoff", "Willing to trade off"),
            ],
            "multi_select": False,
        },
        {
            "id": "geography",
            "question": "Geography preference?",
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
SLIDER_QUESTIONS = ["commute_preferences", "max_commute_time", "max_home_price", "bedrooms_needed"]


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
