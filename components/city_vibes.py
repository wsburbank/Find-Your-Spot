"""
City Vibes Module - Optional AI-generated city descriptions
Uses Claude API to generate qualitative city descriptions and highlights.

This module is optional and can be used to enhance city profiles with:
- Personality descriptions
- Cultural highlights
- "Best for" tags
- Hidden gems / local tips
"""

import json
from pathlib import Path
from typing import Optional

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
VIBES_FILE = DATA_DIR / "city_vibes.json"


def load_vibes() -> dict:
    """Load cached city vibes from JSON file."""
    if VIBES_FILE.exists():
        with open(VIBES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_vibes(vibes: dict):
    """Save city vibes to JSON file."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(VIBES_FILE, "w") as f:
        json.dump(vibes, f, indent=2)


def get_city_vibe(city_id: str) -> Optional[dict]:
    """
    Get cached vibe for a city.

    Returns dict with:
    - description: 2-3 sentence personality description
    - highlights: List of cultural highlights
    - best_for: List of "best for" tags (e.g., "foodies", "outdoor enthusiasts")
    - hidden_gems: List of local tips/hidden gems
    """
    vibes = load_vibes()
    return vibes.get(city_id)


def generate_city_vibe(city_name: str, state: str, city_stats: dict) -> dict:
    """
    Generate a city vibe using Claude API.

    Args:
        city_name: Name of the city
        state: State abbreviation
        city_stats: Dict of city statistics for context

    Returns:
        Dict with description, highlights, best_for, and hidden_gems
    """
    try:
        import anthropic
    except ImportError:
        return {
            "description": f"{city_name}, {state} is a city waiting to be explored.",
            "highlights": [],
            "best_for": [],
            "hidden_gems": [],
            "error": "Anthropic package not installed. Install with: pip install anthropic",
        }

    # Build context from stats
    context_parts = []
    if city_stats.get("population"):
        context_parts.append(f"Population: {city_stats['population']:,}")
    if city_stats.get("region"):
        context_parts.append(f"Region: {city_stats['region']}")
    if city_stats.get("avg_temp_summer"):
        context_parts.append(
            f"Climate: Summer {city_stats['avg_temp_summer']:.0f}F, "
            f"Winter {city_stats.get('avg_temp_winter', 'N/A')}F"
        )
    if city_stats.get("cost_of_living_index"):
        context_parts.append(f"Cost of living index: {city_stats['cost_of_living_index']:.0f}")
    if city_stats.get("has_mountains"):
        context_parts.append("Has mountains nearby")
    if city_stats.get("has_ocean"):
        context_parts.append("Has ocean/beach access")
    if city_stats.get("has_major_university"):
        context_parts.append("Has a major university")

    context = "\n".join(context_parts)

    prompt = f"""Generate a brief, engaging description for {city_name}, {state}.

City stats:
{context}

Please provide:
1. A 2-3 sentence personality description that captures the city's vibe and character
2. 3-5 cultural highlights or notable features
3. 3-4 "best for" tags describing who would love this city (e.g., "outdoor enthusiasts", "foodies", "young professionals")
4. 2-3 hidden gems or local tips that only residents would know

Respond in JSON format:
{{
    "description": "...",
    "highlights": ["...", "..."],
    "best_for": ["...", "..."],
    "hidden_gems": ["...", "..."]
}}
"""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        response_text = message.content[0].text

        # Try to extract JSON from the response
        import re
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            vibe_data = json.loads(json_match.group())
            return vibe_data
        else:
            return {
                "description": response_text[:500],
                "highlights": [],
                "best_for": [],
                "hidden_gems": [],
            }

    except anthropic.APIError as e:
        return {
            "description": f"{city_name}, {state} is a city with its own unique character.",
            "highlights": [],
            "best_for": [],
            "hidden_gems": [],
            "error": f"API error: {str(e)}",
        }
    except json.JSONDecodeError:
        return {
            "description": response_text[:500] if 'response_text' in locals() else "",
            "highlights": [],
            "best_for": [],
            "hidden_gems": [],
            "error": "Could not parse AI response",
        }


def generate_and_cache_vibe(city_id: str, city_name: str, state: str, city_stats: dict) -> dict:
    """
    Generate a city vibe and cache it for future use.

    Args:
        city_id: Unique city identifier
        city_name: Name of the city
        state: State abbreviation
        city_stats: Dict of city statistics

    Returns:
        Dict with vibe data
    """
    # Check cache first
    existing_vibe = get_city_vibe(city_id)
    if existing_vibe and "error" not in existing_vibe:
        return existing_vibe

    # Generate new vibe
    vibe = generate_city_vibe(city_name, state, city_stats)

    # Cache it if successful
    if "error" not in vibe:
        vibes = load_vibes()
        vibes[city_id] = vibe
        save_vibes(vibes)

    return vibe


def batch_generate_vibes(cities: list, max_cities: int = 50):
    """
    Batch generate vibes for multiple cities.

    Args:
        cities: List of dicts with city_id, name, state, and stats
        max_cities: Maximum number of cities to process (to limit API calls)

    Returns:
        Dict mapping city_id to vibe data
    """
    results = {}
    vibes = load_vibes()

    cities_to_process = [c for c in cities if c["city_id"] not in vibes][:max_cities]

    for city in cities_to_process:
        vibe = generate_city_vibe(city["name"], city["state"], city.get("stats", {}))
        if "error" not in vibe:
            vibes[city["city_id"]] = vibe
        results[city["city_id"]] = vibe

    # Save all at once
    save_vibes(vibes)

    return results
