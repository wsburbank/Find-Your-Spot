"""
Collect professional and minor league sports team data from Wikipedia.

Source:
  Wikipedia: "List of professional sports teams in the United States and Canada"
  https://en.wikipedia.org/wiki/List_of_professional_sports_teams_in_the_United_States_and_Canada

Parses wikitable markup via the MediaWiki API to extract team name, location,
and league for all major and minor professional sports leagues in the US.

Major leagues (tier "major"):
  MLB, NBA, NFL, NHL, MLS, WNBA, NWSL

Minor/other leagues (tier "minor"):
  MiLB (all levels: AAA International/Pacific Coast, AA Eastern/Southern/Texas,
    High-A Midwest/Northwest/South Atlantic, Single-A California/Carolina/Florida State),
  AHL, ECHL, NBA G League, USL Championship, USL League One, MLS Next Pro,
  UFL, Indoor Football League, NLL, PLL

Output: data/sports.parquet
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.geo import haversine_miles
from utilities.matching import normalize_city_name, fuzzy_match_city

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "List of professional sports teams in the United States and Canada"
USER_AGENT = "FindYourSpot/0.1 (city-recommendation-app; contact@example.com)"

# Which league sections to extract, with their tier classification.
# Keys = league code (our label), values = (Wikipedia section title, tier).
LEAGUES = {
    # Major leagues
    "MLB": ("Major League Baseball", "major"),
    "NBA": ("National Basketball Association", "major"),
    "NFL": ("National Football League", "major"),
    "NHL": ("National Hockey League", "major"),
    "MLS": ("Major League Soccer", "major"),
    "WNBA": ("Women's National Basketball Association", "major"),
    "NWSL": ("National Women's Soccer League", "major"),
    # Minor/other — MiLB
    "MiLB-IL": ("International League", "minor"),
    "MiLB-PCL": ("Pacific Coast League", "minor"),
    "MiLB-EL": ("Eastern League", "minor"),
    "MiLB-SL": ("Southern League", "minor"),
    "MiLB-TL": ("Texas League", "minor"),
    "MiLB-MWL": ("Midwest League", "minor"),
    "MiLB-NWL": ("Northwest League", "minor"),
    "MiLB-SAL": ("South Atlantic League", "minor"),
    "MiLB-CAL": ("California League", "minor"),
    "MiLB-CL": ("Carolina League", "minor"),
    "MiLB-FSL": ("Florida State League", "minor"),
    # Minor — MLB partner & independent leagues
    "AA": ("American Association of Professional Baseball", "minor"),
    "ALPB": ("Atlantic League of Professional Baseball", "minor"),
    "FL": ("Frontier League", "minor"),
    "PL": ("Pioneer League", "minor"),
    # Minor — Hockey
    "AHL": ("American Hockey League", "minor"),
    "ECHL": ("ECHL", "minor"),
    "SPHL": ("SPHL", "minor"),
    "FPHL": ("Federal Prospects Hockey League", "minor"),
    # Minor — Basketball
    "G League": ("NBA G League", "minor"),
    # Minor — Soccer
    "USL-C": ("USL Championship", "minor"),
    "USL-1": ("USL League One", "minor"),
    "MLS-NP": ("MLS Next Pro", "minor"),
    "MASL": ("Major Arena Soccer League", "minor"),
    # Minor — Football
    "UFL": ("United Football League", "minor"),
    "IFL": ("Indoor Football League", "minor"),
    "NAL": ("National Arena League", "minor"),
    # Minor — Lacrosse
    "NLL": ("National Lacrosse League", "minor"),
    "PLL": ("Premier Lacrosse League", "minor"),
    # Minor — Rugby
    "MLR": ("Major League Rugby", "minor"),
    # Minor — Cricket
    "MLC": ("Major League Cricket", "minor"),
}

# Junior hockey leagues (not on the professional teams page — separate Wikipedia sources).
# These are classified as "minor" for our purposes since they have local fan bases
# and contribute to the sports culture of a city.
JUNIOR_HOCKEY_PAGES = {
    "WHL": {
        "page": "Western Hockey League",
        "tier": "minor",
    },
    "USHL": {
        "page": "United States Hockey League",
        "tier": "minor",
    },
    "NAHL": {
        "page": "North American Hockey League",
        "tier": "minor",
    },
}

US_STATE_NAMES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia", "D.C.",
}

STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "D.C.": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def fetch_wikitext() -> str:
    """Fetch the full wikitext of the sports teams list page (cached)."""
    cache_path = CACHE_DIR / "sports_wiki.txt"
    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    log.info("Fetching Wikipedia sports teams page...")
    params = {
        "action": "parse",
        "page": WIKI_PAGE,
        "prop": "wikitext",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    wikitext = resp.json()["parse"]["wikitext"]["*"]
    log.info("Fetched wikitext: %d characters", len(wikitext))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(wikitext, encoding="utf-8")
    return wikitext


def parse_location(loc_text: str) -> tuple[str, str] | None:
    """Parse a wikitable location cell into (city, state_abbr).

    Handles wiki link formats like:
      [[Baltimore|Baltimore, Maryland]]
      [[St. Petersburg, Florida]]
      [[Orchard Park (town), New York|Orchard Park, New York]]
    """
    link_match = re.search(r'\[\[([^\]]+)\]\]', loc_text)
    if not link_match:
        return None

    link_content = link_match.group(1)
    display = link_content.split("|")[-1]  # use display text if piped
    display = re.sub(r'\[\[|\]\]', '', display).strip()

    if ", " not in display:
        return None

    city, state = display.rsplit(", ", 1)
    city = city.strip()
    state = state.strip()

    if state not in US_STATE_NAMES:
        return None

    state_abbr = STATE_NAME_TO_ABBR.get(state, state)

    # Clean city: remove parentheticals, side qualifiers
    city = re.sub(r'\s*\(.*?\)\s*', '', city)
    city = re.sub(r'^(South|North|West|East)\s+Side\s+', '', city)

    return city, state_abbr


def parse_team_name(cell: str) -> str | None:
    """Extract a clean team name from wiki markup."""
    match = re.search(r"'''?\[\[([^\]]+)\]\]'''?", cell)
    if not match:
        match = re.search(r"\[\[([^\]]+)\]\]", cell)
    if not match:
        return None
    content = match.group(1)
    return content.split("|")[-1].strip()


def extract_teams_from_section(wikitext: str, section_name: str,
                                league_code: str, tier: str) -> list[dict]:
    """Parse teams from a specific league section of the wikitext."""
    # Find section header — handles === Name ===, ==== Name ====, etc.
    # Build pattern without f-strings to avoid {n,m} quantifier conflicts.
    escaped = re.escape(section_name)
    pattern = r"={2,5}\s*" + escaped + r"\s*={2,5}"
    m = re.search(pattern, wikitext)
    start = m.end() if m else -1

    if start < 0:
        log.warning("Section not found: %s", section_name)
        return []

    # Grab text until next section header (any level)
    remaining = wikitext[start:]
    next_section = re.search(r'\n={2,5}[^=]', remaining)  # noqa: no f-string
    section_text = remaining[:next_section.start()] if next_section else remaining[:8000]

    # Parse wikitable rows (split on row delimiter |- )
    teams = []
    rows = re.split(r'\n\|-', section_text)

    for row in rows:
        if not row.strip() or row.strip().startswith('!'):
            continue

        cells = re.split(r'\|\|', row)
        if len(cells) < 2:
            continue

        team_name = None
        location = None

        for cell in cells:
            cell = cell.strip()
            if not team_name and ("'''[[" in cell or ("[[" in cell and "]]" in cell)):
                parsed = parse_team_name(cell)
                if parsed:
                    team_name = parsed
                    continue
            if team_name and "[[" in cell and not location:
                parsed = parse_location(cell)
                if parsed:
                    location = parsed
                    break

        if team_name and location:
            teams.append({
                "team": team_name,
                "city": location[0],
                "state": location[1],
                "league": league_code,
                "tier": tier,
            })

    return teams


def fetch_league_wikitext(page_title: str) -> str:
    """Fetch wikitext for a specific Wikipedia page (cached)."""
    safe_name = re.sub(r'[^a-z0-9]+', '_', page_title.lower()).strip('_')
    cache_path = CACHE_DIR / f"sports_{safe_name}.txt"
    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    log.info("Fetching Wikipedia page: %s", page_title)
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    wikitext = resp.json()["parse"]["wikitext"]["*"]
    log.info("Fetched %s: %d chars", page_title, len(wikitext))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(wikitext, encoding="utf-8")
    return wikitext


def extract_teams_from_table(wikitext: str, league_code: str,
                              tier: str) -> list[dict]:
    """Parse teams from a league page that uses simple wikitable rows.

    Handles formats like:
      | [[Team Name]] || [[City, State]] || ...
    Used for junior hockey leagues (WHL, USHL, NAHL) which have their own pages
    with a different table layout than the pro teams list.
    """
    teams = []
    rows = re.split(r'\n\|-', wikitext)

    for row in rows:
        if not row.strip() or row.strip().startswith('!'):
            continue

        # Find all wiki links in this row
        links = re.findall(r'\[\[([^\]]+)\]\]', row)
        if len(links) < 2:
            continue

        team_name = None
        location = None

        for link in links:
            display = link.split("|")[-1].strip()

            # Skip arena/venue links, date links, and image links
            if any(kw in link.lower() for kw in ["arena", "center", "coliseum",
                    "stadium", "rink", "complex", "file:", "image:"]):
                continue

            # Try as location (City, State)
            if ", " in display and not team_name:
                # Could be a team with comma — check state first
                pass
            if ", " in display:
                parts = display.rsplit(", ", 1)
                state = parts[1].strip()
                if state in US_STATE_NAMES:
                    city = re.sub(r'\s*\(.*?\)\s*', '', parts[0].strip())
                    state_abbr = STATE_NAME_TO_ABBR.get(state, state)
                    location = (city, state_abbr)
                    continue

            # Otherwise treat as team name (take the first non-location link)
            if not team_name and not location:
                clean = re.sub(r'\s*\(.*?\)\s*', '', display)
                if clean and len(clean) > 2:
                    team_name = clean

        if team_name and location:
            teams.append({
                "team": team_name,
                "city": location[0],
                "state": location[1],
                "league": league_code,
                "tier": tier,
            })

    return teams


def fetch_all_teams() -> pd.DataFrame:
    """Fetch and parse all sports teams from Wikipedia."""
    wikitext = fetch_wikitext()
    all_teams = []

    # Parse professional teams from the main list page
    for code, (section_name, tier) in LEAGUES.items():
        teams = extract_teams_from_section(wikitext, section_name, code, tier)
        all_teams.extend(teams)
        log.info("  %s: %d US teams", code, len(teams))

    # Parse junior hockey leagues from their own Wikipedia pages
    for code, info in JUNIOR_HOCKEY_PAGES.items():
        try:
            page_text = fetch_league_wikitext(info["page"])
            teams = extract_teams_from_table(page_text, code, info["tier"])
            all_teams.extend(teams)
            log.info("  %s: %d US teams", code, len(teams))
        except Exception as e:
            log.warning("Failed to fetch %s: %s", code, e)

    df = pd.DataFrame(all_teams)
    df = df.drop_duplicates(subset=["team", "league"])
    log.info("Total unique US teams parsed: %d", len(df))
    return df


# Map suburb/neighborhood/borough names to the main city in our list.
# Format: (parsed_city, state) -> replacement_city OR (replacement_city, replacement_state)
CITY_ALIASES: dict[tuple[str, str], str | tuple[str, str]] = {
    ("Bronx", "NY"): "Brooklyn",
    ("Queens", "NY"): "Brooklyn",
    ("Manhattan", "NY"): "Brooklyn",
    ("Elmont", "NY"): "Brooklyn",
    ("East Rutherford", "NJ"): ("Brooklyn", "NY"),
    ("Harrison", "NJ"): ("Brooklyn", "NY"),
    ("Montclair", "NJ"): ("Brooklyn", "NY"),
    ("Bridgewater", "NJ"): ("Brooklyn", "NY"),
    ("Uniondale", "NY"): "Brooklyn",
    ("Foxborough", "MA"): "Boston",
    ("Foxboro", "MA"): "Boston",
    ("Paradise", "NV"): "Las Vegas",
    ("Inglewood", "CA"): "Los Angeles",
    ("Carson", "CA"): "Los Angeles",
    ("El Segundo", "CA"): "Los Angeles",
    ("Landover", "MD"): "Washington",
    ("Chester", "PA"): "Philadelphia",
    ("Sunrise", "FL"): "Fort Lauderdale",
    ("Commerce City", "CO"): "Denver",
    ("Centennial", "CO"): "Denver",
    ("Loveland", "CO"): "Fort Collins",
    ("Sandy", "UT"): "Salt Lake City",
    ("South Jordan", "UT"): "Salt Lake City",
    ("West Valley City", "UT"): "Salt Lake City",
    ("Saint Paul", "MN"): "Minneapolis",
    ("Blaine", "MN"): "Minneapolis",
    ("Evanston", "IL"): "Chicago",
    ("Bridgeview", "IL"): "Chicago",
    ("Hoffman Estates", "IL"): "Chicago",
    ("Cary", "NC"): "Raleigh",
    ("High Point", "NC"): "Greensboro",
    ("Matthews", "NC"): "Charlotte",
    ("Highland Heights", "KY"): ("Cincinnati", "OH"),
    ("Durham", "NC"): "Raleigh",
    ("Norfolk", "VA"): "Virginia Beach",
    ("Moosic", "PA"): "Scranton",
    ("Lawrenceville", "GA"): "Atlanta",
    ("Duluth", "GA"): "Atlanta",
    ("College Park", "GA"): "Atlanta",
    ("Papillion", "NE"): "Omaha",
    ("Round Rock", "TX"): "Austin",
    ("Sugar Land", "TX"): "Houston",
    ("Cedar Park", "TX"): "Austin",
    ("Arlington", "TX"): "Fort Worth",
    ("Mansfield", "TX"): "Fort Worth",
    ("Celina", "TX"): "Fort Worth",
    ("Springdale", "AR"): "Fayetteville",
    ("Kissimmee", "FL"): "Orlando",
    ("St. Petersburg", "FL"): "Tampa",
    ("Estero", "FL"): "Fort Myers",
    ("Jupiter", "FL"): "West Palm Beach",
    ("Fort Myers", "FL"): "Cape Coral",
    ("Clearwater", "FL"): "Tampa",
    ("Ashwaubenon", "WI"): "Green Bay",
    ("Oshkosh", "WI"): "Appleton",
    ("Moline", "IL"): ("Davenport", "IA"),
    ("Kansas City", "KS"): ("Kansas City", "MO"),
    ("Chandler", "AZ"): "Phoenix",
    ("Tempe", "AZ"): "Phoenix",
    ("New York City", "NY"): "Brooklyn",
    ("Irvine", "CA"): "Anaheim",
    ("Ontario", "CA"): "Rancho Cucamonga",
    ("Rancho Cucamonga", "CA"): "Rancho Cucamonga",
    ("Garden City", "ID"): "Boise",
    ("Fishers", "IN"): "Indianapolis",
    ("Noblesville", "IN"): "Indianapolis",
    ("Coralville", "IA"): "Iowa City",
    ("Independence", "MO"): "Kansas City",
    ("Mount Pleasant", "SC"): "Charleston",
    ("Hamtramck", "MI"): "Detroit",
    ("Pawtucket", "RI"): "Providence",
    ("Uncasville", "CT"): "New London",
    ("Bowie", "MD"): "Annapolis",
    ("Pasco", "WA"): "Kennewick",
    ("Tukwila", "WA"): "Seattle",
    ("Hillsboro", "OR"): "Portland",
    ("Hershey", "PA"): "Harrisburg",
    ("Wilkes-Barre", "PA"): "Scranton",
    ("East Ridge", "TN"): "Chattanooga",
    ("Madison", "AL"): "Huntsville",
    ("Edinburg", "TX"): "McAllen",
    ("Thousand Palms", "CA"): "Palm Springs",
    ("Moraga", "CA"): "Oakland",
    ("Thousand Oaks", "CA"): "Oxnard",
    ("Seaside", "CA"): "Salinas",
    # Junior hockey suburbs/small towns
    ("Geneva", "IL"): "Chicago",
    ("Middleton", "WI"): "Madison",
    ("Plymouth", "MI"): "Detroit",
    ("West Des Moines", "IA"): "Des Moines",
    ("Kent", "WA"): "Seattle",
    ("Forest Lake", "MN"): "Minneapolis",
    ("Mason City", "IA"): "Des Moines",
    ("Auburn", "ME"): "Lewiston",
    ("Cranberry Township", "PA"): "Pittsburgh",
    # Additional unmatched from pro/indie leagues
    ("Newark", "NJ"): ("Brooklyn", "NY"),
    ("Central Islip", "NY"): "Brooklyn",
    ("Hempstead", "NY"): "Brooklyn",
    ("Staten Island", "NY"): "Brooklyn",
    ("Pomona", "NY"): "Brooklyn",
    ("Troy", "NY"): "Albany",
    ("Joliet", "IL"): "Chicago",
    ("Crestwood", "IL"): "Chicago",
    ("Franklin", "WI"): "Milwaukee",
    ("Oconomowoc", "WI"): "Milwaukee",
    ("Pelham", "AL"): "Birmingham",
    ("Towson", "MD"): "Baltimore",
    ("St. Charles", "MO"): "St. Louis",
    ("Quincy", "MA"): "Boston",
    ("Canton", "MA"): "Boston",
    ("Bossier City", "LA"): "Shreveport",
    ("Ralston", "NE"): "Omaha",
    ("Rogers", "AR"): "Fayetteville",
    ("Lakewood Ranch", "FL"): "Sarasota",
    ("Mount Vernon", "NY"): "Brooklyn",
    ("Florence", "KY"): ("Cincinnati", "OH"),
    ("North Richland Hills", "TX"): "Fort Worth",
    ("Grand Prairie", "TX"): "Fort Worth",
    ("Addison", "TX"): "Fort Worth",
    ("Urbandale", "IA"): "Des Moines",
    ("Leesburg", "VA"): "Washington",
    ("Fairport", "NY"): "Rochester",
    ("Bloomington", "MN"): "Minneapolis",
    ("Anoka", "MN"): "Minneapolis",
    ("Dundee", "IL"): "Chicago",
    ("Odenton", "MD"): "Annapolis",
    ("Stateline", "NV"): "Reno",
    ("Lake Elsinore", "CA"): "Riverside",
}


def match_teams_to_cities(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Match teams to the master city list by name + state."""
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    matched_city_ids = []

    for _, team in teams_df.iterrows():
        team_city = team["city"]
        team_state = team["state"]

        # Apply city aliases for suburb/neighborhood -> main city
        alias_key = (team_city, team_state)
        if alias_key in CITY_ALIASES:
            alias = CITY_ALIASES[alias_key]
            if isinstance(alias, tuple):
                team_city, team_state = alias
            else:
                team_city = alias

        state_cities = cities[cities["state"] == team_state]
        best_id = None

        # Exact name match
        norm_team = normalize_city_name(team_city).lower()
        for _, city in state_cities.iterrows():
            norm_city = normalize_city_name(city["name"]).lower()
            if norm_team == norm_city:
                best_id = city["city_id"]
                break
            if norm_team in norm_city or norm_city in norm_team:
                best_id = city["city_id"]

        # Fuzzy fallback
        if best_id is None:
            result = fuzzy_match_city(team_city, team_state, cities)
            if result is not None:
                best_id = result["city_id"]

        matched_city_ids.append(best_id)

    teams_df = teams_df.copy()
    teams_df["city_id"] = matched_city_ids

    matched = teams_df["city_id"].notna().sum()
    log.info("Matched %d/%d teams to cities (%.1f%%)",
             matched, len(teams_df), matched / len(teams_df) * 100)

    unmatched = teams_df[teams_df["city_id"].isna()]
    if len(unmatched) > 0:
        log.info("Unmatched teams (%d):", len(unmatched))
        for _, t in unmatched.iterrows():
            log.info("  %s (%s) — %s, %s", t["team"], t["league"],
                     t["city"], t["state"])

    return teams_df


def compute_city_sports_metrics(
    teams_df: pd.DataFrame,
    cities: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate team data per city."""
    matched = teams_df.dropna(subset=["city_id"]).copy()
    matched["city_id"] = matched["city_id"].astype(int)

    major = matched[matched["tier"] == "major"]
    minor = matched[matched["tier"] == "minor"]

    major_counts = major.groupby("city_id").size().reset_index(name="major_pro_teams")
    minor_counts = minor.groupby("city_id").size().reset_index(name="minor_pro_teams")
    total_counts = matched.groupby("city_id").size().reset_index(name="total_pro_teams")

    major_leagues = (major.groupby("city_id")["league"]
                     .apply(lambda x: ", ".join(sorted(x.unique())))
                     .reset_index(name="pro_leagues"))

    result = cities[["city_id", "name", "state"]].copy()
    result = result.merge(major_counts, on="city_id", how="left")
    result = result.merge(minor_counts, on="city_id", how="left")
    result = result.merge(total_counts, on="city_id", how="left")
    result = result.merge(major_leagues, on="city_id", how="left")

    for col in ["major_pro_teams", "minor_pro_teams", "total_pro_teams"]:
        result[col] = result[col].fillna(0).astype(int)
    result["pro_leagues"] = result["pro_leagues"].fillna("")
    result["has_major_pro_team"] = result["major_pro_teams"] > 0
    result["has_minor_pro_team"] = result["minor_pro_teams"] > 0

    log.info("Cities with major pro teams: %d", result["has_major_pro_team"].sum())
    log.info("Cities with minor pro teams: %d", result["has_minor_pro_team"].sum())
    log.info("Cities with any pro team: %d", (result["total_pro_teams"] > 0).sum())

    return result


def main():
    teams = fetch_all_teams()
    if teams.empty:
        log.error("No teams parsed!")
        sys.exit(1)

    print(f"\nParsed {len(teams)} US teams:")
    print(f"  Major: {(teams['tier'] == 'major').sum()}")
    print(f"  Minor: {(teams['tier'] == 'minor').sum()}")
    print("\nBy league:")
    for league in sorted(teams["league"].unique()):
        count = (teams["league"] == league).sum()
        tier = teams[teams["league"] == league]["tier"].iloc[0]
        print(f"  {league:12s} ({tier}): {count}")

    # Match to city list
    teams = match_teams_to_cities(teams)

    # Aggregate
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    result = compute_city_sports_metrics(teams, cities)

    # Summary
    print(f"\n{'='*60}")
    print("SPORTS DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(result)}")
    print(f"Cities with major pro team: {result['has_major_pro_team'].sum()}")
    print(f"Cities with minor pro team: {result['has_minor_pro_team'].sum()}")
    print(f"Cities with any pro team: {(result['total_pro_teams'] > 0).sum()}")

    print(f"\nTop 20 cities by total teams:")
    top = result.nlargest(20, "total_pro_teams")
    for _, row in top.iterrows():
        print(f"  {row['name']}, {row['state']}: "
              f"{row['major_pro_teams']} major + {row['minor_pro_teams']} minor "
              f"= {row['total_pro_teams']} total"
              f"  [{row['pro_leagues']}]")

    # Save
    output_cols = ["city_id", "name", "state", "major_pro_teams", "minor_pro_teams",
                   "total_pro_teams", "has_major_pro_team", "has_minor_pro_team",
                   "pro_leagues"]
    output_path = DATA_DIR / "sports.parquet"
    write_parquet_with_metadata(
        result[output_cols],
        output_path,
        source_name="Wikipedia — List of professional sports teams in the US and Canada",
        source_url="https://en.wikipedia.org/wiki/List_of_professional_sports_teams_in_the_United_States_and_Canada",
        date_collected="2026-05-02",
        notes=(
            "Professional and minor league sports team counts per city, parsed from "
            "Wikipedia's comprehensive team list via the MediaWiki API. "
            "Major leagues: MLB, NBA, NFL, NHL, MLS, WNBA, NWSL. "
            "Minor leagues: MiLB (AAA through Single-A), AHL, ECHL, G League, "
            "USL Championship, USL League One, MLS Next Pro, UFL, IFL, NLL. "
            "Teams matched to cities by name + state using fuzzy matching."
        ),
    )
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
