"""
Phase 2c: Collect city-level crime data from FBI Crime Data Explorer API v2.

Source: FBI Uniform Crime Reporting (UCR) / Crime Data Explorer
  - Agency list: /cde/agency/byStateAbbr/{state}
  - Offense rates: /cde/summarized/agency/{ORI}/{crime-type}?from=MM-YYYY&to=MM-YYYY

Strategy:
  1. For each state, fetch the list of law-enforcement agencies and their ORIs
  2. Filter to "City" type agencies and match to master cities by name + state
  3. Fetch monthly offense rates (violent-crime, property-crime) per matched agency
  4. Sum monthly rates to get annual rate per 100K; convert to rate per 1,000
  5. Fall back to state-level FBI UCR estimates for cities without a matched agency

Caching: agency lists and offense data are cached to JSON files in data/.cache/
to avoid hitting the rate-limited FBI API on re-runs.

Data year: 2022 (most recent complete year)
Output: data/crime.parquet
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata
from utilities.matching import normalize_city_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FBI_CDE_BASE = "https://api.usa.gov/crime/fbi/cde"
API_KEY = "iiHnOKfno2Mgkt5AynpvPpUQTEyxE77jo1RU8PIv"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "fbi_cde"

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

DATA_YEAR = 2022
FROM_DATE = f"01-{DATA_YEAR}"
TO_DATE = f"12-{DATA_YEAR}"


def load_master_cities() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")


# ---------------------------------------------------------------------------
# HTTP helpers with caching and retry
# ---------------------------------------------------------------------------

def _api_get(url: str, params: dict | None = None, max_retries: int = 4) -> dict | None:
    """GET with exponential backoff. Returns parsed JSON or None on failure."""
    if params is None:
        params = {}
    params["API_KEY"] = API_KEY

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 503):
                wait = min(2 ** (attempt + 2), 60)  # 4, 8, 16, 32 seconds
                log.warning("HTTP %d, retrying in %ds... (%s)", resp.status_code, wait, url)
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                # Bad request — don't retry, endpoint may not support this ORI
                return None
            log.warning("HTTP %d from %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            wait = min(2 ** (attempt + 2), 60)
            log.warning("Request failed (%s), retrying in %ds...", e, wait)
            time.sleep(wait)
    log.error("All retries exhausted for %s", url)
    return None


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
    return CACHE_DIR / f"{safe}.json"


def _read_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _write_cache(key: str, data: dict) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1: Fetch agencies per state
# ---------------------------------------------------------------------------

def fetch_all_agencies() -> pd.DataFrame:
    """Fetch city-type law-enforcement agencies from FBI CDE for all states."""
    log.info("Fetching agency lists for %d states...", len(STATES))
    all_agencies = []

    for state in STATES:
        cache_key = f"agencies_{state}"
        data = _read_cache(cache_key)
        if data is None:
            url = f"{FBI_CDE_BASE}/agency/byStateAbbr/{state}"
            data = _api_get(url)
            if data is None:
                log.warning("No agency data for %s", state)
                continue
            _write_cache(cache_key, data)
            time.sleep(0.5)
        else:
            log.debug("Cache hit: agencies_%s", state)

        for county_name, agencies in data.items():
            for ag in agencies:
                if ag.get("agency_type_name") == "City":
                    raw_name = ag.get("agency_name", "")
                    all_agencies.append({
                        "ori": ag["ori"],
                        "agency_name": raw_name,
                        "city_name": _extract_city_from_agency(raw_name),
                        "state": state,
                    })

        log.info("  %s: loaded agencies", state)

    df = pd.DataFrame(all_agencies)
    log.info("Total city agencies: %d", len(df))
    return df


def _extract_city_from_agency(agency_name: str) -> str:
    """Extract a city name from an agency name.

    'Denver Police Department' -> 'Denver'
    'City of Aurora Police' -> 'Aurora'
    """
    name = agency_name.strip()
    for suffix in [
        "Police Department", "Police Dept", "Police Dept.",
        "Public Safety Department", "Public Safety Dept",
        "Department of Public Safety", "Dept of Public Safety",
        "Marshals Office", "Marshal Office", "Marshal",
        "Law Enforcement", "Police",
    ]:
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)].strip(" -,")
            break
    for prefix in ["City of ", "Town of ", "Village of ", "Borough of "]:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
            break
    return name.strip()


# ---------------------------------------------------------------------------
# Step 2: Match agencies to master cities
# ---------------------------------------------------------------------------

def match_agencies_to_cities(
    agencies: pd.DataFrame,
    cities: pd.DataFrame,
) -> pd.DataFrame:
    """Match FBI agencies to master city list by normalized name + state."""
    from rapidfuzz import fuzz, process

    log.info("Matching %d agencies to %d cities...", len(agencies), len(cities))

    agencies = agencies.copy()
    agencies["norm_name"] = agencies["city_name"].map(normalize_city_name)

    cities_norm = cities[["city_id", "name", "state"]].copy()
    cities_norm["norm_name"] = cities_norm["name"].map(normalize_city_name)

    # Exact match on normalized name + state
    matched = cities_norm.merge(
        agencies[["ori", "state", "norm_name", "agency_name"]],
        on=["state", "norm_name"],
        how="inner",
    )
    log.info("Exact matches: %d", len(matched))

    # Fuzzy match for remaining cities
    matched_ids = set(matched["city_id"])
    unmatched = cities_norm[~cities_norm["city_id"].isin(matched_ids)]

    fuzzy_results = []
    for state in unmatched["state"].unique():
        state_cities = unmatched[unmatched["state"] == state]
        state_agencies = agencies[agencies["state"] == state]
        if state_agencies.empty:
            continue
        choices = dict(zip(state_agencies.index, state_agencies["norm_name"]))
        for _, city_row in state_cities.iterrows():
            result = process.extractOne(
                city_row["norm_name"], choices,
                scorer=fuzz.token_sort_ratio, score_cutoff=85,
            )
            if result is not None:
                _, score, ag_idx = result
                fuzzy_results.append({
                    "city_id": city_row["city_id"],
                    "name": city_row["name"],
                    "state": city_row["state"],
                    "norm_name": city_row["norm_name"],
                    "ori": state_agencies.at[ag_idx, "ori"],
                    "agency_name": state_agencies.at[ag_idx, "agency_name"],
                })

    if fuzzy_results:
        matched = pd.concat([matched, pd.DataFrame(fuzzy_results)], ignore_index=True)
        log.info("Fuzzy matches: %d", len(fuzzy_results))

    matched = matched.drop_duplicates(subset="city_id", keep="first")
    log.info("Total matched: %d/%d cities", len(matched), len(cities))
    return matched[["city_id", "ori", "agency_name"]]


# ---------------------------------------------------------------------------
# Step 3: Fetch offense rates per agency
# ---------------------------------------------------------------------------

def fetch_agency_crime_rates(oris: list[str]) -> pd.DataFrame:
    """Fetch annual violent and property crime rates per agency.

    Uses the /cde/summarized/agency/{ORI}/{crime-type} endpoint which returns
    monthly rates per 100,000 population. Sum monthly rates for annual total.

    If too many consecutive 503s are encountered, stops early and returns
    partial results (cached data will be preserved for future runs).
    """
    log.info("Fetching crime rates for %d agencies...", len(oris))
    results = []
    consecutive_failures = 0
    max_consecutive_failures = 10  # Stop if 10 fresh failures in a row (API down)

    for i, ori in enumerate(oris):
        # Check cache first — skip delay if both are cached
        v_cached = _read_cache(f"rate_{ori}_violent-crime_{DATA_YEAR}")
        p_cached = _read_cache(f"rate_{ori}_property-crime_{DATA_YEAR}")
        needs_fetch = v_cached is None or p_cached is None

        violent_annual = _fetch_annual_rate(ori, "violent-crime")
        if needs_fetch:
            time.sleep(0.5)
        property_annual = _fetch_annual_rate(ori, "property-crime")
        if needs_fetch:
            time.sleep(0.5)

        results.append({
            "ori": ori,
            "violent_rate_per_100k": violent_annual,
            "property_rate_per_100k": property_annual,
        })

        # Track consecutive failures for early-stop detection
        # Only count as failure if we needed to fetch (not cached) and got nothing
        if needs_fetch and violent_annual is None and property_annual is None:
            consecutive_failures += 1
        elif needs_fetch:
            consecutive_failures = 0

        if consecutive_failures >= max_consecutive_failures:
            log.warning(
                "Stopping offense fetch after %d consecutive failures (API likely down). "
                "Re-run later to fetch remaining agencies (cached results preserved).",
                max_consecutive_failures,
            )
            break

        if (i + 1) % 25 == 0:
            log.info("  Progress: %d/%d agencies...", i + 1, len(oris))

    return pd.DataFrame(results)


def _fetch_annual_rate(ori: str, crime_type: str) -> float | None:
    """Fetch annual crime rate per 100K for one agency and crime type.

    Returns the sum of monthly rates, or None if data unavailable.
    """
    cache_key = f"rate_{ori}_{crime_type}_{DATA_YEAR}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached.get("annual_rate")

    url = f"{FBI_CDE_BASE}/summarized/agency/{ori}/{crime_type}"
    params = {"from": FROM_DATE, "to": TO_DATE}
    # Single attempt for offense data — fast fail, rely on cache for persistence
    data = _api_get(url, params, max_retries=1)

    if data is None:
        # Don't cache failures — allows retry on next run
        return None

    # Parse response: find the agency-specific "Offenses" key.
    # Response contains: "{State} Offenses", "United States Offenses",
    # "{Agency Name} Offenses" — we want the LAST one (agency-specific).
    # State names are 2+ words (e.g., "Alaska", "North Carolina"), so we can't
    # reliably filter by word count. Instead: pick the Offenses key that is
    # NOT a US state name and NOT "United States".
    from utilities.matching import STATE_NAMES
    state_name_set = set(STATE_NAMES.values()) | {"United States", "District of Columbia"}

    rates = data.get("offenses", {}).get("rates", {})
    agency_rate = None
    for key, monthly in rates.items():
        if "Offenses" not in key:
            continue
        # Strip " Offenses" suffix to get the entity name
        entity = key.replace(" Offenses", "").strip()
        if entity in state_name_set:
            continue
        # This should be the agency-specific key
        if isinstance(monthly, dict):
            values = [v for v in monthly.values() if isinstance(v, (int, float))]
            if values:
                agency_rate = sum(values)
                break

    _write_cache(cache_key, {"annual_rate": agency_rate})
    return agency_rate


# ---------------------------------------------------------------------------
# State-level fallback (FBI UCR 2022)
# ---------------------------------------------------------------------------

# State crime rates per 100,000 population (2022 FBI UCR)
# Source: FBI Crime Data Explorer, Crime in the United States 2022
STATE_CRIME_RATES = {
    "AL": {"violent": 453.6, "property": 2584.0},
    "AK": {"violent": 837.8, "property": 3577.0},
    "AZ": {"violent": 484.8, "property": 2901.0},
    "AR": {"violent": 671.9, "property": 3256.0},
    "CA": {"violent": 499.5, "property": 2856.0},
    "CO": {"violent": 492.2, "property": 3614.0},
    "CT": {"violent": 183.0, "property": 1626.0},
    "DE": {"violent": 431.8, "property": 2462.0},
    "DC": {"violent": 812.0, "property": 4136.0},
    "FL": {"violent": 383.6, "property": 2121.0},
    "GA": {"violent": 400.1, "property": 2469.0},
    "HI": {"violent": 255.0, "property": 2978.0},
    "ID": {"violent": 233.9, "property": 1454.0},
    "IL": {"violent": 425.2, "property": 1848.0},
    "IN": {"violent": 399.1, "property": 2032.0},
    "IA": {"violent": 300.8, "property": 1820.0},
    "KS": {"violent": 425.0, "property": 2500.0},
    "KY": {"violent": 268.2, "property": 1750.0},
    "LA": {"violent": 639.4, "property": 3010.0},
    "ME": {"violent": 108.6, "property": 1267.0},
    "MD": {"violent": 454.1, "property": 2128.0},
    "MA": {"violent": 308.8, "property": 1262.0},
    "MI": {"violent": 478.3, "property": 1653.0},
    "MN": {"violent": 280.6, "property": 2251.0},
    "MS": {"violent": 291.2, "property": 2101.0},
    "MO": {"violent": 542.7, "property": 2802.0},
    "MT": {"violent": 453.6, "property": 2478.0},
    "NE": {"violent": 310.8, "property": 2023.0},
    "NV": {"violent": 524.8, "property": 2617.0},
    "NH": {"violent": 146.4, "property": 1135.0},
    "NJ": {"violent": 195.4, "property": 1299.0},
    "NM": {"violent": 832.2, "property": 3730.0},
    "NY": {"violent": 363.4, "property": 1556.0},
    "NC": {"violent": 410.4, "property": 2466.0},
    "ND": {"violent": 327.6, "property": 2496.0},
    "OH": {"violent": 339.8, "property": 2092.0},
    "OK": {"violent": 458.6, "property": 2772.0},
    "OR": {"violent": 291.9, "property": 3209.0},
    "PA": {"violent": 346.8, "property": 1400.0},
    "RI": {"violent": 220.6, "property": 1500.0},
    "SC": {"violent": 530.7, "property": 2710.0},
    "SD": {"violent": 501.4, "property": 1780.0},
    "TN": {"violent": 672.7, "property": 2714.0},
    "TX": {"violent": 446.5, "property": 2766.0},
    "UT": {"violent": 260.1, "property": 2914.0},
    "VT": {"violent": 172.8, "property": 1251.0},
    "VA": {"violent": 208.2, "property": 1585.0},
    "WA": {"violent": 367.8, "property": 3596.0},
    "WV": {"violent": 355.2, "property": 1540.0},
    "WI": {"violent": 324.4, "property": 1577.0},
    "WY": {"violent": 234.8, "property": 1634.0},
}


def get_state_fallback_rates() -> dict[str, dict[str, float]]:
    """State-level rates per 1,000 population (converted from per 100K)."""
    return {
        state: {
            "violent_crime_rate": round(rates["violent"] / 100, 2),
            "property_crime_rate": round(rates["property"] / 100, 2),
            "crime_rate_per_1000": round((rates["violent"] + rates["property"]) / 100, 2),
        }
        for state, rates in STATE_CRIME_RATES.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cities = load_master_cities()
    output_path = PROJECT_ROOT / "data" / "crime.parquet"

    # Step 1: Fetch agency lists (cached)
    agencies = fetch_all_agencies()

    city_level_count = 0
    state_fallback_count = 0
    city_crime: pd.DataFrame | None = None

    if not agencies.empty:
        # Step 2: Match agencies to master cities
        matches = match_agencies_to_cities(agencies, cities)

        if not matches.empty:
            # Step 3: Fetch offense rates (cached per ORI)
            rates = fetch_agency_crime_rates(matches["ori"].tolist())

            # Merge rates with city matches
            city_crime = matches.merge(rates, on="ori", how="left")

            # Convert rates from per 100K to per 1,000
            city_crime["violent_crime_rate"] = (
                city_crime["violent_rate_per_100k"] / 100
            ).round(2)
            city_crime["property_crime_rate"] = (
                city_crime["property_rate_per_100k"] / 100
            ).round(2)
            city_crime["crime_rate_per_1000"] = (
                city_crime["violent_crime_rate"].fillna(0)
                + city_crime["property_crime_rate"].fillna(0)
            ).round(2)

            # Mark cities where we got actual data (not all-zero/all-null)
            has_data = (
                city_crime["violent_rate_per_100k"].notna()
                | city_crime["property_rate_per_100k"].notna()
            )
            city_crime.loc[~has_data, ["violent_crime_rate", "property_crime_rate",
                                        "crime_rate_per_1000"]] = np.nan

            city_level_count = city_crime["crime_rate_per_1000"].notna().sum()
            log.info("City-level crime data: %d cities", city_level_count)

    # Step 4: Build output
    out = cities[["city_id", "name", "state"]].copy()

    if city_crime is not None and city_level_count > 0:
        out = out.merge(
            city_crime[["city_id", "violent_crime_rate", "property_crime_rate",
                        "crime_rate_per_1000"]],
            on="city_id", how="left",
        )
        out["crime_data_level"] = np.where(
            out["crime_rate_per_1000"].notna(), "city", None,
        )
    else:
        out["violent_crime_rate"] = np.nan
        out["property_crime_rate"] = np.nan
        out["crime_rate_per_1000"] = np.nan
        out["crime_data_level"] = None

    # Step 5: Fill remaining with state-level fallback
    state_rates = get_state_fallback_rates()
    missing = out["crime_rate_per_1000"].isna()
    if missing.any():
        for idx in out.index[missing]:
            state = out.at[idx, "state"]
            if state in state_rates:
                for col, val in state_rates[state].items():
                    out.at[idx, col] = val
                out.at[idx, "crime_data_level"] = "state"
                state_fallback_count += 1

    coverage = out["crime_rate_per_1000"].notna().sum()
    log.info("Coverage: %d/%d cities (city=%d, state-fallback=%d)",
             coverage, len(cities), city_level_count, state_fallback_count)

    write_parquet_with_metadata(
        out,
        output_path,
        source_name="FBI UCR Crime Data Explorer (2022)",
        source_url="https://api.usa.gov/crime/fbi/cde",
        date_collected="2026-05-04",
        notes=(
            f"City-level crime rates from FBI CDE agency offense data ({city_level_count} cities). "
            f"State-level FBI UCR 2022 fallback for {state_fallback_count} cities. "
            "Rates are per 1,000 population (violent + property = total). "
            "crime_data_level column: 'city' = agency-level FBI data, 'state' = state-level fallback."
        ),
    )

    print(f"\nCrime data coverage: {coverage}/{len(cities)} ({coverage / len(cities) * 100:.1f}%)")
    print(f"  City-level: {city_level_count}")
    print(f"  State-level fallback: {state_fallback_count}")
    print(out[["name", "state", "violent_crime_rate", "property_crime_rate",
               "crime_rate_per_1000", "crime_data_level"]].head(20).to_string())


if __name__ == "__main__":
    main()
