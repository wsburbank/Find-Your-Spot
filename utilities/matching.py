"""
City name normalization and fuzzy matching for cross-source data integration.

Different data sources refer to the same city in wildly different ways:
  Census: "New York city"
  NOAA:   "NEW YORK"
  FBI:    "New York, NY"

This module normalizes these variations and uses rapidfuzz for fuzzy matching
when exact normalization is not enough.
"""
import logging
import re

import pandas as pd
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State lookup tables
# ---------------------------------------------------------------------------
STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# Reverse mapping: full name (lowercased) -> abbreviation.
STATE_ABBREV: dict[str, str] = {name.lower(): abbr for abbr, name in STATE_NAMES.items()}
# Also map abbreviations to themselves for convenience.
STATE_ABBREV.update({abbr.lower(): abbr for abbr in STATE_NAMES})

# Common prefix/suffix replacements applied during normalization.
_PREFIX_MAP = {
    "st.": "saint",
    "st ": "saint ",
    "ft.": "fort",
    "ft ": "fort ",
    "mt.": "mount",
    "mt ": "mount ",
}

# Suffixes stripped from city names.
_SUFFIX_PATTERN = re.compile(
    r"\b(city|town|village|cdp|borough|municipality|urban|metro|metropolitan"
    r"|government|consolidated|unified|balance)\b",
    re.IGNORECASE,
)

# Parenthetical content like "(balance)" or "(pt.)" — removed entirely.
_PAREN_PATTERN = re.compile(r"\s*\([^)]*\)")

# State suffix at end: ", NY" or " NY" (2-letter uppercase at end of string).
_STATE_SUFFIX = re.compile(r"[,\s]+[A-Z]{2}\s*$")


def normalize_city_name(name: str) -> str:
    """Normalize a city name for matching.

    Applies: lowercase, strip state suffix, remove common suffixes (city, town,
    CDP, etc.), handle St./Saint, Ft./Fort, Mt./Mount, remove parentheticals,
    collapse whitespace.

    Examples:
        >>> normalize_city_name("New York city")
        'new york'
        >>> normalize_city_name("Nashville-Davidson metropolitan government (balance)")
        'nashville-davidson'
        >>> normalize_city_name("St. Louis")
        'saint louis'
        >>> normalize_city_name("Urban Honolulu CDP")
        'honolulu'
    """
    s = name.strip()

    # Remove state suffix (", NY" or " NY").
    s = _STATE_SUFFIX.sub("", s)

    # Lowercase.
    s = s.lower()

    # Remove parenthetical content.
    s = _PAREN_PATTERN.sub("", s)

    # Apply prefix replacements (st. -> saint, etc.).
    for old, new in _PREFIX_MAP.items():
        if s.startswith(old):
            s = new + s[len(old):]

    # Remove common suffixes.
    s = _SUFFIX_PATTERN.sub("", s)

    # Collapse whitespace and strip.
    s = re.sub(r"[\s]+", " ", s).strip()
    # Strip trailing punctuation/hyphens.
    s = s.strip(" -,")

    return s


def normalize_state(state: str) -> str:
    """Normalize a state string to its 2-letter abbreviation.

    Args:
        state: Full state name, abbreviation, or mixed case variant.

    Returns:
        2-letter uppercase abbreviation.

    Raises:
        ValueError: If the state cannot be recognized.
    """
    key = state.strip().lower()
    if key in STATE_ABBREV:
        return STATE_ABBREV[key]
    raise ValueError(f"Unrecognized state: {state!r}")


def fuzzy_match_city(
    name: str,
    state: str,
    candidates: pd.DataFrame,
    name_col: str = "name",
    state_col: str = "state",
    threshold: int = 85,
) -> pd.Series | None:
    """Find the best matching city from candidates using fuzzy matching.

    Strategy:
      1. Filter candidates to the matching state (exact).
      2. Normalize both the query name and candidate names.
      3. Try exact match on normalized names.
      4. Fall back to rapidfuzz token_sort_ratio.
      5. Return the best match above *threshold*, or None.

    Args:
        name: City name to match.
        state: 2-letter state abbreviation.
        candidates: DataFrame of candidate cities.
        name_col: Column in *candidates* with city names.
        state_col: Column in *candidates* with state abbreviations.
        threshold: Minimum fuzzy score (0-100) to accept a match.

    Returns:
        The matching row as a Series, or None.
    """
    state = state.upper()
    state_mask = candidates[state_col].str.upper() == state
    state_candidates = candidates.loc[state_mask]

    if state_candidates.empty:
        return None

    normalized_query = normalize_city_name(name)
    normalized_names = state_candidates[name_col].map(normalize_city_name)

    # Exact normalized match.
    exact_mask = normalized_names == normalized_query
    if exact_mask.any():
        return state_candidates.loc[exact_mask.idxmax()]

    # Fuzzy match.
    choices = dict(zip(state_candidates.index, normalized_names))
    result = process.extractOne(
        normalized_query,
        choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result is not None:
        match_str, score, match_idx = result
        return state_candidates.loc[match_idx]

    return None


def batch_match_cities(
    source_df: pd.DataFrame,
    master_df: pd.DataFrame,
    source_name_col: str = "name",
    source_state_col: str = "state",
    master_name_col: str = "name",
    master_state_col: str = "state",
    threshold: int = 85,
) -> pd.DataFrame:
    """Match a source DataFrame's cities against a master city list.

    Returns a copy of *source_df* with added columns:
      - ``matched_name``: The name from master_df that matched.
      - ``matched_state``: The state from master_df that matched.
      - ``match_score``: Fuzzy match score (100 = exact normalized match).
      - ``match_method``: "exact", "fuzzy", or "unmatched".
    """
    results = []
    for idx, row in source_df.iterrows():
        src_name = str(row[source_name_col])
        src_state = str(row[source_state_col])

        try:
            src_state_norm = normalize_state(src_state)
        except ValueError:
            results.append({
                "matched_name": None,
                "matched_state": None,
                "match_score": 0,
                "match_method": "unmatched",
            })
            continue

        normalized_query = normalize_city_name(src_name)

        # Filter master to same state.
        state_mask = master_df[master_state_col].str.upper() == src_state_norm
        state_master = master_df.loc[state_mask]

        if state_master.empty:
            results.append({
                "matched_name": None,
                "matched_state": None,
                "match_score": 0,
                "match_method": "unmatched",
            })
            continue

        normalized_master = state_master[master_name_col].map(normalize_city_name)

        # Try exact.
        exact_mask = normalized_master == normalized_query
        if exact_mask.any():
            match_row = state_master.loc[exact_mask.idxmax()]
            results.append({
                "matched_name": match_row[master_name_col],
                "matched_state": match_row[master_state_col],
                "match_score": 100,
                "match_method": "exact",
            })
            continue

        # Try fuzzy.
        choices = dict(zip(state_master.index, normalized_master))
        result = process.extractOne(
            normalized_query,
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            match_str, score, match_idx = result
            match_row = state_master.loc[match_idx]
            results.append({
                "matched_name": match_row[master_name_col],
                "matched_state": match_row[master_state_col],
                "match_score": int(score),
                "match_method": "fuzzy",
            })
        else:
            results.append({
                "matched_name": None,
                "matched_state": None,
                "match_score": 0,
                "match_method": "unmatched",
            })

    match_df = pd.DataFrame(results, index=source_df.index)
    out = source_df.copy()
    for col in match_df.columns:
        out[col] = match_df[col]
    return out
