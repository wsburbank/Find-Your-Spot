# Find Your Spot

## Overview

A Streamlit app inspired by the original FindYourSpot.com that recommends US cities based on a user's lifestyle preferences. Users take a quiz covering climate, city size, cost of living, outdoor activities, culture, safety, and more. The app scores ~1,800+ real US cities against those preferences and presents the top 10 matches on an interactive map with detailed city profiles.

## Real Data Only

**All data in this project must be real, verifiable, and sourced from public datasets. No data may be fabricated, estimated, or filled in with made-up values.**

- Every dataset used must include a reference to its **source** (organization/agency) and the **date or year** it was collected/published.
- Source references must be maintained in the code (comments or docstrings near data loading) and surfaced to users in the app's Data Documentation page.
- If real data is unavailable for a metric or city, leave it as missing/null rather than inventing a value.
- When adding new data, document:
  - Source name and URL
  - Date/year of the data
  - Any transformations applied
- Preferred public data sources include:
  - **US Census Bureau** (population, demographics, economics, ACS)
  - **NOAA** (climate normals, weather stations)
  - **FBI UCR / Crime Data Explorer** (crime rates)
  - **BLS** (cost of living, employment)
  - **NCES** (education statistics)
  - **FAA / BTS** (airport and transportation data)
  - **Census Gazetteer** (geographic coordinates, land area)

## Data Collection

- Data collection is orchestrated by **Python scripts** that Claude runs via Bash.
- Use **Firecrawl** (self-hosted via Docker at `localhost:3002`) for web scraping and structured data extraction when public APIs are not available. The `firecrawl-py` SDK wraps the local instance.
- Use direct **HTTP requests** (`requests` library) for public APIs (Census, NOAA, FBI, BLS, etc.) that provide structured JSON/CSV endpoints.
- All collected data must still follow the **Real Data Only** policy: source and date must be recorded for every dataset.
- All data sources are stored as **parquet files** with source metadata (name, URL, date) embedded in the file's schema metadata via `utilities/data_io.py`.

## Project Phases

### Phase 0: Tooling & Infrastructure

Set up the tools and utilities needed before any data work begins.

- Configure self-hosted **Firecrawl** (Docker) with `firecrawl-py` SDK wrapper (`utilities/firecrawl_client.py`)
- Build reusable utility functions:
  - `utilities/download.py` — HTTP download with file-based caching
  - `utilities/data_io.py` — Parquet read/write with embedded source metadata
  - `utilities/geo.py` — Haversine distance, city separation enforcement (state-partitioned), Census geocoding
  - `utilities/matching.py` — City name normalization and fuzzy matching with `rapidfuzz`
- Set up project structure, virtual environment, and dependencies

### Phase 1: City Selection

Determine the master list of cities before collecting any data.

**City Selection Rules:**
- USA only (all 50 states including Hawaii and Alaska)
- Include every Census place with population > 50K
- Every state (except Rhode Island) must have at least 10 cities
- Add smaller places as needed to meet the 10-per-state minimum
- Use Census place boundaries (incorporated places + CDPs) which align with widely accepted community boundaries
- Split mega-cities where borough/district-level data is available (e.g., NYC into its 5 boroughs)
- Cities must be at least 20 miles apart — nearby communities like Houston, Sugar Land, Galveston, The Woodlands, and Katy are each treated as separate cities
- Suburbs and CDPs with their own identity are included as distinct entries

### Phase 2: Data Collection & Monitoring

Download real data for each metric and build a Streamlit page for tracking progress.

**Metrics to collect for each city:**
- Weather / climate (temperature, precipitation, sunny days, humidity)
- Seasons and seasonal variation
- Snowfall and proximity to ski areas
- Proximity to water / ocean / lakes
- Airport access and size
- School rankings (K-12 and higher education)
- Cost of living (housing, groceries, utilities, transportation)
- Crime rates and safety
- Outdoor recreation opportunities
- Cultural amenities (museums, theaters, dining, nightlife)
- Job market and employment
- Population size and density
- Demographics and diversity
- Healthcare access
- Commute and transportation options

**Monitoring page:**
- Selectbox of available metrics (weather, crime, cost of living, etc.)
- Selecting a metric displays its current data, source, date collected, and row/city coverage
- Ability to trigger a data source update/refresh on request via MCP + Firecrawl

### Phase 3: Data Integration

Clean, normalize, and join all metric parquets into a unified city dataset.

- Normalize city/place names across sources (Census, NOAA, FBI, BLS all name cities differently)
- Match records by coordinates or FIPS codes where name matching fails
- Handle missing data — leave gaps as null, do not fabricate values
- Produce a single unified parquet with one row per city and all metrics as columns
- Log and surface data coverage (% of cities with data for each metric)

### Phase 4: Quiz & Results

- Build the questionnaire UI — fun, engaging questions that narrow down user preferences
- Each question allows selecting all, some, or none of the options
- Scoring algorithm matches user answers against city metrics
- Results page shows top 10 matching cities on an interactive map
- City detail views with profiles and score breakdowns

## Tech Stack

- **Python 3.13**, **Streamlit** for the UI
- **pandas** for data manipulation
- **plotly** / **pydeck** for visualization and maps
- **geopandas** for geospatial operations
- **pyarrow** for parquet file support with embedded metadata
- **rapidfuzz** for city name matching across data sources
- **Firecrawl** (self-hosted Docker) for web scraping and data extraction
- **requests** for direct API access to public data sources
