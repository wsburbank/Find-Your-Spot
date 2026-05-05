"""
Phase 3: Integrate all metric datasets into a unified cities.parquet.

Reads from:
  - data/cities_master.parquet (master city list)
  - data/climate.parquet (weather normals)
  - data/cost_of_living.parquet (housing, income, taxes)
  - data/sales_tax.parquet (city-level combined sales tax rates)
  - data/rpp.parquet (BEA Regional Price Parities — Goods index)
  - data/crime.parquet (crime rates)
  - data/airports.parquet (airport access)
  - data/walkability.parquet (walkability and transit scores from EPA SLD)
  - data/geography.parquet (geographic features — ocean, mountains, ski, national parks)
  - data/outdoor_recreation.parquet (real outdoor recreation data — state parks, campgrounds,
    hiking/biking trails, climbing areas from PAD-US/USFS/NPS/OpenBeta)
  - data/commute.parquet (commute time and transportation mode from Census ACS)
  - data/education.parquet (college/university data from NCES IPEDS)
  - data/school_ratings.parquet (K-12 school quality from NCES CCD 2022-23)
  - data/museums.parquet (museum counts from IMLS 2018 Museum Data Files)
  - data/sports.parquet (pro/minor league sports teams from official rosters)
  - data/entertainment.parquet (performing arts venues from Census CBP 2022)
  - data/employment.parquet (unemployment rate and job growth from Census ACS)
  - data/air_quality.parquet (EPA AQI annual county summary 2024)
  - data/humidity.parquet (NOAA 1991-2020 average summer dew point)
  - data/gov_finances.parquet (Census of Govts municipal finance per capita FY2023)

Output: data/cities.parquet (unified dataset for scoring)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(name: str) -> pd.DataFrame | None:
    """Load a parquet dataset, returning None if not found."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        log.warning("Dataset not found: %s", path)
        return None
    df = pd.read_parquet(path)
    log.info("Loaded %s: %d rows, %d cols", name, len(df), len(df.columns))
    return df


def main():
    # Load master city list (the base)
    cities = load_dataset("cities_master")
    if cities is None:
        log.error("Master city list not found!")
        sys.exit(1)

    log.info("Starting integration with %d cities...", len(cities))

    # Load all metric datasets
    climate = load_dataset("climate")
    cost = load_dataset("cost_of_living")
    crime = load_dataset("crime")
    airports = load_dataset("airports")
    geography = load_dataset("geography")
    outdoor_rec = load_dataset("outdoor_recreation")
    commute = load_dataset("commute")
    education = load_dataset("education")
    school_ratings = load_dataset("school_ratings")
    employment = load_dataset("employment")
    museums = load_dataset("museums")
    sports = load_dataset("sports")
    entertainment = load_dataset("entertainment")
    demographics = load_dataset("demographics")

    # Start with master list
    unified = cities.copy()

    # Merge climate data
    if climate is not None:
        climate_cols = [c for c in climate.columns
                       if c not in ("name", "state", "station_name", "station_distance_mi")]
        unified = unified.merge(
            climate[climate_cols],
            on="city_id",
            how="left",
        )
        log.info("Climate: %d/%d cities have data",
                 unified["avg_temp_summer"].notna().sum(), len(unified))

    # Merge cost of living data
    if cost is not None:
        cost_cols = [c for c in cost.columns if c not in ("name", "state")]
        unified = unified.merge(
            cost[cost_cols],
            on="city_id",
            how="left",
        )
        log.info("Cost: %d/%d cities have data",
                 unified["median_home_price"].notna().sum(), len(unified))

    # Merge BEA Regional Price Parities (Goods index)
    rpp = load_dataset("rpp")
    if rpp is not None:
        rpp_cols = ["city_id", "goods_rpp", "rpp_level"]
        unified = unified.merge(rpp[rpp_cols], on="city_id", how="left")
        log.info("RPP Goods: %d/%d cities have data (%.0f%% MSA-level)",
                 unified["goods_rpp"].notna().sum(), len(unified),
                 (unified["rpp_level"] == "msa").sum() / len(unified) * 100)

    # Merge city-level sales tax data (overrides state_sales_tax_rate from cost_of_living)
    sales_tax = load_dataset("sales_tax")
    if sales_tax is not None:
        unified = unified.merge(
            sales_tax[["city_id", "combined_sales_tax_rate", "local_sales_tax_rate",
                       "sales_tax_source", "sales_tax_level"]],
            on="city_id",
            how="left",
        )
        # Override state_sales_tax_rate with the combined rate (state + local)
        has_combined = unified["combined_sales_tax_rate"].notna()
        unified.loc[has_combined, "state_sales_tax_rate"] = unified.loc[has_combined, "combined_sales_tax_rate"]
        city_level = (unified["sales_tax_level"] == "city").sum()
        log.info("Sales tax: %d/%d city-level, rest state-avg",
                 city_level, len(unified))

    # Merge crime data
    if crime is not None:
        crime_cols = [c for c in crime.columns if c not in ("name", "state")]
        unified = unified.merge(
            crime[crime_cols],
            on="city_id",
            how="left",
        )
        log.info("Crime: %d/%d cities have data",
                 unified["crime_rate_per_1000"].notna().sum(), len(unified))

    # Merge airports data
    if airports is not None:
        airport_cols = [c for c in airports.columns
                       if c not in ("name", "state", "nearest_airport_name",
                                    "nearest_hub_code")]
        unified = unified.merge(
            airports[airport_cols],
            on="city_id",
            how="left",
        )
        log.info("Airports: %d/%d cities have data",
                 unified["airport_distance_miles"].notna().sum(), len(unified))

    # Merge walkability/transit data (EPA Smart Location Database)
    walkability = load_dataset("walkability")
    if walkability is not None:
        walk_cols = [c for c in walkability.columns if c not in ("name", "state")]
        unified = unified.merge(walkability[walk_cols], on="city_id", how="left")
        log.info("Walkability: %d/%d cities have data",
                 unified["walkability_score"].notna().sum(), len(unified))

    # Merge geography data
    if geography is not None:
        geo_cols = [c for c in geography.columns if c not in ("name", "state")]
        unified = unified.merge(
            geography[geo_cols],
            on="city_id",
            how="left",
        )
        log.info("Geography: %d/%d cities have data",
                 unified["has_ocean"].notna().sum(), len(unified))

    # Merge outdoor recreation data (real data from PAD-US, USFS, NPS, OpenBeta)
    if outdoor_rec is not None:
        rec_cols = [c for c in outdoor_rec.columns if c not in ("name", "state")]
        unified = unified.merge(
            outdoor_rec[rec_cols],
            on="city_id",
            how="left",
        )
        log.info("Outdoor rec: %d/%d cities have data",
                 unified["state_parks_nearby"].notna().sum(), len(unified))

    # Merge commute data
    if commute is not None:
        commute_cols = [c for c in commute.columns if c not in ("name", "state",
                        "fips_state", "fips_place")]
        unified = unified.merge(
            commute[commute_cols],
            on="city_id",
            how="left",
        )
        log.info("Commute: %d/%d cities have data",
                 unified["mean_commute_minutes"].notna().sum(), len(unified))

    # Merge education data (NCES IPEDS — colleges/universities)
    if education is not None:
        edu_cols = [c for c in education.columns if c not in ("name", "state")]
        unified = unified.merge(
            education[edu_cols],
            on="city_id",
            how="left",
        )
        # Fill missing education values for cities with no nearby institutions
        fill_zero = ["university_count", "community_college_count",
                     "total_institutions", "largest_enrollment",
                     "total_student_population"]
        for col in fill_zero:
            if col in unified.columns:
                unified[col] = unified[col].fillna(0).astype(int)
        fill_false = ["has_major_university", "has_r1_r2", "has_community_college"]
        for col in fill_false:
            if col in unified.columns:
                unified[col] = unified[col].fillna(False).infer_objects(copy=False).astype(bool)
        if "college_town_score" in unified.columns:
            unified["college_town_score"] = unified["college_town_score"].fillna(0.0)
        log.info("Education: %d/%d cities have institutions",
                 (unified["total_institutions"] > 0).sum(), len(unified))

    # Merge K-12 school ratings (NCES CCD 2022-23)
    if school_ratings is not None:
        sr_cols = [c for c in school_ratings.columns if c not in ("name", "state")]
        unified = unified.merge(school_ratings[sr_cols], on="city_id", how="left")
        log.info("School ratings: %d/%d cities have data",
                 unified["avg_school_rating"].notna().sum(), len(unified))

    # Merge museum data (IMLS 2018)
    if museums is not None:
        mus_cols = [c for c in museums.columns if c not in ("name", "state")]
        unified = unified.merge(museums[mus_cols], on="city_id", how="left")
        unified["museums_count"] = unified["museums_count"].fillna(0).astype(int)
        log.info("Museums: %d/%d cities have data",
                 (unified["museums_count"] > 0).sum(), len(unified))

    # Merge sports data (Wikipedia — pro and minor league teams)
    if sports is not None:
        sports_cols = [c for c in sports.columns if c not in ("name", "state")]
        unified = unified.merge(sports[sports_cols], on="city_id", how="left")
        for col in ["major_pro_teams", "minor_pro_teams", "total_pro_teams"]:
            if col in unified.columns:
                unified[col] = unified[col].fillna(0).astype(int)
        for col in ["has_major_pro_team", "has_minor_pro_team"]:
            if col in unified.columns:
                unified[col] = unified[col].fillna(False).infer_objects(copy=False).astype(bool)
        if "pro_leagues" in unified.columns:
            unified["pro_leagues"] = unified["pro_leagues"].fillna("")
        log.info("Sports: %d/%d cities have pro teams",
                 unified["has_major_pro_team"].sum(), len(unified))

    # Merge entertainment data (Census CBP 2022)
    if entertainment is not None:
        ent_cols = [c for c in entertainment.columns if c not in ("name", "state")]
        unified = unified.merge(entertainment[ent_cols], on="city_id", how="left")
        for col in ["performing_arts_venues", "concert_venue_count", "museums_cbp_count"]:
            if col in unified.columns:
                unified[col] = unified[col].fillna(0).astype(int)
        log.info("Entertainment: %d/%d cities have performing arts venues",
                 (unified["performing_arts_venues"] > 0).sum(), len(unified))

    # Merge employment data (BLS LAUS — unemployment rate and job growth)
    if employment is not None:
        emp_cols = [c for c in employment.columns if c not in ("name", "state")]
        unified = unified.merge(employment[emp_cols], on="city_id", how="left")
        log.info("Employment (BLS): %d/%d cities have unemployment data",
                 unified["unemployment_rate"].notna().sum(), len(unified))

    # Merge air quality data (EPA AQS 2024)
    air_quality = load_dataset("air_quality")
    if air_quality is not None:
        aq_cols = [c for c in air_quality.columns if c not in ("name", "state")]
        unified = unified.merge(air_quality[aq_cols], on="city_id", how="left")
        log.info("Air quality: %d/%d cities have data",
                 unified["median_aqi"].notna().sum(), len(unified))

    # Merge humidity data (NOAA 1991-2020 hourly dew point normals)
    humidity = load_dataset("humidity")
    if humidity is not None:
        hum_cols = [c for c in humidity.columns if c not in ("name", "state")]
        unified = unified.merge(humidity[hum_cols], on="city_id", how="left")
        log.info("Humidity: %d/%d cities have data",
                 unified["avg_summer_dewpoint"].notna().sum(), len(unified))

    # Merge government finance data (Census of Governments FY2023)
    gov_fin = load_dataset("gov_finances")
    if gov_fin is not None:
        gf_cols = [c for c in gov_fin.columns if c not in ("name", "state")]
        unified = unified.merge(gov_fin[gf_cols], on="city_id", how="left")
        log.info("Gov finances: %d/%d cities have data",
                 unified["police_spending_pc"].notna().sum(), len(unified))

    # Merge demographics data (Census ACS 2022 — diversity, age, education, employment)
    if demographics is not None:
        # Drop unemployment_rate from demographics — employment.parquet has
        # the authoritative version from ACS B23025 at county level
        demo_cols = [c for c in demographics.columns
                     if c not in ("name", "state", "fips_state", "fips_place",
                                  "unemployment_rate")]
        unified = unified.merge(demographics[demo_cols], on="city_id", how="left")
        log.info("Demographics: %d/%d cities have data",
                 unified["diversity_index"].notna().sum(), len(unified))

    # Add derived columns needed by scoring engine
    _add_derived_columns(unified)

    # Report coverage
    print(f"\n{'='*60}")
    print("DATA INTEGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(unified)}")
    print(f"Total columns: {len(unified.columns)}")
    print(f"\nCoverage by metric:")
    key_cols = [
        "avg_temp_summer", "avg_temp_winter", "annual_rainfall", "annual_snow",
        "sunny_days", "median_home_price", "cost_of_living_index",
        "state_income_tax_rate", "crime_rate_per_1000",
        "airport_distance_miles", "walkability_score", "transit_score",
        "has_ocean", "has_mountains", "nearest_boatable_lake_miles",
        "boatable_lakes_within_50mi", "ski_resort_distance_miles",
        "national_parks_within_100mi", "mean_commute_minutes",
        "pct_work_from_home", "university_count", "has_major_university",
        "has_community_college", "college_town_score",
        "avg_school_rating", "avg_pupil_teacher_ratio",
        "major_pro_teams", "minor_pro_teams",
        "museums_count", "performing_arts_venues", "concert_venue_count",
        "diversity_index", "median_age", "pct_bachelors_plus",
        "unemployment_rate", "poverty_rate", "pct_uninsured",
        "median_aqi", "pct_good_days", "days_unhealthy_total",
        "police_spending_pc", "fire_spending_pc", "parks_spending_pc",
        "roads_spending_pc", "total_revenue_pc",
    ]
    for col in key_cols:
        if col in unified.columns:
            n = unified[col].notna().sum()
            pct = n / len(unified) * 100
            print(f"  {col:40s}: {n:4d}/{len(unified)} ({pct:5.1f}%)")
        else:
            print(f"  {col:40s}: MISSING COLUMN")

    # Save
    output_path = DATA_DIR / "cities.parquet"
    write_parquet_with_metadata(
        unified,
        output_path,
        source_name="Find Your Spot - Integrated City Dataset",
        source_url="https://github.com/find-your-spot",
        date_collected="2026-05-02",
        notes=(
            "Unified city dataset integrating: Census population/demographics/housing/income, "
            "NOAA climate normals + hourly cloud cover (sunshine), ACS employment data, "
            "FBI crime rates, OurAirports airport data, EPA Smart Location Database "
            "(walkability/transit), USGS/NPS geographic features, NCES education data, "
            "IMLS museums, Census CBP entertainment, and Wikipedia sports teams. "
            f"Coverage: {len(unified)} cities across {unified['state'].nunique()} states."
        ),
    )

    log.info("Saved unified dataset to %s", output_path)
    print(f"\nSaved: {output_path}")
    print(f"Columns: {sorted(unified.columns.tolist())}")


def _add_derived_columns(df: pd.DataFrame) -> None:
    """Add columns needed by the scoring engine that aren't in raw data."""

    # Swimming access (derived from ocean/lake distance)
    if "has_ocean" in df.columns:
        lake_nearby = df.get("nearest_boatable_lake_miles", pd.Series(999, index=df.index)) < 50
        df["swimming_access"] = np.where(
            df["has_ocean"], "ocean",
            np.where(lake_nearby, "lake", "pool")
        )

    # University/college data now comes from real NCES IPEDS data
    # (merged from data/education.parquet above — no population proxies needed)

    # K-12 school ratings come from NCES CCD data (school_ratings.parquet).
    # If unavailable, derive from pct_bachelors_plus (education attainment proxy).
    if "avg_school_rating" not in df.columns:
        if "pct_bachelors_plus" in df.columns:
            # Scale bachelor's % (6-78%) to a 3-9.5 school rating
            df["avg_school_rating"] = np.clip(
                3.0 + (df["pct_bachelors_plus"].fillna(28) - 6) / (78 - 6) * 6.5,
                3.0, 9.5,
            ).round(1)
        else:
            df["avg_school_rating"] = 6.0

    # Employment — from Census ACS B23025 (2021-2022)
    # Leave missing values as NaN rather than filling with fake averages
    if "unemployment_rate" not in df.columns:
        log.warning("No unemployment_rate column — run scripts/collect_employment.py")
    if "job_growth_rate" not in df.columns:
        log.warning("No job_growth_rate column — run scripts/collect_employment.py")

    # Major industries — real data from Census ACS DP03
    if "major_industries" not in df.columns:
        df["major_industries"] = "Services"
    else:
        df["major_industries"] = df["major_industries"].fillna("Services")


if __name__ == "__main__":
    main()
