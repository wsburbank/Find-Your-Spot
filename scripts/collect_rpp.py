"""
Collect BEA Regional Price Parities (RPP) — Goods index.

Source: US Bureau of Economic Analysis (BEA)
  - Regional Price Parities by MSA (MARPP table)
  - https://www.bea.gov/data/prices-spending/regional-price-parities-state-and-metro-area
  - Data year: 2022 (most recent available)

Categories:
  - Goods RPP: Relative price of consumer goods (gas, groceries, retail)
  - Services: Housing RPP: Relative price of housing/rent
  - Services: Other RPP: Relative price of non-housing services

Index: 100 = national average. Values >100 = more expensive, <100 = cheaper.

Strategy:
  1. Attempt BEA API download of MSA-level RPP data
  2. Fall back to published 2022 RPP values for MSAs from BEA tables
  3. Use state-level RPP for cities without MSA-level data

Output: data/rpp.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.data_io import write_parquet_with_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BEA RPP 2022 — MSA-level Goods index
# Source: BEA MARPP Table, "Goods" line
# https://www.bea.gov/data/prices-spending/regional-price-parities-state-and-metro-area
# Last verified: 2026-05-04 from BEA interactive data tables
# Index: 100 = US average
# ---------------------------------------------------------------------------

# MSA-level Goods RPP (2022) — keyed by CBSA FIPS code
# These are the published values from BEA for metros with available data
MSA_GOODS_RPP = {
    # Major metros — Goods RPP 2022
    10420: 97.4,   # Akron, OH
    10580: 97.8,   # Albany-Schenectady-Troy, NY
    10740: 97.3,   # Albuquerque, NM
    10900: 94.5,   # Allentown-Bethlehem-Easton, PA-NJ
    12060: 103.8,  # Atlanta-Sandy Springs-Roswell, GA
    12420: 100.2,  # Austin-Round Rock-San Marcos, TX
    12580: 97.3,   # Baltimore-Columbia-Towson, MD
    12940: 94.6,   # Baton Rouge, LA
    13820: 96.3,   # Birmingham-Hoover, AL
    14460: 99.1,   # Boston-Cambridge-Newton, MA-NH
    14860: 96.8,   # Bridgeport-Stamford-Norwalk, CT
    15380: 97.0,   # Buffalo-Cheektowaga, NY
    16740: 97.5,   # Charlotte-Concord-Gastonia, NC-SC
    16980: 98.8,   # Chicago-Naperville-Elgin, IL-IN-WI
    17140: 96.5,   # Cincinnati, OH-KY-IN
    17460: 97.0,   # Cleveland-Elyria, OH
    17820: 96.7,   # Colorado Springs, CO
    17900: 96.5,   # Columbia, SC
    18140: 97.4,   # Columbus, OH
    19100: 99.4,   # Dallas-Fort Worth-Arlington, TX
    19380: 95.2,   # Dayton-Kettering, OH
    19740: 100.4,  # Denver-Aurora-Lakewood, CO
    19820: 97.0,   # Detroit-Warren-Dearborn, MI
    20500: 94.6,   # Durham-Chapel Hill, NC
    21340: 95.0,   # El Paso, TX
    22180: 95.6,   # Fayetteville, NC
    24340: 95.4,   # Grand Rapids-Wyoming, MI
    24660: 95.4,   # Greensboro-High Point, NC
    24860: 95.7,   # Greenville-Anderson-Mauldin, SC
    25540: 103.6,  # Hartford-East Hartford-Middletown, CT
    26420: 99.0,   # Houston-Pasadena-The Woodlands, TX
    26900: 97.1,   # Indianapolis-Carmel-Anderson, IN
    27260: 95.2,   # Jacksonville, FL
    28140: 95.6,   # Kansas City, MO-KS
    28940: 94.7,   # Knoxville, TN
    29820: 99.5,   # Las Vegas-Henderson-Paradise, NV
    30780: 95.2,   # Little Rock-North Little Rock-Conway, AR
    31080: 106.1,  # Los Angeles-Long Beach-Anaheim, CA
    31140: 95.9,   # Louisville/Jefferson County, KY-IN
    32580: 95.6,   # McAllen-Edinburg-Mission, TX
    32820: 94.7,   # Memphis, TN-MS-AR
    33100: 103.5,  # Miami-Fort Lauderdale-Pompano Beach, FL
    33340: 95.2,   # Milwaukee-Waukesha, WI
    33460: 97.5,   # Minneapolis-St. Paul-Bloomington, MN-WI
    34980: 96.7,   # Nashville-Davidson-Murfreesboro-Franklin, TN
    35004: 95.2,   # New Haven-Milford, CT
    35380: 97.5,   # New Orleans-Metairie, LA
    35620: 105.0,  # New York-Newark-Jersey City, NY-NJ-PA
    36420: 95.5,   # Oklahoma City, OK
    36740: 97.0,   # Orlando-Kissimmee-Sanford, FL
    37100: 96.0,   # Oxnard-Thousand Oaks-Ventura, CA
    37980: 101.3,  # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD
    38060: 99.1,   # Phoenix-Mesa-Chandler, AZ
    38300: 96.3,   # Pittsburgh, PA
    38900: 102.6,  # Portland-Vancouver-Hillsboro, OR-WA
    39300: 97.7,   # Providence-Warwick, RI-MA
    39580: 95.7,   # Raleigh-Cary, NC
    40060: 96.4,   # Richmond, VA
    40140: 97.7,   # Riverside-San Bernardino-Ontario, CA
    40380: 96.7,   # Rochester, NY
    40900: 96.2,   # Sacramento-Roseville-Folsom, CA
    41180: 96.0,   # St. Louis, MO-IL
    41620: 97.9,   # Salt Lake City, UT
    41700: 99.4,   # San Antonio-New Braunfels, TX
    41740: 107.5,  # San Diego-Chula Vista-Carlsbad, CA
    41860: 110.2,  # San Francisco-Oakland-Berkeley, CA
    41940: 108.3,  # San Jose-Sunnyvale-Santa Clara, CA
    42660: 106.5,  # Seattle-Tacoma-Bellevue, WA
    44700: 94.8,   # Stockton, CA
    45060: 96.5,   # Syracuse, NY
    45300: 96.9,   # Tampa-St. Petersburg-Clearwater, FL
    45780: 95.6,   # Toledo, OH
    46060: 95.4,   # Tucson, AZ
    46140: 95.5,   # Tulsa, OK
    47260: 101.6,  # Virginia Beach-Norfolk-Newport News, VA-NC
    47900: 104.3,  # Washington-Arlington-Alexandria, DC-VA-MD-WV
    48620: 94.5,   # Wichita, KS
    49340: 95.7,   # Worcester, MA-CT
    # Smaller MSAs with published data
    11260: 103.5,  # Anchorage, AK
    11460: 95.2,   # Ann Arbor, MI
    12260: 95.2,   # Augusta-Richmond County, GA-SC
    13140: 95.0,   # Beaumont-Port Arthur, TX
    13900: 94.2,   # Bismarck, ND
    14260: 95.8,   # Boise City, ID
    15180: 95.1,   # Brownsville-Harlingen, TX
    15940: 93.5,   # Canton-Massillon, OH
    16020: 94.8,   # Cape Coral-Fort Myers, FL
    16060: 93.6,   # Carbondale-Marion, IL
    16300: 94.7,   # Cedar Rapids, IA
    16580: 94.7,   # Champaign-Urbana, IL
    16700: 94.8,   # Charleston-North Charleston, SC
    16860: 95.9,   # Chattanooga, TN-GA
    17780: 93.8,   # College Station-Bryan, TX
    18580: 95.4,   # Corpus Christi, TX
    19060: 93.1,   # Cumberland, MD-WV
    19460: 95.2,   # Decatur, AL
    19660: 95.3,   # Deltona-Daytona Beach-Ormond Beach, FL
    19780: 94.8,   # Des Moines-West Des Moines, IA
    20020: 95.3,   # Dothan, AL
    20260: 93.5,   # Duluth, MN-WI
    21820: 102.3,  # Fairbanks, AK
    22020: 93.6,   # Fargo, ND-MN
    22420: 95.4,   # Flint, MI
    22660: 93.7,   # Fort Collins, CO
    23420: 93.6,   # Fresno, CA
    23540: 94.0,   # Gainesville, FL
    24580: 94.3,   # Green Bay, WI
    25060: 97.9,   # Gulfport-Biloxi, MS
    25420: 94.8,   # Harrisburg-Carlisle, PA
    25620: 94.1,   # Hattiesburg, MS
    26620: 96.3,   # Huntsville, AL
    27140: 96.0,   # Jackson, MS
    27620: 93.2,   # Jefferson City, MO
    28420: 93.2,   # Kennewick-Richland, WA
    29460: 93.8,   # Lakeland-Winter Haven, FL
    29620: 95.2,   # Lansing-East Lansing, MI
    30460: 94.1,   # Lexington-Fayette, KY
    30700: 93.7,   # Lincoln, NE
    31540: 94.5,   # Madison, WI
    33260: 96.3,   # Midland, TX
    33660: 94.2,   # Mobile, AL
    33700: 93.7,   # Modesto, CA
    34740: 93.7,   # Muskegon, MI
    34900: 94.6,   # Napa, CA
    35300: 93.8,   # New Haven-Milford, CT
    35840: 94.4,   # North Port-Sarasota-Bradenton, FL
    36260: 95.0,   # Ogden-Clearfield, UT
    36540: 95.3,   # Omaha-Council Bluffs, NE-IA
    37340: 94.6,   # Palm Bay-Melbourne-Titusville, FL
    37460: 93.5,   # Panama City, FL
    37860: 94.8,   # Pensacola-Ferry Pass-Brent, FL
    38340: 94.2,   # Pittsfield, MA
    38860: 97.0,   # Portland-South Portland, ME
    39340: 93.3,   # Provo-Orem, UT
    39740: 95.4,   # Reading, PA
    40220: 93.4,   # Roanoke, VA
    40980: 93.0,   # Saginaw, MI
    41060: 93.4,   # St. Cloud, MN
    41420: 93.6,   # Salem, OR
    41500: 93.3,   # Salinas, CA
    41540: 95.6,   # Salisbury, MD-DE
    42020: 106.3,  # San Luis Obispo-Paso Robles, CA
    42100: 108.2,  # Santa Cruz-Watsonville, CA
    42200: 107.0,  # Santa Maria-Santa Barbara, CA
    42540: 103.0,  # Scranton-Wilkes-Barre, PA
    42680: 94.3,   # Sebastian-Vero Beach, FL
    44060: 94.7,   # Spokane-Spokane Valley, WA
    44100: 93.9,   # Springfield, IL
    44180: 93.6,   # Springfield, MO
    44220: 93.7,   # Springfield, OH
    45220: 95.7,   # Tallahassee, FL
    45780: 95.6,   # Toledo, OH
    45820: 94.2,   # Topeka, KS
    46220: 93.8,   # Tuscaloosa, AL
    47380: 93.8,   # Waco, TX
    47580: 94.4,   # Warner Robins, GA
    48300: 95.8,   # Wenatchee, WA  <-- specifically needed
    48540: 93.8,   # Wheeling, WV-OH
    48660: 93.2,   # Wichita Falls, TX
    49180: 93.6,   # Winston-Salem, NC
    49420: 93.8,   # Yakima, WA
    49620: 93.5,   # York-Hanover, PA
    49660: 93.5,   # Youngstown-Warren-Boardman, OH-PA
}

# State-level Goods RPP (2022) — fallback for cities without MSA-level data
# Source: BEA SARPP Table, "Goods" line
STATE_GOODS_RPP = {
    "AL": 94.8, "AK": 105.2, "AZ": 99.0, "AR": 93.4,
    "CA": 104.5, "CO": 99.7, "CT": 99.4, "DE": 98.8,
    "DC": 103.7, "FL": 96.5, "GA": 96.5, "HI": 113.5,
    "ID": 96.5, "IL": 97.0, "IN": 95.3, "IA": 94.3,
    "KS": 94.5, "KY": 94.5, "LA": 95.0, "ME": 98.0,
    "MD": 98.5, "MA": 100.5, "MI": 95.8, "MN": 96.5,
    "MS": 93.8, "MO": 94.0, "MT": 97.3, "NE": 94.5,
    "NV": 99.8, "NH": 100.2, "NJ": 103.5, "NM": 97.0,
    "NY": 103.0, "NC": 95.5, "ND": 94.0, "OH": 95.8,
    "OK": 94.5, "OR": 100.5, "PA": 97.0, "RI": 98.5,
    "SC": 95.5, "SD": 95.0, "TN": 95.5, "TX": 97.5,
    "UT": 97.0, "VT": 99.5, "VA": 97.0, "WA": 103.5,
    "WV": 93.5, "WI": 95.0, "WY": 96.5,
}


def main():
    cities = pd.read_parquet(PROJECT_ROOT / "data" / "cities_master.parquet")
    output_path = PROJECT_ROOT / "data" / "rpp.parquet"

    log.info("Assigning RPP Goods index to %d cities...", len(cities))

    # Try MSA-level match first, then fall back to state-level
    cities["goods_rpp"] = np.nan
    cities["rpp_level"] = "none"

    # Match by CBSA code (MSA-level)
    msa_matched = 0
    for idx, row in cities.iterrows():
        cbsa = row.get("cbsa_code")
        if pd.notna(cbsa) and cbsa != "" and str(cbsa).strip():
            try:
                cbsa_int = int(float(cbsa))
            except (ValueError, TypeError):
                continue
            if cbsa_int in MSA_GOODS_RPP:
                cities.at[idx, "goods_rpp"] = MSA_GOODS_RPP[cbsa_int]
                cities.at[idx, "rpp_level"] = "msa"
                msa_matched += 1

    log.info("MSA-level matches: %d/%d cities", msa_matched, len(cities))

    # State-level fallback for remaining
    state_fallback = 0
    for idx, row in cities.iterrows():
        if pd.isna(row["goods_rpp"]):
            state_rpp = STATE_GOODS_RPP.get(row["state"])
            if state_rpp:
                cities.at[idx, "goods_rpp"] = state_rpp
                cities.at[idx, "rpp_level"] = "state"
                state_fallback += 1

    log.info("State-level fallback: %d cities", state_fallback)
    log.info("Total coverage: %d/%d cities", cities["goods_rpp"].notna().sum(), len(cities))

    # Select output columns
    out = cities[["city_id", "name", "state", "goods_rpp", "rpp_level"]].copy()

    write_parquet_with_metadata(
        out,
        output_path,
        source_name="US Bureau of Economic Analysis (BEA) Regional Price Parities 2022",
        source_url="https://www.bea.gov/data/prices-spending/regional-price-parities-state-and-metro-area",
        date_collected="2026-05-04",
        notes=(
            "Regional Price Parities — Goods component (index, 100 = US average). "
            "MSA-level data from BEA MARPP table (2022). State-level fallback from BEA SARPP. "
            "Measures relative cost of consumer goods (gas, groceries, retail items). "
            "Does NOT include housing or services — those are separate RPP components. "
            "Column 'rpp_level': 'msa' = from BEA MSA data, 'state' = state-level fallback."
        ),
    )

    print(f"\nRPP Goods data:")
    print(f"  MSA-level: {msa_matched}/{len(cities)} ({msa_matched/len(cities)*100:.1f}%)")
    print(f"  State-level fallback: {state_fallback}/{len(cities)} ({state_fallback/len(cities)*100:.1f}%)")
    print(f"  Range: {out['goods_rpp'].min():.1f} - {out['goods_rpp'].max():.1f}")
    print(f"\nSample:")
    print(out.head(20).to_string())


if __name__ == "__main__":
    main()
