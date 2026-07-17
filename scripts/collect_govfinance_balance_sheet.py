"""
Collect municipal balance sheet data from Reason Foundation GovFinance Dashboard.

Source: Reason Foundation, Government Financial Transparency Project
  - https://govfinance.reason.org/
  - Data year: Fiscal Year 2023
  - Derived from city Annual Comprehensive Financial Reports (ACFRs)
  - Covers 8,630+ U.S. municipalities

This script extracts accrual-basis balance sheet data that complements the
cash-basis Census of Governments data in collect_gov_finances.py.

Metrics collected (per capita):
  - total_liabilities_pc: All liabilities per capita
  - pension_liability_pc: Net pension liability per capita
  - opeb_liability_pc: Net OPEB (retiree healthcare) liability per capita
  - bonds_outstanding_pc: Outstanding bonds per capita
  - total_assets_pc: Total assets per capita
  - debt_ratio: Total liabilities / total assets (0-1 scale)
  - current_ratio: Current assets / current liabilities
  - free_cash_flow_pc: Free cash flow per capita
  - net_position_pc: Net position (assets - liabilities) per capita

Output: data/govfinance_balance_sheet.parquet
"""

import json
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# The GovFinance municipal data is embedded in a SvelteKit JS bundle.
# This URL points to the chunk containing all 8,600+ municipal records.
GOVFINANCE_APP_URL = "https://govfinance.reason.org/_app/immutable/entry/app.DV6bqJpK.js"
GOVFINANCE_BASE = "https://govfinance.reason.org/_app/immutable/"


def find_data_chunk_url() -> str:
    """Discover the municipal data chunk URL from the app manifest.

    The GovFinance SvelteKit app bundles data in JS chunk files. The largest
    chunk (>5MB) with 'General Purpose' entity data is what we need.
    We identify it by checking file sizes and content.
    """
    cache_path = CACHE_DIR / "govfinance_app.js"

    log.info("Fetching app manifest to locate data chunk...")
    resp = requests.get(GOVFINANCE_APP_URL, timeout=30)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")

    # Extract all chunk file references from the app bundle
    all_files = re.findall(r'"(\.\./[^"]+\.js)"', resp.text)
    chunk_files = [f.replace("../", "") for f in all_files if "chunks/" in f]

    log.info("Found %d chunk files in manifest", len(chunk_files))

    # Find the large data chunk (>5MB) containing municipal data
    for chunk in chunk_files:
        url = GOVFINANCE_BASE + chunk
        try:
            head = requests.head(url, timeout=10)
            size = int(head.headers.get("content-length", 0))
            if 5_000_000 < size < 10_000_000:
                # Verify it contains municipal data by checking first bytes
                partial = requests.get(url, timeout=10, headers={"Range": "bytes=0-200"})
                if "General Purpose" in partial.text or "entity_type" in partial.text:
                    log.info("Found municipal data chunk: %s (%d MB)", chunk, size // 1_000_000)
                    return url
        except requests.RequestException:
            continue

    raise RuntimeError(
        "Could not locate municipal data chunk. The GovFinance site may have updated "
        "its bundle hashes. Check govfinance.reason.org and update the script."
    )


def download_municipal_data() -> list[dict]:
    """Download and parse the municipal balance sheet data from GovFinance.

    Returns:
        List of dicts, one per municipality, with balance sheet fields.
    """
    cache_path = CACHE_DIR / "govfinance_municipal_raw.js"

    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        text = cache_path.read_text(encoding="utf-8")
    else:
        url = find_data_chunk_url()
        log.info("Downloading municipal data from %s ...", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8")
        log.info("Cached to %s (%d MB)", cache_path, len(text) // 1_000_000)

    # Extract JSON array from JS module: const t=[...];export{t as m};
    match = re.search(r"const\s+\w\s*=\s*(\[.*?\])\s*;\s*export", text, re.DOTALL)
    if not match:
        raise ValueError("Could not find data array in JS bundle")

    json_str = match.group(1)

    # Convert JS object notation to valid JSON:
    # 1. Quote unquoted keys (word characters before colon)
    fixed = re.sub(r"(?<=[\{,])(\w+):", r'"\1":', json_str)
    # 2. Fix bare decimals like .6289 -> 0.6289
    fixed = re.sub(r":\.(\d)", r":0.\1", fixed)
    fixed = re.sub(r":-\.(\d)", r":-0.\1", fixed)

    data = json.loads(fixed)
    log.info("Parsed %d municipal records from GovFinance", len(data))
    return data


def match_to_cities(gf_df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Match GovFinance data to our master city list using FIPS geo_id.

    GovFinance provides geo_id as state FIPS (2 chars) + place FIPS (5 chars).

    Returns:
        DataFrame indexed by city_id with per-capita balance sheet metrics.
    """
    # Build geo_id for our cities
    cities = cities.copy()
    cities["geo_id"] = (
        cities["fips_state"].astype(str).str.zfill(2)
        + cities["fips_place"].astype(str).str.zfill(5)
    )

    # Clean GovFinance geo_ids
    gf_df = gf_df.copy()
    gf_df["geo_id_clean"] = gf_df["geo_id"].astype(str).str.strip()

    # Filter to records with population > 0 and actual data
    gf_df = gf_df[(gf_df["population"] > 0) & (gf_df["total_assets"] != 0)].copy()

    # Compute per-capita metrics
    pop = gf_df["population"]
    gf_df["total_liabilities_pc"] = (gf_df["total_liabilities"] / pop).round(0)
    gf_df["pension_liability_pc"] = (gf_df["pension_liability"] / pop).round(0)
    gf_df["opeb_liability_pc"] = (gf_df["opeb_liability"] / pop).round(0)
    gf_df["bonds_outstanding_pc"] = (gf_df["bonds_outstanding"] / pop).round(0)
    gf_df["total_assets_pc"] = (gf_df["total_assets"] / pop).round(0)
    gf_df["free_cash_flow_pc"] = (gf_df["free_cash_flow"] / pop).round(0)
    gf_df["net_position_pc"] = (gf_df["net_position"] / pop).round(0)
    # debt_ratio and current_ratio are already ratios, not per-capita
    gf_df["debt_ratio"] = gf_df["debt_ratio"].round(4)
    gf_df["current_ratio"] = gf_df["current_ratio"].round(4)

    # Replace zeros with NaN for liability fields where 0 likely means missing
    for col in ["pension_liability_pc", "opeb_liability_pc"]:
        gf_df[col] = gf_df[col].replace(0, np.nan)

    # Select columns for merge
    merge_cols = [
        "geo_id_clean",
        "total_liabilities_pc",
        "pension_liability_pc",
        "opeb_liability_pc",
        "bonds_outstanding_pc",
        "total_assets_pc",
        "debt_ratio",
        "current_ratio",
        "free_cash_flow_pc",
        "net_position_pc",
    ]

    # Merge on geo_id
    result = cities[["city_id", "geo_id"]].merge(
        gf_df[merge_cols].drop_duplicates(subset=["geo_id_clean"], keep="first"),
        left_on="geo_id",
        right_on="geo_id_clean",
        how="left",
    )

    result = result.drop(columns=["geo_id", "geo_id_clean"])
    return result


def main():
    # Download GovFinance data
    raw_data = download_municipal_data()
    gf_df = pd.DataFrame(raw_data)

    log.info("GovFinance dataset: %d municipalities, year=%s", len(gf_df), gf_df["year"].unique())
    log.info("Entity types: %s", gf_df["entity_type"].value_counts().to_dict())

    # Load master city list
    cities = pd.read_parquet(DATA_DIR / "cities_master.parquet")
    log.info("Master city list: %d cities", len(cities))

    # Match and compute per-capita metrics
    result = match_to_cities(gf_df, cities)

    # Report coverage
    metrics = [
        "total_liabilities_pc", "pension_liability_pc", "opeb_liability_pc",
        "bonds_outstanding_pc", "total_assets_pc", "debt_ratio",
    ]
    print(f"\nGovFinance Balance Sheet Data (FY2023):")
    print(f"  Source: Reason Foundation GovFinance Dashboard (ACFRs)")
    for col in metrics:
        count = result[col].notna().sum()
        print(f"  {col:30s}: {count}/{len(result)} ({count / len(result) * 100:.1f}%)")

    # Median values for sanity check
    print(f"\n  Per-capita metrics (median for matched cities):")
    for col in metrics:
        valid = result[col].dropna()
        if len(valid) > 0:
            if col in ("debt_ratio", "current_ratio"):
                print(f"    {col:30s}: {valid.median():.3f}")
            else:
                print(f"    {col:30s}: ${valid.median():>10,.0f}")

    # Write output
    output_cols = ["city_id"] + [
        "total_liabilities_pc", "pension_liability_pc", "opeb_liability_pc",
        "bonds_outstanding_pc", "total_assets_pc", "debt_ratio",
        "current_ratio", "free_cash_flow_pc", "net_position_pc",
    ]
    output = result[output_cols].copy()

    matched = output["total_liabilities_pc"].notna().sum()
    write_parquet_with_metadata(
        output,
        DATA_DIR / "govfinance_balance_sheet.parquet",
        source_name="Reason Foundation GovFinance Dashboard (ACFR-derived) FY2023",
        source_url="https://govfinance.reason.org/",
        notes=(
            "Municipal balance sheet data derived from Annual Comprehensive Financial "
            "Reports (ACFRs). Includes total liabilities, net pension liability, net OPEB "
            "liability, bonds outstanding, total assets, debt ratio, current ratio, free "
            f"cash flow, and net position. All per-capita except ratios. "
            f"Coverage: {matched}/{len(output)} cities ({matched/len(output)*100:.1f}%). "
            "Data year: FY2023. Pension/OPEB zeros treated as missing."
        ),
    )

    print(f"\n  Output: data/govfinance_balance_sheet.parquet")
    print(f"  Total cities matched: {matched}/{len(output)} ({matched / len(output) * 100:.1f}%)")


if __name__ == "__main__":
    main()
