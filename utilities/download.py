"""
HTTP download utilities with simple file-based caching.

Downloads from public APIs and data portals with time-based cache invalidation.
If a file already exists on disk and is newer than *max_age_days*, the download
is skipped entirely.
"""
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_USER_AGENT = "FindYourSpot/0.1 (city-recommendation-app; +https://github.com)"
_CONNECT_TIMEOUT = 30  # seconds
_READ_TIMEOUT = 120    # seconds


def _is_fresh(path: Path, max_age_days: int) -> bool:
    """Return True if *path* exists and was modified within *max_age_days*."""
    if not path.exists():
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_days * 86400


def download_file(
    url: str,
    dest_path: str | Path,
    max_age_days: int = 30,
    force: bool = False,
    headers: dict | None = None,
    params: dict | None = None,
) -> Path:
    """Download a file from *url* with time-based caching.

    If *dest_path* exists and is newer than *max_age_days*, the download is
    skipped unless *force* is True.

    Args:
        url: URL to download.
        dest_path: Local path to save the file.
        max_age_days: Maximum age in days before re-downloading.
        force: If True, always download regardless of cache.
        headers: Extra HTTP headers.
        params: URL query parameters.

    Returns:
        The resolved Path of the downloaded (or cached) file.
    """
    dest_path = Path(dest_path)

    if not force and _is_fresh(dest_path, max_age_days):
        log.info("Cache hit: %s (age < %d days)", dest_path.name, max_age_days)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    all_headers = {"User-Agent": _USER_AGENT}
    if headers:
        all_headers.update(headers)

    log.info("Downloading %s -> %s", url, dest_path.name)
    resp = requests.get(
        url,
        params=params,
        headers=all_headers,
        stream=True,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    )
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info("Downloaded %s (%d bytes)", dest_path.name, dest_path.stat().st_size)
    return dest_path


def download_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    cache_path: str | Path | None = None,
    max_age_days: int = 30,
    force: bool = False,
) -> dict | list:
    """Download JSON from a URL with optional file caching.

    Args:
        url: URL returning JSON.
        params: URL query parameters.
        headers: Extra HTTP headers.
        cache_path: If provided, cache the JSON response to this file.
        max_age_days: Maximum cache age in days.
        force: If True, bypass cache.

    Returns:
        Parsed JSON (dict or list).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if not force and _is_fresh(cache_path, max_age_days):
            log.info("Cache hit: %s", cache_path.name)
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

    all_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if headers:
        all_headers.update(headers)

    log.info("Fetching JSON from %s", url)
    resp = requests.get(
        url,
        params=params,
        headers=all_headers,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    )
    resp.raise_for_status()

    # Some government APIs return HTTP 200 with an error message in the body.
    text = resp.text.strip()
    if text.lower().startswith("error") or text.lower().startswith("<!doctype"):
        raise ValueError(f"API returned non-JSON response: {text[:200]}")

    data = resp.json()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Cached JSON to %s", cache_path.name)

    return data


def download_csv(
    url: str,
    dest_path: str | Path,
    max_age_days: int = 30,
    force: bool = False,
    encoding: str = "utf-8",
    headers: dict | None = None,
    params: dict | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Download a CSV file and return it as a DataFrame.

    The raw CSV is saved to *dest_path* for caching.  Subsequent calls within
    *max_age_days* read from the cached file without re-downloading.

    Args:
        url: URL to the CSV file.
        dest_path: Local path to cache the CSV.
        max_age_days: Maximum cache age in days.
        force: If True, bypass cache.
        encoding: Character encoding of the CSV.
        headers: Extra HTTP headers.
        params: URL query parameters.
        **read_csv_kwargs: Extra keyword arguments passed to ``pd.read_csv()``.

    Returns:
        DataFrame parsed from the CSV.
    """
    dest_path = Path(dest_path)

    download_file(
        url, dest_path,
        max_age_days=max_age_days,
        force=force,
        headers=headers,
        params=params,
    )

    return pd.read_csv(dest_path, encoding=encoding, **read_csv_kwargs)
