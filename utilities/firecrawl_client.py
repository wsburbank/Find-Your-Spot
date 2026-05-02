"""
Thin wrapper around the firecrawl-py SDK for the self-hosted Firecrawl instance.

Firecrawl runs locally in Docker (default: http://localhost:3002) and provides
web scraping, crawling, and search capabilities for data collection.

Environment variables:
    FIRECRAWL_API_URL  -- Base URL of the Firecrawl instance (default: http://localhost:3002)
    FIRECRAWL_API_KEY  -- API key (default: "fc-" for self-hosted; not enforced locally)
"""
import logging
import os

import requests as _requests

log = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:3002"
_DEFAULT_KEY = "fc-"


def get_firecrawl_client():
    """Return a FirecrawlApp instance pointed at the local self-hosted Firecrawl.

    Reads ``FIRECRAWL_API_URL`` and ``FIRECRAWL_API_KEY`` from the environment.

    Returns:
        A ``firecrawl.FirecrawlApp`` instance.

    Raises:
        ImportError: If firecrawl-py is not installed.
    """
    from firecrawl import FirecrawlApp

    api_url = os.environ.get("FIRECRAWL_API_URL", _DEFAULT_URL)
    api_key = os.environ.get("FIRECRAWL_API_KEY", _DEFAULT_KEY)

    return FirecrawlApp(api_key=api_key, api_url=api_url)


def is_firecrawl_available() -> bool:
    """Check if the Firecrawl service is reachable.

    Performs a simple HTTP GET against the Firecrawl base URL. Returns False
    if the service is unreachable or returns an error.
    """
    api_url = os.environ.get("FIRECRAWL_API_URL", _DEFAULT_URL)
    try:
        resp = _requests.get(api_url, timeout=3)
        return resp.status_code < 500
    except Exception:
        return False


def scrape_page(url: str, formats: list[str] | None = None) -> dict:
    """Scrape a single page and return structured data.

    Args:
        url: The URL to scrape.
        formats: Output formats (e.g. ``["markdown", "html"]``).
            Defaults to ``["markdown"]``.

    Returns:
        Dict with keys like ``markdown``, ``html``, ``metadata`` depending on
        the requested formats.

    Raises:
        RuntimeError: If Firecrawl is not available.
    """
    if not is_firecrawl_available():
        raise RuntimeError(
            "Firecrawl is not available. Ensure Docker is running with "
            "'docker compose up' in the firecrawl repo."
        )

    app = get_firecrawl_client()
    params = {}
    if formats:
        params["formats"] = formats

    log.info("Scraping %s", url)
    result = app.scrape_url(url, params=params)
    return result


def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Search the web via Firecrawl and return results.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of dicts with URL, title, description, and content.
    """
    if not is_firecrawl_available():
        raise RuntimeError("Firecrawl is not available.")

    app = get_firecrawl_client()
    log.info("Searching: %s", query)
    result = app.search(query, params={"limit": num_results})
    return result


def crawl_site(
    url: str,
    max_pages: int = 10,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """Crawl a website starting from *url*, up to *max_pages*.

    Args:
        url: Starting URL.
        max_pages: Maximum number of pages to crawl.
        include_patterns: URL patterns to include (glob-style).
        exclude_patterns: URL patterns to exclude (glob-style).

    Returns:
        List of page dicts with ``url``, ``markdown``, ``metadata`` keys.
    """
    if not is_firecrawl_available():
        raise RuntimeError("Firecrawl is not available.")

    app = get_firecrawl_client()
    params = {"limit": max_pages}
    if include_patterns:
        params["includePaths"] = include_patterns
    if exclude_patterns:
        params["excludePaths"] = exclude_patterns

    log.info("Crawling %s (max %d pages)", url, max_pages)
    result = app.crawl_url(url, params=params)
    return result
