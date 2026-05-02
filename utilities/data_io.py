"""
Parquet I/O with embedded source metadata.

Every parquet file written by this module carries provenance information
(source name, URL, collection date) in the file's schema metadata so the
origin of the data is always recoverable without a separate manifest.
"""
import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# Keys stored in parquet schema metadata (all values are UTF-8 bytes).
_META_KEYS = ("source_name", "source_url", "date_collected", "notes", "row_count", "column_names")


def write_parquet_with_metadata(
    df: pd.DataFrame,
    path: str | Path,
    source_name: str,
    source_url: str,
    date_collected: str | None = None,
    notes: str = "",
) -> Path:
    """Write a DataFrame to parquet with source metadata embedded in the file.

    Args:
        df: The data to write.
        path: Destination file path (directories are created automatically).
        source_name: Human-readable name of the data source (e.g. "US Census Bureau ACS 5-Year").
        source_url: URL where the data was obtained.
        date_collected: ISO date string (YYYY-MM-DD). Defaults to today.
        notes: Free-text notes about transformations applied, filters, etc.

    Returns:
        The resolved Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if date_collected is None:
        date_collected = datetime.date.today().isoformat()

    table = pa.Table.from_pandas(df)

    custom_meta = {
        b"source_name": source_name.encode("utf-8"),
        b"source_url": source_url.encode("utf-8"),
        b"date_collected": date_collected.encode("utf-8"),
        b"notes": notes.encode("utf-8"),
        b"row_count": str(len(df)).encode("utf-8"),
        b"column_names": json.dumps(list(df.columns)).encode("utf-8"),
    }

    # Preserve any existing pyarrow/pandas metadata and merge ours in.
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, **custom_meta}
    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, str(path))
    log.info("Wrote %d rows to %s (source: %s)", len(df), path, source_name)
    return path


def read_parquet_with_metadata(
    path: str | Path,
    metadata_only: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    """Read a parquet file and return (DataFrame, metadata_dict).

    Args:
        path: Path to the parquet file.
        metadata_only: If True, only read schema metadata without loading data.
            Returns (None, metadata_dict).

    Returns:
        Tuple of (DataFrame or None, dict of source metadata).
    """
    path = Path(path)
    pf = pq.ParquetFile(str(path))
    raw_meta = pf.schema_arrow.metadata or {}

    metadata = {}
    for key in _META_KEYS:
        val = raw_meta.get(key.encode("utf-8"))
        if val is not None:
            decoded = val.decode("utf-8")
            if key == "column_names":
                decoded = json.loads(decoded)
            elif key == "row_count":
                decoded = int(decoded)
            metadata[key] = decoded

    if metadata_only:
        return None, metadata

    df = pf.read().to_pandas()
    return df, metadata


def list_parquet_sources(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Scan a directory tree for .parquet files and return a summary of their metadata.

    Args:
        data_dir: Root directory to scan. Defaults to the project's ``data/`` directory.

    Returns:
        DataFrame with columns: file, source_name, source_url, date_collected, notes, row_count.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    data_dir = Path(data_dir)

    records = []
    for pq_path in sorted(data_dir.rglob("*.parquet")):
        try:
            _, meta = read_parquet_with_metadata(pq_path, metadata_only=True)
            records.append({
                "file": str(pq_path.relative_to(data_dir)),
                "source_name": meta.get("source_name", ""),
                "source_url": meta.get("source_url", ""),
                "date_collected": meta.get("date_collected", ""),
                "notes": meta.get("notes", ""),
                "row_count": meta.get("row_count", 0),
            })
        except Exception:
            log.warning("Could not read metadata from %s", pq_path, exc_info=True)
            records.append({"file": str(pq_path.relative_to(data_dir)), "source_name": "ERROR"})

    return pd.DataFrame(records)
