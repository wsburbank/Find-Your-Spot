"""Tests for utilities/data_io.py — parquet I/O with source metadata."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from utilities.data_io import (
    list_parquet_sources,
    read_parquet_with_metadata,
    write_parquet_with_metadata,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "city": ["Denver", "Boulder", "Pueblo"],
        "state": ["CO", "CO", "CO"],
        "population": [715522, 108250, 112361],
    })


@pytest.fixture
def tmp_parquet(tmp_path, sample_df):
    path = tmp_path / "test.parquet"
    write_parquet_with_metadata(
        sample_df, path,
        source_name="US Census Bureau",
        source_url="https://api.census.gov/data/2022/acs/acs5",
        date_collected="2024-01-15",
        notes="Test data only",
    )
    return path


class TestWriteAndRead:
    def test_round_trip_data(self, tmp_parquet, sample_df):
        df, meta = read_parquet_with_metadata(tmp_parquet)
        assert len(df) == len(sample_df)
        assert list(df.columns) == list(sample_df.columns)
        assert df["city"].tolist() == sample_df["city"].tolist()

    def test_round_trip_metadata(self, tmp_parquet):
        _, meta = read_parquet_with_metadata(tmp_parquet)
        assert meta["source_name"] == "US Census Bureau"
        assert meta["source_url"] == "https://api.census.gov/data/2022/acs/acs5"
        assert meta["date_collected"] == "2024-01-15"
        assert meta["notes"] == "Test data only"
        assert meta["row_count"] == 3
        assert meta["column_names"] == ["city", "state", "population"]

    def test_metadata_only_does_not_load_data(self, tmp_parquet):
        df, meta = read_parquet_with_metadata(tmp_parquet, metadata_only=True)
        assert df is None
        assert meta["source_name"] == "US Census Bureau"
        assert meta["row_count"] == 3

    def test_default_date_collected(self, tmp_path, sample_df):
        import datetime
        path = tmp_path / "auto_date.parquet"
        write_parquet_with_metadata(
            sample_df, path,
            source_name="Test", source_url="http://example.com",
        )
        _, meta = read_parquet_with_metadata(path)
        assert meta["date_collected"] == datetime.date.today().isoformat()

    def test_creates_parent_directories(self, tmp_path, sample_df):
        path = tmp_path / "deep" / "nested" / "dir" / "test.parquet"
        write_parquet_with_metadata(
            sample_df, path,
            source_name="Test", source_url="http://example.com",
        )
        assert path.exists()


class TestListParquetSources:
    def test_finds_parquet_files(self, tmp_path, sample_df):
        for name in ["a.parquet", "b.parquet"]:
            write_parquet_with_metadata(
                sample_df, tmp_path / name,
                source_name=f"Source {name}",
                source_url=f"http://example.com/{name}",
            )
        summary = list_parquet_sources(tmp_path)
        assert len(summary) == 2
        assert "source_name" in summary.columns
        assert set(summary["source_name"]) == {"Source a.parquet", "Source b.parquet"}

    def test_empty_directory(self, tmp_path):
        summary = list_parquet_sources(tmp_path)
        assert len(summary) == 0
