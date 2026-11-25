from pathlib import Path

import polars as pl

from cestameans.ingestion.csv import CSVDataSource as _CSVDataSource


class CSVDataSource:
    def __init__(self, path: Path) -> None:
        # Store the source path and any required reader configuration.
        self._source = _CSVDataSource(path)

    def read(self) -> pl.LazyFrame:
        # Scan CSV with explicit raw schema; add source row ID and file provenance.
        return self._source.read()
