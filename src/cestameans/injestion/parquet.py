from pathlib import Path

import polars as pl

from cestameans.ingestion.parquet import ParquetDataSource as _ParquetDataSource


class ParquetDataSource:
    def __init__(self, path: Path) -> None:
        # Store the source path and any required reader configuration.
        self._source = _ParquetDataSource(path)

    def read(self) -> pl.LazyFrame:
        # Scan Parquet lazily; preserve existing observation IDs.
        return self._source.read()
