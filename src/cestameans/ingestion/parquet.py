from pathlib import Path

import polars as pl


class ParquetDataSource:
    def __init__(self, path: Path) -> None:
        # Store the source path and any required reader configuration.
        self.path = path

    def read(self) -> pl.LazyFrame:
        # Scan Parquet lazily; preserve existing observation IDs.
        return pl.scan_parquet(self.path)
