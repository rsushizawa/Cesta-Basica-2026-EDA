from pathlib import Path
import polars as pl

class CSVDataSource:
    def __init__(self, path: Path) -> None:
        # Store the source path and any required reader configuration.
        raise NotImplementedError

    def read(self) -> pl.LazyFrame:
        # Scan CSV with explicit raw schema; add source row ID and file provenance.
        raise NotImplementedError
