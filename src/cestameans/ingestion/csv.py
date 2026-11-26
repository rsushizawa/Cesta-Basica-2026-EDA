from pathlib import Path

import polars as pl

from cestameans.ingestion.schema import RAW_SCHEMA


class CSVDataSource:
    def __init__(self, path: Path) -> None:
        # Store the source path and any required reader configuration.
        self.path = path

    def read(self) -> pl.LazyFrame:
        # Scan CSV with explicit raw schema; add source row ID and file provenance.
        return (
            pl.scan_csv(
                self.path, schema=RAW_SCHEMA, try_parse_dates=False, infer_schema=False
            )
            .with_row_index("source_row_id")
            .with_columns(pl.lit(str(self.path)).alias("source_file"))
        )
