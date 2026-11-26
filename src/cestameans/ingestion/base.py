from typing import Protocol

import polars as pl


class DataSource(Protocol):
    def read(self) -> pl.LazyFrame:
        # Return a lazy source without dropping rows.
        ...
