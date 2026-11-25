from typing import Protocol

import polars as pl


class DataSource(Protocol):
    def read(self) -> pl.LazyFrame:
        # return lazy source without froping rows
        ...
