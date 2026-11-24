import polars as pl


def rename_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    # Use COLUMN_MAPPING; verify required headers; preserve observation ID.
    raise NotImplementedError


def parse_dates(df: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve raw strings; parse dd/mm/yy (and verified four-digit variants).
    # Normalize reference month to first day; distinguish missing from malformed dates.
    raise NotImplementedError


def parse_numbers(df: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve raw price and quantity strings; cast to Float64; flag parse failures.
    raise NotImplementedError


