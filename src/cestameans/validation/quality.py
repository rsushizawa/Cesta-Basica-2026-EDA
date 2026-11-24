import polars as pl


def summarize_quality(df: pl.LazyFrame) -> pl.DataFrame:
    # Count rows, nulls, malformed values, duplicates and flags; report by month where useful.
    raise NotImplementedError


def assert_unique_key(df: pl.LazyFrame, columns: tuple[str, ...]) -> None:
    # Raise a descriptive error if keys are null or repeated.
    raise NotImplementedError


