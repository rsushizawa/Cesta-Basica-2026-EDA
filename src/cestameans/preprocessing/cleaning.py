import polars as pl


def normalize_text(df: pl.LazyFrame) -> pl.LazyFrame:
    # Trim whitespace and convert empty strings to null; retain original product and brand names.
    raise NotImplementedError


def calculate_unit_price(df: pl.LazyFrame) -> pl.LazyFrame:
    # Divide package_price by quantity only for finite positive inputs; otherwise null.
    raise NotImplementedError


def apply_duplicate_policy(df: pl.LazyFrame, *, drop_confirmed: bool = False) -> pl.LazyFrame:
    # Only remove duplicates under a documented policy; keep audit IDs.
    raise NotImplementedError


