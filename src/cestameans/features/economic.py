import polars as pl


def add_market_reference(df: pl.LazyFrame, min_stores: int = 5) -> pl.LazyFrame:
    # Median of store-level median prices per product-month; count unique stores; insufficient coverage yields null benchmark.
    raise NotImplementedError


def add_relative_prices(df: pl.LazyFrame) -> pl.LazyFrame:
    # Create relative_price = median_price / market_median, log_relative_price and price_premium_pct; guard invalid benchmarks.
    raise NotImplementedError


