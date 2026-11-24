import polars as pl


def build_product_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Construct one row per canonical product; summarize market price returns, temporal volatility and coverage; avoid mixing units.
    raise NotImplementedError


