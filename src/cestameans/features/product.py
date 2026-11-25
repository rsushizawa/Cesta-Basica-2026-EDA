import polars as pl


def build_product_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Construct one row per canonical product; summarize market price returns, temporal volatility and coverage; avoid mixing units.
    return df.group_by("product").agg(
        pl.col("median_price").mean().alias("mean_market_price"),
        pl.col("log_relative_price").std().alias("price_volatility"),
        pl.col("establishment_id").n_unique().alias("establishments"),
        pl.col("reference_month").n_unique().alias("months"),
    )
