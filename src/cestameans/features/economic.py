import polars as pl


def add_market_reference(df: pl.LazyFrame, min_stores: int = 5) -> pl.LazyFrame:
    # Median of store-level median prices per product-month; count unique stores; insufficient coverage yields null benchmark.
    market = (
        df.group_by(["reference_month", "product"])
        .agg(
            pl.col("median_price").median().alias("market_median"),
            pl.col("establishment_id").n_unique().alias("market_store_count"),
        )
        .with_columns(
            pl.when(pl.col("market_store_count") >= min_stores)
            .then(pl.col("market_median"))
            .otherwise(None)
            .alias("market_median")
        )
    )
    return df.join(market, on=["reference_month", "product"], how="left")


def add_relative_prices(df: pl.LazyFrame) -> pl.LazyFrame:
    # Create relative_price = median_price / market_median, log_relative_price and price_premium_pct; guard invalid benchmarks.
    return df.with_columns(
        (pl.col("median_price") / pl.col("market_median")).alias("relative_price")
    ).with_columns(
        pl.when(pl.col("relative_price") > 0)
        .then(pl.col("relative_price").log())
        .otherwise(None)
        .alias("log_relative_price"),
        ((pl.col("relative_price") - 1) * 100).alias("price_premium_pct"),
    )
