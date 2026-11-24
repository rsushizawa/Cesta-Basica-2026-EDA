import polars as pl


def aggregate_monthly_prices(df: pl.LazyFrame) -> pl.LazyFrame:
    # One row per establishment_id, reference_month, product.
    # Compute median_price, mean_price, min_price, max_price, price_std, price_iqr, sample_count, observed_brands.
    # Use valid observations and documented duplicate/outlier policy; preserve equal store weight downstream.
    raise NotImplementedError


