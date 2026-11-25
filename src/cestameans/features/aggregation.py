import polars as pl


def aggregate_monthly_prices(df: pl.LazyFrame) -> pl.LazyFrame:
    # One row per establishment_id, reference_month, product.
    # Compute median_price, mean_price, min_price, max_price, price_std, price_iqr, sample_count, observed_brands.
    # Use valid observations and documented duplicate/outlier policy; preserve equal store weight downstream.
    return (
        df.filter(
            pl.col("unit_price").is_not_null()
            & ~pl.col("is_price_outlier").fill_null(False)
        )
        .group_by(["establishment_id", "reference_month", "product"])
        .agg(
            pl.col("unit_price").median().alias("median_price"),
            pl.col("unit_price").mean().alias("mean_price"),
            pl.col("unit_price").min().alias("min_price"),
            pl.col("unit_price").max().alias("max_price"),
            pl.col("unit_price").std().alias("price_std"),
            pl.col("unit_price")
            .quantile(0.75)
            .sub(pl.col("unit_price").quantile(0.25))
            .alias("price_iqr"),
            pl.len().alias("sample_count"),
            pl.col("brand_normalized").n_unique().alias("observed_brands"),
        )
    )
