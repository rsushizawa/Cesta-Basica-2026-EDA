import polars as pl


def flag_price_outliers(
    df: pl.LazyFrame, threshold: float = 3.5, min_group_size: int = 10
) -> pl.LazyFrame:
    # Group by product and reference_month; compute median, MAD and modified z-score.
    # Leave small or zero-MAD groups unassessed; flag rather than automatically delete.
    stats = df.group_by(["product", "reference_month"]).agg(
        pl.col("unit_price").median().alias("price_median"),
        (pl.col("unit_price") - pl.col("unit_price").median())
        .abs()
        .median()
        .alias("price_mad"),
        pl.len().alias("price_group_size"),
    )
    return (
        df.join(stats, on=["product", "reference_month"], how="left")
        .with_columns(
            (
                (pl.col("unit_price") - pl.col("price_median")).abs()
                / (pl.col("price_mad") * 1.4826)
            ).alias("modified_z_score")
        )
        .with_columns(
            (
                (pl.col("price_group_size") >= min_group_size)
                & (pl.col("price_mad") > 0)
                & (pl.col("modified_z_score") > threshold)
            )
            .fill_null(False)
            .alias("is_price_outlier")
        )
    )
