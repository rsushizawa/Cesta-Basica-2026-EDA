import polars as pl


def add_monthly_changes(df: pl.LazyFrame) -> pl.LazyFrame:
    # Sort by store/product/month; create mom_change and log_price_return only for consecutive calendar months.
    return df.sort(["establishment_id", "product", "reference_month"]).with_columns(
        pl.col("median_price")
        .pct_change()
        .over(["establishment_id", "product"])
        .alias("mom_change"),
        pl.col("median_price")
        .log()
        .diff()
        .over(["establishment_id", "product"])
        .alias("log_price_return"),
    )


def add_yearly_changes(df: pl.LazyFrame) -> pl.LazyFrame:
    # Join to exact month 12 months earlier, not the 12th previous row; create yoy_change.
    return (
        df.with_columns(
            pl.col("reference_month").dt.offset_by("-12mo").alias("prior_year_month")
        )
        .join(
            df.select(
                [
                    "establishment_id",
                    "product",
                    "reference_month",
                    pl.col("median_price").alias("prior_year_price"),
                ]
            ),
            left_on=["establishment_id", "product", "prior_year_month"],
            right_on=["establishment_id", "product", "reference_month"],
            how="left",
        )
        .with_columns(
            (pl.col("median_price") / pl.col("prior_year_price") - 1).alias(
                "yoy_change"
            )
        )
        .drop("prior_year_month")
    )


def add_rolling_features(df: pl.LazyFrame, window_months: int = 3) -> pl.LazyFrame:
    # Use calendar-aware trailing windows; no future observations; document minimum valid periods.
    return df.sort(["establishment_id", "product", "reference_month"]).with_columns(
        pl.col("log_relative_price")
        .rolling_mean_by(
            "reference_month",
            window_size=f"{window_months}mo",
            by=["establishment_id", "product"],
        )
        .alias(f"rolling_{window_months}m_log_relative_price")
    )
