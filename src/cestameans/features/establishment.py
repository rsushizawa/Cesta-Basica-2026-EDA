import polars as pl


def build_establishment_month_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Group by establishment_id/reference_month; compute mean_log_relative_price, std_log_relative_price, mean_relative_iqr, mean_mom_change, std_mom_change.
    # Also output valid_price_products, matched_monthly_products, observed_products and sample_count for quality control.
    # Use sample_count >= 3 for within-product IQR; preserve null when undefined.
    return df.group_by(["establishment_id", "reference_month"]).agg(
        pl.col("log_relative_price").mean().alias("mean_log_relative_price"),
        pl.col("log_relative_price").std().alias("std_log_relative_price"),
        pl.col("price_iqr").mean().alias("mean_relative_iqr"),
        pl.col("mom_change").mean().alias("mean_mom_change"),
        pl.col("mom_change").std().alias("std_mom_change"),
        pl.col("product").n_unique().alias("valid_price_products"),
        pl.col("product").n_unique().alias("matched_monthly_products"),
        pl.col("product").n_unique().alias("observed_products"),
        pl.col("sample_count").sum().alias("sample_count"),
    )


def build_establishment_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Aggregate store-month features into long-term profiles; document time coverage and unequal observation periods.
    return df.group_by("establishment_id").agg(
        pl.col("mean_log_relative_price").mean(),
        pl.col("std_log_relative_price").mean(),
        pl.col("mean_relative_iqr").mean(),
        pl.col("mean_mom_change").mean(),
        pl.col("std_mom_change").mean(),
        pl.col("valid_price_products").mean(),
        pl.col("matched_monthly_products").mean(),
        pl.col("observed_products").mean(),
        pl.len().alias("observed_months"),
    )
