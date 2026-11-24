import polars as pl


def build_establishment_month_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Group by establishment_id/reference_month; compute mean_log_relative_price, std_log_relative_price, mean_relative_iqr, mean_mom_change, std_mom_change.
    # Also output valid_price_products, matched_monthly_products, observed_products and sample_count for quality control.
    # Use sample_count >= 3 for within-product IQR; preserve null when undefined.
    raise NotImplementedError


def build_establishment_features(df: pl.LazyFrame) -> pl.LazyFrame:
    # Aggregate store-month features into long-term profiles; document time coverage and unequal observation periods.
    raise NotImplementedError


