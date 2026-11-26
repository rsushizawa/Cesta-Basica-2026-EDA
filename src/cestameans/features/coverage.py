import polars as pl


def filter_eligible_features(
    df: pl.LazyFrame, min_products: int, min_matched_products: int = 0
) -> pl.LazyFrame:
    # Apply explicit coverage requirements; report excluded rows separately.
    return df.filter(
        (pl.col("valid_price_products") >= min_products)
        & (pl.col("matched_monthly_products") >= min_matched_products)
    )


def restrict_common_basket(df: pl.LazyFrame, products: tuple[str, ...]) -> pl.LazyFrame:
    # Restrict to a fixed verified product set; report missing basket products; do not silently treat missing as zero.
    return df.filter(pl.col("product").is_in(products))
