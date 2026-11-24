import polars as pl


def filter_eligible_features(df: pl.LazyFrame, min_products: int, min_matched_products: int = 0) -> pl.LazyFrame:
    # Apply explicit coverage requirements; report excluded rows separately.
    raise NotImplementedError


def restrict_common_basket(df: pl.LazyFrame, products: tuple[str, ...]) -> pl.LazyFrame:
    # Restrict to a fixed verified product set; report missing basket products; do not silently treat missing as zero.
    raise NotImplementedError


