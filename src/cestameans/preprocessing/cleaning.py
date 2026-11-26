import polars as pl


def normalize_text(df: pl.LazyFrame) -> pl.LazyFrame:
    # Trim whitespace and convert empty strings to null; retain original product and brand names.
    cols = [
        c
        for c in ("establishment_id", "product_name", "brand_name")
        if c in df.collect_schema().names()
    ]
    return df.with_columns(
        [pl.col(c).str.strip_chars().replace("", None).alias(c) for c in cols]
    )


def calculate_unit_price(df: pl.LazyFrame) -> pl.LazyFrame:
    # Divide package_price by quantity only for finite positive inputs; otherwise null.
    return df.with_columns(
        pl.when((pl.col("package_price_value") > 0) & (pl.col("quantity_value") > 0))
        .then(pl.col("package_price_value") / pl.col("quantity_value"))
        .otherwise(None)
        .alias("unit_price")
    )


def apply_duplicate_policy(
    df: pl.LazyFrame, *, drop_confirmed: bool = False
) -> pl.LazyFrame:
    # Only remove duplicates under a documented policy; keep audit IDs.
    keys = [
        c
        for c in (
            "reference_month",
            "establishment_id",
            "product_name",
            "brand_name",
            "package_price_value",
            "quantity_value",
        )
        if c in df.collect_schema().names()
    ]
    return (
        df.with_columns(pl.struct(keys).is_duplicated().alias("is_duplicate"))
        if drop_confirmed
        else df
    )
