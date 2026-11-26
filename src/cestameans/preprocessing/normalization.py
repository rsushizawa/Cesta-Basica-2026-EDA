import polars as pl


def normalize_products(df: pl.LazyFrame, aliases: pl.LazyFrame) -> pl.LazyFrame:
    # Validate unique raw_name keys; left join verified aliases; create product column; preserve unmapped names and original rows.
    left = df.with_columns(
        pl.col("product_name").str.to_lowercase().str.strip_chars().alias("product_key")
    )
    if aliases is None:
        return left.with_columns(pl.col("product_name").alias("product"))
    names = aliases.collect_schema().names()
    if not {"raw_name", "canonical_name"}.issubset(names):
        return left.with_columns(pl.col("product_name").alias("product"))
    a = aliases.select(
        [
            pl.col("raw_name")
            .str.to_lowercase()
            .str.strip_chars()
            .alias("product_key"),
            pl.col("canonical_name"),
        ]
    ).unique("product_key")
    return (
        left.join(a, on="product_key", how="left")
        .with_columns(
            pl.coalesce([pl.col("canonical_name"), pl.col("product_name")]).alias(
                "product"
            )
        )
        .drop("canonical_name")
    )


def normalize_brands(df: pl.LazyFrame) -> pl.LazyFrame:
    # Normalize whitespace and casing for comparisons; never invent missing brands.
    return df.with_columns(
        pl.col("brand_name")
        .str.to_lowercase()
        .str.strip_chars()
        .alias("brand_normalized")
    )
