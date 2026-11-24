import polars as pl


def normalize_products(df: pl.LazyFrame, aliases: pl.LazyFrame) -> pl.LazyFrame:
    # Validate unique raw_name keys; left join verified aliases; create product column; preserve unmapped names and original rows.
    raise NotImplementedError


def normalize_brands(df: pl.LazyFrame) -> pl.LazyFrame:
    # Normalize whitespace and casing for comparisons; never invent missing brands.
    raise NotImplementedError


