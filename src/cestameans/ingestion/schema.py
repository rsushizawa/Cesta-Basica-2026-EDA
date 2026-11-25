import polars as pl

# Read raw values as strings to preserve malformed values for auditing.
RAW_SCHEMA = {
    name: pl.String
    for name in (
        "Mes_Ano_Data",
        "Data_Coleta",
        "Estab",
        "Produto",
        "Marca",
        "Preco",
        "Quantidade",
        " PPK ",
    )
}
COLUMN_MAPPING = {
    "Mes_Ano_Data": "reference_month",
    "Data_Coleta": "collection_date",
    "Estab": "establishment_id",
    "Produto": "product_name",
    "Marca": "brand_name",
    "Preco": "package_price",
    "Quantidade": "quantity",
    " PPK ": "reported_unit_price",
}
