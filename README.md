# Cesta Means

Reproducible exploratory analysis and establishment clustering for the cesta básica dataset.

The active model is scikit-learn `KMeans`. The pipeline uses Polars and writes auditable data layers:

- `experiments/<run_id>/data/`: bronze, silver, rejected, and gold Parquet tables.
- `experiments/<run_id>/data/basket_cube.parquet`: product-category facts and standardized national basket costs.
- `experiments/<run_id>/eda/`: quality summaries, feature summaries, K diagnostics, and PNG charts.
- `experiments/<run_id>/model/`: assignments, cluster profiles, and serialized model/preprocessor.
- `experiments/<run_id>/metadata/`: metrics and run metadata.
- `latest.json`: pointer to the most recently recorded run.

Each experiment is self-contained, so runs can be compared without overwriting one another.

## Run

```bash
uv sync
PYTHONPATH=src uv run python -m cestameans \
  --input data/ICB_DADOS_2026.csv \
  --output artifacts \
  --k 4
```

Each execution creates a timestamped directory under `artifacts/`.

The clustering is descriptive for the observed period; it is not a future-time evaluation or forecast.

The basket analysis uses the national quantities from the DIEESE methodology based on Decreto-Lei 399/1938. The raw `Quantidade` field is treated as an observed package/unit size for price normalization, not as the household consumption quantity. Rows with incomplete national basket coverage keep a partial cost and coverage percentage but receive a null complete `basket_cost`.

## Notebook

Open [notebooks/cesta_basica_pipeline.ipynb](notebooks/cesta_basica_pipeline.ipynb) for a guided walkthrough. It explains each pipeline stage, displays intermediate tables, plots the main distributions, evaluates candidate values of K, and interprets the final cluster profiles.
