from pathlib import Path
from typing import Any
import polars as pl
from matplotlib.figure import Figure

class ArtifactRepository:
    def __init__(self, root: Path) -> None:
        # Store output root; create a distinct directory per run.
        raise NotImplementedError

    def save_table(self, name: str, df: pl.DataFrame | pl.LazyFrame) -> Path:
        # Persist Parquet; use lazy sink where appropriate; return path.
        raise NotImplementedError

    def save_metadata(self, name: str, metadata: dict[str, Any]) -> Path:
        # Write JSON including config, versions, dataset checksum and seed.
        raise NotImplementedError

    def save_model(self, name: str, model: Any) -> Path:
        # Choose appropriate serialization; document how to reload fitted state.
        raise NotImplementedError

    def save_figure(self, name: str, figure: Figure) -> Path:
        # Save publication-quality figure; return path.
        raise NotImplementedError
