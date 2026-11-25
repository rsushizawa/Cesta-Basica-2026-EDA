import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from matplotlib.figure import Figure


class ArtifactRepository:
    def __init__(self, root: Path) -> None:
        """Create the directory layout for one reproducible experiment run."""
        self.root = root
        self.run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.run_dir = root / "experiments" / self.run_id
        self.data_dir = self.run_dir / "data"
        self.eda_dir = self.run_dir / "eda"
        self.model_dir = self.run_dir / "model"
        self.metadata_dir = self.run_dir / "metadata"
        for directory in (
            self.data_dir,
            self.eda_dir,
            self.model_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_table(self, name: str, df: pl.DataFrame | pl.LazyFrame) -> Path:
        # Persist Parquet; use lazy sink where appropriate; return path.
        path = self._path_for_table(name)
        (df if isinstance(df, pl.DataFrame) else df.collect()).write_parquet(path)
        return path

    def save_metadata(self, name: str, metadata: dict[str, Any]) -> Path:
        # Write JSON including config, versions, dataset checksum and seed.
        path = self.metadata_dir / f"{name}.json"
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        self._write_latest_pointer()
        return path

    def save_model(self, name: str, model: Any) -> Path:
        # Choose appropriate serialization; document how to reload fitted state.
        path = self.model_dir / f"{name}.joblib"
        joblib.dump(model, path)
        return path

    def save_figure(self, name: str, figure: Figure) -> Path:
        # Save publication-quality figure; return path.
        path = self.eda_dir / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        return path

    def _path_for_table(self, name: str) -> Path:
        if name in {
            "bronze",
            "silver",
            "silver_rejected",
            "gold",
            "gold_model_input",
            "basket_cube",
        }:
            directory = self.data_dir
        elif name in {"assignments", "cluster_profiles"}:
            directory = self.model_dir
        else:
            directory = self.eda_dir
        return directory / f"{name}.parquet"

    def _write_latest_pointer(self) -> None:
        pointer = self.root / "latest.json"
        pointer.write_text(
            json.dumps({"run_id": self.run_id, "path": str(self.run_dir)}, indent=2),
            encoding="utf-8",
        )
