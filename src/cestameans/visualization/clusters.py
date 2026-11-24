import numpy as np
import polars as pl
from matplotlib.figure import Figure


def plot_clusters(X: np.ndarray, labels: np.ndarray) -> Figure:
    # Plot 2D features or PCA projection; label axes and explain projection.
    raise NotImplementedError


def plot_k_diagnostics(metrics: pl.DataFrame) -> Figure:
    # Plot inertia and silhouette versus K; return figure without saving.
    raise NotImplementedError


def plot_cluster_profiles(profiles: pl.DataFrame) -> Figure:
    # Visualize interpretable cluster feature summaries; label units.
    raise NotImplementedError


