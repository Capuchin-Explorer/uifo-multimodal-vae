"""
Author: Raphael Jontofsohn

Post-training diagnostics and latent-space analysis for the structured multimodal
UIFO VAE. The module reconstructs validation data, exports posterior statistics,
creates UMAP visualizations, and optionally performs HDBSCAN clustering in the
full latent space.
"""
import json
import platform
import shutil
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from pathlib import Path
import umap
import argparse
import joblib
from sklearn.model_selection import train_test_split

from model import MultimodalTopologyVAE


def order_points_best_last(values: np.ndarray) -> np.ndarray:
    """Returns indices that draw high values first and low values last."""
    return np.argsort(values)[::-1]


def determine_latent_color_limits(
    values: np.ndarray,
    mode: str,
    loss_cap: float,
    low_percentile: float,
    high_percentile: float,
) -> tuple[float, float]:
    """Determine loss-color limits using the thesis post-processing policy."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite loss values are available.")

    if mode == "fixed_cap":
        vmin, vmax = float(finite.min()), float(loss_cap)
    elif mode == "percentile":
        vmin, vmax = np.percentile(
            finite,
            [low_percentile, high_percentile],
        ).astype(float)
    elif mode == "full_range":
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        raise ValueError(
            "latent_color_limit_mode must be 'fixed_cap', 'percentile', or "
            f"'full_range', got {mode!r}."
        )

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid latent-plot color limits: vmin={vmin}, vmax={vmax}")
    return vmin, vmax


def apply_latent_colorbar_ticks(
    colorbar,
    vmin: float,
    vmax: float,
    mode: str,
) -> None:
    """Apply the notebook's automatic, integer, or half-step tick policy."""
    if mode == "auto":
        return
    if mode not in {"integer", "half"}:
        raise ValueError(
            "latent_colorbar_tick_mode must be 'auto', 'integer', or 'half', "
            f"got {mode!r}."
        )

    step = 1.0 if mode == "integer" else 0.5
    first = np.ceil((vmin - 1e-12) / step) * step
    last = np.floor((vmax + 1e-12) / step) * step
    ticks = np.arange(first, last + 0.5 * step, step)
    colorbar.locator = FixedLocator(ticks)
    colorbar.formatter = FuncFormatter(lambda value, position: f"{value:g}")
    colorbar.update_ticks()


def load_model_state(model_path: Path, device: torch.device) -> dict:
    """Load either the legacy state dict or the new self-describing checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def fit_density_clusters(
    values: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
):
    """Fit HDBSCAN while supporting either the external or sklearn backend."""
    try:
        import hdbscan as hdbscan_package

        clusterer = hdbscan_package.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(values)
        probabilities = clusterer.probabilities_
        outlier_scores = clusterer.outlier_scores_
        backend = f"hdbscan {getattr(hdbscan_package, '__version__', 'unknown')}"
        return clusterer, labels, probabilities, outlier_scores, backend
    except ModuleNotFoundError:
        try:
            from sklearn import __version__ as sklearn_version
            from sklearn.cluster import HDBSCAN
        except ImportError:
            print(
                "[!] HDBSCAN is unavailable. Latent and UMAP data will still be exported, "
                "but cluster labels will be set to -1."
            )
            n_runs = len(values)
            return None, np.full(n_runs, -1), np.zeros(n_runs), np.full(n_runs, np.nan), "unavailable"

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(values)
        probabilities = getattr(clusterer, "probabilities_", np.ones(len(values)))
        outlier_scores = getattr(clusterer, "outlier_scores_", np.full(len(values), np.nan))
        return clusterer, labels, probabilities, outlier_scores, f"sklearn {sklearn_version}"


def build_cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Create one compact, metadata-aware row per density cluster."""
    rows = []
    for label, group in assignments.groupby("cluster_label", sort=True):
        row = {
            "cluster_label": int(label),
            "is_noise": bool(label == -1),
            "run_count": len(group),
            "mean_cluster_probability": float(group["cluster_probability"].mean()),
        }
        for column in ("loss_senspow", "complexity", "latent_norm", "posterior_std_mean"):
            if column in group.columns:
                row[f"median_{column}"] = float(group[column].median())
        for column in ("setup_graph", "suffix"):
            if column in group.columns and group[column].notna().any():
                counts = group[column].astype(str).value_counts()
                row[f"dominant_{column}"] = counts.index[0]
                row[f"dominant_{column}_count"] = int(counts.iloc[0])
                row[f"dominant_{column}_share"] = float(counts.iloc[0] / len(group))
                row[f"distinct_{column}_count"] = int(counts.size)
        rows.append(row)
    return pd.DataFrame(rows)


def load_aligned_visualization_data(parquet_path: Path, npy_path: Path, index_path: Path | None = None):
    """Loads topology matrix and aligns sensitivity rows to matrix order if an index is available."""
    df = pd.read_parquet(parquet_path)
    x_params_raw = np.load(npy_path)

    if index_path is not None:
        if not index_path.exists():
            raise FileNotFoundError(f"[!] Index file missing: {index_path}")

        if index_path.suffix == ".parquet":
            df_index = pd.read_parquet(index_path)
        elif index_path.suffix == ".csv":
            df_index = pd.read_csv(index_path)
        elif index_path.suffix == ".json":
            df_index = pd.read_json(index_path)
        else:
            raise ValueError("index_path must be a .parquet, .csv, or .json file")

        required_cols = {"hash", "run_id"}
        if not required_cols.issubset(df.columns) or not required_cols.issubset(df_index.columns):
            raise ValueError("Both parquet data and index file must contain hash and run_id columns.")
        if len(df_index) != len(x_params_raw):
            raise ValueError(
                f"Index row count mismatch: index has {len(df_index)} rows, "
                f"matrix has {len(x_params_raw)} rows."
            )

        df_index = df_index[["hash", "run_id"]].copy()
        df_index["__matrix_order"] = np.arange(len(df_index))
        df = df.merge(df_index, on=["hash", "run_id"], how="inner")
        if len(df) != len(df_index):
            raise ValueError(
                f"Alignment failed: only {len(df)} of {len(df_index)} matrix rows matched parquet hash/run_id keys."
            )
        df = df.sort_values("__matrix_order").drop(columns=["__matrix_order"]).reset_index(drop=True)
        print(f"[+] Aligned visualization rows via {index_path.name}.")
    elif {"hash", "run_id"}.issubset(df.columns):
        print("[!] No index_path provided. Assuming parquet row order already matches matrix row order.")

    if len(df) != len(x_params_raw):
        raise ValueError(
            f"Row count mismatch: parquet has {len(df)} rows, matrix has {len(x_params_raw)} rows."
        )

    sens_cols = [c for c in df.columns if c.startswith("sens_")]
    x_sens_raw = df[sens_cols].values
    physical_losses = df['loss_senspow'].values if 'loss_senspow' in df.columns else np.zeros(len(df))
    return df, x_params_raw, x_sens_raw, physical_losses


def plot_training_history(history_path: Path, output_dir: Path, latent_dim: int, repr_type: str) -> None:
    """
    Plots the empirical loss progression across epochs for both total optimization
    objectives and isolated multimodal reconstruction criteria.
    """
    print("-" * 80)
    print(f"Executing Loss Progression Analysis ({latent_dim}D) - {repr_type}")
    print("-" * 80)

    with open(history_path, "r") as f:
        history = json.load(f)

    total_epochs = len(history['train_total'])
    start_epoch = 45 if total_epochs > 50 else 0
    x_epochs = list(range(start_epoch, total_epochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # -------------------------------------------------------------------
    # Axis 1: Total Loss & Regularization Schedule
    # -------------------------------------------------------------------
    ax1.plot(x_epochs, history['train_total'][start_epoch:], label='Train Total', color='tab:blue', linewidth=1.5)
    ax1.plot(x_epochs, history['val_total'][start_epoch:], label='Val Total', color='tab:orange', linewidth=1.5)

    max_beta = max(history['beta'])
    warmup_end_epoch = history['beta'].index(max_beta)
    search_start = min(warmup_end_epoch, total_epochs - 1)
    
    post_warmup_val_losses = history['val_total'][search_start:]
    best_val_loss = min(post_warmup_val_losses)
    best_epoch_idx = post_warmup_val_losses.index(best_val_loss) + search_start

    if best_epoch_idx >= start_epoch:
        ax1.plot(best_epoch_idx, best_val_loss, marker='o', color='red', markersize=6,
                 linestyle='None', label=f'Min Val: {best_val_loss:.4f} (Ep {best_epoch_idx})', zorder=5)

    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Objective Value', fontsize=11)
    ax1.set_title(f'Global Optimization Sequence ({latent_dim}D) - {repr_type}', fontsize=12, pad=12)
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax1_beta = ax1.twinx()
    ax1_beta.plot(x_epochs, history['beta'][start_epoch:], color='tab:green', linestyle=':', linewidth=1.5)
    ax1_beta.set_ylabel('KL Weight (Beta)', color='tab:green', fontsize=11)
    ax1_beta.tick_params(axis='y', labelcolor='tab:green')
    ax1_beta.set_ylim([-0.01, (max_beta * 1.2 if max_beta > 0 else 0.1)])

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    # -------------------------------------------------------------------
    # Axis 2: Objective Deconstruction
    # -------------------------------------------------------------------
    ax2.plot(x_epochs, history['val_sens'][start_epoch:], label='Sensitivity (MSE)', color='tab:red', linewidth=1.5)
    ax2.plot(x_epochs, history['val_params'][start_epoch:], label='Topology (BCE)', color='tab:brown', linewidth=1.5)
    ax2.plot(x_epochs, history['val_kl'][start_epoch:], label='Divergence (KL)', color='tab:purple', linewidth=1.5)

    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Loss Metric (Log Scale)', fontsize=11)
    ax2.set_title(f'Validation Component Decomposition ({latent_dim}D) - {repr_type}', fontsize=12, pad=12)
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)
    
    # Repositioned legend to center right to avoid overlapping with the KL divergence trajectory
    ax2.legend(loc='center right', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    out_path = output_dir / f"01_vae_losses_{latent_dim}D_{repr_type}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Export] Saved objective trajectory: {out_path.name}")


def plot_latent_space(
    model_path: Path,
    parquet_path: Path,
    npy_path: Path,
    vocab_path: Path,
    output_dir: Path,
    latent_dim: int,
    repr_type: str,
    index_path: Path | None = None,
    split_path: Path | None = None,
    random_seed: int = 42,
    val_split: float = 0.15,
    sens_weight: float = 100.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    cluster_space: str = "latent",
    min_cluster_size: int = 50,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    skip_clustering: bool = False,
    latent_point_size: float = 8.0,
    latent_point_alpha: float = 0.60,
    latent_point_edge_mode: str = "none",
    latent_point_edge_width: float = 0.25,
    latent_color_limit_mode: str = "percentile",
    latent_loss_cap: float = 4.0,
    latent_low_percentile: float = 1.0,
    latent_high_percentile: float = 95.0,
    latent_explicit_color_clipping: bool = True,
    latent_colorbar_tick_mode: str = "auto",
) -> None:
    """
    Maps multimodal physical parameters to bottleneck coordinates
    and projects the topology to a 2D UMAP subspace.
    """
    print("\n" + "-" * 80)
    print(f"Executing Latent Subspace Projection ({latent_dim}D) - {repr_type}")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(vocab_path, "r") as f:
        global_vocab = json.load(f)

    df, x_params_raw, x_sens_raw, physical_losses = load_aligned_visualization_data(
        parquet_path, npy_path, index_path
    )
    x_sens_scaled = (np.log10(x_sens_raw + 1e-35) + 30.0) / 30.0

    model = MultimodalTopologyVAE(
        global_vocab=global_vocab,
        latent_dim=latent_dim,
        sens_weight=sens_weight,
    ).to(device)
    model.load_state_dict(load_model_state(model_path, device))
    model.eval()

    latent_vectors = []
    logvar_vectors = []
    sensitivity_mse = []
    parameter_class_bce = []
    parameter_value_mse = []
    parameter_edge_mse = []
    kl_divergence = []
    batch_size = 512

    class_mask = model.param_class_mask.unsqueeze(0)
    value_mask = model.param_value_mask.unsqueeze(0)
    edge_mask = model.param_edge_mask.unsqueeze(0)

    def masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum(dim=1) / mask.sum().clamp_min(1.0)

    with torch.no_grad():
        for i in range(0, len(x_sens_scaled), batch_size):
            b_sens = torch.tensor(x_sens_scaled[i:i+batch_size], dtype=torch.float32).to(device)
            b_params = torch.tensor(x_params_raw[i:i+batch_size], dtype=torch.float32).to(device)
            mu, logvar = model.encode(b_sens, b_params)
            recon_sens, recon_params = model.decode(mu)

            sens_error = F.mse_loss(recon_sens, b_sens, reduction="none").mean(dim=1)
            class_error = F.binary_cross_entropy(recon_params, b_params, reduction="none")
            value_error = F.mse_loss(recon_params, b_params, reduction="none")
            kl_error = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - logvar.exp(),
                dim=1,
            )

            latent_vectors.append(mu.cpu().numpy())
            logvar_vectors.append(logvar.cpu().numpy())
            sensitivity_mse.append(sens_error.cpu().numpy())
            parameter_class_bce.append(masked_row_mean(class_error, class_mask).cpu().numpy())
            parameter_value_mse.append(masked_row_mean(value_error, value_mask).cpu().numpy())
            parameter_edge_mse.append(masked_row_mean(value_error, edge_mask).cpu().numpy())
            kl_divergence.append(kl_error.cpu().numpy())

    latent_matrix = np.vstack(latent_vectors)
    logvar_matrix = np.vstack(logvar_vectors)
    sensitivity_mse = np.concatenate(sensitivity_mse)
    parameter_class_bce = np.concatenate(parameter_class_bce)
    parameter_value_mse = np.concatenate(parameter_value_mse)
    parameter_edge_mse = np.concatenate(parameter_edge_mse)
    kl_divergence = np.concatenate(kl_divergence)

    if latent_dim > 2:
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_seed,
        )
        plot_coords = reducer.fit_transform(latent_matrix)
        x_label, y_label = "UMAP component 1", "UMAP component 2"
        reducer_path = output_dir / f"umap_reducer_{latent_dim}D_{repr_type}.joblib"
        joblib.dump(reducer, reducer_path)
    else:
        plot_coords = latent_matrix
        if plot_coords.shape[1] == 1:
            plot_coords = np.column_stack([plot_coords[:, 0], np.zeros(len(plot_coords))])
        x_label, y_label = "Latent Mean Coordinate $\mu_0$", "Latent Mean Coordinate $\mu_1$"
        reducer = None
        reducer_path = None

    cluster_input = latent_matrix if cluster_space == "latent" else plot_coords
    if skip_clustering:
        clusterer = None
        cluster_labels = np.full(len(df), -1)
        cluster_probabilities = np.zeros(len(df))
        outlier_scores = np.full(len(df), np.nan)
        cluster_backend = "disabled"
    else:
        clusterer, cluster_labels, cluster_probabilities, outlier_scores, cluster_backend = fit_density_clusters(
            cluster_input,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

    cluster_model_path = None
    if clusterer is not None:
        cluster_model_path = output_dir / f"hdbscan_{cluster_space}_{latent_dim}D_{repr_type}.joblib"
        joblib.dump(clusterer, cluster_model_path)

    assignments = df.copy().reset_index(drop=True)
    assignments.insert(0, "matrix_row", np.arange(len(assignments), dtype=np.int64))
    assignments.insert(1, "run_key", assignments["hash"].astype(str) + "_" + assignments["run_id"].astype(str))
    assignments["representation"] = repr_type

    if split_path is not None and split_path.exists():
        split_df = pd.read_parquet(split_path)
        required_split_columns = {"hash", "run_id", "dataset_split"}
        if not required_split_columns.issubset(split_df.columns):
            raise ValueError(f"Split file is missing columns: {sorted(required_split_columns - set(split_df.columns))}")
        split_df = split_df[["hash", "run_id", "dataset_split"]]
        assignments = assignments.merge(
            split_df,
            on=["hash", "run_id"],
            how="left",
            validate="one_to_one",
        )
        if assignments["dataset_split"].isna().any():
            raise ValueError("Some latent rows could not be matched to dataset_split.parquet.")
    else:
        _, validation_indices = train_test_split(
            np.arange(len(assignments)),
            test_size=val_split,
            random_state=random_seed,
        )
        split_labels = np.full(len(assignments), "train", dtype=object)
        split_labels[validation_indices] = "validation"
        assignments["dataset_split"] = split_labels

    for dimension in range(latent_dim):
        assignments[f"mu_{dimension:02d}"] = latent_matrix[:, dimension]
        assignments[f"logvar_{dimension:02d}"] = logvar_matrix[:, dimension]

    assignments["latent_norm"] = np.linalg.norm(latent_matrix, axis=1)
    assignments["posterior_std_mean"] = np.exp(0.5 * logvar_matrix).mean(axis=1)
    assignments["umap_1"] = plot_coords[:, 0]
    assignments["umap_2"] = plot_coords[:, 1]
    assignments["cluster_label"] = cluster_labels.astype(np.int64)
    assignments["cluster_probability"] = cluster_probabilities
    assignments["cluster_outlier_score"] = outlier_scores
    assignments["active_parameter_features"] = np.count_nonzero(x_params_raw, axis=1)
    assignments["sensitivity_reconstruction_mse"] = sensitivity_mse
    assignments["weighted_sensitivity_reconstruction"] = sens_weight * sensitivity_mse
    assignments["parameter_class_bce"] = parameter_class_bce
    assignments["parameter_value_mse"] = parameter_value_mse
    assignments["parameter_edge_mse"] = parameter_edge_mse
    assignments["parameter_reconstruction_total"] = (
        parameter_class_bce + parameter_value_mse + parameter_edge_mse
    )
    assignments["kl_divergence"] = kl_divergence

    assignments_path = output_dir / f"latent_run_assignments_{latent_dim}D_{repr_type}.parquet"
    assignments.to_parquet(assignments_path, index=False)

    arrays_path = output_dir / f"latent_arrays_{latent_dim}D_{repr_type}.npz"
    np.savez_compressed(
        arrays_path,
        mu=latent_matrix,
        logvar=logvar_matrix,
        umap=plot_coords,
        cluster_label=cluster_labels,
        cluster_probability=cluster_probabilities,
        matrix_row=np.arange(len(assignments), dtype=np.int64),
    )

    parameter_feature_columns = [f"param_feature__{feature}" for feature in global_vocab]
    parameter_features = pd.DataFrame(x_params_raw, columns=parameter_feature_columns)
    parameter_features.insert(0, "run_key", assignments["run_key"].to_numpy())
    parameter_features.insert(0, "matrix_row", np.arange(len(assignments), dtype=np.int64))
    parameter_features_path = output_dir / f"parameter_features_{repr_type}.parquet"
    parameter_features.to_parquet(parameter_features_path, index=False)

    analysis_vocab_path = output_dir / f"analysis_vocabulary_{repr_type}.json"
    with open(analysis_vocab_path, "w") as handle:
        json.dump(global_vocab, handle, indent=2)

    cluster_summary = build_cluster_summary(assignments)
    summary_path = output_dir / f"cluster_summary_{latent_dim}D_{repr_type}.parquet"
    cluster_summary.to_parquet(summary_path, index=False)

    visualization_snapshot = output_dir / "analysis_code_snapshot_visualize.py"
    model_snapshot = output_dir / "analysis_code_snapshot_model.py"
    latent_plot_pdf_path = output_dir / f"02_vae_latent_space_{latent_dim}D_{repr_type}_thesis.pdf"
    latent_plot_preview_path = output_dir / f"02_vae_latent_space_{latent_dim}D_{repr_type}.png"
    shutil.copy2(Path(__file__).resolve(), visualization_snapshot)
    shutil.copy2(Path(__file__).resolve().parent / "model.py", model_snapshot)

    analysis_manifest = {
        "schema_version": 1,
        "representation": repr_type,
        "run_count": len(assignments),
        "latent_dim": latent_dim,
        "parameter_dim": x_params_raw.shape[1],
        "sensitivity_dim": x_sens_raw.shape[1],
        "model_path": str(model_path.resolve()),
        "matrix_path": str(npy_path.resolve()),
        "vocabulary_path": str(vocab_path.resolve()),
        "index_path": str(index_path.resolve()) if index_path else None,
        "parquet_path": str(parquet_path.resolve()),
        "split_path": str(split_path.resolve()) if split_path and split_path.exists() else None,
        "sensitivity_scaling": {"epsilon": 1e-35, "offset": 30.0, "divisor": 30.0},
        "umap": {
            "n_neighbors": umap_n_neighbors,
            "min_dist": umap_min_dist,
            "metric": "euclidean",
            "random_state": random_seed,
            "model_path": str(reducer_path.resolve()) if reducer_path else None,
        },
        "clustering": {
            "enabled": not skip_clustering,
            "space": cluster_space,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "metric": "euclidean",
            "backend": cluster_backend,
            "model_path": str(cluster_model_path.resolve()) if cluster_model_path else None,
            "cluster_count_excluding_noise": int(len(set(cluster_labels)) - (-1 in set(cluster_labels))),
            "noise_run_count": int(np.sum(cluster_labels == -1)),
        },
        "latent_plot": {
            "colormap": "magma_r",
            "point_size": latent_point_size,
            "point_alpha": latent_point_alpha,
            "point_edge_mode": latent_point_edge_mode,
            "point_edge_width": latent_point_edge_width,
            "color_limit_mode": latent_color_limit_mode,
            "loss_colorbar_cap": latent_loss_cap,
            "low_percentile": latent_low_percentile,
            "high_percentile": latent_high_percentile,
            "explicit_color_clipping": latent_explicit_color_clipping,
            "colorbar_tick_mode": latent_colorbar_tick_mode,
            "point_order": "descending_loss_best_drawn_last",
            "figure_size_inches": [7.0, 6.2],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "umap": getattr(umap, "__version__", "unknown"),
        },
        "outputs": {
            "assignments": str(assignments_path.resolve()),
            "arrays": str(arrays_path.resolve()),
            "parameter_features": str(parameter_features_path.resolve()),
            "vocabulary_snapshot": str(analysis_vocab_path.resolve()),
            "cluster_summary": str(summary_path.resolve()),
            "latent_plot_pdf": str(latent_plot_pdf_path.resolve()),
            "latent_plot_preview": str(latent_plot_preview_path.resolve()),
            "visualization_code_snapshot": str(visualization_snapshot.resolve()),
            "model_code_snapshot": str(model_snapshot.resolve()),
        },
    }
    manifest_path = output_dir / f"latent_analysis_manifest_{latent_dim}D_{repr_type}.json"
    with open(manifest_path, "w") as handle:
        json.dump(analysis_manifest, handle, indent=2)

    print(f"[Export] Saved per-run post-cluster table: {assignments_path.name}")
    print(f"[Export] Saved latent arrays: {arrays_path.name}")
    print(f"[Export] Saved aligned parameter features: {parameter_features_path.name}")
    print(f"[Export] Saved cluster summary: {summary_path.name}")
    print(
        f"[Clusters] backend={cluster_backend}, "
        f"clusters={analysis_manifest['clustering']['cluster_count_excluding_noise']}, "
        f"noise={analysis_manifest['clustering']['noise_run_count']}"
    )

    valid_edge_modes = {"none", "same", "black"}
    if latent_point_edge_mode not in valid_edge_modes:
        raise ValueError(
            "latent_point_edge_mode must be one of "
            f"{sorted(valid_edge_modes)}, got {latent_point_edge_mode!r}."
        )
    if not 0.0 < latent_point_alpha <= 1.0:
        raise ValueError("latent_point_alpha must lie in the interval (0, 1].")
    if latent_point_size <= 0.0 or latent_point_edge_width < 0.0:
        raise ValueError("Point size must be positive and edge width must be non-negative.")
    if not 0.0 <= latent_low_percentile < latent_high_percentile <= 100.0:
        raise ValueError(
            "Latent color percentiles must satisfy "
            "0 <= low_percentile < high_percentile <= 100."
        )
    if latent_colorbar_tick_mode not in {"auto", "integer", "half"}:
        raise ValueError(
            "latent_colorbar_tick_mode must be 'auto', 'integer', or 'half'."
        )

    # Match the thesis post-processing notebook: remove non-finite rows, draw
    # high-loss points first, and keep the best points visible on top.
    valid = np.isfinite(physical_losses) & np.isfinite(plot_coords).all(axis=1)
    plot_coords_valid = plot_coords[valid]
    losses_valid = physical_losses[valid]
    sort_idx = order_points_best_last(losses_valid)
    plot_coords_sorted = plot_coords_valid[sort_idx]
    losses_sorted = losses_valid[sort_idx]

    vmin, vmax = determine_latent_color_limits(
        losses_sorted,
        mode=latent_color_limit_mode,
        loss_cap=latent_loss_cap,
        low_percentile=latent_low_percentile,
        high_percentile=latent_high_percentile,
    )
    color_values = (
        np.clip(losses_sorted, vmin, vmax)
        if latent_explicit_color_clipping
        else losses_sorted
    )

    edgecolors = {
        "none": "none",
        "same": "face",
        "black": "black",
    }[latent_point_edge_mode]
    linewidth = 0.0 if latent_point_edge_mode == "none" else latent_point_edge_width

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    scatter = ax.scatter(
        plot_coords_sorted[:, 0],
        plot_coords_sorted[:, 1],
        c=color_values,
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        s=latent_point_size,
        alpha=latent_point_alpha,
        edgecolors=edgecolors,
        linewidths=linewidth,
        rasterized=False,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Loss", fontsize=11)
    apply_latent_colorbar_ticks(
        cbar,
        vmin=vmin,
        vmax=vmax,
        mode=latent_colorbar_tick_mode,
    )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(latent_plot_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(latent_plot_preview_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(
        f"[Latent plot] raw loss range={losses_sorted.min():.6g} to "
        f"{losses_sorted.max():.6g}; color range={vmin:.6g} to {vmax:.6g}; "
        f"below_vmin={int(np.sum(losses_sorted < vmin))}; "
        f"above_vmax={int(np.sum(losses_sorted > vmax))}; "
        f"mode={latent_color_limit_mode}; ticks={latent_colorbar_tick_mode}; "
        f"edge_mode={latent_point_edge_mode}; point_size={latent_point_size:g}"
    )
    print(f"[Export] Saved thesis subspace embedding: {latent_plot_pdf_path.name}")
    print(f"[Export] Saved PNG preview: {latent_plot_preview_path.name}")


def plot_latent_space_by_nonzero_count(
    model_path: Path,
    parquet_path: Path,
    npy_path: Path,
    vocab_path: Path,
    output_dir: Path,
    latent_dim: int,
    repr_type: str,
    index_path: Path | None = None,
) -> None:
    """
    Maps multimodal samples to latent coordinates and colors each point by the
    number of topology-vector entries above several activity thresholds.
    """
    print("\n" + "-" * 80)
    print(f"Executing Latent Subspace Projection by Non-Zero Count ({latent_dim}D) - {repr_type}")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(vocab_path, "r") as f:
        global_vocab = json.load(f)

    _, x_params_raw, x_sens_raw, _ = load_aligned_visualization_data(parquet_path, npy_path, index_path)
    x_sens_scaled = (np.log10(x_sens_raw + 1e-35) + 30.0) / 30.0
    epsilons = [0.5, 0.75, 0.99]
    activity_counts = {
        eps: np.count_nonzero(np.abs(x_params_raw) > eps, axis=1)
        for eps in epsilons
    }

    model = MultimodalTopologyVAE(global_vocab=global_vocab, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    latent_vectors = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(x_sens_scaled), batch_size):
            b_sens = torch.tensor(x_sens_scaled[i:i+batch_size], dtype=torch.float32).to(device)
            b_params = torch.tensor(x_params_raw[i:i+batch_size], dtype=torch.float32).to(device)
            mu, _ = model.encode(b_sens, b_params)
            latent_vectors.append(mu.cpu().numpy())

    latent_matrix = np.vstack(latent_vectors)

    if latent_dim > 2:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        plot_coords = reducer.fit_transform(latent_matrix)
        x_label, y_label = "UMAP Coordinate 1", "UMAP Coordinate 2"
    else:
        plot_coords = latent_matrix
        x_label, y_label = "Latent Mean Coordinate $\mu_0$", "Latent Mean Coordinate $\mu_1$"

    fig, axes = plt.subplots(1, len(epsilons), figsize=(18, 5.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"Latent Representation by Feature Sparsity ({latent_dim}D Subspace) - {repr_type}",
        fontsize=13,
        y=1.02,
    )

    for ax, eps in zip(axes, epsilons):
        counts = activity_counts[eps]
        sort_idx = order_points_best_last(counts)
        plot_coords_sorted = plot_coords[sort_idx]
        counts_sorted = counts[sort_idx]

        scatter = ax.scatter(
            plot_coords_sorted[:, 0], plot_coords_sorted[:, 1],
            c=counts_sorted, cmap='viridis', s=10, alpha=0.7, edgecolors='none'
        )

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Active Entries', fontsize=10)

        ax.set_title(f"|value| > {eps:g}", fontsize=11, pad=10)
        ax.set_xlabel(x_label, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.2)

    axes[0].set_ylabel(y_label, fontsize=10)

    plt.tight_layout()
    out_path = output_dir / f"02b_vae_latent_space_nonzero_count_{latent_dim}D_{repr_type}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Export] Saved non-zero count subspace embedding: {out_path.name}")


def compare_posterior_convergence(
    current_model_path: Path,
    baseline_model_path: Path,
    parquet_path: Path,
    npy_path: Path,
    vocab_path: Path,
    output_dir: Path,
    latent_dim: int,
    repr_type: str,
    index_path: Path | None = None,
) -> None:
    """
    DIAGNOSTIC CRITERION: Comparatively analyzes variance scaling behavior between 
    a zero-KL baseline (Deterministic AE) and the current true VAE formulation.
    """
    print("\n" + "-" * 80)
    print(f"Executing Comparative Posterior Convergence Analysis ({latent_dim}D) - {repr_type}")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(vocab_path, "r") as f:
        global_vocab = json.load(f)

    _, x_params_raw, x_sens_raw, _ = load_aligned_visualization_data(parquet_path, npy_path, index_path)
    x_sens_scaled = (np.log10(x_sens_raw + 1e-35) + 30.0) / 30.0

    b_sens = torch.tensor(x_sens_scaled[:512], dtype=torch.float32).to(device)
    b_params = torch.tensor(x_params_raw[:512], dtype=torch.float32).to(device)

    def extract_metrics(model_path: Path, is_baseline=False):
        model = MultimodalTopologyVAE(global_vocab=global_vocab, latent_dim=latent_dim).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            if is_baseline:
                print(f"[!] Baseline dimension mismatch (different vocabulary size). Skipping baseline comparison.")
                return None, None, None
            raise e
            
        model.eval() 

        with torch.no_grad():
            _, logvar = model.encode(b_sens, b_params)
            logvar_np = logvar.cpu().numpy().flatten()

        logvar_weights = model.fc_logvar.weight.detach().cpu().numpy().flatten()

        single_sens, single_params = b_sens[0:1], b_params[0:1]
        generations = []
        
        with torch.no_grad():
            mu, logvar = model.encode(single_sens, single_params)
            for _ in range(100):
                z = model.reparameterize(mu, logvar)
                _, recon_p = model.decode(z)
                generations.append(recon_p.cpu().numpy())
                
        generations_stacked = np.vstack(generations)
        empirical_variances = np.var(generations_stacked, axis=0)

        return logvar_np, logvar_weights, empirical_variances

    print("[*] Evaluating True VAE (Current Run)...")
    vae_logvar, vae_weights, vae_variances = extract_metrics(current_model_path, is_baseline=False)

    base_logvar, base_weights, base_variances = None, None, None
    if baseline_model_path and baseline_model_path.exists():
        print("[*] Evaluating Deterministic AE (Baseline Run)...")
        base_logvar, base_weights, base_variances = extract_metrics(baseline_model_path, is_baseline=True)
    else:
        print(f"[!] Baseline model not found at {baseline_model_path}. Plotting VAE only.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Metric 1: Log-Variance Predictions
    if base_logvar is not None:
        axes[0].hist(base_logvar, bins='auto', color='tab:red', alpha=0.5, edgecolor='none', label=r'Baseline AE ($\beta=0.0$)')
    axes[0].hist(vae_logvar, bins='auto', color='tab:blue', alpha=0.7, edgecolor='none', label=f'True VAE ({repr_type})')
    axes[0].set_title(f'Predicted $\log(\sigma^2)$ Density - {repr_type}', fontsize=11, pad=10)
    axes[0].set_xlabel('Log-Variance Output', fontsize=10)
    axes[0].set_ylabel('Density Count (Linear)', fontsize=10)
    axes[0].axvline(-15, color='black', linestyle=':', linewidth=1.2, label='Lower Clamp Bound')
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.3)

    # Metric 2: Weights Spectrum
    if base_weights is not None:
        axes[1].hist(base_weights, bins='auto', color='tab:red', alpha=0.5, edgecolor='none', label=r'Baseline AE ($\beta=0.0$)')
    axes[1].hist(vae_weights, bins='auto', color='tab:blue', alpha=0.7, edgecolor='none', label=f'True VAE ({repr_type})')
    axes[1].set_title(f'Linear Layer Weights Spectrum - {repr_type}', fontsize=11, pad=10)
    axes[1].set_xlabel('Weight Magnitude', fontsize=10)
    axes[1].set_ylabel('Count (Linear)', fontsize=10)
    axes[1].axvline(0, color='black', linestyle=':', linewidth=1.2, label='Origin Line')
    axes[1].legend(loc='upper right', frameon=False, fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.3)

    # Metric 3: Sample Reconstruction Dispersion
    if base_variances is not None:
        axes[2].plot(base_variances, marker='o', linestyle='None', color='tab:red', markersize=3.5, alpha=0.4, label=r'Baseline AE ($\beta=0.0$)')
    axes[2].plot(vae_variances, marker='o', linestyle='None', color='tab:blue', markersize=3.5, alpha=0.7, label=f'True VAE ({repr_type})')
    axes[2].set_title(f'Stochastic Reconstruction Variance - {repr_type}', fontsize=11, pad=10)
    axes[2].set_xlabel('Parameter Coordinate Index', fontsize=10)
    axes[2].set_ylabel('Variance Value $\sigma^2_{100}$ (Log10 Scale)', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.3)
    axes[2].legend(loc='upper right', frameon=False, fontsize=9)
    axes[2].set_yscale('log')

    plt.tight_layout()
    out_path = output_dir / f"03_comparative_posterior_convergence_{latent_dim}D_{repr_type}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Export] Saved comparative convergence analytics: {out_path.name}")


def plot_cross_repr_loss_trajectories(parent_dir: Path, latent_dim: int, job_id: str) -> None:
    """
    NEW: Compares the temporal loss trajectories (MSE, BCE, KL) across 
    the different vector representations over epochs in distinct subplots,
    filtered by the cluster SLURM job ID.
    """
    print("\n" + "-" * 80)
    print(f"Executing Cross-Representation Loss Trajectory Analysis ({latent_dim}D) [Job: {job_id}]")
    print("-" * 80)

    trajectories = {}
    colors = {'FLAT': 'tab:blue', 'GRID': 'tab:orange', 'ALIASED': 'tab:green'}

    # Updated search pattern to mandate the job ID
    search_pattern = f"*vae_dim_{latent_dim}_*_repr_*_job{job_id}/training_history.json"
    
    for hist_path in parent_dir.rglob(search_pattern):
        dir_name = hist_path.parent.name
        try:
            repr_str = dir_name.split('_repr_')[1].split('_')[0].upper()
        except IndexError:
            continue

        with open(hist_path, "r") as f:
            history = json.load(f)
            
        # Start plotting from epoch 45 to skip initial massive spikes
        start_epoch = 45 if len(history['val_sens']) > 50 else 0
        
        trajectories[repr_str] = {
            'epochs': list(range(start_epoch, len(history['val_sens']))),
            'mse': history['val_sens'][start_epoch:],
            'bce': history['val_params'][start_epoch:],
            'kl': history['val_kl'][start_epoch:]
        }

    if not trajectories:
        print("[!] No sweep data found for temporal comparison.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ensure a consistent plotting order if available
    ordered_keys = [k for k in ['FLAT', 'GRID', 'ALIASED'] if k in trajectories] + [k for k in trajectories.keys() if k not in ['FLAT', 'GRID', 'ALIASED']]

    for repr_type in ordered_keys:
        data = trajectories[repr_type]
        c = colors.get(repr_type, 'tab:gray')
        
        axes[0].plot(data['epochs'], data['mse'], label=repr_type, color=c, alpha=0.8, linewidth=1.5)
        axes[1].plot(data['epochs'], data['bce'], label=repr_type, color=c, alpha=0.8, linewidth=1.5)
        axes[2].plot(data['epochs'], data['kl'], label=repr_type, color=c, alpha=0.8, linewidth=1.5)

    # Styling Axis 0: MSE
    axes[0].set_title('Sensitivity Fidelity (MSE) Trajectories', fontsize=12, pad=10)
    axes[0].set_xlabel('Epochs', fontsize=11)
    axes[0].set_ylabel('Validation MSE (Log Scale)', fontsize=11)
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', frameon=True)

    # Styling Axis 1: BCE
    axes[1].set_title('Topological Reconstruction (BCE) Trajectories', fontsize=12, pad=10)
    axes[1].set_xlabel('Epochs', fontsize=11)
    axes[1].set_ylabel('Validation BCE (Log Scale)', fontsize=11)
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', frameon=True)

    # Styling Axis 2: KL
    axes[2].set_title('Prior Matching (KL Divergence) Trajectories', fontsize=12, pad=10)
    axes[2].set_xlabel('Epochs', fontsize=11)
    axes[2].set_ylabel('Validation KL (Log Scale)', fontsize=11)
    axes[2].set_yscale('log')
    axes[2].grid(True, which="both", linestyle='--', alpha=0.3)
    axes[2].legend(loc='lower right', frameon=True)

    plt.tight_layout()
    out_path = parent_dir / f"05_cross_representation_loss_trajectories_{latent_dim}D.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Export] Saved Loss Trajectory Comparison: {out_path}")


def plot_repr_sweep_summary(parent_dir: Path, latent_dim: int, job_id: str) -> None:
    """
    Scans the parent directory for completed runs of the Representation sweep
    (Flat vs Grid vs Aliased) filtered by SLURM job ID and visualizes 
    the comparative performance via bar charts.
    """
    print("\n" + "-" * 80)
    print(f"Executing Cross-Run Vector Representation Analysis ({latent_dim}D) [Job: {job_id}]")
    print("-" * 80)

    sweep_data = []

    # Updated search pattern to mandate the job ID
    search_pattern = f"*vae_dim_{latent_dim}_*_repr_*_job{job_id}/training_history.json"

    for hist_path in parent_dir.rglob(search_pattern):
        dir_name = hist_path.parent.name
        try:
            repr_str = dir_name.split('_repr_')[1].split('_')[0].upper()
        except IndexError:
            continue

        with open(hist_path, "r") as f:
            history = json.load(f)

        max_beta_hist = max(history['beta'])
        warmup_end = history['beta'].index(max_beta_hist)
        
        post_warmup_val_total = history['val_total'][warmup_end:]
        best_val_total = min(post_warmup_val_total)
        best_idx = post_warmup_val_total.index(best_val_total) + warmup_end

        recon_loss = history['val_sens'][best_idx] + history['val_params'][best_idx]
        kl_loss = history['val_kl'][best_idx]

        sweep_data.append({
            'repr': repr_str,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': best_val_total
        })

    if not sweep_data:
        print(f"[!] Insufficient sweep data found in {parent_dir} to generate schema comparison.")
        return

    sweep_data.sort(key=lambda x: x['total_loss'])
    
    reprs = [r['repr'] for r in sweep_data]
    recons = [r['recon_loss'] for r in sweep_data]
    kls = [r['kl_loss'] for r in sweep_data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['tab:blue' if r == 'FLAT' else 'tab:orange' if r == 'GRID' else 'tab:green' for r in reprs]

    axes[0].bar(reprs, recons, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_title('Reconstruction Fidelity by Schema', fontsize=12, pad=10)
    axes[0].set_ylabel('Validation Recon Loss (MSE + BCE)', fontsize=11)
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)

    axes[1].bar(reprs, kls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_title('Latent Space Regularization by Schema', fontsize=12, pad=10)
    axes[1].set_ylabel('Validation KL Divergence', fontsize=11)
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)

    axes[2].scatter(kls, recons, s=150, c=colors, edgecolor='black', zorder=5)
    for i, txt in enumerate(reprs):
        axes[2].annotate(txt, (kls[i], recons[i]), textcoords="offset points", xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
    
    axes[2].set_title('Topological Representation Pareto Trade-off', fontsize=12, pad=10)
    axes[2].set_xlabel('Validation KL Divergence (Prior Matching)', fontsize=11)
    axes[2].set_ylabel('Validation Recon Loss (Data Fidelity)', fontsize=11)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_path = parent_dir / f"04_representation_sweep_comparison_{latent_dim}D.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Export] Saved Representation Sweep Comparison: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the .npy feature matrix")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the JSON vocabulary file")
    parser.add_argument("--index_path", type=str, required=True, help="Required hash/run_id index file for reproducible row alignment")
    parser.add_argument("--parquet_path", type=str, default="data/master_small_filter.parquet", help="Run metadata and sensitivity curves")
    parser.add_argument("--repr_type", type=str, required=True, help="The representation schema (e.g. FLAT, GRID, ALIASED)")
    
    parser.add_argument("--out_dir", type=str, default="results/vae_multimodal")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--baseline_dir", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--sens_weight", type=float, default=100.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--cluster_space", choices=("latent", "umap"), default="latent")
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--skip_clustering", action="store_true")
    parser.add_argument("--latent_point_size", type=float, default=8.0)
    parser.add_argument("--latent_point_alpha", type=float, default=0.60)
    parser.add_argument(
        "--latent_point_edge_mode",
        choices=("none", "same", "black"),
        default="none",
        help="Point-outline style for the thesis-ready latent-space plot.",
    )
    parser.add_argument("--latent_point_edge_width", type=float, default=0.25)
    parser.add_argument(
        "--latent_color_limit_mode",
        choices=("fixed_cap", "percentile", "full_range"),
        default="percentile",
        help=(
            "Loss color-limit policy. The thesis-notebook default is the 1st "
            "to 95th percentile range."
        ),
    )
    parser.add_argument(
        "--latent_loss_cap",
        type=float,
        default=4.0,
        help="Upper color limit used when --latent_color_limit_mode=fixed_cap.",
    )
    parser.add_argument("--latent_low_percentile", type=float, default=1.0)
    parser.add_argument("--latent_high_percentile", type=float, default=95.0)
    parser.add_argument(
        "--latent_explicit_color_clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip loss values explicitly to the selected color range.",
    )
    parser.add_argument(
        "--latent_colorbar_tick_mode",
        choices=("auto", "integer", "half"),
        default="auto",
        help="Colorbar ticks as chosen automatically, at integers, or every 0.5.",
    )
    args = parser.parse_args()
    
    OUT_DIR = Path(args.out_dir)
    PLOT_DIR = Path(args.plot_dir) if args.plot_dir else OUT_DIR
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    BASELINE_DIR = Path(args.baseline_dir) if args.baseline_dir else None
    BASELINE_MODEL_PATH = BASELINE_DIR / "best_multimodal_vae.pt" if BASELINE_DIR else None
    
    HISTORY_PATH = OUT_DIR / "training_history.json"
    MODEL_PATH = OUT_DIR / "best_multimodal_vae.pt"
    
    PARQUET_PATH = Path(args.parquet_path)
    NPY_PATH = Path(args.matrix_path)
    VOCAB_PATH = Path(args.vocab_path)
    INDEX_PATH = Path(args.index_path) if args.index_path else None
    SPLIT_PATH = OUT_DIR / "dataset_split.parquet"
    
    # Extract clean representation string for titles
    REPR_STR = args.repr_type.upper()
    
    if HISTORY_PATH.exists():
        plot_training_history(HISTORY_PATH, PLOT_DIR, latent_dim=args.latent_dim, repr_type=REPR_STR)
    else:
        print(f"[!] History file missing: {HISTORY_PATH}")

    if MODEL_PATH.exists() and PARQUET_PATH.exists():
        plot_latent_space(
            MODEL_PATH,
            PARQUET_PATH,
            NPY_PATH,
            VOCAB_PATH,
            PLOT_DIR,
            latent_dim=args.latent_dim,
            repr_type=REPR_STR,
            index_path=INDEX_PATH,
            split_path=SPLIT_PATH,
            random_seed=args.random_seed,
            val_split=args.val_split,
            sens_weight=args.sens_weight,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            cluster_space=args.cluster_space,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            skip_clustering=args.skip_clustering,
            latent_point_size=args.latent_point_size,
            latent_point_alpha=args.latent_point_alpha,
            latent_point_edge_mode=args.latent_point_edge_mode,
            latent_point_edge_width=args.latent_point_edge_width,
            latent_color_limit_mode=args.latent_color_limit_mode,
            latent_loss_cap=args.latent_loss_cap,
            latent_low_percentile=args.latent_low_percentile,
            latent_high_percentile=args.latent_high_percentile,
            latent_explicit_color_clipping=args.latent_explicit_color_clipping,
            latent_colorbar_tick_mode=args.latent_colorbar_tick_mode,
        )
        
        compare_posterior_convergence(
            current_model_path=MODEL_PATH, 
            baseline_model_path=BASELINE_MODEL_PATH, 
            parquet_path=PARQUET_PATH, 
            npy_path=NPY_PATH, 
            vocab_path=VOCAB_PATH, 
            output_dir=PLOT_DIR, 
            latent_dim=args.latent_dim,
            repr_type=REPR_STR,
            index_path=INDEX_PATH,
        )
    else:
        print("[!] Execution halted: Core weights or structural assets missing.")

    # Always attempt to generate the Pareto summary at the parent directory level
    PARENT_DIR = OUT_DIR.parent
    if PARENT_DIR.exists():
        # Safely extract the job ID from the current output directory name
        # Fallback to wildcard '*' if the naming convention is missing
        current_job_id = OUT_DIR.name.split('_job')[-1] if '_job' in OUT_DIR.name else "*"
        
        plot_repr_sweep_summary(PARENT_DIR, latent_dim=args.latent_dim, job_id=current_job_id)
        plot_cross_repr_loss_trajectories(PARENT_DIR, latent_dim=args.latent_dim, job_id=current_job_id)
