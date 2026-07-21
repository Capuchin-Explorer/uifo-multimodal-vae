# Reproducible post-cluster exports

## Why this change was necessary

The previous pipeline saved model weights, aggregate epoch histories, and PNG
figures. The 32-dimensional posterior representations and UMAP coordinates were
discarded after plotting. Consequently, a point in the figure could not be
traced back reliably to its `hash` and `run_id`, and the visible islands had no
persisted formal cluster assignment.

## Training outputs

`train.py` now requires `--index_path` and additionally saves:

- `dataset_split.parquet`: matrix row, `hash`, `run_id`, and deterministic
  train/validation membership;
- `training_vocabulary.json`: the exact ordered vocabulary used by the model;
- `best_multimodal_vae_checkpoint.pt`: state dict plus model dimensions and
  best-epoch metadata;
- `training_manifest.json`: arguments, dimensions, seeds, environment versions,
  resolved paths, SHA-256 input fingerprints, and output paths.

The legacy `best_multimodal_vae.pt` state dict and `training_history.json` remain
unchanged for compatibility.

## Visualization and cluster-analysis outputs

`visualize.py` still creates the original latent-space PNG. It now also saves:

- `latent_run_assignments_<D>D_<REP>.parquet`;
- `latent_arrays_<D>D_<REP>.npz`;
- `parameter_features_<REP>.parquet`;
- `analysis_vocabulary_<REP>.json`;
- `cluster_summary_<D>D_<REP>.parquet`;
- `umap_reducer_<D>D_<REP>.joblib`;
- `hdbscan_<SPACE>_<D>D_<REP>.joblib` when HDBSCAN is available;
- `latent_analysis_manifest_<D>D_<REP>.json`.

The per-run Parquet table contains all columns from the aligned source Parquet,
including sensitivity samples and physical metadata, plus:

- stable `matrix_row`, `run_key`, and `dataset_split` identifiers;
- every posterior mean `mu_*` and log-variance `logvar_*` coordinate;
- posterior-scale and latent-norm summaries;
- UMAP coordinates;
- HDBSCAN label, membership probability, and outlier score;
- active input-feature count;
- deterministic per-run sensitivity, parameter, and KL diagnostics.

The separate parameter-feature table preserves every model input coordinate
under its exact vocabulary name and aligns it through `matrix_row` and
`run_key`. This allows cluster-wise feature-distribution analysis without
depending on an external copy of the original NumPy matrix.

## Clustering policy

HDBSCAN is applied to the full posterior-mean space by default. UMAP remains a
visualization and is not the default clustering space. Use `--cluster_space
umap` only when the explicit research question concerns the visual projection.
All HDBSCAN settings are command-line arguments and are recorded in the analysis
manifest.

If neither the external `hdbscan` package nor `sklearn.cluster.HDBSCAN` is
available, the latent export still completes, labels are set to `-1`, and the
console reports that clustering was unavailable.

## Important interpretation

Cluster labels are algorithmic results, not physical explanations. The exported
identifiers and metadata make it possible to compare topology composition,
suffixes, simplification status, complexity, loss, posterior uncertainty,
reconstruction quality, and sensitivity profiles without inferring membership
from PNG coordinates.
