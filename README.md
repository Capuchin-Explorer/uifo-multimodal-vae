# Structured Multimodal Representation Learning for UIFO Configurations

**Author:** Raphael Jontofsohn

This repository contains the final structured multimodal variational autoencoder developed for a bachelor’s thesis on recurring structure in optimized quasi-universal interferometer (UIFO) configurations. The pipeline combines three sources of information:

- a discrete and continuous detector configuration,
- a 50-point broadband strain-sensitivity curve, and
- one of three alternative configuration representations: **Flat**, **Grid**, or **Aliased**.

The main methodological contribution is a shared topology-aware tokenization. Representation-specific feature vectors are mapped to a common sequence of node, edge, and global tokens, processed by a transformer encoder, fused with a sensitivity embedding, and compressed into a 32-dimensional variational latent space.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── representations/
├── scripts/
│   ├── build_representations.py
│   └── train_VAE.sh
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── analysis/
│   └── plot_flat_linear_vs_scaled_bce.py
├── docs/
│   └── POST_CLUSTER_EXPORTS.md
├── results/
└── logs/
```

IDE metadata, Python bytecode, checkpoints, logs, and generated result folders are intentionally excluded from version control.

## Data artifacts

To train the models, place the released files in `data/representations/` using the following names:

```text
uifo_metadata.parquet
uifo_flat_matrix.npy
uifo_flat_vocab.json
uifo_flat_index.parquet
uifo_grid_matrix.npy
uifo_grid_vocab.json
uifo_grid_index.parquet
uifo_aliased_matrix.npy
uifo_aliased_vocab.json
uifo_aliased_index.parquet
```

The matrix, vocabulary, and index of each representation form one inseparable data artifact. The index preserves the row-wise mapping to `hash` and `run_id`; do not reorder any file independently.

Large `.npy` and `.parquet` files may exceed GitHub’s regular file limit. Store them with Git LFS or attach them to a GitHub Release. See `data/README.md` for the recommended procedure.

## Environment

Python 3.11 was used during development. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the CUDA version available on the target cluster.

## Representation construction

`scripts/build_representations.py` documents the extraction workflow for the Flat, Grid, and Aliased representations. The public version intentionally disables two thesis-specific selection steps:

1. filtering through a private fractional-run blacklist;
2. retaining only the five best runs per exact topology.

The script therefore demonstrates representation construction without depending on private run-selection metadata. It expects the original HDF5 simulation files, aligned lightweight/heavyweight Parquet tables, and the three extractor modules named in the script header. The precomputed representation artifacts are supplied separately so that model training does not require rerunning this extraction stage.

Example:

```bash
python scripts/build_representations.py --representation all
```

## Local training

The training entry point accepts one representation at a time:

```bash
python src/train.py \
  --matrix_path data/representations/uifo_aliased_matrix.npy \
  --vocab_path data/representations/uifo_aliased_vocab.json \
  --index_path data/representations/uifo_aliased_index.parquet \
  --parquet_path data/representations/uifo_metadata.parquet \
  --latent_dim 32 \
  --max_beta 0.001 \
  --sens_weight 100 \
  --out_dir results/aliased_run
```

## SLURM training

Adapt the resource directives in `scripts/train_VAE.sh` to the target cluster. The script can train all three representations sequentially or a selected subset:

```bash
sbatch scripts/train_VAE.sh
sbatch scripts/train_VAE.sh --vectors aliased
sbatch scripts/train_VAE.sh --vectors aliased,grid,flat
```

The following environment variables can be used instead of editing project paths in the script:

```bash
export PROJECT_ROOT=/path/to/uifo-multimodal-vae
export CONDA_ENV=uifo_env
```

## Post-training analysis

`src/visualize.py` reconstructs the validation data, exports posterior means and log-variances, creates UMAP projections, and optionally runs HDBSCAN in the full latent space. UMAP is used for visualization; it is not the default clustering space.

The exported manifests, stable train/validation assignments, per-run latent coordinates, and clustering metadata are described in `docs/POST_CLUSTER_EXPORTS.md`.

## Scope and interpretation

The repository is intended to document and reproduce the final machine-learning pipeline. The learned clusters are exploratory structures and should not automatically be interpreted as distinct physical mechanisms. Their meaning depends on the available data, representation, architecture, regularization, and post-hoc analysis.

## Citation

When using this code or the released representations, please cite the accompanying bachelor’s thesis and credit Raphael Jontofsohn as the author of the implementation.