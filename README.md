# Structured Multimodal Representation Learning for UIFO Configurations

**Author:** Raphael Jontofsohn

This repository contains the structured multimodal variational autoencoder developed for a bachelor's thesis on recurring structure in optimized quasi-universal interferometer (UIFO) configurations.

The model jointly processes:

- 50-point strain-sensitivity curves;
- discrete interferometer topologies; and
- continuous component parameters.

Flat, Grid, and Aliased configuration vectors are converted into topology-aware token sequences, processed by a transformer encoder, fused with the sensitivity representation, and compressed into a 32-dimensional variational latent space.

## Installation

The released datasets are included through Git LFS. Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/Capuchin-Explorer/uifo-multimodal-vae.git
cd uifo-multimodal-vae
```

If the repository was cloned before Git LFS was installed, retrieve the data with:

```bash
git lfs pull
```

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

Python 3.11 was used during development. For GPU training, install the PyTorch build appropriate for the available CUDA version.

## Included data

All files required for training are available in `data/representations/`:

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

No raw simulation files, separate dataset download, or representation-building step is required. See [`data/README.md`](data/README.md) for details.

## Training

Train one representation by passing its matrix, vocabulary, and index to `src/train.py`:

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

Replace `aliased` with `grid` or `flat` to train another representation.

For SLURM training:

```bash
sbatch scripts/train_VAE.sh
sbatch scripts/train_VAE.sh --vectors aliased
sbatch scripts/train_VAE.sh --vectors aliased,grid,flat
```

## Analysis

`src/visualize.py` reconstructs validation samples, exports latent representations, creates two-dimensional UMAP projections, and optionally performs HDBSCAN clustering in the full latent space.

UMAP is used for visualization and is not the default clustering space. The post-training exports are documented in [`docs/POST_CLUSTER_EXPORTS.md`](docs/POST_CLUSTER_EXPORTS.md).

## Optional representation construction

The released representations are ready for training and do not need to be regenerated.

`scripts/build_representations.py` documents the original extraction workflow for methodological reference. Running it requires the original simulation data and additional extractor modules that are not needed when using the released artifacts.

## Interpretation

Latent-space clusters are exploratory structures and should not automatically be interpreted as distinct physical mechanisms. Their interpretation depends on the dataset, representation, architecture, regularization, and post-hoc analysis.

## Citation

When using this code or the released datasets, please cite the accompanying bachelor's thesis and credit Raphael Jontofsohn as the author of the implementation.
