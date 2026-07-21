# Data directory

The repository expects the precomputed Flat, Grid, and Aliased representation artifacts in `data/representations/`. Raw simulation files are not required for training when these artifacts are available.

## Required files

For each representation, keep the matrix, vocabulary, and index together:

- `uifo_<representation>_matrix.npy` — encoded feature matrix;
- `uifo_<representation>_vocab.json` — ordered feature vocabulary;
- `uifo_<representation>_index.parquet` — row alignment by `hash` and `run_id`;
- `uifo_metadata.parquet` — aligned run metadata and sensitivity curves.

## Recommended release strategy

Use regular Git tracking for JSON files and small documentation. Use **Git LFS** for large NumPy and Parquet files:

```bash
git lfs install
git lfs track "data/representations/*.npy"
git lfs track "data/representations/*.parquet"
git add .gitattributes data/representations/
git commit -m "Add released UIFO representations"
```

Alternatively, publish the large files as a versioned GitHub Release and include checksums in the release notes. Avoid committing private raw simulation files, blacklists, cluster outputs containing unpublished metadata, or absolute HPC paths.
