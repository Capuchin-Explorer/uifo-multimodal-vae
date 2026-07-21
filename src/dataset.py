"""
Author: Raphael Jontofsohn

Data loading and preprocessing utilities for the structured multimodal UIFO VAE.
The module aligns representation matrices with sensitivity curves, applies the
training-time scaling, creates deterministic train/validation splits, and returns
PyTorch DataLoaders.
"""
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. CUSTOM PYTORCH DATASET
# ==============================================================================
class MultimodalSensTopoDataset(Dataset):
    """
    PyTorch dataset wrapper for paired sensitivity and configuration data.
    Returns a tuple of (Sensitivity_Curve, Parameter_Vector) for each index.
    """

    def __init__(self, sens_matrix, param_matrix):
        # Ensure identical lengths to prevent silent data misalignment
        assert len(sens_matrix) == len(param_matrix), "Mismatch between sensitivity and parameter counts!"

        # Convert NumPy arrays to PyTorch Tensors
        self.sens_data = torch.tensor(sens_matrix, dtype=torch.float32)
        self.param_data = torch.tensor(param_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.sens_data)

    def __getitem__(self, idx):
        return self.sens_data[idx], self.param_data[idx]


# ==============================================================================
# 2. MAIN DATA PIPELINE FUNCTION
# ==============================================================================
def get_dataloaders(
        matrix_path: str,
        vocab_path: str,
        index_path: str | None = None,
        parquet_path: str = "data/master_small_filter.parquet",
        batch_size: int = 256,
        val_split: float = 0.15,
        random_seed: int = 42,
        num_workers: int = 2,
        return_split_assignments: bool = False,
):
    """
    Orchestrates the data loading, scaling, splitting, and DataLoader creation.
    Dynamically adapts to the dimensions of the provided matrix_path.

    Args:
        matrix_path (str): Path to the .npy feature matrix (Flat, Grid, or Aliased).
        vocab_path (str): Path to the corresponding .json vocabulary file.
        index_path (str | None): Optional table with hash/run_id rows matching the matrix order.
        parquet_path (str): Path to the master parquet file containing sensitivities.
        batch_size (int): Training batch size.
        val_split (float): Fraction of data to reserve for validation.
        random_seed (int): Global seed for deterministic splitting.
        num_workers (int): Number of CPU threads for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        global_vocab (list): The list of feature names (needed to initialize the model).
        split_assignments (DataFrame, optional): Stable matrix-row and
            train/validation membership information for post-training analysis.
    """
    print("=" * 80)
    print(" INITIALIZING MULTIMODAL DATA PIPELINE")
    print("=" * 80)

    parquet_file = Path(parquet_path)
    matrix_file = Path(matrix_path)
    vocab_file = Path(vocab_path)

    # Validate that all required files exist before loading into memory
    for file_obj in [parquet_file, matrix_file, vocab_file]:
        if not file_obj.exists():
            raise FileNotFoundError(f"[!] Critical Error: Required file not found at {file_obj.resolve()}")

    # --------------------------------------------------------------------------
    # Step 1: Load Global Vocabulary
    # --------------------------------------------------------------------------
    with open(vocab_file, "r") as f:
        global_vocab = json.load(f)
    print(f"[+] Loaded dynamic vocabulary: {len(global_vocab)} dimensions.")

    # --------------------------------------------------------------------------
    # Step 2: Load Topological Parameter Matrix [0, 1]
    # --------------------------------------------------------------------------
    X_params_raw = np.load(matrix_file)
    print(f"[+] Loaded parameter matrix. Shape: {X_params_raw.shape}")

    if X_params_raw.shape[1] != len(global_vocab):
        raise ValueError(
            f"Feature dimension mismatch! Matrix has {X_params_raw.shape[1]} columns, "
            f"but vocabulary has {len(global_vocab)} entries."
        )

    # --------------------------------------------------------------------------
    # Step 3: Load and Extract Sensitivity Curves
    # --------------------------------------------------------------------------
    print(f"[*] Reading Parquet file: {parquet_file.name}...")
    df = pd.read_parquet(parquet_file)

    if index_path is not None:
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"[!] Index file not found at {index_file.resolve()}")

        if index_file.suffix == ".parquet":
            df_index = pd.read_parquet(index_file)
        elif index_file.suffix == ".csv":
            df_index = pd.read_csv(index_file)
        elif index_file.suffix == ".json":
            df_index = pd.read_json(index_file)
        else:
            raise ValueError("index_path must be a .parquet, .csv, or .json file")

        required_cols = {"hash", "run_id"}
        if not required_cols.issubset(df_index.columns) or not required_cols.issubset(df.columns):
            raise ValueError("Both parquet data and index file must contain hash and run_id columns.")

        if len(df_index) != len(X_params_raw):
            raise ValueError(
                f"Index row count mismatch! Index has {len(df_index)} rows, "
                f"but matrix has {len(X_params_raw)} rows."
            )

        df_index = df_index[["hash", "run_id"]].copy()
        df_index["__matrix_order"] = np.arange(len(df_index))

        df = df.merge(df_index, on=["hash", "run_id"], how="inner")
        if len(df) != len(df_index):
            raise ValueError(
                f"Alignment failed: only {len(df)} of {len(df_index)} matrix rows matched parquet hash/run_id keys."
            )

        df = df.sort_values("__matrix_order").drop(columns=["__matrix_order"]).reset_index(drop=True)
        print(f"[+] Aligned parquet rows to matrix order via {index_file.name}.")
    elif {"hash", "run_id"}.issubset(df.columns):
        print("[!] No index_path provided. Assuming parquet row order already matches matrix row order.")

    # Extract only the columns that represent the sensitivity frequency bins
    sens_cols = [c for c in df.columns if c.startswith("sens_")]
    X_sens_raw = df[sens_cols].values

    print(f"[+] Extracted raw sensitivity curves. Shape: {X_sens_raw.shape}")

    # Sanity Check: Ensure rows match identically
    if len(X_sens_raw) != len(X_params_raw):
        raise ValueError(
            f"Row count mismatch! Parquet has {len(X_sens_raw)} rows, "
            f"but NPY Matrix has {len(X_params_raw)} rows. Data must be aligned."
        )

    # --------------------------------------------------------------------------
    # Step 4: Physics Scaling for Sensitivities
    # Formula: N(x) = (log10(x + eps) + b) / S
    # Constants: eps = 1e-35, b = 30, S = 30
    # --------------------------------------------------------------------------
    epsilon = 1e-35
    b = 30.0
    S = 30.0

    X_sens_scaled = (np.log10(X_sens_raw + epsilon) + b) / S
    print(f"[+] Applied Log-Scaling. Sensitivities bounded within approx [0, 1].")

    # --------------------------------------------------------------------------
    # Step 5: Train / Validation Split
    # --------------------------------------------------------------------------
    indices = np.arange(len(X_sens_scaled))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=random_seed)

    X_train_sens, X_val_sens = X_sens_scaled[train_idx], X_sens_scaled[val_idx]
    X_train_params, X_val_params = X_params_raw[train_idx], X_params_raw[val_idx]

    split_labels = np.full(len(df), "train", dtype=object)
    split_labels[val_idx] = "validation"
    identity_columns = [column for column in ("hash", "run_id") if column in df.columns]
    split_assignments = df.loc[:, identity_columns].copy()
    split_assignments.insert(0, "matrix_row", np.arange(len(df), dtype=np.int64))
    split_assignments["dataset_split"] = split_labels

    print(f"[*] Data Split ({100 - val_split * 100:.0f}/{val_split * 100:.0f}):")
    print(f"    Train samples: {len(X_train_sens)}")
    print(f"    Val samples  : {len(X_val_sens)}")

    # --------------------------------------------------------------------------
    # Step 6: Create PyTorch DataLoaders
    # --------------------------------------------------------------------------
    train_dataset = MultimodalSensTopoDataset(X_train_sens, X_train_params)
    val_dataset = MultimodalSensTopoDataset(X_val_sens, X_val_params)

    # drop_last=True in train_loader helps stabilize LayerNorm during training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[+] DataLoaders ready (num_workers={num_workers}). Batches per epoch: Train ({len(train_loader)}), Val ({len(val_loader)})")
    print("=" * 80)

    if return_split_assignments:
        return train_loader, val_loader, global_vocab, split_assignments
    return train_loader, val_loader, global_vocab


# Optional smoke test executed only when this module is run directly.
if __name__ == "__main__":
    # Provide fallback paths for local testing
    TEST_MATRIX = "data/master_flat_matrix.npy"
    TEST_VOCAB = "data/master_flat_vocab.json"
    
    if Path(TEST_MATRIX).exists() and Path(TEST_VOCAB).exists():
        t_loader, v_loader, vocab = get_dataloaders(matrix_path=TEST_MATRIX, vocab_path=TEST_VOCAB)

        # Check the shapes of the first batch
        sens_batch, param_batch = next(iter(t_loader))
        print(f"\nTest Batch - Sensitivities Shape : {sens_batch.shape}")
        print(f"Test Batch - Parameters Shape    : {param_batch.shape}")
    else:
        print("[!] Test data not found. Please run the extraction scripts first.")
