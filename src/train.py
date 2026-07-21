"""
Author: Raphael Jontofsohn

HPC-oriented training entry point for the structured multimodal UIFO VAE.
The script loads one precomputed representation, trains the model with KL warm-up,
early stopping, learning-rate scheduling, and reproducible split tracking, and
exports checkpoints, histories, and run metadata.
"""
import os
import time
import json
import random
import argparse
import hashlib
import platform
import shutil
import sys
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.utils as utils

# Import the project-specific architecture modules.
from dataset import get_dataloaders
from model import (
    MultimodalTopologyVAE, 
    compute_multimodal_vae_loss, 
    get_beta, 
    EarlyStopping
)


def describe_file(path_value: str | None) -> dict | None:
    """Return an immutable file fingerprint for reproducible downstream analysis."""
    if path_value is None:
        return None
    path = Path(path_value).resolve()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def set_global_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across all computing libraries 
    and hardware backends (CPU/GPU).
    
    Args:
        seed (int): The integer value to use for all random number generators.
    """
    # 1. Set Python built-in random seed (e.g., for standard libraries)
    random.seed(seed)
    
    # 2. Set environment variable for hash seed (ensures deterministic dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Set NumPy random seed (used for data processing/arrays)
    np.random.seed(seed)
    
    # 4. Set PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # 5. Set PyTorch seed for all available GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Essential for multi-GPU HPC environments
        
        # 6. Force cuDNN backend to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        
        # 7. Disable cuDNN benchmarking
        # (If True, cuDNN auto-tunes algorithms dynamically, introducing randomness)
        torch.backends.cudnn.benchmark = False

    print(f"\n[*] Global Seed set to {seed}. Training is strictly deterministic.")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training configuration."""
    parser = argparse.ArgumentParser(description="Train the Multimodal Topology VAE")
    
    # Dynamic Data Inputs for Representation Sweep
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the precomputed .npy feature matrix")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the JSON vocabulary file")
    parser.add_argument("--index_path", type=str, required=True, help="Required hash/run_id index file for reproducible row alignment")
    parser.add_argument("--parquet_path", type=str, default="data/master_small_filter.parquet", help="Aligned run metadata and sensitivity curves")
    
    # Standard Hyperparameters
    parser.add_argument("--epochs", type=int, default=2000, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimensionality of the latent bottleneck")
    parser.add_argument("--warmup_epochs", type=int, default=250, help="Epochs for KL-divergence beta warmup")
    parser.add_argument("--max_beta", type=float, default=0.2, help="Maximum Beta penalty for KL divergence")
    parser.add_argument("--patience", type=int, default=150, help="Early stopping patience (epochs)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--random_seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--sens_weight", type=float, default=100.0, help="Sensitivity reconstruction weight")
    parser.add_argument("--out_dir", type=str, default="results/vae_multimodal", help="Directory to save model weights and logs")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_seed)

    # 1. Setup Device & Output Directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_snapshot_dir = out_dir / "code_snapshot"
    code_snapshot_dir.mkdir(exist_ok=True)
    for source_name in ("train.py", "dataset.py", "model.py"):
        shutil.copy2(Path(__file__).resolve().parent / source_name, code_snapshot_dir / source_name)
    
    print("=" * 80)
    print(f" STARTING MULTIMODAL VAE TRAINING ON {device.type.upper()}")
    print("=" * 80)
    print(f"Vector Matrix : {args.matrix_path}")
    print(f"Vocabulary    : {args.vocab_path}")
    print("-" * 80)
    print(f"Hyperparameters: Latent Dim={args.latent_dim}, Max Beta={args.max_beta}, LR={args.lr}, Batch Size={args.batch_size}")
    print(f"Schedules: Warmup={args.warmup_epochs} Ep, Patience={args.patience} Ep")

    # 2. Load Data 
    # Note: Ensure dataset.py is updated to accept these new arguments
    train_loader, val_loader, global_vocab, split_assignments = get_dataloaders(
        matrix_path=args.matrix_path,
        vocab_path=args.vocab_path,
        index_path=args.index_path,
        parquet_path=args.parquet_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        return_split_assignments=True,
    )

    split_path = out_dir / "dataset_split.parquet"
    split_assignments.to_parquet(split_path, index=False)
    with open(out_dir / "training_vocabulary.json", "w") as handle:
        json.dump(global_vocab, handle, indent=2)
    print(f"-> Saved stable dataset split to: {split_path}")

    # 3. Initialize Model, Optimizer, Scheduler, and Early Stopping
    model = MultimodalTopologyVAE(
        global_vocab=global_vocab, 
        latent_dim=args.latent_dim,
        sens_weight=args.sens_weight,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning Rate Scheduler: Reduces the LR by a factor of 0.5 (halving) 
    # if the validation loss plateaus for 40 epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=25, 
        min_lr=1e-6
    )
    
    early_stopping = EarlyStopping(patience=args.patience)

    # 4. History Tracking for Visualization
    history = {
        'train_total': [], 'train_sens': [], 'train_params': [], 'train_kl': [],
        'val_total': [], 'val_sens': [], 'val_params': [], 'val_kl': [],
        'beta': [], 'learning_rate': []
    }

    # 5. Training Loop
    start_time = time.time()
    
    for epoch in range(args.epochs):
        current_beta = get_beta(epoch, warmup_epochs=args.warmup_epochs, max_beta=args.max_beta)
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- TRAINING PHASE ---
        model.train()
        t_loss, t_sens, t_params, t_kl = 0.0, 0.0, 0.0, 0.0
        
        for batch_sens, batch_params in train_loader:
            batch_sens, batch_params = batch_sens.to(device), batch_params.to(device)
            
            optimizer.zero_grad()
            
            recon_sens, recon_params, mu, logvar = model(batch_sens, batch_params)
            total_loss, loss_sens, loss_params, kl_loss = compute_multimodal_vae_loss(
                recon_sens,
                batch_sens,
                recon_params,
                batch_params,
                mu,
                logvar,
                beta=current_beta,
                sens_weight=args.sens_weight,
                param_masks={
                    "class": model.param_class_mask,
                    "value": model.param_value_mask,
                    "edge": model.param_edge_mask,
                },
            )
            
            total_loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            t_loss += total_loss.item()
            t_sens += loss_sens.item()
            t_params += loss_params.item()
            t_kl += kl_loss.item()

        n_train = len(train_loader)
        history['train_total'].append(t_loss / n_train)
        history['train_sens'].append(t_sens / n_train)
        history['train_params'].append(t_params / n_train)
        history['train_kl'].append(t_kl / n_train)
        history['beta'].append(current_beta)
        history['learning_rate'].append(current_lr)

        # --- VALIDATION PHASE ---
        model.eval()
        v_loss, v_sens, v_params, v_kl = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_sens, batch_params in val_loader:
                batch_sens, batch_params = batch_sens.to(device), batch_params.to(device)
                
                recon_sens, recon_params, mu, logvar = model(batch_sens, batch_params)
                total_loss, loss_sens, loss_params, kl_loss = compute_multimodal_vae_loss(
                    recon_sens,
                    batch_sens,
                    recon_params,
                    batch_params,
                    mu,
                    logvar,
                    beta=current_beta,
                    sens_weight=args.sens_weight,
                    param_masks={
                        "class": model.param_class_mask,
                        "value": model.param_value_mask,
                        "edge": model.param_edge_mask,
                    },
                )
                
                v_loss += total_loss.item()
                v_sens += loss_sens.item()
                v_params += loss_params.item()
                v_kl += kl_loss.item()
                
        n_val = len(val_loader)
        avg_val_loss = v_loss / n_val
        history['val_total'].append(avg_val_loss)
        history['val_sens'].append(v_sens / n_val)
        history['val_params'].append(v_params / n_val)
        history['val_kl'].append(v_kl / n_val)

        # Learning Rate Scheduler Step 
        # (Waits until the beta warmup phase is complete before evaluating plateaus)
        if epoch >= args.warmup_epochs:
            scheduler.step(avg_val_loss)

        # Logging output for HPC stdout
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:>4}/{args.epochs}] | LR: {current_lr:.2e} | Beta: {current_beta:.4f} | "
                  f"Train Loss: {t_loss / n_train:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"ES Patience: {early_stopping.counter}/{early_stopping.patience}")

        # --- EARLY STOPPING CHECK ---
        if early_stopping(avg_val_loss, model, epoch, current_beta, args.max_beta):
            print(f"\n[!] Early Stopping triggered at Epoch {epoch+1}.")
            break

    # 6. Post-Training & Export
    elapsed = (time.time() - start_time) / 60
    print("=" * 80)
    print(f" TRAINING COMPLETE IN {elapsed:.2f} MINUTES")
    print("=" * 80)

    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        print(f"-> Restored best weights from Epoch {early_stopping.best_epoch + 1} (Val Loss: {early_stopping.best_loss:.4f})")
    
    model_path = out_dir / "best_multimodal_vae.pt"
    torch.save(model.state_dict(), model_path)
    print(f"-> Saved model state dict to: {model_path}")

    checkpoint_path = out_dir / "best_multimodal_vae_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "latent_dim": args.latent_dim,
                "sens_dim": 50,
                "param_dim": len(global_vocab),
                "sens_weight": args.sens_weight,
            },
            "best_epoch_zero_based": early_stopping.best_epoch,
            "best_validation_loss": early_stopping.best_loss,
            "completed_epochs": len(history["train_total"]),
        },
        checkpoint_path,
    )
    print(f"-> Saved self-describing checkpoint to: {checkpoint_path}")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"-> Saved training history to: {history_path}")

    manifest = {
        "schema_version": 1,
        "command_line_arguments": vars(args),
        "model_config": {
            "architecture": "MultimodalTopologyVAE",
            "latent_dim": args.latent_dim,
            "sensitivity_dim": 50,
            "parameter_dim": len(global_vocab),
            "sensitivity_weight": args.sens_weight,
        },
        "training_result": {
            "completed_epochs": len(history["train_total"]),
            "best_epoch_zero_based": early_stopping.best_epoch,
            "best_validation_loss": early_stopping.best_loss,
            "elapsed_minutes": elapsed,
            "train_runs": int((split_assignments["dataset_split"] == "train").sum()),
            "validation_runs": int((split_assignments["dataset_split"] == "validation").sum()),
        },
        "input_files": {
            "matrix": describe_file(args.matrix_path),
            "vocabulary": describe_file(args.vocab_path),
            "index": describe_file(args.index_path),
            "parquet": describe_file(args.parquet_path),
        },
        "code_files": {
            name: describe_file(str(Path(__file__).resolve().parent / name))
            for name in ("train.py", "dataset.py", "model.py")
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
        "outputs": {
            "state_dict": str(model_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "history": str(history_path.resolve()),
            "dataset_split": str(split_path.resolve()),
            "vocabulary_snapshot": str((out_dir / "training_vocabulary.json").resolve()),
            "code_snapshot_directory": str(code_snapshot_dir.resolve()),
        },
    }
    manifest_path = out_dir / "training_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"-> Saved reproducibility manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
