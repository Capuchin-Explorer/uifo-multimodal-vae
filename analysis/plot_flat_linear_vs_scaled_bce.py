"""
Author: Raphael Jontofsohn

Post-hoc comparison of the original scaled Flat representation and a linear Flat
encoding. The script compares validation trajectories and reconstruction losses
for the parameter types whose preprocessing differs between both variants.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from model import MultimodalTopologyVAE


PARAM_TYPES = ["angle", "tuning", "db", "power", "mass", "length"]


def load_history(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def load_vocab(path: Path) -> list[str]:
    with path.open("r") as handle:
        return json.load(handle)


def load_aligned_arrays(
    matrix_path: Path,
    index_path: Path,
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    params = np.load(matrix_path).astype(np.float32)
    df_sens = pd.read_parquet(parquet_path)

    if index_path.exists():
        df_index = pd.read_parquet(index_path)
        alignment = df_index[["hash", "run_id"]].copy()
        alignment["matrix_row"] = np.arange(len(alignment))
        df_sens = alignment.merge(df_sens, on=["hash", "run_id"], how="inner", validate="one_to_one")
        if len(df_sens) != len(alignment):
            raise RuntimeError(
                f"Index alignment kept {len(df_sens)} rows, expected {len(alignment)}."
            )
        params = params[df_sens["matrix_row"].to_numpy()]
    else:
        df_sens = df_sens.iloc[: len(params)].copy()

    sens_cols = [col for col in df_sens.columns if col.startswith("sens_")]
    if not sens_cols:
        raise RuntimeError("No sensitivity columns with prefix 'sens_' found.")

    sens_raw = df_sens[sens_cols].to_numpy(dtype=np.float32)
    sens_scaled = (np.log10(sens_raw + 1e-35) + 30.0) / 30.0
    return sens_scaled.astype(np.float32), params


def parameter_columns(vocab: list[str], prop_type: str) -> list[int]:
    cols = []
    for idx, feature in enumerate(vocab):
        if not feature.startswith("PROP_"):
            continue

        if prop_type in {"angle", "tuning"}:
            if (
                feature.endswith(f"_{prop_type}")
                or feature.endswith(f"_{prop_type}_sin")
                or feature.endswith(f"_{prop_type}_cos")
            ):
                cols.append(idx)
        elif feature.endswith(f"_{prop_type}"):
            cols.append(idx)

    return cols


def bce_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = np.clip(pred.astype(np.float64), eps, 1.0 - eps)
    target = np.clip(target.astype(np.float64), 0.0, 1.0)
    loss = -(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))
    return float(np.mean(loss))


def reconstruct_validation_params(
    model_path: Path,
    vocab: list[str],
    sens_val: np.ndarray,
    params_val: np.ndarray,
    latent_dim: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = MultimodalTopologyVAE(global_vocab=vocab, latent_dim=latent_dim, sens_weight=100.0).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    recon_batches = []
    with torch.no_grad():
        for start in range(0, len(params_val), batch_size):
            end = start + batch_size
            batch_sens = torch.from_numpy(sens_val[start:end]).to(device)
            batch_params = torch.from_numpy(params_val[start:end]).to(device)
            _, recon_params, _, _ = model(batch_sens, batch_params)
            recon_batches.append(recon_params.detach().cpu().numpy())

    return np.vstack(recon_batches)


def evaluate_final_bce_by_type(
    model_path: Path,
    matrix_path: Path,
    vocab_path: Path,
    index_path: Path,
    parquet_path: Path,
    latent_dim: int,
    batch_size: int,
    val_split: float,
    random_seed: int,
    device: torch.device,
) -> dict[str, float]:
    vocab = load_vocab(vocab_path)
    sens, params = load_aligned_arrays(matrix_path, index_path, parquet_path)

    indices = np.arange(len(params))
    _, val_idx = train_test_split(indices, test_size=val_split, random_state=random_seed)
    sens_val = sens[val_idx]
    params_val = params[val_idx]

    recon_val = reconstruct_validation_params(
        model_path=model_path,
        vocab=vocab,
        sens_val=sens_val,
        params_val=params_val,
        latent_dim=latent_dim,
        batch_size=batch_size,
        device=device,
    )

    out = {}
    for prop_type in PARAM_TYPES:
        cols = parameter_columns(vocab, prop_type)
        if not cols:
            out[prop_type] = np.nan
            continue
        out[prop_type] = bce_loss(recon_val[:, cols], params_val[:, cols])
    return out


def plot_comparison(args: argparse.Namespace) -> None:
    scaled_history = load_history(args.scaled_history)
    linear_history = load_history(args.linear_history)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Evaluating final saved weights on {device.type.upper()}.")

    scaled_bce = evaluate_final_bce_by_type(
        model_path=args.scaled_model,
        matrix_path=args.scaled_matrix,
        vocab_path=args.scaled_vocab,
        index_path=args.scaled_index,
        parquet_path=args.parquet_path,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed,
        device=device,
    )
    linear_bce = evaluate_final_bce_by_type(
        model_path=args.linear_model,
        matrix_path=args.linear_matrix,
        vocab_path=args.linear_vocab,
        index_path=args.linear_index,
        parquet_path=args.parquet_path,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed,
        device=device,
    )

    start_epoch = args.start_epoch
    scaled_val = scaled_history["val_total"][start_epoch:]
    linear_val = linear_history["val_total"][start_epoch:]
    scaled_epochs = np.arange(start_epoch, start_epoch + len(scaled_val))
    linear_epochs = np.arange(start_epoch, start_epoch + len(linear_val))

    fig, (ax_loss, ax_bce) = plt.subplots(1, 2, figsize=(15, 5.7))

    ax_loss.plot(
        scaled_epochs,
        scaled_val,
        color="#d95f02",
        linewidth=1.7,
        label=f"Scaled Flat validation loss ({args.scaled_label})",
    )
    ax_loss.plot(
        linear_epochs,
        linear_val,
        color="#fdb863",
        linewidth=1.7,
        label=f"Linear Flat validation loss ({args.linear_label})",
    )
    ax_loss.set_title("Validation Loss Trajectories")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation total loss")
    ax_loss.grid(True, linestyle="--", alpha=0.25)
    ax_loss.legend(frameon=True)

    x = np.arange(len(PARAM_TYPES))
    width = 0.38
    scaled_values = [scaled_bce[prop] for prop in PARAM_TYPES]
    linear_values = [linear_bce[prop] for prop in PARAM_TYPES]

    ax_bce.bar(
        x - width / 2,
        scaled_values,
        width=width,
        color="#d95f02",
        label=f"Scaled Flat final BCE ({args.scaled_label})",
    )
    ax_bce.bar(
        x + width / 2,
        linear_values,
        width=width,
        color="#fdb863",
        label=f"Linear Flat final BCE ({args.linear_label})",
    )
    ax_bce.set_title("Final Validation BCE on Changed Parameter Types")
    ax_bce.set_xlabel("Parameter type")
    ax_bce.set_ylabel("BCE of saved best weights")
    ax_bce.set_xticks(x)
    ax_bce.set_xticklabels(PARAM_TYPES, rotation=30, ha="right")
    ax_bce.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_bce.legend(frameon=True)

    for offset, values in [(-width / 2, scaled_values), (width / 2, linear_values)]:
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax_bce.text(idx + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Flat Representation: Scaled Encoding vs Linear Encoding", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved comparison plot:", args.output)
    print("[OK] Final validation BCE by parameter type:")
    for prop in PARAM_TYPES:
        print(f"  {prop:<8} scaled={scaled_bce[prop]:.6f}  linear={linear_bce[prop]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final Flat scaled-vs-linear parameter BCE.")

    parser.add_argument("--scaled_history", type=Path, default=Path("results/vae_dim_32_beta_0.001_repr_flat_job2647847/training_history.json"))
    parser.add_argument("--linear_history", type=Path, default=Path("results/vae_dim_32_beta_0.001_repr_flat_job2651217/training_history.json"))
    parser.add_argument("--scaled_model", type=Path, default=Path("results/vae_dim_32_beta_0.001_repr_flat_job2647847/best_multimodal_vae.pt"))
    parser.add_argument("--linear_model", type=Path, default=Path("results/vae_dim_32_beta_0.001_repr_flat_job2651217/best_multimodal_vae.pt"))

    parser.add_argument("--scaled_matrix", type=Path, default=Path("data/master_flat_matrix.npy"))
    parser.add_argument("--scaled_vocab", type=Path, default=Path("data/master_flat_vocab.json"))
    parser.add_argument("--scaled_index", type=Path, default=Path("data/master_flat_index.parquet"))
    parser.add_argument("--linear_matrix", type=Path, default=Path("data/master_flat_linear_matrix.npy"))
    parser.add_argument("--linear_vocab", type=Path, default=Path("data/master_flat_linear_vocab.json"))
    parser.add_argument("--linear_index", type=Path, default=Path("data/master_flat_linear_index.parquet"))
    parser.add_argument("--parquet_path", type=Path, default=Path("data/master_small_filter.parquet"))

    parser.add_argument("--output", type=Path, default=Path("results/flat_scaled_vs_linear_final_bce_job2647847_vs_job2651217.png"))
    parser.add_argument("--start_epoch", type=int, default=45)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--scaled_label", type=str, default="job2647847")
    parser.add_argument("--linear_label", type=str, default="job2651217")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    plot_comparison(parse_args())
