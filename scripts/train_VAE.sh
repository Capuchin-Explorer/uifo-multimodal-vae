#!/usr/bin/env bash
# Author: Raphael Jontofsohn
#
# SLURM submission script for training and evaluating the structured multimodal
# UIFO VAE on the Flat, Grid, and Aliased representations. Site-specific paths
# are configurable through environment variables; model and training logic are
# delegated to the Python modules in src/.
#
# Example SLURM directives are provided below. Adapt partition, resources, and
# log paths to the target cluster before submission.
#SBATCH --job-name=uifo_vae_sweep
#SBATCH --output=logs/uifo_vae_sweep_%j.out
#SBATCH --error=logs/uifo_vae_sweep_%j.err
#SBATCH --time=0-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1


# Abort on command failures and failed pipelines.
# nounset (-u) is deliberately omitted because some Conda activation
# scripts are incompatible with it.
set -eo pipefail

# ==============================================================================
# 1. Project paths
#
# Set PROJECT_ROOT and CONDA_ENV before submission, or edit the defaults below.
# ==============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/src}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/representations}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-uifo_env}"

PARQUET_PATH="${PARQUET_PATH:-$DATA_DIR/uifo_metadata.parquet}"

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ==============================================================================
# 2. Training configuration
# ==============================================================================

LATENT_DIM=32
MAX_BETA=0.001
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
WARMUP_EPOCHS=250
PATIENCE=50

RANDOM_SEED=42
VAL_SPLIT=0.15
NUM_WORKERS=2
SENS_WEIGHT=100.0

# ==============================================================================
# 3. UMAP configuration
#
# UMAP is used only to visualize the latent-space representation.
# HDBSCAN is not fitted on these two-dimensional coordinates.
# ==============================================================================

UMAP_N_NEIGHBORS=15
UMAP_MIN_DIST=0.1

# ==============================================================================
# 4. HDBSCAN configuration
#
# HDBSCAN is fitted on the complete 32-dimensional latent mean vectors.
# ==============================================================================

CLUSTER_SPACE="latent"
MIN_CLUSTER_SIZE=50
MIN_SAMPLES=10
CLUSTER_SELECTION_EPSILON=0.0

# ==============================================================================
# 5. Thesis-ready latent-space plot configuration
#
# No loss cap is passed, so the complete finite loss range is shown.
# ==============================================================================

LATENT_POINT_SIZE=8
LATENT_POINT_ALPHA=0.60
LATENT_POINT_EDGE_MODE="none"
LATENT_POINT_EDGE_WIDTH=0.25

# ==============================================================================
# 6. Representations and optional command-line arguments
#
# Examples:
#   sbatch shell_script/train_VAE.sh
#   sbatch shell_script/train_VAE.sh --vectors flat
#   sbatch shell_script/train_VAE.sh --vectors aliased,grid
# ==============================================================================

VECTOR_TYPES=("aliased" "grid" "flat")
RESULT_JOB_ID="${SLURM_JOB_ID:-local}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vectors)
            if [[ $# -lt 2 ]]; then
                echo "[!] --vectors requires a comma-separated argument."
                exit 1
            fi
            IFS=',' read -r -a VECTOR_TYPES <<< "$2"
            shift 2
            ;;
        --result-job-id)
            if [[ $# -lt 2 ]]; then
                echo "[!] --result-job-id requires an argument."
                exit 1
            fi
            RESULT_JOB_ID="$2"
            shift 2
            ;;
        --use-real-job-id)
            RESULT_JOB_ID="${SLURM_JOB_ID:-local}"
            shift
            ;;
        *)
            echo "[!] Unknown argument: $1"
            echo "Usage:"
            echo "  sbatch shell_script/train_VAE.sh \\"
            echo "    [--vectors flat|grid|aliased|aliased,grid,flat] \\"
            echo "    [--result-job-id ID] [--use-real-job-id]"
            exit 1
            ;;
    esac
done

# Reject unknown representation names.
for VECTOR in "${VECTOR_TYPES[@]}"; do
    case "$VECTOR" in
        flat|grid|aliased)
            ;;
        *)
            echo "[!] Unknown representation: $VECTOR"
            exit 1
            ;;
    esac
done

echo "================================================================="
echo " Multimodal VAE representation sweep"
echo "================================================================="
echo "SLURM job ID    : ${SLURM_JOB_ID:-local}"
echo "Result job ID   : $RESULT_JOB_ID"
echo "Compute node    : ${SLURMD_NODENAME:-local}"
echo "Project root    : $PROJECT_ROOT"
echo "Representations : ${VECTOR_TYPES[*]}"
echo "Latent dimension: $LATENT_DIM"
echo "Maximum beta    : $MAX_BETA"
echo "================================================================="

TIMESTAMP_START_TOTAL=$(date +%s)

# ==============================================================================
# 7. Environment initialization
# ==============================================================================

echo -e "\n[*] Activating Conda environment..."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HWLOC_HIDE_ERRORS=1
export PYTHONPATH="$PROJECT_ROOT:$CODE_DIR:${PYTHONPATH:-}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "[*] Python executable: $(which python)"
python --version

# Verify that the code and shared metadata exist.
for REQUIRED_FILE in \
    "$CODE_DIR/train.py" \
    "$CODE_DIR/visualize.py" \
    "$CODE_DIR/dataset.py" \
    "$CODE_DIR/model.py" \
    "$PARQUET_PATH"
do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "[!] Required file is missing: $REQUIRED_FILE"
        exit 1
    fi
done

# Verify packages and GPU access before starting a long training run.
python - <<'PY'
import importlib.metadata

import hdbscan
import joblib
import matplotlib
import numpy
import pandas
import pyarrow
import sklearn
import torch
import umap

print("[OK] Required Python packages imported.")
print("[OK] PyTorch:", torch.__version__)
print("[OK] scikit-learn:", sklearn.__version__)
print("[OK] UMAP:", importlib.metadata.version("umap-learn"))
print("[OK] HDBSCAN:", importlib.metadata.version("hdbscan"))
print("[OK] HDBSCAN path:", hdbscan.__file__)
print("[OK] PyArrow:", pyarrow.__version__)
print("[OK] CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable although an A100 GPU was requested.")

print("[OK] CUDA device:", torch.cuda.get_device_name(0))
PY

# ==============================================================================
# 8. Sequential representation runs
# ==============================================================================

for VECTOR in "${VECTOR_TYPES[@]}"; do
    echo -e "\n================================================================="
    echo " Starting representation: ${VECTOR^^}"
    echo "================================================================="

    TIMESTAMP_START_VECTOR=$(date +%s)

    MATRIX_PATH="$DATA_DIR/uifo_${VECTOR}_matrix.npy"
    VOCAB_PATH="$DATA_DIR/uifo_${VECTOR}_vocab.json"
    INDEX_PATH="$DATA_DIR/uifo_${VECTOR}_index.parquet"

    OUT_DIR="$RESULTS_DIR/vae_dim_${LATENT_DIM}_beta_${MAX_BETA}_repr_${VECTOR}_job${RESULT_JOB_ID}"

    # Matrix, vocabulary, index, and metadata are jointly required.
    for REQUIRED_FILE in \
        "$MATRIX_PATH" \
        "$VOCAB_PATH" \
        "$INDEX_PATH" \
        "$PARQUET_PATH"
    do
        if [[ ! -f "$REQUIRED_FILE" ]]; then
            echo "[!] Required input is missing: $REQUIRED_FILE"
            exit 1
        fi
    done

    # Avoid silently overwriting an earlier run.
    if [[ -e "$OUT_DIR" ]]; then
        echo "[!] Output directory already exists:"
        echo "    $OUT_DIR"
        echo "[!] Submit a new job or specify a new --result-job-id."
        exit 1
    fi

    mkdir -p "$OUT_DIR"

    if [[ ! -w "$OUT_DIR" ]]; then
        echo "[!] Output directory is not writable: $OUT_DIR"
        exit 1
    fi

    echo "Matrix     : $MATRIX_PATH"
    echo "Vocabulary : $VOCAB_PATH"
    echo "Index      : $INDEX_PATH"
    echo "Metadata   : $PARQUET_PATH"
    echo "Output     : $OUT_DIR"

    # --------------------------------------------------------------------------
    # 8.1 Train model
    # --------------------------------------------------------------------------

    echo -e "\n[*] Training ${VECTOR^^} model..."
    TIMESTAMP_START_TRAIN=$(date +%s)

    srun python "$CODE_DIR/train.py" \
        --matrix_path "$MATRIX_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --index_path "$INDEX_PATH" \
        --parquet_path "$PARQUET_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_dim "$LATENT_DIM" \
        --max_beta "$MAX_BETA" \
        --lr "$LEARNING_RATE" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --patience "$PATIENCE" \
        --val_split "$VAL_SPLIT" \
        --random_seed "$RANDOM_SEED" \
        --num_workers "$NUM_WORKERS" \
        --sens_weight "$SENS_WEIGHT" \
        --out_dir "$OUT_DIR"

    TIMESTAMP_END_TRAIN=$(date +%s)
    TRAIN_MINUTES=$(( (TIMESTAMP_END_TRAIN - TIMESTAMP_START_TRAIN) / 60 ))

    echo "[OK] Training completed in ${TRAIN_MINUTES} minutes."

    # Verify the central training outputs before visualization.
    for TRAINING_OUTPUT in \
        "$OUT_DIR/best_multimodal_vae.pt" \
        "$OUT_DIR/training_history.json" \
        "$OUT_DIR/dataset_split.parquet"
    do
        if [[ ! -f "$TRAINING_OUTPUT" ]]; then
            echo "[!] Expected training output is missing: $TRAINING_OUTPUT"
            exit 1
        fi
    done

    # --------------------------------------------------------------------------
    # 8.2 Post-training analysis
    # --------------------------------------------------------------------------

    echo -e "\n[*] Analyzing ${VECTOR^^} model..."
    TIMESTAMP_START_ANALYSIS=$(date +%s)

    srun python "$CODE_DIR/visualize.py" \
        --matrix_path "$MATRIX_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --index_path "$INDEX_PATH" \
        --parquet_path "$PARQUET_PATH" \
        --out_dir "$OUT_DIR" \
        --latent_dim "$LATENT_DIM" \
        --repr_type "${VECTOR^^}" \
        --random_seed "$RANDOM_SEED" \
        --val_split "$VAL_SPLIT" \
        --sens_weight "$SENS_WEIGHT" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --cluster_space "$CLUSTER_SPACE" \
        --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --min_samples "$MIN_SAMPLES" \
        --cluster_selection_epsilon "$CLUSTER_SELECTION_EPSILON" \
        --latent_point_size "$LATENT_POINT_SIZE" \
        --latent_point_alpha "$LATENT_POINT_ALPHA" \
        --latent_point_edge_mode "$LATENT_POINT_EDGE_MODE" \
        --latent_point_edge_width "$LATENT_POINT_EDGE_WIDTH"

    TIMESTAMP_END_ANALYSIS=$(date +%s)
    ANALYSIS_MINUTES=$(( (TIMESTAMP_END_ANALYSIS - TIMESTAMP_START_ANALYSIS) / 60 ))

    echo "[OK] Analysis completed in ${ANALYSIS_MINUTES} minutes."

    # Verify central post-analysis exports.
    for ANALYSIS_OUTPUT in \
        "$OUT_DIR/latent_arrays_${LATENT_DIM}D_${VECTOR^^}.npz" \
        "$OUT_DIR/latent_run_assignments_${LATENT_DIM}D_${VECTOR^^}.parquet" \
        "$OUT_DIR/latent_analysis_manifest_${LATENT_DIM}D_${VECTOR^^}.json" \
        "$OUT_DIR/02_vae_latent_space_${LATENT_DIM}D_${VECTOR^^}_thesis.pdf"
    do
        if [[ ! -f "$ANALYSIS_OUTPUT" ]]; then
            echo "[!] Expected analysis output is missing: $ANALYSIS_OUTPUT"
            exit 1
        fi
    done

    TIMESTAMP_END_VECTOR=$(date +%s)
    VECTOR_MINUTES=$(( (TIMESTAMP_END_VECTOR - TIMESTAMP_START_VECTOR) / 60 ))

    echo "================================================================="
    echo " ${VECTOR^^} completed successfully in ${VECTOR_MINUTES} minutes."
    echo " Results: $OUT_DIR"
    echo "================================================================="
done

# ==============================================================================
# 9. Completion
# ==============================================================================

conda deactivate

TIMESTAMP_END_TOTAL=$(date +%s)
TOTAL_MINUTES=$(( (TIMESTAMP_END_TOTAL - TIMESTAMP_START_TOTAL) / 60 ))

echo -e "\n================================================================="
echo " Representation sweep completed successfully."
echo " Representations : ${VECTOR_TYPES[*]}"
echo " Result job ID   : $RESULT_JOB_ID"
echo " Total runtime   : ${TOTAL_MINUTES} minutes"
echo " Results         : $RESULTS_DIR"
echo " Logs            : $LOG_DIR"
echo "================================================================="
