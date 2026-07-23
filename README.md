# Structured Multimodal Representation Learning for UIFO Configurations

**Author:** Raphael Jontofsohn

This repository contains the final structured multimodal variational autoencoder developed for a bachelor's thesis on recurring structure in optimized quasi-universal interferometer (UIFO) configurations.

The model combines three sources of information:

- a 50-point broadband strain-sensitivity curve;
- discrete interferometer topology; and
- continuous component parameters.

Its main methodological contribution is a shared topology-aware tokenization. Representation-specific feature vectors are converted into a common sequence of node, edge, and global tokens. A transformer encoder processes these tokens, combines them with an encoded sensitivity curve, and maps the resulting representation to a 32-dimensional variational latent space.

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
