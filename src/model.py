"""
Author: Raphael Jontofsohn

Structured multimodal variational autoencoder for UIFO configurations.
Flat, Grid, and Aliased vocabularies are mapped into a shared sequence of node,
edge, and global tokens before transformer-based encoding and multimodal fusion
with the corresponding sensitivity curve.
"""
import copy
import math
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_NODES = [
    "01", "02", "03", "10", "11", "12", "13", "14",
    "20", "21", "22", "23", "24", "30", "31", "32",
    "33", "34", "41", "42", "43"
]

CANONICAL_EDGES = [
    ("10", "11"), ("11", "12"), ("12", "13"), ("13", "14"),
    ("20", "21"), ("21", "22"), ("22", "23"), ("23", "24"),
    ("30", "31"), ("31", "32"), ("32", "33"), ("33", "34"),
    ("01", "11"), ("11", "21"), ("21", "31"), ("31", "41"),
    ("02", "12"), ("12", "22"), ("22", "32"), ("32", "42"),
    ("03", "13"), ("13", "23"), ("23", "33"), ("33", "43"),
]

NODE_SUBPOSITIONS = [
    "CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM", "MAIN",
    "BOUNDARY", "LO", "BHBS", "DETECTOR"
]


def _subposition_from_name(name: str) -> str:
    name = str(name).lower()

    if "bhbs" in name:
        return "BHBS"
    if "lo" in name:
        return "LO"
    if "detector" in name:
        return "DETECTOR"
    if "boundary" in name:
        return "BOUNDARY"
    if "center" in name:
        return "CENTER"
    if "ml" in name:
        return "LEFT"
    if "mr" in name:
        return "RIGHT"
    if "mt" in name:
        return "TOP"
    if "mb" in name:
        return "BOTTOM"
    if re.search(r"m\d{2}", name):
        return "MAIN"
    return "BOUNDARY"


def _normalize_grid_subposition(sub: str) -> str:
    # Old extractors represented suspensions as separate *_SUS subpositions.
    # For the shared encoder we fold mass/free_mass information into the parent
    # optical token, matching the newer representation design.
    return str(sub).replace("_SUS", "")


def _coords_from_text(text: str) -> list[str]:
    return [coord for coord in re.findall(r"\d{2}", str(text)) if coord in VALID_NODES]


class StructuredTopologyEmbedder(nn.Module):
    """
    Converts a representation-specific feature vector into shared UIFO tokens.

    Output shape: [batch, num_tokens, d_model]
    """

    def __init__(self, global_vocab, d_model=256):
        super().__init__()
        self.global_vocab = list(global_vocab)
        self.param_dim = len(global_vocab)

        self.token_names = (
            [f"NODE_{coord}_{sub}" for coord in VALID_NODES for sub in NODE_SUBPOSITIONS]
            + [f"EDGE_{u}_{v}" for u, v in CANONICAL_EDGES]
            + ["GLOBAL"]
        )
        self.token_to_id = {name: idx for idx, name in enumerate(self.token_names)}

        token_to_indices = self._build_token_index_lists(global_vocab)
        self.p_max = max(1, max(len(indices) for indices in token_to_indices.values()))

        index_matrix = torch.zeros((len(self.token_names), self.p_max), dtype=torch.long)
        mask_matrix = torch.zeros((len(self.token_names), self.p_max), dtype=torch.float32)

        for token_name, indices in token_to_indices.items():
            token_idx = self.token_to_id[token_name]
            for feature_pos, vocab_idx in enumerate(indices[: self.p_max]):
                index_matrix[token_idx, feature_pos] = vocab_idx
                mask_matrix[token_idx, feature_pos] = 1.0

        self.register_buffer("token_indices", index_matrix)
        self.register_buffer("token_mask", mask_matrix)

        self.feature_projection = nn.Sequential(
            nn.Linear(self.p_max, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.token_embedding = nn.Parameter(torch.randn(1, len(self.token_names), d_model) * 0.02)

    def _edge_token_from_coords(self, coords: list[str]) -> str:
        for left, right in zip(coords, coords[1:]):
            for u, v in CANONICAL_EDGES:
                if (left == u and right == v) or (left == v and right == u):
                    return f"EDGE_{u}_{v}"
        return "GLOBAL"

    def _node_token(self, coord: str, sub: str) -> str:
        sub = _normalize_grid_subposition(sub)
        token = f"NODE_{coord}_{sub}"
        return token if token in self.token_to_id else "GLOBAL"

    def _flat_token_from_feature(self, feat_name: str) -> str:
        coords = _coords_from_text(feat_name)
        if not coords:
            return "GLOBAL"

        # Flat edge/length features usually contain two or more coordinates.
        if "length" in feat_name.lower() and len(coords) >= 2:
            return self._edge_token_from_coords(coords)

        coord = coords[0]
        sub = _subposition_from_name(feat_name)
        return self._node_token(coord, sub)

    def _pos_token_from_feature(self, feat_name: str) -> str:
        match = re.match(r"POS_(\d{2})_([A-Z_]+?)_(CLASS|TYPE|PROP|SHARED_SLOT)_", feat_name)
        if not match:
            return "GLOBAL"
        coord, sub, _kind = match.groups()
        return self._node_token(coord, sub)

    def _build_token_index_lists(self, global_vocab):
        token_to_indices = defaultdict(list)
        for vocab_idx, feat_name in enumerate(global_vocab):
            if feat_name.startswith("POS_"):
                token_name = self._pos_token_from_feature(feat_name)
            elif feat_name.startswith("EDGE_"):
                token_name = self._edge_token_from_coords(_coords_from_text(feat_name))
            else:
                token_name = self._flat_token_from_feature(feat_name)

            token_to_indices[token_name].append(vocab_idx)

        for token_name in self.token_names:
            token_to_indices.setdefault(token_name, [])
        return token_to_indices

    def forward(self, x_flat):
        token_values = x_flat[:, self.token_indices]
        token_values = token_values * self.token_mask.unsqueeze(0)
        return self.feature_projection(token_values) + self.token_embedding


class TopologyTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_engine = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((x, cls_tokens), dim=1)
        return self.transformer_engine(sequence)[:, -1, :]


def build_param_loss_masks(global_vocab):
    class_mask = []
    value_mask = []
    edge_mask = []

    for feat in global_vocab:
        is_edge = feat.startswith("EDGE_") or ("length" in feat.lower() and feat.startswith("PROP_"))
        is_class = (
            "_CLASS_" in feat
            or "_TYPE_" in feat
            or feat.startswith("NODE_")
        )
        is_value = not is_class

        class_mask.append(float(is_class))
        value_mask.append(float(is_value and not is_edge))
        edge_mask.append(float(is_edge))

    return {
        "class": torch.tensor(class_mask, dtype=torch.float32),
        "value": torch.tensor(value_mask, dtype=torch.float32),
        "edge": torch.tensor(edge_mask, dtype=torch.float32),
    }


class MultimodalTopologyVAE(nn.Module):
    def __init__(self, global_vocab, latent_dim=32, sens_dim=50, sens_weight=100.0):
        super().__init__()
        self.sens_weight = sens_weight
        self.param_dim = len(global_vocab)

        masks = build_param_loss_masks(global_vocab)
        self.register_buffer("param_class_mask", masks["class"])
        self.register_buffer("param_value_mask", masks["value"])
        self.register_buffer("param_edge_mask", masks["edge"])

        self.topology_embedder = StructuredTopologyEmbedder(global_vocab, d_model=256)
        self.transformer = TopologyTransformer(
            d_model=256,
            nhead=4,
            dim_feedforward=384,
            num_layers=8,
        )
        self.topo_compressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )

        self.sens_encoder = nn.Sequential(nn.Linear(sens_dim, 64), nn.SiLU())

        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )
        self.sens_decoder_head = nn.Linear(128, sens_dim)

        # Structured parameter decoder heads. Classes/existence and continuous
        # values are decoded separately and merged into the original vector shape.
        self.param_class_head = nn.Linear(128, self.param_dim)
        self.param_value_head = nn.Linear(128, self.param_dim)

    def encode(self, sens_x, flat_params_x):
        tokens = self.topology_embedder(flat_params_x)
        cls_output = self.transformer(tokens)
        topo_abstract = self.topo_compressor(cls_output)
        sens_abstract = self.sens_encoder(sens_x)
        fused = torch.cat([sens_abstract, topo_abstract], dim=1)
        mixed = self.fusion_mlp(fused)

        mu = self.fc_mu(mixed)
        logvar = torch.clamp(self.fc_logvar(mixed), min=-15.0, max=15.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        shared_features = self.shared_decoder(z)
        recon_sens = self.sens_decoder_head(shared_features)

        class_probs = torch.sigmoid(self.param_class_head(shared_features))
        values = torch.sigmoid(self.param_value_head(shared_features))

        class_mask = self.param_class_mask.unsqueeze(0)
        value_mask = self.param_value_mask.unsqueeze(0)
        edge_mask = self.param_edge_mask.unsqueeze(0)
        recon_params = class_probs * class_mask + values * (value_mask + edge_mask)
        return recon_sens, recon_params

    def forward(self, sens_x, flat_params_x):
        mu, logvar = self.encode(sens_x, flat_params_x)
        z = self.reparameterize(mu, logvar)
        recon_sens, recon_params = self.decode(z)
        return recon_sens, recon_params, mu, logvar


def _masked_mean(loss_values, mask):
    denom = mask.sum().clamp_min(1.0)
    return (loss_values * mask.unsqueeze(0)).sum() / (loss_values.size(0) * denom)


def compute_multimodal_vae_loss(
    recon_sens,
    sens,
    recon_params,
    params,
    mu,
    logvar,
    beta=1.0,
    sens_weight=100.0,
    param_masks=None,
    class_weight=1.0,
    value_weight=1.0,
    edge_weight=1.0,
):
    loss_sens = F.mse_loss(recon_sens, sens, reduction="mean")
    weighted_loss_sens = loss_sens * sens_weight

    if param_masks is None:
        loss_params = F.binary_cross_entropy(recon_params, params, reduction="mean")
    else:
        class_mask = param_masks["class"].to(params.device)
        value_mask = param_masks["value"].to(params.device)
        edge_mask = param_masks["edge"].to(params.device)

        class_loss_raw = F.binary_cross_entropy(recon_params, params, reduction="none")
        value_loss_raw = F.mse_loss(recon_params, params, reduction="none")

        loss_class = _masked_mean(class_loss_raw, class_mask)
        loss_value = _masked_mean(value_loss_raw, value_mask)
        loss_edge = _masked_mean(value_loss_raw, edge_mask)
        loss_params = class_weight * loss_class + value_weight * loss_value + edge_weight * loss_edge

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (mu.size(0) * mu.size(1))

    total_loss = weighted_loss_sens + loss_params + beta * kl_loss
    return total_loss, loss_sens, loss_params, kl_loss


def get_beta(current_epoch, warmup_epochs=50, max_beta=0.2):
    if current_epoch < warmup_epochs:
        return max_beta * (current_epoch / warmup_epochs)
    return max_beta


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_weights = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch, current_beta, max_beta):
        is_warmup_finished = math.isclose(current_beta, max_beta, rel_tol=1e-5)
        if not is_warmup_finished:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
