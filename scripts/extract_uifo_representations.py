#!/usr/bin/env python3
"""
Unified UIFO Representation Extraction Pipeline
================================================

Author
------
Raphael Jontofsohn

Purpose
-------
This script documents and implements the complete extraction pipeline for the
three UIFO configuration representations used by the structured multimodal VAE:

1. Flat
   Preserves representation-specific feature names in one global vector.

2. Grid
   Assigns component classes and continuous properties to explicit grid
   positions and local subpositions.

3. Aliased
   Retains the same spatial structure as Grid while storing continuous
   properties in class-dependent shared slots.

The public pipeline intentionally omits project-specific fractional-run
blacklists and does not restrict the data to the five best runs per topology.
Those thesis-specific filtering steps are documented below but remain disabled.

Inputs
------
- A lightweight Parquet table containing run metadata.
- A heavyweight Parquet table containing sensitivity curves and related fields.
- HDF5 files containing setup graphs and optimized parameter vectors.
- ``reconstruct_optimization_pairs.py`` from the Differometor data pipeline.

Outputs
-------
For each selected representation, the script writes:

- ``uifo_<representation>_matrix.npy``
- ``uifo_<representation>_vocab.json``
- ``uifo_<representation>_index.parquet``

It also writes one shared metadata table:

- ``uifo_metadata.parquet``

The matrix, vocabulary, and index files must always be kept together because
the vocabulary defines the matrix columns and the index defines the matrix rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import warnings
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from reconstruct_optimization_pairs import reconstruct_optimization_pairs

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        """Fallback used when tqdm is unavailable."""
        return iterable if iterable is not None else []


warnings.filterwarnings("ignore")


# ==============================================================================
# 1. SHARED CONFIGURATION
# ==============================================================================

DEFAULT_H5_DIR = Path("data/raw/qamplfreq_flat")
DEFAULT_LIGHTWEIGHT = Path("data/raw/broadband_metadata_light.parquet")
DEFAULT_HEAVYWEIGHT = Path("data/raw/broadband_heavy_data.parquet")
DEFAULT_OUTPUT_DIR = Path("data/representations")

TARGET_PROPERTIES = {
    "reflectivity",
    "tuning",
    "db",
    "angle",
    "power",
    "mass",
    "length",
}

IGNORED_NODE_COMPONENTS = {
    "signal",
    "qnoised",
    "frequency",
    "incoherent_noise",
    "qhd",
}

VALID_POSITIONS = [
    "01", "02", "03", "10", "11", "12", "13", "14",
    "20", "21", "22", "23", "24", "30", "31", "32",
    "33", "34", "41", "42", "43",
]

NODE_CLASSES = [
    "laser",
    "squeezer",
    "mirror",
    "beamsplitter_left",
    "beamsplitter_right",
    "beamsplitter_top",
    "beamsplitter_bottom",
    "directional_beamsplitter_0",
    "directional_beamsplitter_90",
    "detector",
    "qnoised",
    "signal",
    "free_mass",
    "frequency",
    "bhbs",
]

SLOT_MAP = {
    "laser": ["power"],
    "squeezer": ["db", "angle_sin", "angle_cos"],
    "mirror": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
    "beamsplitter_left": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
    "beamsplitter_right": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
    "beamsplitter_top": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
    "beamsplitter_bottom": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
    "bhbs": ["reflectivity", "tuning_sin", "tuning_cos"],
}


# ==============================================================================
# 2. FLAT REPRESENTATION
# ==============================================================================

class FlatNonlinearBS4Extractor:
    """
    Flat extractor with original nonlinear-style scaling and a four-way
    orientation split for standard beamsplitters.
    """

    VALID_TOPOLOGY = {
        "CENTER_PATCHES": {
            "coordinates": ["11", "12", "13", "21", "22", "23", "31", "32", "33"],
            "sub_positions": {
                "LEFT": ["mirror"],
                "RIGHT": ["mirror"],
                "TOP": ["mirror"],
                "BOTTOM": ["mirror"],
                "CENTER": [
                    "beamsplitter_left",
                    "beamsplitter_right",
                    "beamsplitter_top",
                    "beamsplitter_bottom",
                    "directional_beamsplitter_0",
                    "directional_beamsplitter_90",
                ],
            },
        },
        "BOUNDARY_PATCHES": {
            "coordinates": ["01", "02", "03", "10", "14", "20", "24", "30", "34", "41", "42", "43"],
            "sub_positions": {
                "MAIN": ["mirror"],
                "BOUNDARY": ["laser", "squeezer", "detector"],
                "LO": ["lo_laser"],
                "DETECTOR": ["detector"],
                "BHBS": ["bhbs"],
            },
        },
    }

    COMPONENT_PROPERTIES = {
        "laser": ["power"],
        "squeezer": ["db", "angle_sin", "angle_cos"],
        "mirror": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
        "beamsplitter_left": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
        "beamsplitter_right": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
        "beamsplitter_top": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
        "beamsplitter_bottom": ["reflectivity", "tuning_sin", "tuning_cos", "mass"],
        "bhbs": ["reflectivity", "tuning_sin", "tuning_cos"],
    }

    def __init__(self):
        self.vocabulary: list[str] = []
        self.feat_to_idx: dict[str, int] = {}
        self.vector_size = 0
        self.is_fitted = False

    def _source_port_for_node(self, node_name: str, edges: dict) -> str:
        coord = "".join(filter(str.isdigit, node_name))
        preferred_targets = [f"ml{coord}", f"mr{coord}", f"mt{coord}", f"mb{coord}"] if coord else []

        for target in preferred_targets:
            edge_info = edges.get(f"{node_name}_{target}", {})
            source_port = str(edge_info.get("source_port", "")).lower()
            if source_port in {"left", "right", "top", "bottom"}:
                return source_port

        for edge_name, edge_info in edges.items():
            if edge_name.startswith("__") or not isinstance(edge_info, dict):
                continue

            source = str(edge_info.get("source", edge_info.get("source_node", ""))).lower()
            target = str(edge_info.get("target", edge_info.get("target_node", ""))).lower()
            edge_name_lower = str(edge_name).lower()
            node_name_lower = node_name.lower()

            node_is_source = source == node_name_lower or edge_name_lower.startswith(f"{node_name_lower}_")
            node_is_target = target == node_name_lower or edge_name_lower.endswith(f"_{node_name_lower}")

            if node_is_source:
                source_port = str(edge_info.get("source_port", "")).lower()
                if source_port in {"left", "right", "top", "bottom"}:
                    return source_port

            if node_is_target:
                target_port = str(edge_info.get("target_port", "")).lower()
                if target_port in {"left", "right", "top", "bottom"}:
                    return target_port

        return ""

    def _clean_component_name(self, comp_type: str, node_name: str, edges: dict) -> str:
        comp_type = comp_type.lower()
        node_name_lower = node_name.lower()

        if comp_type == "laser" and "lo" in node_name_lower:
            return "lo_laser"

        if comp_type == "beamsplitter":
            if "bhbs" in node_name_lower:
                return "bhbs"

            source_port = self._source_port_for_node(node_name, edges)
            if source_port in {"left", "right", "top", "bottom"}:
                return f"beamsplitter_{source_port}"

            return "beamsplitter_left"

        if comp_type == "directional_beamsplitter":
            source_port = self._source_port_for_node(node_name, edges)
            if source_port in {"left", "right"}:
                return "directional_beamsplitter_0"
            if source_port in {"top", "bottom"}:
                return "directional_beamsplitter_90"

            return "directional_beamsplitter_0"

        return comp_type

    def _map_to_physical(self, raw_val: float, bounds: list[float]) -> float:
        min_val, max_val = bounds
        clipped = np.clip(raw_val, -500.0, 500.0)
        sig = 1.0 / (1.0 + np.exp(-clipped))
        return min_val + (max_val - min_val) * sig

    def _linear_minmax(self, phys_val: float, bounds: list[float]) -> float:
        min_phys, max_phys = bounds
        norm = (phys_val - min_phys) / (max_phys - min_phys + 1e-9)
        return float(np.clip(norm, 0.0, 1.0))

    def _log_minmax(self, phys_val: float, bounds: list[float]) -> float:
        min_phys, max_phys = bounds
        eps = 1e-12
        safe_val = np.clip(phys_val, min_phys, max_phys)
        log_val = np.log10(safe_val + eps)
        log_min = np.log10(min_phys + eps)
        log_max = np.log10(max_phys + eps)
        norm = (log_val - log_min) / (log_max - log_min + 1e-12)
        return float(np.clip(norm, 0.0, 1.0))

    def _trig_transform(self, prop_type: str, phys_val: float) -> dict[str, float]:
        radians = np.deg2rad(phys_val)
        return {
            f"{prop_type}_sin": float((np.sin(radians) + 1.0) / 2.0),
            f"{prop_type}_cos": float((np.cos(radians) + 1.0) / 2.0),
        }

    def _apply_ml_transform(self, prop_type: str, phys_val: float, bounds: list[float]) -> dict[str, float]:
        if prop_type in {"angle", "tuning"}:
            return self._trig_transform(prop_type, phys_val)

        if prop_type == "mass":
            return {prop_type: self._log_minmax(phys_val, bounds)}

        if prop_type in {"db", "length", "power", "reflectivity"}:
            return {prop_type: self._linear_minmax(phys_val, bounds)}

        return {}

    def _format_optimization_pair_name(self, pair_keys) -> str:
        return "_".join(str(k) for k in pair_keys) if isinstance(pair_keys, list) else str(pair_keys)

    def _expand_length_pair_names(self, pair_keys) -> list[str]:
        if not isinstance(pair_keys, list):
            return [str(pair_keys)]

        if len(pair_keys) == 2 and pair_keys[1] == "length" and not isinstance(pair_keys[0], (list, tuple)):
            return [f"{pair_keys[0]}_length"]

        expanded_names = []
        for item in pair_keys:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] == "length":
                expanded_names.append(f"{item[0]}_length")

        if expanded_names:
            return expanded_names

        return [self._format_optimization_pair_name(pair_keys)]

    def _load_graph(self, h5_file, run_id_str: str) -> dict:
        raw_graph = h5_file[f"runs/{run_id_str}/setup_graph"][()]
        try:
            graph_str = zlib.decompress(raw_graph).decode("utf-8")
        except Exception:
            graph_str = raw_graph.decode("utf-8")
        return json.loads(graph_str)

    @staticmethod
    def _validate_encoded_matrix(
        matrix: np.ndarray,
        df_index: pd.DataFrame,
        representation_name: str,
        *,
        print_summary: bool = True,
    ) -> dict[str, float | int]:
        """Report matrix occupancy and prevent silently corrupted exports."""
        if matrix.ndim != 2:
            raise ValueError(
                f"{representation_name} matrix must be two-dimensional, got shape {matrix.shape}."
            )
        if matrix.shape[0] != len(df_index):
            raise ValueError(
                f"{representation_name} matrix has {matrix.shape[0]} rows, "
                f"but the index contains {len(df_index)} runs."
            )
        if matrix.shape[1] == 0:
            raise ValueError(f"{representation_name} vocabulary is empty.")
        if not np.isfinite(matrix).all():
            invalid_count = int(np.size(matrix) - np.isfinite(matrix).sum())
            raise ValueError(
                f"{representation_name} matrix contains {invalid_count} NaN or infinite values."
            )

        nonzero_count = int(np.count_nonzero(matrix))
        nonzero_fraction = nonzero_count / matrix.size if matrix.size else 0.0
        all_zero_mask = ~np.any(matrix != 0.0, axis=1)
        all_zero_indices = np.flatnonzero(all_zero_mask)
        all_zero_count = int(all_zero_indices.size)
        all_zero_fraction = all_zero_count / matrix.shape[0] if matrix.shape[0] else 0.0

        if print_summary:
            print("\n" + "=" * 80)
            print(f"{representation_name.upper()} MATRIX CONTROL SUMMARY")
            print(f"   Matrix shape       : {matrix.shape}")
            print(f"   Nonzero fraction   : {nonzero_fraction:.6f} ({nonzero_fraction:.2%})")
            print(
                f"   All-zero rows      : {all_zero_count}/{matrix.shape[0]} "
                f"({all_zero_fraction:.2%})"
            )
            print("=" * 80)

        if all_zero_count:
            id_columns = [column for column in ("hash", "run_id") if column in df_index.columns]
            preview = df_index.iloc[all_zero_indices[:5]][id_columns].to_dict("records")
            raise ValueError(
                f"{representation_name} extraction produced {all_zero_count} all-zero rows. "
                f"First affected runs: {preview}. The matrix was not saved."
            )

        return {
            "nonzero_count": nonzero_count,
            "nonzero_fraction": nonzero_fraction,
            "all_zero_count": all_zero_count,
            "all_zero_fraction": all_zero_fraction,
        }

    def fit(self, df: pd.DataFrame, h5_files: list[Path]):
        print("\n[+] Phase 1: Discovering Global Semantic Vocabulary (Flat BS4 Nonlinear)...")
        global_vocab = set()
        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for hash_id, group in tqdm(df.groupby("hash"), desc="Scanning Topologies"):
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(f"No H5 file found for hash {hash_id!r} during Flat fitting.")

            for run_id in group["run_id"]:
                run_id_str = str(run_id)

                try:
                    with h5py.File(h5_path, "r") as h5_file:
                        graph = self._load_graph(h5_file, run_id_str)
                        edges = graph.get("edges", {})

                        for node_name, info in graph.get("nodes", {}).items():
                            comp_type = info.get("component", "unknown").lower()
                            if comp_type in IGNORED_NODE_COMPONENTS:
                                continue

                            comp_type = self._clean_component_name(comp_type, node_name, edges)
                            global_vocab.add(f"NODE_{comp_type}_{node_name}")

                    mapping_result = reconstruct_optimization_pairs(str(h5_path), run_id_str)
                    pairs = mapping_result.get("optimization_pairs", [])
                    if not isinstance(pairs, list):
                        raise TypeError(
                            f"optimization_pairs must be a list, got {type(pairs).__name__}."
                        )

                    for row in pairs:
                        prop_type = row.get("property", "")
                        if prop_type not in TARGET_PROPERTIES:
                            continue

                        pair_keys = row.get("optimization_pair", [])
                        mock_transform = self._apply_ml_transform(prop_type, 0.0, row.get("bounds", [0, 1]))

                        if prop_type == "length":
                            for length_name in self._expand_length_pair_names(pair_keys):
                                global_vocab.add(f"PROP_{length_name}")
                            continue

                        flat_name = self._format_optimization_pair_name(pair_keys)
                        for suffix in mock_transform.keys():
                            if prop_type in suffix and suffix != prop_type:
                                feature_name = f"PROP_{flat_name.replace(prop_type, suffix)}"
                            else:
                                feature_name = f"PROP_{flat_name}"
                            global_vocab.add(feature_name)

                except Exception as exc:
                    raise RuntimeError(
                        f"Flat vocabulary extraction failed for hash={hash_id!r}, "
                        f"run_id={run_id_str!r}."
                    ) from exc

        self.vocabulary = sorted(global_vocab)
        self.vector_size = len(self.vocabulary)
        if self.vector_size == 0:
            raise ValueError("Flat vocabulary discovery produced no features.")
        self.feat_to_idx = {feat: i for i, feat in enumerate(self.vocabulary)}
        self.is_fitted = True
        print(f"[+] Phase 1 Complete. Distilled to {self.vector_size} dimensions.")
        return self

    def transform(self, df: pd.DataFrame, h5_files: list[Path]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transformation.")

        print("\n[+] Phase 2: Encoding Semantic Matrix (Flat BS4 Nonlinear)...")
        matrix = np.zeros((len(df), self.vector_size), dtype=np.float32)
        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for row_idx, (_, df_row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Encoding Matrix")):
            hash_id = df_row["hash"]
            run_id_str = str(df_row["run_id"])
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(
                    f"No H5 file found for hash {hash_id!r}, run_id={run_id_str!r} "
                    "during Flat transformation."
                )

            try:
                with h5py.File(h5_path, "r") as h5_file:
                    raw_opt = h5_file[f"runs/{run_id_str}/best_parameters"][:]
                    opt_params = raw_opt[0] if raw_opt.ndim >= 2 else raw_opt
                    graph = self._load_graph(h5_file, run_id_str)
                    edges = graph.get("edges", {})

                for node_name, info in graph.get("nodes", {}).items():
                    comp_type = info.get("component", "unknown").lower()
                    if comp_type in IGNORED_NODE_COMPONENTS:
                        continue

                    comp_type = self._clean_component_name(comp_type, node_name, edges)
                    node_key = f"NODE_{comp_type}_{node_name}"
                    if node_key in self.feat_to_idx:
                        matrix[row_idx, self.feat_to_idx[node_key]] = 1.0

                mapping_result = reconstruct_optimization_pairs(str(h5_path), run_id_str)
                pairs = mapping_result.get("optimization_pairs", [])
                if not isinstance(pairs, list):
                    raise TypeError(
                        f"optimization_pairs must be a list, got {type(pairs).__name__}."
                    )

                for row in pairs:
                    prop_type = row.get("property", "")
                    idx = row.get("index")
                    bounds = row.get("bounds")

                    if prop_type not in TARGET_PROPERTIES or idx is None or bounds is None:
                        continue
                    if idx >= len(opt_params):
                        continue

                    raw_val = float(opt_params[idx])
                    phys_val = self._map_to_physical(raw_val, bounds)
                    transformed_vals = self._apply_ml_transform(prop_type, phys_val, bounds)

                    pair_keys = row.get("optimization_pair", [])
                    if prop_type == "length":
                        for length_name in self._expand_length_pair_names(pair_keys):
                            f_name = f"PROP_{length_name}"
                            if f_name in self.feat_to_idx:
                                matrix[row_idx, self.feat_to_idx[f_name]] = float(transformed_vals["length"])
                        continue

                    flat_name = self._format_optimization_pair_name(pair_keys)
                    for suffix, t_val in transformed_vals.items():
                        if prop_type in suffix and suffix != prop_type:
                            f_name = f"PROP_{flat_name.replace(prop_type, suffix)}"
                        else:
                            f_name = f"PROP_{flat_name}"

                        if f_name in self.feat_to_idx:
                            matrix[row_idx, self.feat_to_idx[f_name]] = float(t_val)

            except Exception as exc:
                raise RuntimeError(
                    f"Flat transformation failed for row={row_idx}, hash={hash_id!r}, "
                    f"run_id={run_id_str!r}."
                ) from exc

        self._validate_encoded_matrix(matrix, df, "Flat")
        return matrix

    def save_assets(
        self,
        matrix: np.ndarray,
        matrix_path: Path,
        vocab_path: Path,
        df_index: pd.DataFrame,
        index_path: Path,
    ) -> None:
        # A final guard ensures that invalid matrices cannot be persisted even if
        # save_assets is called independently of transform.
        self._validate_encoded_matrix(
            matrix,
            df_index,
            self.__class__.__name__,
            print_summary=False,
        )

        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(matrix_path, matrix)
        with open(vocab_path, "w") as f:
            json.dump(self.vocabulary, f, indent=4)

        required_cols = ["hash", "run_id"]
        missing_cols = [col for col in required_cols if col not in df_index.columns]
        if missing_cols:
            raise ValueError(f"Cannot save index file. Missing columns: {missing_cols}")
        if len(df_index) != matrix.shape[0]:
            raise ValueError(f"Index length ({len(df_index)}) does not match matrix rows ({matrix.shape[0]}).")

        df_index.loc[:, required_cols].reset_index(drop=True).to_parquet(index_path, index=False)

        print("\n" + "=" * 80)
        print("EXPORT COMPLETE")
        print(f"   Matrix Shape : {matrix.shape}")
        print(f"   Matrix File  : {matrix_path}")
        print(f"   Vocab File   : {vocab_path}")
        print(f"   Index File   : {index_path}")
        print("=" * 80)

    def print_sanity_check(self, matrix: np.ndarray) -> None:
        bs4_labels = [
            "beamsplitter_left",
            "beamsplitter_right",
            "beamsplitter_top",
            "beamsplitter_bottom",
        ]
        directional_labels = [
            "directional_beamsplitter_0",
            "directional_beamsplitter_90",
        ]

        bs4_features = [feat for feat in self.vocabulary if any(label in feat for label in bs4_labels)]
        directional_features = [feat for feat in self.vocabulary if any(label in feat for label in directional_labels)]

        print("\n" + "=" * 80)
        print("SANITY CHECK")
        print(f"Matrix shape                         : {matrix.shape}")
        print(f"Vocabulary size                      : {len(self.vocabulary)}")
        print(f"Features with beamsplitter L/R/T/B   : {len(bs4_features)}")
        print(f"Features with directional BS 0/90    : {len(directional_features)}")
        print("Vocabulary sample:")
        for feat in self.vocabulary[:20]:
            print(f"  - {feat}")
        print("=" * 80)

# ==============================================================================
# 3. GRID REPRESENTATION
# ==============================================================================

class GridFeatureExtractor(FlatVectorExtractor):
    """
    A position-specific Spatial Grid Extractor solving the Spatial Collapse bug.
    Maps properties strictly via Coordinate AND Sub-Position (e.g., POS_11_LEFT_PROP_tuning_sin).
    """

    def _get_sub_pos(self, name: str) -> str:
        name = str(name).lower()
        suffix = "_SUS" if "sus" in name else ""

        if 'center' in name:
            base = 'CENTER'
        elif 'ml' in name:
            base = 'LEFT'
        elif 'mr' in name:
            base = 'RIGHT'
        elif 'mt' in name:
            base = 'TOP'
        elif 'mb' in name:
            base = 'BOTTOM'
        elif 'bhbs' in name:
            base = 'BHBS'
        elif 'lo' in name:
            base = 'LO'
        elif 'boundary' in name:
            base = 'BOUNDARY'
        elif 'detector' in name:
            base = 'DETECTOR'
        elif re.search(r'm\d{2}', name):
            base = 'MAIN'
        else:
            base = 'DEFAULT'

        return base + suffix

    def _get_property_sub_pos(self, name: str, prop_type: str) -> str:
        sub_pos = self._get_sub_pos(name)
        if prop_type == "mass" and sub_pos.endswith("_SUS"):
            return sub_pos.removesuffix("_SUS")
        return sub_pos


    def fit(self, df: pd.DataFrame, h5_files: list):
        print("\n[+] Phase 1: Building Constrained Spatial Grid Vocabulary...")
        vocab = []

        # 1. Iterate over constrained topology rules
        for patch_type, rules in self.VALID_TOPOLOGY.items():
            for pos in rules["coordinates"]:
                for sub_pos, allowed_components in rules["sub_positions"].items():
                    # Nodes (Types)
                    for comp in allowed_components:
                        vocab.append(f"POS_{pos}_{sub_pos}_TYPE_{comp}")

                        # Continuous Properties based on component capabilities
                        if comp in self.COMPONENT_PROPERTIES:
                            for prop in self.COMPONENT_PROPERTIES[comp]:
                                prop_feat = f"POS_{pos}_{sub_pos}_PROP_{prop}"
                                if prop_feat not in vocab:
                                    vocab.append(prop_feat)

        print("    -> Discovering dynamic length edges (Rule 5)...")
        # Edges (Längen) ermitteln wir weiterhin datengetrieben, da sie sich
        # immer automatisch auf die existierenden Kanten des Graphen beschränken.
        edge_vocab = set()
        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for hash_id, group in df.groupby("hash"):
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(f"No H5 file found for hash {hash_id!r} during Grid fitting.")

            for run_id in group["run_id"]:
                try:
                    mapping_result = reconstruct_optimization_pairs(str(h5_path), str(run_id))
                    pairs = mapping_result.get("optimization_pairs", [])
                    if not isinstance(pairs, list):
                        raise TypeError(
                            f"optimization_pairs must be a list, got {type(pairs).__name__}."
                        )

                    for row in pairs:
                        if row.get('property') == 'length':
                            pair_keys = row.get('optimization_pair', [])
                            for length_name in self._expand_length_pair_names(pair_keys):
                                edge_vocab.add(f"EDGE_{length_name}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Grid vocabulary extraction failed for hash={hash_id!r}, "
                        f"run_id={str(run_id)!r}."
                    ) from exc

        for edge in sorted(list(edge_vocab)):
            vocab.append(edge)

        self.vocabulary = vocab
        self.vector_size = len(vocab)
        if self.vector_size == 0:
            raise ValueError("Grid vocabulary discovery produced no features.")
        self.feat_to_idx = {feat: i for i, feat in enumerate(vocab)}
        self.is_fitted = True

        print(f"[+] Constrained Grid Vocabulary built: {self.vector_size} dimensions.")
        return self

    def transform(self, df: pd.DataFrame, h5_files: list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transformation.")

        print("\n[+] Phase 2: Encoding Masked Spatial Matrix...")
        num_runs = len(df)
        matrix = np.zeros((num_runs, self.vector_size), dtype=np.float32)
        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for row_idx, (_, df_row) in enumerate(tqdm(df.iterrows(), total=num_runs, desc="Encoding Grid")):
            hash_id = df_row["hash"]
            run_id_str = str(df_row["run_id"])
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(
                    f"No H5 file found for hash {hash_id!r}, run_id={run_id_str!r} "
                    "during Grid transformation."
                )

            try:
                with h5py.File(h5_path, 'r') as h5_file:
                    raw_graph = h5_file[f'runs/{run_id_str}/setup_graph'][()]
                    try:
                        graph_str = zlib.decompress(raw_graph).decode('utf-8')
                    except Exception:
                        graph_str = raw_graph.decode('utf-8')
                    graph = json.loads(graph_str)

                    raw_opt = h5_file[f'runs/{run_id_str}/best_parameters'][:]
                    opt_params = raw_opt[0] if raw_opt.ndim >= 2 else raw_opt


                    # 1. Encode Static Nodes
                    edges = graph.get('edges', {})  # <--- Kanten laden
                    for node_name, info in graph.get('nodes', {}).items():
                        comp_type = info.get('component', '').lower()

                        if comp_type in IGNORED_NODE_COMPONENTS:
                            continue

                        edges["__current_node_info__"] = info

                        target_val = info.get('target', '')
                        search_str = target_val if target_val else node_name

                        coord_match = re.search(r'\d{2}', str(search_str))
                        if not coord_match or coord_match.group(0) not in VALID_POSITIONS: continue
                        coord = coord_match.group(0)

                        sub_pos = self._get_sub_pos(node_name)

                        # ========================================================
                        # NEUE, ZENTRALE REGEL (mit Node-Injection für BHBS)
                        # ========================================================
                        edges["__current_node_info__"] = info
                        comp_type = self._clean_component_name(comp_type, node_name, edges)

                        type_feat = f"POS_{coord}_{sub_pos}_TYPE_{comp_type}"
                        if type_feat in self.feat_to_idx:
                            matrix[row_idx, self.feat_to_idx[type_feat]] = 1.0

                # 2. Encode Optimized Parameters
                mapping_result = reconstruct_optimization_pairs(str(h5_path), run_id_str)
                pairs = mapping_result.get("optimization_pairs", [])
                if not isinstance(pairs, list):
                    raise TypeError(
                        f"optimization_pairs must be a list, got {type(pairs).__name__}."
                    )

                for row in pairs:
                    prop_type = row.get('property', '')
                    idx = row.get('index')
                    bounds = row.get('bounds')

                    if prop_type in TARGET_PROPERTIES and idx is not None and bounds is not None and idx < len(
                            opt_params):
                        raw_val = float(opt_params[idx])

                        phys_val = self._map_to_physical(raw_val, bounds)
                        transformed_vals = self._apply_ml_transform(prop_type, phys_val, bounds)

                        pair_keys = row.get('optimization_pair', [])

                        if prop_type == 'length':
                            for length_name in self._expand_length_pair_names(pair_keys):
                                edge_feat = f"EDGE_{length_name}"
                                if edge_feat in self.feat_to_idx:
                                    matrix[row_idx, self.feat_to_idx[edge_feat]] = float(transformed_vals["length"])
                        else:
                            flat_name = self._format_optimization_pair_name(pair_keys)
                            coord_match = re.search(r'\d{2}', flat_name)
                            if coord_match and coord_match.group(0) in VALID_POSITIONS:
                                coord = coord_match.group(0)
                                sub_pos = self._get_property_sub_pos(flat_name, prop_type)

                                for suffix, t_val in transformed_vals.items():
                                    prop_feat = f"POS_{coord}_{sub_pos}_PROP_{suffix}"
                                    if prop_feat in self.feat_to_idx:
                                        matrix[row_idx, self.feat_to_idx[prop_feat]] = float(t_val)
            except Exception as exc:
                raise RuntimeError(
                    f"Grid transformation failed for row={row_idx}, hash={hash_id!r}, "
                    f"run_id={run_id_str!r}."
                ) from exc

        self._validate_encoded_matrix(matrix, df, "Grid")
        return matrix

# ==============================================================================
# 4. ALIASED REPRESENTATION
# ==============================================================================

class AliasedFeatureExtractor(FlatVectorExtractor):
    """
    A highly compressed Extractor preventing Spatial Collapse while retaining 1:1 capacity.
    Injects Sub-Positions & conditional Shared Slots (multiplexed representation).
    """

    def _get_sub_pos(self, name: str) -> str:
        name = name.lower()
        suffix = "_SUS" if "sus" in name else ""

        if 'center' in name:
            base = 'CENTER'
        elif 'ml' in name:
            base = 'LEFT'
        elif 'mr' in name:
            base = 'RIGHT'
        elif 'mt' in name:
            base = 'TOP'
        elif 'mb' in name:
            base = 'BOTTOM'
        elif 'bhbs' in name:
            base = 'BHBS'
        elif 'lo' in name:
            base = 'LO'
        elif 'noise' in name:
            base = 'NOISE'
        elif 'boundary' in name:
            base = 'BOUNDARY'
        elif 'detector' in name:
            base = 'DETECTOR'
        elif re.search(r'm\d{2}', name):
            base = 'MAIN'
        else:
            base = 'DEFAULT'

        return base + suffix

    def _get_property_sub_pos(self, name: str, prop_type: str) -> str:
        sub_pos = self._get_sub_pos(name)
        if prop_type == "mass" and sub_pos.endswith("_SUS"):
            return sub_pos.removesuffix("_SUS")
        return sub_pos

    # -----------------------------------------------------------------------
    # PHASE 1: VOCABULARY GENERATION
    # -----------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, h5_files: list):
        print("\n[+] Phase 1: Building Constrained Aliased Vocabulary...")
        vocab = []

        # 1. Iterate over constrained topology rules
        for patch_type, rules in self.VALID_TOPOLOGY.items():
            for pos in rules["coordinates"]:
                for sub_pos, allowed_components in rules["sub_positions"].items():
                    # Class Flags
                    for comp in allowed_components:
                        vocab.append(f"POS_{pos}_{sub_pos}_CLASS_{comp}")

                    # Allocate maximum needed slots for this specific sub_position
                    max_slots_needed = 0
                    for comp in allowed_components:
                        if comp in self.COMPONENT_PROPERTIES:
                            max_slots_needed = max(max_slots_needed, len(self.COMPONENT_PROPERTIES[comp]))

                    for i in range(max_slots_needed):
                        vocab.append(f"POS_{pos}_{sub_pos}_SHARED_SLOT_{i}")

        print("    -> Discovering dynamic length edges...")
        edge_vocab = set()
        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for hash_id, group in df.groupby("hash"):
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(f"No H5 file found for hash {hash_id!r} during Aliased fitting.")

            for run_id in group["run_id"]:
                try:
                    mapping_result = reconstruct_optimization_pairs(str(h5_path), str(run_id))
                    pairs = mapping_result.get("optimization_pairs", [])
                    if not isinstance(pairs, list):
                        raise TypeError(
                            f"optimization_pairs must be a list, got {type(pairs).__name__}."
                        )

                    for row in pairs:
                        if row.get('property') == 'length':
                            pair_keys = row.get('optimization_pair', [])
                            for length_name in self._expand_length_pair_names(pair_keys):
                                edge_vocab.add(f"EDGE_{length_name}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Aliased vocabulary extraction failed for hash={hash_id!r}, "
                        f"run_id={str(run_id)!r}."
                    ) from exc

        for edge in sorted(list(edge_vocab)):
            vocab.append(edge)

        self.vocabulary = vocab
        self.vector_size = len(vocab)
        if self.vector_size == 0:
            raise ValueError("Aliased vocabulary discovery produced no features.")
        self.feat_to_idx = {feat: i for i, feat in enumerate(vocab)}
        self.is_fitted = True

        print(f"[+] Aliased Vocabulary built: {self.vector_size} highly compressed dimensions.")
        return self

    # -----------------------------------------------------------------------
    # PHASE 2: ALIASED MATRIX ENCODING
    # -----------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, h5_files: list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transformation.")

        print("\n[+] Phase 2: Encoding Aliased Spatial Matrix...")
        num_runs = len(df)
        matrix = np.zeros((num_runs, self.vector_size), dtype=np.float32)

        hash_to_path = {f.stem.split("_")[-1]: f for f in h5_files}

        for row_idx, (_, df_row) in enumerate(tqdm(df.iterrows(), total=num_runs, desc="Encoding Aliased Grid")):
            hash_id = df_row["hash"]
            run_id_str = str(df_row["run_id"])
            h5_path = hash_to_path.get(hash_id)
            if not h5_path:
                raise FileNotFoundError(
                    f"No H5 file found for hash {hash_id!r}, run_id={run_id_str!r} "
                    "during Aliased transformation."
                )

            try:
                with h5py.File(h5_path, 'r') as h5_file:
                    raw_graph = h5_file[f'runs/{run_id_str}/setup_graph'][()]
                    try:
                        graph_str = zlib.decompress(raw_graph).decode('utf-8')
                    except Exception:
                        graph_str = raw_graph.decode('utf-8')
                    graph = json.loads(graph_str)

                    raw_opt = h5_file[f'runs/{run_id_str}/best_parameters'][:]
                    opt_params = raw_opt[0] if raw_opt.ndim >= 2 else raw_opt

                node_cache = {}

                # 1. Encode Static Nodes
                edges = graph.get('edges', {})  # <--- Kanten laden
                for node_name, info in graph.get('nodes', {}).items():
                    comp_type = info.get('component', '').lower()

                    if comp_type in IGNORED_NODE_COMPONENTS:
                        continue

                    edges["__current_node_info__"] = info

                    target_val = info.get('target', '')
                    search_str = target_val if target_val else node_name

                    coord_match = re.search(r'\d{2}', str(search_str))
                    if not coord_match or coord_match.group(0) not in VALID_POSITIONS: continue
                    coord = coord_match.group(0)

                    sub_pos = self._get_sub_pos(node_name)

                    # ========================================================
                    # NEUE, ZENTRALE REGEL (mit Node-Injection für BHBS)
                    # ========================================================
                    edges["__current_node_info__"] = info
                    comp_type = self._clean_component_name(comp_type, node_name, edges)

                    # Hier wird der Cache befüllt, damit die Slots später mappen können!
                    if comp_type in NODE_CLASSES:
                        node_cache[f"{coord}_{sub_pos}"] = comp_type

                    class_feat = f"POS_{coord}_{sub_pos}_CLASS_{comp_type}"
                    if class_feat in self.feat_to_idx:
                        matrix[row_idx, self.feat_to_idx[class_feat]] = 1.0

                # 2. Encode Optimized Parameters into Shared Slots
                mapping_result = reconstruct_optimization_pairs(str(h5_path), run_id_str)
                pairs = mapping_result.get("optimization_pairs", [])
                if not isinstance(pairs, list):
                    raise TypeError(
                        f"optimization_pairs must be a list, got {type(pairs).__name__}."
                    )

                for row in pairs:
                    prop_type = row.get('property', '')
                    idx = row.get('index')
                    bounds = row.get('bounds')

                    if prop_type in TARGET_PROPERTIES and idx is not None and bounds is not None and idx < len(
                            opt_params):
                        raw_val = float(opt_params[idx])

                        phys_val = self._map_to_physical(raw_val, bounds)
                        transformed_vals = self._apply_ml_transform(prop_type, phys_val, bounds)

                        pair_keys = row.get('optimization_pair', [])

                        if prop_type == 'length':
                            for length_name in self._expand_length_pair_names(pair_keys):
                                edge_feat = f"EDGE_{length_name}"
                                if edge_feat in self.feat_to_idx:
                                    matrix[row_idx, self.feat_to_idx[edge_feat]] = float(transformed_vals["length"])
                        else:
                            flat_name = self._format_optimization_pair_name(pair_keys)
                            coord_match = re.search(r'\d{2}', flat_name)
                            if coord_match and coord_match.group(0) in VALID_POSITIONS:
                                coord = coord_match.group(0)
                                sub_pos = self._get_property_sub_pos(flat_name, prop_type)
                                comp_type = node_cache.get(f"{coord}_{sub_pos}")

                                if comp_type in SLOT_MAP:
                                    for suffix, t_val in transformed_vals.items():
                                        if suffix in SLOT_MAP[comp_type]:
                                            slot_idx = SLOT_MAP[comp_type].index(suffix)
                                            slot_feat = f"POS_{coord}_{sub_pos}_SHARED_SLOT_{slot_idx}"
                                            if slot_feat in self.feat_to_idx:
                                                matrix[row_idx, self.feat_to_idx[slot_feat]] = float(t_val)

            except Exception as exc:
                raise RuntimeError(
                    f"Aliased transformation failed for row={row_idx}, hash={hash_id!r}, "
                    f"run_id={run_id_str!r}."
                ) from exc

        self._validate_encoded_matrix(matrix, df, "Aliased")
        return matrix

# ==============================================================================
# 5. DATA LOADING AND PUBLIC SELECTION POLICY
# ==============================================================================

def load_source_runs(
    lightweight_path: Path,
    heavyweight_path: Path,
) -> pd.DataFrame:
    """Load, align, validate, and deterministically order the source runs."""
    lightweight = pd.read_parquet(lightweight_path)
    heavyweight = pd.read_parquet(heavyweight_path)

    keys = ["hash", "run_id"]
    for name, frame in (("lightweight", lightweight), ("heavyweight", heavyweight)):
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} input contains duplicate hash/run_id keys.")

    runs = lightweight.merge(
        heavyweight,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    required = {"hash", "run_id", "uifo_size", "setup_graph", "loss_senspow"}
    missing = required - set(runs.columns)
    if missing:
        raise KeyError(f"Merged input is missing required columns: {sorted(missing)}")

    # The final model was developed for 3 x 3 UIFO configurations.
    runs = runs.loc[runs["uifo_size"] == 3].copy()

    if runs["setup_graph"].isna().any():
        raise ValueError("setup_graph contains missing values.")
    if runs["loss_senspow"].isna().any():
        raise ValueError("loss_senspow contains missing values.")

    # Deterministic ordering ensures that all three representation jobs receive
    # the same run sequence before their representation-specific encoding.
    runs = runs.sort_values(["hash", "run_id"], kind="mergesort").reset_index(drop=True)

    print("=" * 80)
    print("SOURCE DATA READY")
    print("=" * 80)
    print(f"Aligned 3 x 3 UIFO runs : {len(runs):,}")
    print(f"Distinct exact topologies: {runs['setup_graph'].nunique():,}")
    print("Fractional-run blacklist : disabled")
    print("Top-five topology filter : disabled")
    print("=" * 80)
    return runs


# ------------------------------------------------------------------------------
# Thesis-specific filtering retained only as documentation
# ------------------------------------------------------------------------------
#
# Fractional-run blacklist
# ------------------------
# The thesis pipeline removed project-specific run keys stored in a private JSON
# blacklist. That file is intentionally not part of the public repository.
#
# Top-five runs per exact topology
# --------------------------------
# A separate robustness analysis retained the five lowest-loss runs for every
# unchanged raw setup_graph. The corresponding operation was:
#
# runs = runs.sort_values(
#     ["loss_senspow", "hash", "run_id"],
#     ascending=[True, True, True],
#     kind="mergesort",
# )
# runs = (
#     runs.groupby("setup_graph", sort=False, group_keys=False)
#     .head(5)
#     .reset_index(drop=True)
# )
#
# Neither operation is required to understand or reproduce the construction of
# the Flat, Grid, and Aliased feature representations.
# ------------------------------------------------------------------------------


# ==============================================================================
# 6. OUTPUT MANAGEMENT
# ==============================================================================

def output_paths(output_dir: Path, representation: str) -> dict[str, Path]:
    """Return the three mutually dependent output paths for one representation."""
    prefix = f"uifo_{representation}"
    return {
        "matrix": output_dir / f"{prefix}_matrix.npy",
        "vocab": output_dir / f"{prefix}_vocab.json",
        "index": output_dir / f"{prefix}_index.parquet",
    }


def save_compact_metadata(runs: pd.DataFrame, output_dir: Path) -> Path:
    """Save aligned run metadata and sensitivity curves without raw graph JSON."""
    graph_hash_by_value = {
        graph: hashlib.sha256(str(graph).encode("utf-8")).hexdigest()
        for graph in runs["setup_graph"].drop_duplicates()
    }

    sensitivity_columns = [
        column for column in runs.columns if column.startswith("sens_")
    ]
    metadata_columns = [
        "hash",
        "run_id",
        "loss_senspow",
        *sensitivity_columns,
    ]
    metadata = runs.loc[:, metadata_columns].copy()
    metadata.insert(
        2,
        "exact_setup_graph_sha256",
        runs["setup_graph"].map(graph_hash_by_value).to_numpy(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "uifo_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False, compression="zstd")

    print(f"Shared metadata file : {metadata_path}")
    print(f"Metadata rows        : {len(metadata):,}")
    print(f"Sensitivity columns  : {len(sensitivity_columns)}")
    return metadata_path


# ==============================================================================
# 7. REPRESENTATION BUILDING
# ==============================================================================

EXTRACTORS = {
    "flat": FlatNonlinearBS4Extractor,
    "grid": GridFeatureExtractor,
    "aliased": AliasedFeatureExtractor,
}


def build_representation(
    representation: str,
    runs: pd.DataFrame,
    h5_files: list[Path],
    output_dir: Path,
) -> None:
    """Fit one vocabulary, encode all runs, validate, and save its artifacts."""
    paths = output_paths(output_dir, representation)
    extractor = EXTRACTORS[representation]()

    print("\n" + "=" * 80)
    print(f"BUILDING {representation.upper()} REPRESENTATION")
    print("=" * 80)
    print(f"Runs: {len(runs):,}")

    print(f"[{representation.upper()} 1/3] Fitting vocabulary")
    extractor.fit(runs, h5_files)

    print(f"[{representation.upper()} 2/3] Encoding feature matrix")
    matrix = extractor.transform(runs, h5_files)

    print(f"[{representation.upper()} 3/3] Saving matrix, vocabulary, and index")
    extractor.save_assets(
        matrix,
        paths["matrix"],
        paths["vocab"],
        runs,
        paths["index"],
    )

    del matrix, extractor
    gc.collect()


# ==============================================================================
# 8. COMMAND-LINE INTERFACE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the unified extraction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the Flat, Grid, and Aliased UIFO configuration "
            "representations used by the structured multimodal VAE."
        )
    )
    parser.add_argument(
        "--representation",
        choices=["flat", "grid", "aliased", "all"],
        default="all",
        help="Representation to build. The default builds all three sequentially.",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=DEFAULT_H5_DIR,
        help="Directory containing the run-level HDF5 files.",
    )
    parser.add_argument(
        "--lightweight",
        type=Path,
        default=DEFAULT_LIGHTWEIGHT,
        help="Lightweight Parquet table containing run metadata.",
    )
    parser.add_argument(
        "--heavyweight",
        type=Path,
        default=DEFAULT_HEAVYWEIGHT,
        help="Heavyweight Parquet table containing sensitivity data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for matrices, vocabularies, indices, and shared metadata.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of deterministically ordered runs for a quick test.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write only the shared metadata table and skip feature extraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the complete public extraction workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_source_runs(args.lightweight, args.heavyweight)

    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be a positive integer.")
        runs = runs.head(args.sample).reset_index(drop=True)
        print(f"Sample mode enabled: processing {len(runs):,} runs.")

    save_compact_metadata(runs, args.output_dir)

    if args.metadata_only:
        print("Metadata-only mode complete.")
        return

    h5_files = sorted(args.h5_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {args.h5_dir.resolve()}")

    representations = (
        ["flat", "grid", "aliased"]
        if args.representation == "all"
        else [args.representation]
    )

    for representation in representations:
        build_representation(
            representation=representation,
            runs=runs,
            h5_files=h5_files,
            output_dir=args.output_dir,
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED REPRESENTATIONS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
