"""IsoHC GNN: Fixed-Vector Isometric Operator for Graph Neural Networks."""

from .utils import (
    generate_sbm_graph,
    generate_node_features,
    normalize_adjacency,
    graph_propagation,
    split_data,
    compute_accuracy,
    compute_energy_ratio,
    compute_centered_variance,
    compute_dirichlet_energy,
    compute_pairwise_cosine,
    compute_invariant_error,
    compute_invariant_error_norm,
    compute_v_centered_variance,
)
from .projection import construct_orthogonal_complement_v, iso_ns_project_v, newton_schulz_polar_v
from .models import GCN, ResGCN, IsoStreamGCN, IsoResGCN

__all__ = [
    "generate_sbm_graph",
    "generate_node_features",
    "normalize_adjacency",
    "graph_propagation",
    "split_data",
    "compute_accuracy",
    "compute_energy_ratio",
    "compute_centered_variance",
    "compute_dirichlet_energy",
    "compute_pairwise_cosine",
    "compute_invariant_error",
    "compute_invariant_error_norm",
    "compute_v_centered_variance",
    "construct_orthogonal_complement_v",
    "iso_ns_project_v",
    "newton_schulz_polar_v",
    "GCN",
    "ResGCN",
    "IsoStreamGCN",
    "IsoResGCN",
]
