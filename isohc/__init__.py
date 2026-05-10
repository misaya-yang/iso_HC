"""IsoHC: Isotropic Hypercube Connectivity for Deep Learning."""

from .projection import iso_ns_project, newton_schulz_polar, polar_decomposition_svd, construct_orthogonal_complement
from .layers import IsoHCResidualMixing, RMSNorm, CausalSelfAttention, MLP
from .transformer import IsoHCTransformer, BaselineTransformer, UnconstrainedHCTransformer

__all__ = [
    "iso_ns_project",
    "newton_schulz_polar",
    "polar_decomposition_svd",
    "construct_orthogonal_complement",
    "IsoHCResidualMixing",
    "RMSNorm",
    "CausalSelfAttention",
    "MLP",
    "IsoHCTransformer",
    "BaselineTransformer",
    "UnconstrainedHCTransformer",
]
