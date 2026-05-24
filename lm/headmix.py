"""Head-output mixing modules for MHA head communication experiments."""

import torch
import torch.nn as nn

from isohc.projection import construct_orthogonal_complement, iso_ns_project
from .mixing import MHCMixing


class HeadOutputMixing(nn.Module):
    """Mix attention head outputs before concat/projection.

    Input and output shape: (B, T, H, Dh).  The IsoHC variant preserves the
    average over heads and the full energy in the head-complement subspace.
    """

    def __init__(
        self,
        num_heads,
        mixing_type="identity",
        ns_steps=5,
        sinkhorn_iters=10,
        init_scale=0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mixing_type = mixing_type
        self.ns_steps = ns_steps

        if mixing_type in (None, "none", "identity", "baseline"):
            self.register_buffer("_eye", torch.eye(num_heads, dtype=torch.float32))
        elif mixing_type == "unconstrained":
            self.H_raw = nn.Parameter(
                torch.eye(num_heads) + torch.randn(num_heads, num_heads) * init_scale
            )
        elif mixing_type == "isohc":
            self.H_raw = nn.Parameter(
                torch.eye(num_heads) + torch.randn(num_heads, num_heads) * init_scale
            )
            U = construct_orthogonal_complement(num_heads, device="cpu", dtype=torch.float32)
            self.register_buffer("U", U)
        elif mixing_type == "fixed-random-iso":
            raw = torch.eye(num_heads) + torch.randn(num_heads, num_heads) * init_scale
            U = construct_orthogonal_complement(num_heads, device="cpu", dtype=torch.float32)
            H = iso_ns_project(raw, U=U, steps=ns_steps, use_svd=True)
            self.register_buffer("H_fixed", H.detach().to(dtype=torch.float32))
        elif mixing_type in ("birkhoff", "mhc"):
            self.mhc = MHCMixing(
                num_heads,
                sinkhorn_iters=sinkhorn_iters,
                diag_bias=4.0,
                noise_std=init_scale,
            )
        else:
            raise ValueError(f"Unknown head mixing type: {mixing_type}")

    def matrix(self):
        """Return the current head mixing matrix H with shape (H, H)."""
        if self.mixing_type in (None, "none", "identity", "baseline"):
            return self._eye
        if self.mixing_type == "unconstrained":
            return self.H_raw
        if self.mixing_type == "isohc":
            return iso_ns_project(
                self.H_raw,
                U=self.U,
                steps=self.ns_steps,
                use_svd=False,
                svd_fallback=True,
                fallback_tolerance=1e-6,
            )
        if self.mixing_type == "fixed-random-iso":
            return self.H_fixed
        if self.mixing_type in ("birkhoff", "mhc"):
            return self.mhc()
        raise AssertionError("unreachable")

    def forward(self, x, return_stats=False):
        """Apply head mixing to x.

        Args:
            x: Tensor with shape (B, T, H, Dh).
            return_stats: if True, also return structure-preservation metrics.
        """
        H = self.matrix().to(device=x.device, dtype=torch.float32)
        x_float = x.float()
        y = torch.einsum("ij,btjd->btid", H, x_float).to(dtype=x.dtype)

        if not return_stats:
            return y

        stats = head_mix_stats(x_float, y.float())
        return y, stats


def head_mix_stats(x, y, eps=1e-12):
    """Measure mean drift and complement-energy ratio for mixed head outputs."""
    x_mean = x.mean(dim=2)
    y_mean = y.mean(dim=2)
    x_perp = x - x_mean.unsqueeze(2)
    y_perp = y - y_mean.unsqueeze(2)

    mean_drift = torch.norm(y_mean - x_mean) / (torch.norm(x_mean) + eps)
    complement_energy_ratio = (
        y_perp.norm().square() / (x_perp.norm().square() + eps)
    )
    head_vectors = y.permute(2, 0, 1, 3).reshape(y.shape[2], -1)
    head_norm = head_vectors / (head_vectors.norm(dim=1, keepdim=True) + eps)
    cosine = head_norm @ head_norm.T
    mask = ~torch.eye(y.shape[2], dtype=torch.bool, device=y.device)

    gram = (head_vectors @ head_vectors.T) / max(y.shape[0] * y.shape[1] * y.shape[3], 1)
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    probs = eigvals / (eigvals.sum() + eps)
    effective_rank = torch.exp(-(probs * torch.log(probs + eps)).sum())

    return {
        "mean_drift": mean_drift.item(),
        "complement_energy_ratio": complement_energy_ratio.item(),
        "head_offdiag_cosine": cosine[mask].mean().item(),
        "head_effective_rank": effective_rank.item(),
    }
