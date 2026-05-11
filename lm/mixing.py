"""Stream mixing modules for multi-stream HC Transformer.

Mixing types:
  - identity: H = I (no mixing)
  - unconstrained: H = A (free matrix)
  - orthogonal: H^T H = I (plain orthogonal, no fixed-vector constraint)
  - isohc: H^T H = I, H @ 1 = 1 (fixed-vector isometry)
  - mhc: H @ 1 = 1, 1^T H = 1^T, H >= 0 (doubly stochastic via Sinkhorn)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isohc.projection import iso_ns_project, construct_orthogonal_complement


class StreamMixing(nn.Module):
    """Base class for stream mixing. Returns H of shape (s, s)."""

    def __init__(self, n_streams):
        super().__init__()
        self.n_streams = n_streams

    def forward(self):
        """Return mixing matrix H: (s, s)."""
        raise NotImplementedError

    def get_diagnostics(self):
        """Return dict of diagnostic metrics."""
        return {}


class IdentityMixing(StreamMixing):
    """Identity mixing: H = I."""

    def forward(self):
        return torch.eye(
            self.n_streams,
            device=next(self.parameters()).device if list(self.parameters()) else 'cpu',
            dtype=torch.float32
        )

    def get_diagnostics(self):
        return {'orth_error': 0.0, 'fix_error': 0.0}


class UnconstrainedMixing(StreamMixing):
    """Unconstrained learnable mixing: H = A."""

    def __init__(self, n_streams, init_scale=0.01):
        super().__init__(n_streams)
        self.H_raw = nn.Parameter(
            torch.eye(n_streams) + torch.randn(n_streams, n_streams) * init_scale
        )

    def forward(self):
        return self.H_raw

    def get_diagnostics(self):
        H = self.H_raw.detach()
        n = self.n_streams
        ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
        orth_error = torch.norm(H.T @ H - torch.eye(n, device=H.device), p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()
        return {'orth_error': orth_error, 'fix_error': fix_error}


class OrthogonalMixing(StreamMixing):
    """Plain orthogonal mixing: H^T H = I (no fixed-vector constraint)."""

    def __init__(self, n_streams, init_scale=0.01):
        super().__init__(n_streams)
        self.H_raw = nn.Parameter(
            torch.eye(n_streams) + torch.randn(n_streams, n_streams) * init_scale
        )

    def forward(self):
        # Polar decomposition via SVD
        U, s, Vh = torch.linalg.svd(self.H_raw, full_matrices=False)
        return U @ Vh

    def get_diagnostics(self):
        H = self.forward().detach()
        n = self.n_streams
        orth_error = torch.norm(H.T @ H - torch.eye(n, device=H.device), p='fro').item()
        ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
        fix_error = torch.norm(H @ ones - ones, p=2).item()
        return {'orth_error': orth_error, 'fix_error': fix_error}


class IsoHCMixing(StreamMixing):
    """IsoHC mixing: H^T H = I, H @ 1 = 1 (fixed-vector isometry).

    Uses Newton-Schulz polar decomposition with optional SVD fallback.
    Projection runs in fp32 for numerical stability (bf16 causes mean drift).
    """

    def __init__(self, n_streams, ns_steps=5, init_scale=0.01,
                 use_svd=False, svd_fallback=True, fallback_tol=1e-3):
        super().__init__(n_streams)
        self.ns_steps = ns_steps
        self.use_svd = use_svd
        self.svd_fallback = svd_fallback
        self.fallback_tol = fallback_tol

        # Raw parameter: near identity
        self.H_raw = nn.Parameter(
            torch.eye(n_streams) + torch.randn(n_streams, n_streams) * init_scale
        )

        # Precompute U for 1-vector orthogonal complement
        U = construct_orthogonal_complement(n_streams, device='cpu', dtype=torch.float32)
        self.register_buffer('U', U)

    def forward(self):
        return iso_ns_project(
            self.H_raw,
            U=self.U,
            steps=self.ns_steps,
            use_svd=self.use_svd,
            return_diagnostics=False,
            svd_fallback=self.svd_fallback,
            fallback_tolerance=self.fallback_tol,
        )

    def get_diagnostics(self):
        H = self.forward().detach()
        n = self.n_streams
        device = H.device
        ones = torch.ones(n, 1, device=device, dtype=torch.float32)
        I = torch.eye(n, device=device, dtype=torch.float32)

        orth_error = torch.norm(H.T @ H - I, p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()

        # Singular values on 1_perp
        U = self.U.to(device=device, dtype=torch.float32)
        A = U.T @ H @ U  # (n-1, n-1)
        s = torch.linalg.svdvals(A)

        return {
            'orth_error': orth_error,
            'fix_error': fix_error,
            'sv_min_1perp': s.min().item(),
            'sv_max_1perp': s.max().item(),
            'sv_mean_1perp': s.mean().item(),
        }


class MHCMixing(StreamMixing):
    """mHC mixing: doubly stochastic via log-Sinkhorn projection.

    H @ 1 = 1, 1^T H = 1^T, H >= 0.
    Initialized near-identity to avoid unfair diffusion bias.
    """

    def __init__(self, n_streams, sinkhorn_iters=10, temperature=1.0,
                 diag_bias=4.0, noise_std=0.01):
        super().__init__(n_streams)
        self.sinkhorn_iters = sinkhorn_iters
        self.temperature = temperature

        # Near-identity initialization: strong diagonal bias + small noise
        logits_init = torch.eye(n_streams) * diag_bias
        logits_init += torch.randn(n_streams, n_streams) * noise_std
        self.logits = nn.Parameter(logits_init)

    def sinkhorn(self, logits):
        """Log-space Sinkhorn: project logits to doubly-stochastic."""
        logP = logits / self.temperature
        for _ in range(self.sinkhorn_iters):
            # Row normalize
            logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)
            # Column normalize
            logP = logP - torch.logsumexp(logP, dim=0, keepdim=True)
        return torch.exp(logP)

    def forward(self):
        return self.sinkhorn(self.logits)

    def get_diagnostics(self):
        H = self.forward().detach()
        n = self.n_streams
        device = H.device
        ones = torch.ones(n, 1, device=device, dtype=torch.float32)

        row_sum_err = torch.norm(H.sum(dim=1, keepdim=True) - ones, p=2).item()
        col_sum_err = torch.norm(H.sum(dim=0, keepdim=True) - ones, p=2).item()

        # Check non-negativity
        neg_ratio = (H < 0).float().mean().item()

        # Entropy: higher = more uniform / more diffusion
        H_safe = H.clamp(min=1e-12)
        entropy = -(H_safe * torch.log(H_safe)).sum(dim=1).mean().item()
        max_entropy = math.log(n)
        normalized_entropy = entropy / max_entropy

        # Singular values on 1_perp (contraction indicator)
        e0 = ones / (n ** 0.5)
        P_perp = torch.eye(n, device=device) - e0 @ e0.T
        A = P_perp @ H @ P_perp
        s = torch.linalg.svdvals(A)
        # Exclude the zero singular value corresponding to 1-vector
        s_nonzero = s[s > 1e-6]

        return {
            'row_sum_err': row_sum_err,
            'col_sum_err': col_sum_err,
            'neg_ratio': neg_ratio,
            'entropy': normalized_entropy,
            'sv_min_1perp': s_nonzero.min().item() if len(s_nonzero) > 0 else 0.0,
            'sv_max_1perp': s_nonzero.max().item() if len(s_nonzero) > 0 else 0.0,
            'sv_mean_1perp': s_nonzero.mean().item() if len(s_nonzero) > 0 else 0.0,
        }


def create_mixing(n_streams, mixing_type, **kwargs):
    """Factory for creating mixing modules.

    Args:
        n_streams: number of streams
        mixing_type: 'identity', 'unconstrained', 'orthogonal', 'isohc', 'mhc'
        **kwargs: passed to mixing constructor
    """
    if mixing_type == 'identity':
        return IdentityMixing(n_streams)
    elif mixing_type == 'unconstrained':
        return UnconstrainedMixing(n_streams, init_scale=kwargs.get('init_scale', 0.01))
    elif mixing_type == 'orthogonal':
        return OrthogonalMixing(n_streams, init_scale=kwargs.get('init_scale', 0.01))
    elif mixing_type == 'isohc':
        return IsoHCMixing(
            n_streams,
            ns_steps=kwargs.get('ns_steps', 5),
            init_scale=kwargs.get('init_scale', 0.01),
            use_svd=kwargs.get('use_svd', False),
            svd_fallback=kwargs.get('svd_fallback', True),
            fallback_tol=kwargs.get('fallback_tol', 1e-3),
        )
    elif mixing_type == 'mhc':
        return MHCMixing(
            n_streams,
            sinkhorn_iters=kwargs.get('sinkhorn_iters', 10),
            temperature=kwargs.get('temperature', 1.0),
            diag_bias=kwargs.get('diag_bias', 4.0),
            noise_std=kwargs.get('noise_std', 0.01),
        )
    else:
        raise ValueError(f"Unknown mixing_type: {mixing_type}")
