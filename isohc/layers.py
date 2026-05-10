"""Neural network layers for IsoHC experiments."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import iso_ns_project, construct_orthogonal_complement


class RMSNorm(nn.Module):
    """RMSNorm as used in modern LLMs (Llama, etc.)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        Returns: (batch, seq_len, hidden_dim)
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)

        return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = int(hidden_dim * mlp_ratio)

        self.gate_proj = nn.Linear(hidden_dim, self.mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, self.mlp_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU-like: gate(x) * up(x)
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class IsoHCResidualMixing(nn.Module):
    """
    IsoHC residual mixing layer.

    Projects a learnable raw matrix to M_iso and applies it as residual mixing:
      X_out = H X_in
    where H ∈ M_iso, i.e., H^T H = I and H @ 1 = 1.

    For multiple streams (n_streams > 1), each stream gets its own mixing matrix.
    """

    def __init__(self, n_streams, ns_steps=10, init_identity=True):
        super().__init__()
        self.n_streams = n_streams
        self.ns_steps = ns_steps

        # One raw parameter per stream
        if init_identity:
            H_raw_init = torch.eye(n_streams).unsqueeze(0).repeat(n_streams, 1, 1)
        else:
            H_raw_init = torch.randn(n_streams, n_streams, n_streams) * 0.01
            # Add identity for each
            for i in range(n_streams):
                H_raw_init[i] = torch.eye(n_streams) + H_raw_init[i]

        self.H_raw = nn.Parameter(H_raw_init)

        # Precompute U
        U = construct_orthogonal_complement(n_streams, device='cpu', dtype=torch.float32)
        self.register_buffer('U', U)

    def get_H(self, stream_idx=None):
        """Get projected H for given stream, or all streams."""
        if stream_idx is not None:
            return iso_ns_project(self.H_raw[stream_idx], U=self.U, steps=self.ns_steps)
        else:
            # Project all
            H_list = []
            for i in range(self.n_streams):
                H_list.append(iso_ns_project(self.H_raw[i], U=self.U, steps=self.ns_steps))
            return torch.stack(H_list, dim=0)

    def forward(self, x):
        """
        Apply IsoHC residual mixing.

        x: (batch, seq_len, n_streams, feature_dim_per_stream)
        Returns: same shape
        """
        B, T, n, d = x.shape
        assert n == self.n_streams

        # Reshape to apply per-feature-dim mixing
        # x: (B, T, n, d) -> (B*T*d, n)
        x_flat = x.permute(0, 1, 3, 2).reshape(-1, n)  # (B*T*d, n)

        # For each stream, compute H and apply mixing
        # Actually, we want: for each position and feature dim,
        #   out[j] = H[i] @ x[j] where i is the feature dim index
        # But here all streams share the same feature dim.

        # Simpler approach: treat streams as independent channels
        # For each stream i, H[i] mixes across streams
        # We have n_streams mixing matrices (one per "feature group")

        # Actually, re-read the paper: there's one H per layer, not per stream
        # Let me reconsider...

        # Standard interpretation: single H matrix for all streams
        H = iso_ns_project(self.H_raw[0], U=self.U, steps=self.ns_steps)  # (n, n)

        # Apply: x_out[b,t,:,d] = H @ x[b,t,:,d]
        # x: (B, T, n, d)
        # H: (n, n)
        # output: (B, T, n, d)
        x_out = torch.einsum('nm,btmd->btnd', H, x)

        return x_out

    def get_diagnostics(self):
        """Return projection diagnostics for all stream mixing matrices."""
        H = self.get_H(0)  # Currently only one H
        n = self.n_streams
        device = H.device
        ones = torch.ones(n, 1, device=device, dtype=torch.float32)

        orth_error = torch.norm(H.T @ H - torch.eye(n, device=device, dtype=torch.float32), p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()

        # Energy preservation: ||HX|| / ||X|| for random X
        X_test = torch.randn(n, 128, device=device, dtype=torch.float32)
        energy_ratio = (torch.norm(H @ X_test, p='fro') / torch.norm(X_test, p='fro')).item()

        return {
            'orth_error': orth_error,
            'fix_error': fix_error,
            'energy_ratio': energy_ratio,
        }


class UnconstrainedHCResidualMixing(nn.Module):
    """Unconstrained residual mixing for comparison."""

    def __init__(self, n_streams, init_scale=0.01):
        super().__init__()
        self.n_streams = n_streams
        self.H = nn.Parameter(torch.eye(n_streams) + torch.randn(n_streams, n_streams) * init_scale)

    def forward(self, x):
        """x: (batch, seq_len, n_streams, feature_dim_per_stream)"""
        return torch.einsum('nm,btmd->btnd', self.H, x)


class MHCLiteResidualMixing(nn.Module):
    """
    mHC-lite: Sinkhorn doubly-stochastic projection for comparison.

    Projects to doubly-stochastic matrices (row and column sums = 1).
    """

    def __init__(self, n_streams, sinkhorn_iters=20, init_identity=True):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        if init_identity:
            self.H_raw = nn.Parameter(torch.eye(n_streams) + torch.randn(n_streams, n_streams) * 0.01)
        else:
            self.H_raw = nn.Parameter(torch.randn(n_streams, n_streams) * 0.01)

    def sinkhorn(self, M):
        """Normalize M to doubly-stochastic via Sinkhorn iteration."""
        A = M.abs() + 1e-8
        for _ in range(self.sinkhorn_iters):
            # Row normalize
            A = A / A.sum(dim=1, keepdim=True)
            # Column normalize
            A = A / A.sum(dim=0, keepdim=True)
        return A

    def forward(self, x):
        H = self.sinkhorn(self.H_raw)
        return torch.einsum('nm,btmd->btnd', H, x)
