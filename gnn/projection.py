"""GNN-specific Iso-NS projection with fixed vector v = sqrt(d_tilde)."""

import torch
import torch.nn as nn


def construct_orthogonal_complement_v(n, v, device=None, dtype=torch.float32):
    """
    Construct orthonormal basis U for the orthogonal complement of v.

    Uses QR on [v | rand] with v as the *first column* (not normalised).
    After QR, Q[:, 0] is parallel to v and the remaining columns form the
    complement, giving U^T v = 0 up to QR round-off only.

    U satisfies:
      U^T U = I_{n-1}
      U^T v = 0   (up to QR round-off)

    Args:
        n: dimension
        v: (n,) fixed vector (need not be normalised)
        device, dtype: torch device/dtype

    Returns U of shape (n, n-1).
    """
    v = v.flatten()
    if device is not None:
        v = v.to(device=device, dtype=dtype)
    else:
        v = v.to(dtype=dtype)

    # Build [v, rand] and QR.  Q[:, 0] is parallel to v.
    M = torch.cat([v.unsqueeze(1), torch.randn(n, n - 1, device=v.device, dtype=dtype)], dim=1)
    Q, _ = torch.linalg.qr(M)

    # Ensure first column points in the same direction as v
    if (Q[:, 0] * v).sum() < 0:
        Q[:, 0] = -Q[:, 0]

    U = Q[:, 1:]  # (n, n-1)
    return U


def newton_schulz_polar_v(A, steps=10):
    """
    Newton-Schulz iteration for polar decomposition.

    Args:
        A: (m, m) matrix
        steps: number of iterations

    Returns:
        R: (m, m) approximately orthogonal factor
    """
    norm = torch.linalg.matrix_norm(A, ord=2)
    X = A / (norm + 1e-8)
    for _ in range(steps):
        XTX = X.T @ X
        X = 0.5 * X @ (3.0 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype) - XTX)
    return X


def polar_decomposition_svd_v(A):
    """Exact polar decomposition via SVD."""
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh


def iso_ns_project_v(H_raw, v, steps=10, use_svd=True, return_diagnostics=False, U=None):
    """
    Project H_raw onto M_v = {Q ∈ O(n) : Qv = v}.

    Algorithm:
      1. e0 = v / ||v||
      2. A = U^T H_raw U  (project to orthogonal complement)
      3. R = polar(A)
      4. Q = e0 e0^T + U R U^T

    Args:
        H_raw: (n, n) raw matrix
        v: (n,) fixed vector (need not be normalized)
        steps: NS iterations (when use_svd=False)
        use_svd: use exact SVD polar
        return_diagnostics: return orth_error, fix_error

    Returns:
        Q: (n, n) projected orthogonal matrix preserving v
    """
    n = H_raw.shape[0]
    device = H_raw.device
    dtype = H_raw.dtype

    # Use fp64 internally for numerical accuracy on large matrices (n ~ 512)
    # fp32 QR on 512×512 gives U^T v ~ 1e-4; fp64 gives ~ 1e-13.
    use_fp64 = True
    compute_dtype = torch.float64 if use_fp64 else dtype

    H_raw_c = H_raw.to(compute_dtype)
    v_c = v.to(compute_dtype)

    # Step 1: normalize fixed vector
    e0 = v_c / (torch.norm(v_c) + 1e-12)  # (n,)

    # Step 2: construct orthogonal complement (use precomputed U if provided)
    if U is None:
        U_comp = construct_orthogonal_complement_v(n, v_c, device=device, dtype=compute_dtype)
    else:
        U_comp = U.to(device=device, dtype=compute_dtype)

    # Step 3: project to subspace
    A = U_comp.T @ H_raw_c @ U_comp  # (n-1, n-1)

    # Step 4: polar decomposition
    if use_svd:
        R = polar_decomposition_svd_v(A)
    else:
        R = newton_schulz_polar_v(A, steps=steps)

    # Step 5: reconstruct
    e0_mat = e0.unsqueeze(1)  # (n, 1)
    Q = e0_mat @ e0_mat.T + U_comp @ R @ U_comp.T

    Q = Q.to(dtype)

    if return_diagnostics:
        v_col = v_c.unsqueeze(1)  # (n, 1)
        I = torch.eye(n, device=device, dtype=compute_dtype)
        orth_error = torch.norm(Q.T @ Q - I, p='fro').item()
        fix_error = torch.norm(Q.to(compute_dtype) @ v_col - v_col, p=2).item()
        return Q, orth_error, fix_error

    return Q


class IsoNodeProjection(nn.Module):
    """
    Learnable IsoNode projection for GNN.

    Projects a raw matrix to M_v where v is a fixed graph-dependent vector
    (e.g., sqrt of degree).
    """

    def __init__(self, n, v, ns_steps=10, init_identity=True, use_svd=False):
        super().__init__()
        self.n = n
        self.ns_steps = ns_steps
        self.use_svd = use_svd

        # Register fixed vector in fp64 for numerical accuracy
        v64 = v.double()
        self.register_buffer('v', v64)

        # Raw unconstrained parameter (kept in fp32 for training efficiency)
        if init_identity:
            self.H_raw = nn.Parameter(torch.eye(n, dtype=torch.float32))
        else:
            self.H_raw = nn.Parameter(torch.randn(n, n, dtype=torch.float32) * 0.01)
            self.H_raw.data += torch.eye(n, dtype=torch.float32)

        # Precompute U in fp64 for arithmetic-exact orthogonality.
        # fp32 QR on large n gives U^T v ~ 1e-4; fp64 gives ~ 1e-13.
        U = construct_orthogonal_complement_v(n, v64, device='cpu', dtype=torch.float64)
        self.register_buffer('U', U)

    def forward(self):
        """Return projected Q."""
        return iso_ns_project_v(self.H_raw, self.v, steps=self.ns_steps, use_svd=self.use_svd, U=self.U)

    def get_diagnostics(self):
        """Return projection error diagnostics."""
        Q = self.forward()
        v_col = self.v.to(Q.dtype).unsqueeze(1)
        I = torch.eye(self.n, device=Q.device, dtype=Q.dtype)
        orth_error = torch.norm(Q.T @ Q - I, p='fro').item()
        fix_error = torch.norm(Q @ v_col - v_col, p=2).item()
        return {
            'orth_error': orth_error,
            'fix_error': fix_error,
        }
