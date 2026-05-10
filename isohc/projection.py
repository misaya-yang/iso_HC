"""Iso-NS projection: project arbitrary matrix to M_iso manifold."""

import torch
import torch.nn as nn


# Global cache for orthogonal complement bases
_U_CACHE = {}


def get_cached_U(n, device, dtype=torch.float32):
    """Get cached orthogonal complement basis U for dimension n."""
    key = (n, device, dtype)
    if key not in _U_CACHE:
        _U_CACHE[key] = construct_orthogonal_complement(n, device=device, dtype=dtype)
    return _U_CACHE[key]


def construct_orthogonal_complement(n, device=None, dtype=torch.float32):
    """
    Construct orthonormal basis U for the orthogonal complement of e0 = 1/sqrt(n).

    U satisfies:
      U^T U = I_{n-1}
      U^T e0 = 0

    Returns U of shape (n, n-1).
    """
    e0 = torch.ones(n, 1, device=device, dtype=dtype) / (n ** 0.5)

    # Deterministic construction: concatenate e0 with standard basis columns
    eye_rest = torch.eye(n, device=device, dtype=dtype)[:, 1:]
    M = torch.cat([e0, eye_rest], dim=1)  # (n, n)

    Q, _ = torch.linalg.qr(M)

    # Ensure first column aligns with e0 (same direction, not opposite)
    if (Q[:, 0:1] * e0).sum() < 0:
        Q[:, 0] = -Q[:, 0]

    U = Q[:, 1:]  # (n, n-1)
    return U


def newton_schulz_polar(A, steps=10):
    """
    Newton-Schulz iteration for polar decomposition approximation.

    Given A (..., m, m), returns R_K ≈ polar(A) where A = R P.

    The iteration:
      X_0 = A / ||A||_2
      X_{k+1} = X_k (3I - X_k^T X_k) / 2

    For matrices near-orthogonal (e.g., initialized from identity),
    this converges very quickly. The backward pass is fully
    differentiable and numerically stable.

    Args:
        A: (..., m, m) matrix
        steps: number of NS iterations (default 10)

    Returns:
        R: (..., m, m) approximately orthogonal factor
    """
    if A.dim() == 2:
        norm = torch.linalg.matrix_norm(A, ord=2)
        X = A / (norm + 1e-8)
        for _ in range(steps):
            XTX = X.T @ X
            X = 0.5 * X @ (3.0 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype) - XTX)
        return X
    else:
        # Batch mode
        norm = torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1), keepdim=True)
        X = A / (norm + 1e-8)
        I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
        for _ in range(steps):
            XTX = torch.matmul(X.mT, X)
            X = 0.5 * torch.matmul(X, 3.0 * I - XTX)
        return X


def polar_decomposition_svd(A):
    """
    Exact polar decomposition via SVD.

    Given A (m, m), returns R = polar(A) where A = R P.
    Uses SVD: A = U Σ V^T, then R = U V^T.

    This is the most accurate method but SVD backward can be
    numerically unstable for ill-conditioned matrices. Use this
    for forward-only evaluation (e.g., sanity checks).

    Args:
        A: (m, m) matrix

    Returns:
        R: (m, m) orthogonal factor
    """
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh


def iso_ns_project(
    H_raw,
    U=None,
    steps=10,
    use_svd=False,
    return_diagnostics=False,
    svd_fallback=True,
    fallback_tolerance=1e-6,
):
    """
    Project arbitrary matrix H_raw onto M_iso manifold.

    M_iso = {H ∈ R^{n×n} : H^T H = I, H @ 1 = 1}

    Algorithm:
      1. e0 = 1 / sqrt(n)
      2. A = U^T H_raw U  (project to orthogonal complement subspace)
      3. R = polar(A)     (exact SVD or NS approximation)
      4. H = e0 e0^T + U R U^T  (reconstruct)

    Args:
        H_raw: (n, n) raw unconstrained matrix
        U: optional precomputed orthogonal complement basis (n, n-1)
        steps: number of Newton-Schulz iterations (when use_svd=False)
        use_svd: if True, use exact SVD polar (forward-only, highest accuracy)
        return_diagnostics: if True, also return orth_error, fix_error
        svd_fallback: if True, use exact polar when K-step NS is not accurate enough
        fallback_tolerance: Frobenius orthogonality tolerance for the fallback

    Returns:
        H: (n, n) projected matrix
        Optionally (orth_error, fix_error) if return_diagnostics=True
    """
    n = H_raw.shape[-1]
    device = H_raw.device
    dtype = H_raw.dtype

    # Internal computation uses float64 for the tiny stream matrices in Stage 1.
    # Returning to the caller's dtype preserves the model interface while keeping
    # fixed-vector drift small across hundreds of residual-only layers.
    projection_dtype = torch.float64
    H_raw_f = H_raw.to(projection_dtype)

    if U is None:
        U = get_cached_U(n, device, dtype=projection_dtype)

    # Ensure U is on the right device and kept in the projection dtype.
    if U.device != device or U.dtype != projection_dtype:
        U = U.to(device=device, dtype=projection_dtype)

    # Step 2: project to orthogonal complement subspace
    A = U.T @ H_raw_f @ U  # (n-1, n-1)

    # Step 3: polar decomposition
    if use_svd:
        R = polar_decomposition_svd(A)
    else:
        R = newton_schulz_polar(A, steps=steps)
        if svd_fallback and steps >= 5:
            I_sub = torch.eye(R.shape[-1], device=device, dtype=projection_dtype)
            orth_error = torch.norm(R.T @ R - I_sub, p='fro')
            orth_error_value = orth_error.item()
            if (not torch.isfinite(orth_error).item()) or orth_error_value > fallback_tolerance:
                R = polar_decomposition_svd(A)

    # Step 4: reconstruct
    e0 = torch.ones(n, 1, device=device, dtype=projection_dtype) / (n ** 0.5)
    H = e0 @ e0.T + U @ R @ U.T

    H = H.to(dtype)

    if return_diagnostics:
        ones = torch.ones(n, 1, device=device, dtype=torch.float32)
        orth_error = torch.norm(H.T @ H - torch.eye(n, device=device, dtype=torch.float32), p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()
        return H, orth_error, fix_error

    return H


class IsoNSProject(nn.Module):
    """
    Learnable Iso-NS projection layer.

    Maintains a raw unconstrained parameter and projects it to M_iso
    during forward pass. Gradient flows through the projection.
    """

    def __init__(self, n, ns_steps=10, init_identity=True):
        super().__init__()
        self.n = n
        self.ns_steps = ns_steps

        # Raw unconstrained parameter
        if init_identity:
            self.H_raw = nn.Parameter(torch.eye(n))
        else:
            self.H_raw = nn.Parameter(torch.randn(n, n) * 0.01)

        # Precompute and register U as buffer (non-trainable)
        U = construct_orthogonal_complement(n, device='cpu', dtype=torch.float32)
        self.register_buffer('U', U)

    def forward(self):
        """Return projected H."""
        return iso_ns_project(self.H_raw, U=self.U, steps=self.ns_steps, use_svd=False)

    def get_diagnostics(self):
        """Return projection error diagnostics."""
        H = self.forward()
        n = self.n
        ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
        orth_error = torch.norm(H.T @ H - torch.eye(n, device=H.device, dtype=torch.float32), p='fro').item()
        fix_error = torch.norm(H @ ones - ones, p=2).item()
        return {
            'orth_error': orth_error,
            'fix_error': fix_error,
        }
