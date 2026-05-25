"""Posthoc analysis for multi-stream residual transport.

The core paper question is not only whether a mixer preserves the stream mean,
but what happens to the transported signal inside the mean-zero subspace.  This
module keeps that analysis small, explicit, and reusable from training runners
and checkpoint analysis scripts.
"""

import torch

from isohc.projection import construct_orthogonal_complement


def mean_zero_basis(n_streams, device=None, dtype=torch.float32):
    """Return an orthonormal basis U for the 1_perp stream subspace."""
    return construct_orthogonal_complement(
        n_streams,
        device=device or "cpu",
        dtype=dtype,
    )


def complement_spectrum(H, U=None):
    """Singular values of a stream mixer restricted to 1_perp."""
    H = H.detach().float()
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"Expected square H, got {tuple(H.shape)}")
    if U is None:
        U = mean_zero_basis(n, device=H.device, dtype=torch.float32)
    else:
        U = U.to(device=H.device, dtype=torch.float32)

    B = U.T @ H @ U
    s = torch.linalg.svdvals(B)
    return {
        "sv_min": s.min().item(),
        "sv_mean": s.mean().item(),
        "sv_max": s.max().item(),
        "identity_distance": torch.norm(
            B - torch.eye(n - 1, device=H.device, dtype=torch.float32),
            p="fro",
        ).item(),
    }


def _coerce_named_matrices(source):
    if hasattr(source, "get_named_mixing_matrices"):
        return source.get_named_mixing_matrices()
    if isinstance(source, (list, tuple)):
        matrices = []
        for index, item in enumerate(source):
            if isinstance(item, dict):
                matrices.append({
                    "branch": item.get("branch", "transport"),
                    "layer": item.get("layer", index),
                    "index": item.get("index", index),
                    "H": item["H"],
                })
            else:
                matrices.append({
                    "branch": "transport",
                    "layer": index,
                    "index": index,
                    "H": item,
                })
        return matrices
    raise TypeError("Expected a model with get_named_mixing_matrices() or a list of matrices.")


def collect_transport_report(source):
    """Collect per-step and prefix-product complement gain diagnostics.

    Returns a JSON-serializable dict with:
      - steps: single-mixer 1_perp singular values.
      - prefix: actual singular values of U^T H_k...H_1 U and the simpler
        product-of-single-step summaries.
    """
    matrices = _coerce_named_matrices(source)
    if not matrices:
        return {"steps": [], "prefix": []}

    first_H = matrices[0]["H"].detach().float()
    n = first_H.shape[0]
    U = mean_zero_basis(n, device=first_H.device, dtype=torch.float32)
    composite = torch.eye(n - 1, device=first_H.device, dtype=torch.float32)
    product_sv_min = 1.0
    product_sv_mean = 1.0
    product_sv_max = 1.0
    eps = 1e-12

    steps = []
    prefix = []
    for index, item in enumerate(matrices):
        H = item["H"].detach().float()
        B = U.T @ H @ U
        s = torch.linalg.svdvals(B)
        s_min = s.min().item()
        s_mean = s.mean().item()
        s_max = s.max().item()

        product_sv_min *= max(s_min, eps)
        product_sv_mean *= max(s_mean, eps)
        product_sv_max *= max(s_max, eps)
        composite = B @ composite
        c = torch.linalg.svdvals(composite)

        step = {
            "index": int(item.get("index", index)),
            "branch": item.get("branch", "transport"),
            "layer": int(item.get("layer", index)),
            "sv_min": s_min,
            "sv_mean": s_mean,
            "sv_max": s_max,
            "identity_distance": torch.norm(
                B - torch.eye(n - 1, device=H.device, dtype=torch.float32),
                p="fro",
            ).item(),
        }
        steps.append(step)
        prefix.append({
            "index": step["index"],
            "branch": step["branch"],
            "layer": step["layer"],
            "composite_sv_min": c.min().item(),
            "composite_sv_mean": c.mean().item(),
            "composite_sv_max": c.max().item(),
            "product_sv_min": product_sv_min,
            "product_sv_mean": product_sv_mean,
            "product_sv_max": product_sv_max,
        })

    return {
        "n_streams": n,
        "num_transports": len(matrices),
        "steps": steps,
        "prefix": prefix,
        "final": prefix[-1],
    }
