"""Posthoc analysis for multi-stream residual transport.

The core paper question is not only whether a mixer preserves the stream mean,
but what happens to the transported signal inside the mean-zero subspace.  This
module keeps that analysis small, explicit, and reusable from training runners
and checkpoint analysis scripts.
"""

import math

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
    """Singular values of the compression U^T H U to 1_perp.

    This is an invariant-subspace restriction only if H maps 1_perp into itself.
    Otherwise the compression omits signal that H transports into the mean.
    """
    H = H.detach().float()
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"Expected square H, got {tuple(H.shape)}")
    if U is None:
        U = mean_zero_basis(n, device=H.device, dtype=torch.float32)
    else:
        U = U.to(device=H.device, dtype=torch.float32)

    ones = torch.ones(n, 1, device=H.device, dtype=torch.float32)
    v = ones / (n ** 0.5)
    B = U.T @ H @ U
    s = torch.linalg.svdvals(B)
    return {
        "singular_values": s.cpu().tolist(),
        "sv_min": s.min().item(),
        "sv_mean": s.mean().item(),
        "sv_max": s.max().item(),
        "identity_distance": torch.norm(
            B - torch.eye(n - 1, device=H.device, dtype=torch.float32),
            p="fro",
        ).item(),
        "mean_to_perp_leakage": torch.norm(U.T @ H @ v).item(),
        "perp_to_mean_leakage": torch.norm(v.T @ H @ U).item(),
        "row_sum_error": torch.norm(H @ ones - ones).item(),
        "col_sum_error": torch.norm(ones.T @ H - ones.T).item(),
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
      - steps: singular values of each compression B_i = U^T H_i U, plus
        mean_preservation_error = ||H_i^T e0 - e0||_2 for unit mean vector e0.
      - prefix: composite_sv_* describes U^T (H_k...H_1) U, while
        projected_step_product_sv_* describes B_k...B_1.  These agree when
        each H_i preserves 1_perp; otherwise only the full product retains
        excursions through the mean direction.

    The historical product_sv_* fields multiply per-step singular-value
    summaries, with each factor floored at 1e-12.  They are diagnostics, not
    general bounds on composite_sv_*; even without the floor the mean product
    is not a spectral bound.  Min/max product bounds apply to B_k...B_1 before
    flooring, and also describe the full compression when 1_perp is invariant.

    Schema version 2 corrects composite_sv_*: older reports used B_k...B_1
    under that name and must be recomputed for non-invariant mixers.
    """
    matrices = [
        {**item, "H": item["H"].detach().cpu()}
        for item in _coerce_named_matrices(source)
    ]
    metadata = {
        "schema_version": 2,
        "operators": {
            "composite": "U^T (H_k ... H_1) U",
            "projected_step_product": "(U^T H_k U) ... (U^T H_1 U)",
        },
        "product_sv_floor": 1e-12,
    }
    if not matrices:
        return {**metadata, "steps": [], "prefix": []}

    first_H = matrices[0]["H"].detach().float()
    n = first_H.shape[0]
    U = mean_zero_basis(n, device=first_H.device, dtype=torch.float32)
    e0 = torch.ones(n, device=first_H.device, dtype=torch.float64) / n ** 0.5
    U64 = U.double()
    composite = torch.eye(n, device=first_H.device, dtype=torch.float64)
    projected_step_product = torch.eye(
        n - 1, device=first_H.device, dtype=torch.float64
    )
    product_sv_min = 1.0
    product_sv_mean = 1.0
    product_sv_max = 1.0
    eps = metadata["product_sv_floor"]

    steps = []
    prefix = []
    for index, item in enumerate(matrices):
        H = item["H"].detach().float()
        stats = complement_spectrum(H, U)
        B = U.T @ H @ U
        s_min = stats["sv_min"]
        s_mean = stats["sv_mean"]
        s_max = stats["sv_max"]

        product_sv_min *= max(s_min, eps)
        product_sv_mean *= max(s_mean, eps)
        product_sv_max *= max(s_max, eps)
        composite = H.double() @ composite
        c = torch.linalg.svdvals(U64.T @ composite @ U64)
        projected_step_product = B.double() @ projected_step_product
        projected_s = torch.linalg.svdvals(projected_step_product)

        step = {
            "index": int(item.get("index", index)),
            "branch": item.get("branch", "transport"),
            "layer": int(item.get("layer", index)),
            **stats,
            "mean_preservation_error": torch.norm(
                H.double().T @ e0 - e0
            ).item(),
        }
        steps.append(step)
        prefix.append({
            "index": step["index"],
            "branch": step["branch"],
            "layer": step["layer"],
            "composite_singular_values": c.cpu().tolist(),
            "composite_sv_min": c.min().item(),
            "composite_sv_mean": c.mean().item(),
            "composite_sv_max": c.max().item(),
            "projected_step_product_singular_values": projected_s.cpu().tolist(),
            "projected_step_product_sv_min": projected_s.min().item(),
            "projected_step_product_sv_mean": projected_s.mean().item(),
            "projected_step_product_sv_max": projected_s.max().item(),
            "product_sv_min": product_sv_min,
            "product_sv_mean": product_sv_mean,
            "product_sv_max": product_sv_max,
        })

    final = dict(prefix[-1])
    for key in (
        "mean_to_perp_leakage",
        "perp_to_mean_leakage",
        "row_sum_error",
        "col_sum_error",
        "mean_preservation_error",
    ):
        final[f"{key}_max"] = max(step[key] for step in steps)

    return {
        **metadata,
        "n_streams": n,
        "num_transports": len(matrices),
        "steps": steps,
        "prefix": prefix,
        "final": final,
    }


def _gramian_summary(matrix, eps):
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
    if not eigenvalues.numel():
        return {
            "eigenvalues": [],
            "rank": 0,
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "logdet_regularized": 0.0,
        }
    largest = eigenvalues.max()
    tolerance = max(float(eps), largest.item() * 1e-8)
    return {
        "eigenvalues": eigenvalues.cpu().tolist(),
        "rank": int((eigenvalues > tolerance).sum().item()),
        "lambda_min": eigenvalues.min().item(),
        "lambda_max": largest.item(),
        "logdet_regularized": torch.log(eigenvalues + eps).sum().item(),
    }


@torch.no_grad()
def collect_accessibility_report(model, eps=1e-12):
    """Linear transport-and-gate accessibility proxy for an HC model.

    The Transformer updates are nonlinear, so these are structural stream-space
    Gramians, not full-model controllability or observability claims.
    """
    required = (
        "get_named_mixing_matrices",
        "attn_readout_weights",
        "attn_injection_weights",
        "mlp_readout_weights",
        "mlp_injection_weights",
        "readout_final",
    )
    if any(not hasattr(model, name) for name in required):
        raise TypeError("Expected a TwoBranchHCTransformer-style model")

    n = int(model.n_streams)
    complement_dim = max(0, n - 1)
    device = torch.device("cpu")
    U = mean_zero_basis(n, device=device, dtype=torch.float64)
    matrices = {
        (item["branch"], int(item["layer"])): (
            item["H"].detach().cpu().double()
        )
        for item in model.get_named_mixing_matrices()
    }

    def stream_vector(weights, lambda_param):
        weights = weights.detach().cpu().double()
        scale = lambda_param.detach().cpu().double()
        return torch.ones(n, dtype=torch.float64) + scale * (
            weights - weights.mean()
        )

    steps = []
    Bs = []
    alphas = []
    betas = []

    for layer in range(model.num_layers):
        for branch, readouts, injections in (
            ("attn", model.attn_readout_weights, model.attn_injection_weights),
            ("mlp", model.mlp_readout_weights, model.mlp_injection_weights),
        ):
            a = stream_vector(readouts[layer], model.readout_lambda)
            b = stream_vector(injections[layer], model.injection_lambda)
            a_perp = U.T @ (a - a.mean())
            b_perp = U.T @ (b - b.mean())
            alpha = a_perp / n
            beta = b_perp
            B = U.T @ matrices[(branch, layer)] @ U
            Bs.append(B)
            alphas.append(alpha)
            betas.append(beta)
            steps.append({
                "index": len(steps),
                "branch": branch,
                "layer": layer,
                "readout_gate_norm": (
                    torch.linalg.vector_norm(a_perp) / math.sqrt(n)
                ).item(),
                "injection_gate_norm": (
                    torch.linalg.vector_norm(b_perp) / math.sqrt(n)
                ).item(),
            })

    final_a = stream_vector(
        model.readout_final, model.readout_final_lambda
    )
    final_a_perp = U.T @ (final_a - final_a.mean())
    final_alpha = final_a_perp / n
    alphas.append(final_alpha)

    Wc = torch.zeros(
        complement_dim, complement_dim, device=device, dtype=torch.float64
    )
    Phi = torch.eye(complement_dim, device=device, dtype=torch.float64)
    for index in range(len(Bs) - 1, -1, -1):
        mapped = Phi @ betas[index]
        Wc += torch.outer(mapped, mapped)
        Phi = Phi @ Bs[index]

    Wo = torch.zeros_like(Wc)
    Phi = torch.eye(complement_dim, device=device, dtype=torch.float64)
    for index, B in enumerate(Bs):
        mapped = alphas[index] @ Phi
        Wo += torch.outer(mapped, mapped)
        Phi = B @ Phi
    mapped_final = final_alpha @ Phi
    Wo += torch.outer(mapped_final, mapped_final)

    if complement_dim:
        wo_eigenvalues, wo_eigenvectors = torch.linalg.eigh(
            0.5 * (Wo + Wo.T)
        )
        sqrt_wo = (
            wo_eigenvectors
            @ torch.diag(wo_eigenvalues.clamp_min(0.0).sqrt())
            @ wo_eigenvectors.T
        )
        balanced = sqrt_wo @ Wc @ sqrt_wo
        hankel_squared = 0.5 * (balanced + balanced.T)
        hankel = torch.linalg.eigvalsh(
            hankel_squared
        ).clamp_min(0.0).sqrt()
    else:
        hankel = torch.empty(0, dtype=torch.float64, device=device)

    readout_norms = [row["readout_gate_norm"] for row in steps]
    injection_norms = [row["injection_gate_norm"] for row in steps]
    return {
        "kind": "linear_transport_gate_proxy",
        "nonlinear_full_model_claim": False,
        "n_streams": n,
        "complement_dim": complement_dim,
        "regularization_eps": float(eps),
        "steps": steps,
        "readout_gate_norm_mean": sum(readout_norms) / max(len(steps), 1),
        "injection_gate_norm_mean": (
            sum(injection_norms) / max(len(steps), 1)
        ),
        "final_readout_gate_norm": (
            torch.linalg.vector_norm(final_a_perp) / math.sqrt(n)
        ).item(),
        "controllability": _gramian_summary(Wc, eps),
        "observability": _gramian_summary(Wo, eps),
        "hankel_singular_values": hankel.cpu().tolist(),
    }
