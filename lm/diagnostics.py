"""Diagnostic metrics for LM training.

Tracks:
  - H geometry (orth_error, fix_error, singular values)
  - Mean-zero norm ratio and squared energy
  - Stream cosine similarity
  - Gradient profile by layer
  - Training stability (loss spikes, grad norm, activation stats)
"""

import torch
import numpy as np
from collections import defaultdict


class DiagnosticsCollector:
    """Collect and aggregate diagnostics during training.

    Use with collect_every=N to only collect every N steps (for efficiency).
    """

    def __init__(self, collect_every=100):
        self.collect_every = collect_every
        self.step_count = 0
        self.history = defaultdict(list)
        self.snapshots = defaultdict(list)

    def should_collect(self):
        return self.step_count % self.collect_every == 0

    def step(self):
        self.step_count += 1

    def record(self, **metrics):
        """Record metrics (only if should_collect)."""
        if not self.should_collect():
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history[key].append(value)
            elif isinstance(value, torch.Tensor):
                self.history[key].append(value.detach().cpu().item())

    def record_dict(self, prefix, d):
        """Record all items from a dict with prefix."""
        if not self.should_collect():
            return
        for key, value in d.items():
            full_key = f"{prefix}/{key}"
            if isinstance(value, (int, float)):
                self.history[full_key].append(value)
            elif isinstance(value, torch.Tensor):
                self.history[full_key].append(value.detach().cpu().item())

    def record_snapshot(self, name, value):
        """Record a structured JSON-ready snapshot at collection steps."""
        if self.should_collect():
            self.snapshots[name].append(value)

    def get_summary(self):
        """Get mean of all recorded metrics."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = np.mean(values[-100:])  # last 100 records
        return summary

    def get_latest(self):
        """Get most recent values."""
        return {k: v[-1] if v else None for k, v in self.history.items()}

    def clear(self):
        self.history.clear()
        self.snapshots.clear()
        self.step_count = 0


def compute_mean_zero_norm_ratio(X, eps=1e-12):
    """Compute ||P_perp X||_F / ||X||_F."""
    mean = X.mean(dim=0, keepdim=True)
    X_perp = X - mean
    return (
        torch.norm(X_perp, p='fro')
        / (torch.norm(X, p='fro') + eps)
    ).item()


def compute_mean_zero_energy(X, eps=1e-12):
    """Compute squared mean-zero energy ratio.

    X: (s, B, T, d) stream states
    Returns: scalar in [0, 1]
    """
    ratio = compute_mean_zero_norm_ratio(X, eps=eps)
    return ratio * ratio


def compute_complement_participation_rank(X, eps=1e-12):
    """Participation rank of covariance across mean-zero stream modes."""
    s = X.shape[0]
    if s <= 1:
        return 0.0
    centered = (
        X - X.mean(dim=0, keepdim=True)
    ).reshape(s, -1).detach().cpu().double()
    covariance = centered @ centered.T / max(centered.shape[1], 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if total.item() <= eps:
        return 0.0
    probabilities = eigenvalues / total
    return (1.0 / probabilities.square().sum()).item()


def compute_stream_cosine(X, eps=1e-12):
    """Compute average pairwise cosine similarity between streams.

    X: (s, B, T, d) stream states
    Returns: scalar in [-1, 1], higher = more similar = more collapsed
    """
    s = X.shape[0]
    if s <= 1:
        return 1.0

    # Flatten B, T, d -> each stream is a vector
    X_flat = X.reshape(s, -1)  # (s, B*T*d)
    X_norm = X_flat / (torch.norm(X_flat, dim=1, keepdim=True) + eps)

    # Cosine similarity matrix
    cos_sim = X_norm @ X_norm.T  # (s, s)

    # Average of off-diagonal elements
    mask = ~torch.eye(s, dtype=torch.bool, device=X.device)
    return cos_sim[mask].mean().item()


def compute_centered_stream_cosine(X, eps=1e-12):
    """Average pairwise cosine after removing the stream mean."""
    s = X.shape[0]
    if s <= 1:
        return 1.0
    flat = (X - X.mean(dim=0, keepdim=True)).reshape(s, -1)
    normed = flat / (torch.norm(flat, dim=1, keepdim=True) + eps)
    cosine = normed @ normed.T
    mask = ~torch.eye(s, dtype=torch.bool, device=X.device)
    return cosine[mask].mean().item()


def compute_head_output_stats(O, eps=1e-12):
    """Compute head-output diversity metrics.

    Args:
        O: attention head outputs with shape (B, T, H, Dh).

    Returns:
        Dict with average off-diagonal cosine and effective rank of the
        head Gram matrix.  Higher cosine means more head redundancy; higher
        effective rank means more diverse head usage.
    """
    if O.ndim != 4:
        raise ValueError(f"Expected O with shape (B, T, H, Dh), got {tuple(O.shape)}")

    B, T, H, Dh = O.shape
    if H <= 1:
        return {
            "head_offdiag_cosine": 1.0,
            "head_effective_rank": 1.0,
        }

    # Head vectors: (H, B*T*Dh)
    heads = O.permute(2, 0, 1, 3).reshape(H, -1).float()
    heads_norm = heads / (heads.norm(dim=1, keepdim=True) + eps)
    cos = heads_norm @ heads_norm.T
    mask = ~torch.eye(H, dtype=torch.bool, device=O.device)
    offdiag_cosine = cos[mask].mean()

    gram = (heads @ heads.T) / max(B * T * Dh, 1)
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    probs = eigvals / (eigvals.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    effective_rank = torch.exp(entropy)

    return {
        "head_offdiag_cosine": offdiag_cosine.item(),
        "head_effective_rank": effective_rank.item(),
    }


def compute_gradient_profile(model):
    """Compute gradient norm per layer.

    Returns dict: {layer_name: grad_norm}
    """
    profile = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            profile[name] = param.grad.norm().item()
    return profile


def compute_gradient_stats_by_layer(model, num_layers):
    """Aggregate gradient norms by layer.

    Returns dict with top/bottom layer grad norms and ratio.
    """
    layer_grads = []
    for l in range(num_layers):
        patterns = (
            f'blocks.{l}.',
            f'mixings.{l}.',
            f'attns.{l}.',
            f'mlps.{l}.',
            f'attn_norms.{l}.',
            f'mlp_norms.{l}.',
            f'attn_mixings.{l}.',
            f'mlp_mixings.{l}.',
        )
        layer_params = [p for n, p in model.named_parameters()
                        if any(pattern in n for pattern in patterns)]
        if layer_params:
            total_norm = sum(p.grad.norm().item() for p in layer_params
                           if p.grad is not None)
            layer_grads.append(total_norm)

    if not layer_grads:
        return {}

    top = layer_grads[0]
    bottom = layer_grads[-1]
    ratio = bottom / (top + 1e-12)

    return {
        'grad_top': top,
        'grad_bottom': bottom,
        'grad_ratio_bottom_top': ratio,
        'grad_mean': np.mean(layer_grads),
        'grad_std': np.std(layer_grads),
    }


def compute_activation_stats(x):
    """Compute activation statistics.

    x: tensor of any shape
    Returns dict with rms, max, min
    """
    return {
        'act_rms': x.pow(2).mean().sqrt().item(),
        'act_max': x.abs().max().item(),
        'act_min': x.min().item(),
    }


def collect_hc_diagnostics(model):
    """Collect HC-specific diagnostics from model.

    For HCTransformer: H geometry, stream states.
    For BaselineTransformer: empty (no HC).
    """
    results = {}

    if hasattr(model, 'get_diagnostics'):
        diags = model.get_diagnostics()
        for l, diag in enumerate(diags):
            for key, value in diag.items():
                results[f'layer{l}/{key}'] = value

    if hasattr(model, 'get_stream_states'):
        # Don't call this during training (expensive)
        # Only for evaluation hooks
        pass

    if hasattr(model, "get_named_mixing_matrices"):
        from .transport_analysis import collect_transport_report

        report = collect_transport_report(model)
        results["transport_report"] = report
        final = report.get("final", {})
        for key in (
            "composite_sv_min",
            "composite_sv_mean",
            "composite_sv_max",
            "product_sv_min",
            "product_sv_mean",
            "product_sv_max",
            "mean_to_perp_leakage_max",
            "perp_to_mean_leakage_max",
            "row_sum_error_max",
            "col_sum_error_max",
        ):
            if key in final:
                results[f"transport/{key}"] = final[key]

    for result_key, attribute in (
        ("readout_lambda", "readout_lambda"),
        ("injection_lambda", "injection_lambda"),
        ("final_readout_lambda", "readout_final_lambda"),
    ):
        parameter = getattr(model, attribute, None)
        if parameter is not None:
            results[f"gates/{result_key}"] = (
                parameter.detach().float().item()
            )

    return results
