"""Diagnostic metrics for LM training.

Tracks:
  - H geometry (orth_error, fix_error, singular values)
  - Mean-zero energy: ||P_perp X||_F / ||X||_F
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
        self.step_count = 0


def compute_mean_zero_energy(X, eps=1e-12):
    """Compute mean-zero energy ratio: ||P_perp X||_F / ||X||_F.

    X: (s, B, T, d) stream states
    Returns: scalar in [0, 1]
    """
    s = X.shape[0]
    # Mean over stream dimension
    mean = X.mean(dim=0, keepdim=True)  # (1, B, T, d)
    X_perp = X - mean  # (s, B, T, d)

    norm_perp = torch.norm(X_perp, p='fro')
    norm_total = torch.norm(X, p='fro')

    return (norm_perp / (norm_total + eps)).item()


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
        layer_params = [p for n, p in model.named_parameters()
                        if f'blocks.{l}.' in n or f'mixings.{l}.' in n]
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

    return results
