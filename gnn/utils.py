"""GNN utility functions: graph generation, metrics, data splitting."""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Graph Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, self_loops=True, seed=None):
    """
    Generate a two-block stochastic block model graph.

    Returns:
        adj: (N, N) adjacency matrix (float, includes self-loops if requested)
        labels: (N,) node labels in {0, 1, ..., num_classes-1}
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = num_nodes
    C = num_classes
    nodes_per_class = N // C

    labels = torch.arange(C).repeat_interleave(nodes_per_class)
    if len(labels) < N:
        labels = torch.cat([labels, torch.zeros(N - len(labels), dtype=torch.long)])

    # Build adjacency
    adj = torch.zeros(N, N)
    for i in range(N):
        for j in range(i + 1, N):
            same_class = (labels[i] == labels[j]).item()
            p = p_in if same_class else p_out
            if torch.rand(1).item() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    if self_loops:
        adj = adj + torch.eye(N)

    return adj, labels


def generate_node_features(labels, feature_dim, class_signal=1.0, noise_std=0.5, seed=None):
    """
    Generate node features with class-dependent signal.

    X_i = mu_{y_i} + epsilon_i

    Returns:
        X: (N, d) feature matrix
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = len(labels)
    C = labels.max().item() + 1
    d = feature_dim

    # Class means: random unit vectors scaled by class_signal
    class_means = torch.randn(C, d)
    class_means = F.normalize(class_means, p=2, dim=1) * class_signal

    X = class_means[labels] + torch.randn(N, d) * noise_std
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Graph Operations
# ──────────────────────────────────────────────────────────────────────────────

def normalize_adjacency(adj):
    """
    Compute symmetric normalized adjacency: S = D^{-1/2} A D^{-1/2}.

    Args:
        adj: (N, N) adjacency matrix (may include self-loops)

    Returns:
        S: (N, N) normalized adjacency
        d_tilde: (N,) degree vector (with self-loops)
    """
    N = adj.shape[0]
    d = adj.sum(dim=1)  # degree (includes self-loops if present)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    S = D_inv_sqrt @ adj @ D_inv_sqrt
    return S, d


def graph_propagation(S, X, method='gcn', alpha=0.1, Q=None):
    """
    Single-step graph propagation.

    Args:
        S: (N, N) normalized adjacency
        X: (N, d) features
        method: 'gcn', 'residual', or 'isonode'
        alpha: residual coefficient (for 'residual')
        Q: (N, N) isometric operator (for 'isonode')

    Returns:
        X_next: (N, d) propagated features
    """
    if method == 'gcn':
        return S @ X
    elif method == 'residual':
        return (1 - alpha) * (S @ X) + alpha * X
    elif method == 'isonode':
        return Q @ X
    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_energy_ratio(X, X_ref):
    """r = ||X||_F / ||X_ref||_F"""
    return (torch.norm(X, p='fro') / (torch.norm(X_ref, p='fro') + 1e-12)).item()


def compute_centered_variance(X, X_ref):
    """V = ||X - mean(X)||_F / ||X_ref - mean(X_ref)||_F"""
    N = X.shape[0]
    mean_X = X.mean(dim=0, keepdim=True)
    mean_ref = X_ref.mean(dim=0, keepdim=True)
    centered = X - mean_X
    centered_ref = X_ref - mean_ref
    num = torch.norm(centered, p='fro')
    den = torch.norm(centered_ref, p='fro') + 1e-12
    return (num / den).item()


def compute_dirichlet_energy(X, S):
    """E = tr(X^T L X) where L = I - S"""
    N = S.shape[0]
    L = torch.eye(N, device=S.device, dtype=S.dtype) - S
    energy = torch.trace(X.T @ L @ X).item()
    return energy


def compute_pairwise_cosine(X, num_samples=1000):
    """
    Average pairwise cosine similarity between random node pairs.

    Args:
        X: (N, d) features
        num_samples: number of random pairs to sample

    Returns:
        mean_cosine: float
    """
    N = X.shape[0]
    if N <= 1:
        return 1.0

    # Sample random pairs without replacement if possible
    num_samples = min(num_samples, N * (N - 1) // 2)

    i = torch.randint(0, N, (num_samples,))
    j = torch.randint(0, N, (num_samples,))
    mask = i != j
    i = i[mask]
    j = j[mask]

    if len(i) == 0:
        return 1.0

    X_i = F.normalize(X[i], p=2, dim=1)
    X_j = F.normalize(X[j], p=2, dim=1)
    cosines = (X_i * X_j).sum(dim=1)
    return cosines.mean().item()


def compute_invariant_error(X, X_ref, v):
    """
    m = ||v^T X - v^T X_ref||_2.

    Args:
        X: (N, d) current features
        X_ref: (N, d) reference features
        v: (N,) invariant vector

    Returns:
        error: float
    """
    vTX = v @ X
    vTX_ref = v @ X_ref
    return torch.norm(vTX - vTX_ref, p=2).item()


def compute_invariant_error_norm(X, X_ref, v):
    """
    Normalized invariant error: ||v^T(X - X_ref)||_2 / ||X_ref||_F.

    This is invariant_error scaled by reference feature norm,
    making it comparable across feature dimensions and scales.
    """
    vTX = v @ X
    vTX_ref = v @ X_ref
    num = torch.norm(vTX - vTX_ref, p=2)
    den = torch.norm(X_ref, p='fro') + 1e-12
    return (num / den).item()


def compute_v_centered_variance(X, X_ref, v):
    """
    Graph-native centered variance: variance after projecting away the
    invariant direction v = sqrt(d).

    For symmetric normalized adjacency S = D^{-1/2} A D^{-1/2},
    the natural invariant direction is v = sqrt(d), not the all-ones vector.

    We compute:
      e = v / ||v||
      X_perp = X - e @ (e^T @ X)
      ratio = ||X_perp||_F / ||X_ref_perp||_F

    Args:
        X: (N, d) current features
        X_ref: (N, d) reference features
        v: (N,) invariant vector (e.g., sqrt of degree)

    Returns:
        ratio: float
    """
    e = v / (torch.norm(v) + 1e-12)
    e_mat = e.unsqueeze(1)  # (N, 1)

    # Project away v direction
    X_proj = e_mat @ (e_mat.T @ X)
    X_perp = X - X_proj

    X_ref_proj = e_mat @ (e_mat.T @ X_ref)
    X_ref_perp = X_ref - X_ref_proj

    num = torch.norm(X_perp, p='fro')
    den = torch.norm(X_ref_perp, p='fro') + 1e-12
    return (num / den).item()


# ──────────────────────────────────────────────────────────────────────────────
# Data Splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_data(labels, train_per_class, val_per_class, seed=None):
    """
    Split nodes into train/val/test.

    Args:
        labels: (N,) node labels
        train_per_class: number of training nodes per class
        val_per_class: number of validation nodes per class

    Returns:
        train_mask, val_mask, test_mask: (N,) boolean tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = len(labels)
    C = labels.max().item() + 1

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.ones(N, dtype=torch.bool)

    for c in range(C):
        c_nodes = (labels == c).nonzero(as_tuple=True)[0]
        c_nodes = c_nodes[torch.randperm(len(c_nodes))]

        n_train = min(train_per_class, len(c_nodes))
        n_val = min(val_per_class, len(c_nodes) - n_train)

        train_idx = c_nodes[:n_train]
        val_idx = c_nodes[n_train:n_train + n_val]

        train_mask[train_idx] = True
        val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


def compute_accuracy(logits, labels, mask):
    """Compute classification accuracy on masked nodes."""
    if mask.sum() == 0:
        return 0.0
    preds = logits[mask].argmax(dim=1)
    correct = (preds == labels[mask]).sum().item()
    return correct / mask.sum().item()
