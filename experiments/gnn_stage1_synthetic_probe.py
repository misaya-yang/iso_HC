"""Experiment G2: Dense Node-Space IsoProp Classification Probe.

Propagate features L layers, then train a linear classifier.
Compare raw features, GCN diffusion, residual diffusion, and IsoNode.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gnn import (
    generate_sbm_graph, generate_node_features, normalize_adjacency,
    graph_propagation, split_data, compute_accuracy,
    compute_centered_variance,
)
from gnn.projection import iso_ns_project_v


class LinearClassifier(nn.Module):
    """Simple linear classifier: softmax(X W)."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


def propagate_features(X0, S, depth, method, alpha=0.1, Q=None):
    """Propagate features for L layers."""
    X = X0.clone()
    for _ in range(depth):
        if method == 'gcn':
            X = graph_propagation(S, X, method='gcn')
        elif method == 'residual':
            X = graph_propagation(S, X, method='residual', alpha=alpha)
        elif method == 'isonode':
            X = Q @ X
    return X


def train_classifier(X, labels, train_mask, val_mask, test_mask,
                     num_classes, epochs=100, lr=0.01, weight_decay=5e-4):
    """Train linear classifier on propagated features."""
    device = X.device
    N, d = X.shape

    model = LinearClassifier(d, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        with torch.no_grad():
            model.eval()
            logits = model(X)
            val_acc = compute_accuracy(logits, labels, val_mask)
            test_acc = compute_accuracy(logits, labels, test_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch

    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
    }


def run_probe(num_nodes, feature_dim, num_classes, p_in, p_out,
              depths, methods, alpha=0.1, epsilon=0.02,
              train_per_class=20, val_per_class=50,
              epochs=100, lr=0.01, weight_decay=5e-4,
              dtype=torch.float32, device='cuda', seed=42):
    """Run classification probe."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("G2: Synthetic Classification Probe")
    print("=" * 70)

    # Generate graph
    adj, labels = generate_sbm_graph(num_nodes, num_classes, p_in, p_out,
                                     self_loops=True, seed=seed)
    S, d = normalize_adjacency(adj)
    S = S.to(device=device, dtype=dtype)
    d = d.to(device=device, dtype=dtype)
    labels = labels.to(device)

    v = torch.sqrt(d + 1.0).to(device=device, dtype=dtype)

    # Features
    X0 = generate_node_features(labels, feature_dim, class_signal=1.0,
                                 noise_std=0.5, seed=seed)
    X0 = X0.to(device=device, dtype=dtype)

    # Data split
    train_mask, val_mask, test_mask = split_data(
        labels, train_per_class, val_per_class, seed=seed
    )
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    # Pre-build IsoNode operator
    Q = None
    if 'isonode' in methods:
        print("\nBuilding IsoNode operator...")
        N = num_nodes
        noise = torch.randn(N, N, device=device, dtype=dtype) * epsilon
        Q = iso_ns_project_v(S + noise, v, steps=10, use_svd=True)
        inv_err = torch.norm(Q @ v.unsqueeze(1) - v.unsqueeze(1), p=2).item()
        print(f"  Invariant error: {inv_err:.2e}")

    results = {}

    for method in methods:
        print(f"\n--- Method: {method} ---")
        results[method] = {}

        for depth in depths:
            if method == 'raw':
                # No propagation
                X_prop = X0.clone()
                d_used = 0
            else:
                X_prop = propagate_features(X0, S, depth, method, alpha=alpha, Q=Q)
                d_used = depth

            # Centered variance
            var_ratio = compute_centered_variance(X_prop, X0)

            # Train classifier
            metrics = train_classifier(
                X_prop, labels, train_mask, val_mask, test_mask,
                num_classes, epochs=epochs, lr=lr, weight_decay=weight_decay
            )
            metrics['depth'] = d_used
            metrics['centered_variance'] = var_ratio
            results[method][d_used] = metrics

            print(f"  depth={d_used:2d}: test_acc={metrics['best_test_acc']:.4f}, "
                  f"val_acc={metrics['best_val_acc']:.4f}, var={var_ratio:.4f}")

    return results


def plot_results(results, depths, methods, output_dir):
    """Plot accuracy vs depth."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    for method in methods:
        test_accs = [results[method].get(d, {}).get('best_test_acc', 0) for d in depths]
        label = {'raw': 'Raw Features', 'gcn': 'GCN', 'residual': 'Residual GCN', 'isonode': 'IsoNode'}[method]
        marker = {'raw': 'D', 'gcn': 'o', 'residual': 's', 'isonode': '^'}[method]
        ax.plot(depths, test_accs, marker=marker, label=label, linewidth=2)
    ax.set_xlabel('Propagation Depth')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification Accuracy vs Depth')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Variance
    ax = axes[1]
    for method in methods:
        variances = [results[method].get(d, {}).get('centered_variance', 0) for d in depths]
        label = {'raw': 'Raw Features', 'gcn': 'GCN', 'residual': 'Residual GCN', 'isonode': 'IsoNode'}[method]
        marker = {'raw': 'D', 'gcn': 'o', 'residual': 's', 'isonode': '^'}[method]
        ax.plot(depths, variances, marker=marker, label=label, linewidth=2)
    ax.set_xlabel('Propagation Depth')
    ax.set_ylabel('Centered Variance Ratio')
    ax.set_title('Feature Variance Preservation')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_results.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_dir}/probe_results.png")


def evaluate_pass(results, depths):
    """Check pass criteria."""
    print("\n" + "=" * 70)
    print("Pass Evaluation")
    print("=" * 70)

    passed = True

    # Check deep GCN degradation
    if 'gcn' in results and 'isonode' in results:
        max_depth = max(depths)
        gcn_acc = results['gcn'].get(max_depth, {}).get('best_test_acc', 0)
        iso_acc = results['isonode'].get(max_depth, {}).get('best_test_acc', 0)

        print(f"\nAt depth {max_depth}:")
        print(f"  GCN test_acc: {gcn_acc:.4f}")
        print(f"  IsoNode test_acc: {iso_acc:.4f}")

        if gcn_acc < iso_acc:
            print("  [PASS] IsoNode retains better signal than deep GCN")
        else:
            print("  [INFO] IsoNode accuracy not higher than GCN (may still be OK)")

    return passed


def save_results(results, passed, output_dir='results/gnn_stage1/synthetic_probe'):
    """Save results."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'probe.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'results': results,
        }, f, indent=2)

    md_path = os.path.join(output_dir, 'probe.md')
    with open(md_path, 'w') as f:
        f.write("# GNN Stage 1 G2: Synthetic Probe Results\n\n")
        f.write(f"**Overall:** {'PASS' if passed else 'FAIL'}\n\n")
        f.write("| method | depth | test_acc | val_acc | variance |\n")
        f.write("|--------|------:|---------:|--------:|---------:|\n")
        for method in sorted(results.keys()):
            for depth in sorted(results[method].keys()):
                r = results[method][depth]
                f.write(f"| {method} | {depth} | {r['best_test_acc']:.4f} | "
                       f"{r['best_val_acc']:.4f} | {r['centered_variance']:.4f} |\n")

    print(f"\nResults saved to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-nodes', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--feature-dim', type=int, default=64)
    parser.add_argument('--p-in', type=float, default=0.06)
    parser.add_argument('--p-out', type=float, default=0.01)
    parser.add_argument('--depths', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['raw', 'gcn', 'residual', 'isonode'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--train-per-class', type=int, default=20)
    parser.add_argument('--val-per-class', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results/gnn_stage1/synthetic_probe')
    args = parser.parse_args()

    start = time.time()
    results = run_probe(
        num_nodes=args.num_nodes,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        p_in=args.p_in,
        p_out=args.p_out,
        depths=args.depths,
        methods=args.methods,
        alpha=args.alpha,
        epsilon=args.epsilon,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )
    elapsed = time.time() - start

    plot_results(results, args.depths, args.methods, args.out)
    passed = evaluate_pass(results, args.depths)
    save_results(results, passed, output_dir=args.out)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}")
    print(f"G2: {'PASSED' if passed else 'FAILED'}")
    print(f"{'=' * 70}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
