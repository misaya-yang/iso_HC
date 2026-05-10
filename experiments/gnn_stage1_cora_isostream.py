"""Experiment G3: Cora IsoStream-GNN Smoke Test.

Train GCN, ResGCN, and IsoStream-GCN on Cora.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gnn import normalize_adjacency, compute_accuracy, compute_centered_variance, compute_pairwise_cosine, compute_dirichlet_energy
from gnn.models import GCN, ResGCN, IsoStreamGCN


def load_cora_pyg(root='data'):
    """Try loading Cora via PyTorch Geometric."""
    try:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name='Cora')
        data = dataset[0]
        return data.x, data.edge_index, data.y, data.train_mask, data.val_mask, data.test_mask
    except ImportError:
        return None


def load_cora_manual(root='data'):
    """Manual Cora loader if PyG is not available."""
    data_dir = os.path.join(root, 'cora')
    content_file = os.path.join(data_dir, 'cora.content')
    cites_file = os.path.join(data_dir, 'cora.cites')

    if not os.path.exists(content_file):
        print("Downloading Cora dataset...")
        os.makedirs(data_dir, exist_ok=True)
        url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
        tar_path = os.path.join(data_dir, 'cora.tgz')
        try:
            urllib.request.urlretrieve(url, tar_path)
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            extracted = os.path.join(data_dir, 'cora')
            for f in os.listdir(extracted):
                src, dst = os.path.join(extracted, f), os.path.join(data_dir, f)
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
            os.rmdir(extracted)
            os.remove(tar_path)
        except Exception as e:
            print(f"Failed to download Cora: {e}")
            return None

    paper_ids, features, labels_list = [], [], []
    label_map = {}
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            paper_ids.append(int(parts[0]))
            features.append([float(x) for x in parts[1:-1]])
            label_str = parts[-1]
            if label_str not in label_map:
                label_map[label_str] = len(label_map)
            labels_list.append(label_map[label_str])

    N = len(paper_ids)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.long)

    edge_list = []
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                src, dst = int(parts[0]), int(parts[1])
                if src in id_to_idx and dst in id_to_idx:
                    i, j = id_to_idx[src], id_to_idx[dst]
                    edge_list.extend([[i, j], [j, i]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    num_classes = y.max().item() + 1
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    for c in range(num_classes):
        c_nodes = (y == c).nonzero(as_tuple=True)[0]
        c_nodes = c_nodes[torch.randperm(len(c_nodes))]
        train_mask[c_nodes[:20]] = True

    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    remaining = remaining[torch.randperm(len(remaining))]
    val_mask[remaining[:500]] = True
    test_mask[remaining[500:]] = True

    return x, edge_index, y, train_mask, val_mask, test_mask


def load_cora(root='data'):
    result = load_cora_pyg(root)
    if result is not None:
        print("Loaded Cora via PyTorch Geometric")
        return result
    result = load_cora_manual(root)
    if result is not None:
        print("Loaded Cora manually")
        return result
    raise RuntimeError("Failed to load Cora dataset")


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(num_nodes)
    return (adj > 0).float()


def train_epoch(model, X, S, labels, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(X, S)
    loss = criterion(logits[train_mask], labels[train_mask])
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item(), grad_norm.item()


@torch.no_grad()
def evaluate(model, X, S, labels, mask):
    model.eval()
    logits = model(X, S)
    return compute_accuracy(logits, labels, mask)


@torch.no_grad()
def compute_oversmoothing_metrics(model, X, S):
    states = model.get_hidden_states(X, S)
    X0 = states[0]
    metrics = []
    for l, X_l in enumerate(states):
        metrics.append({
            'layer': l,
            'centered_variance': compute_centered_variance(X_l, X0),
            'pairwise_cosine': compute_pairwise_cosine(X_l, num_samples=2000),
            'dirichlet_energy': compute_dirichlet_energy(X_l, S),
        })
    return metrics


def run_cora_experiment(models_to_run, layers_list, hidden_dim, streams,
                        beta, ns_steps, dropout, epochs, lr, weight_decay,
                        device, seed, out_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("G3: Cora IsoStream-GNN Smoke Test")
    print("=" * 70)

    x, edge_index, labels, train_mask, val_mask, test_mask = load_cora()
    N = x.shape[0]
    in_dim = x.shape[1]
    num_classes = labels.max().item() + 1

    adj = edge_index_to_adj(edge_index, N)
    S, d = normalize_adjacency(adj)

    x = x.to(device)
    S = S.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"Nodes: {N}, Features: {in_dim}, Classes: {num_classes}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    all_results = {}

    for model_name in models_to_run:
        for num_layers in layers_list:
            config_key = f"{model_name}_L{num_layers}"
            print(f"\n--- {config_key} ---")

            if model_name == 'gcn':
                model = GCN(in_dim, hidden_dim, num_classes, num_layers, dropout)
            elif model_name == 'resgcn':
                model = ResGCN(in_dim, hidden_dim, num_classes, num_layers, beta, dropout)
            elif model_name == 'isostream':
                model = IsoStreamGCN(in_dim, hidden_dim, num_classes, num_layers,
                                     streams, beta, dropout, ns_steps)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            model = model.to(device)
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            history = []
            best_val_acc = 0.0
            best_test_acc = 0.0
            best_epoch = 0
            nan_count = 0

            for epoch in range(epochs):
                loss, grad_norm = train_epoch(model, x, S, labels, train_mask, optimizer, criterion)
                if np.isnan(loss) or np.isinf(loss):
                    nan_count += 1

                train_acc = evaluate(model, x, S, labels, train_mask)
                val_acc = evaluate(model, x, S, labels, val_mask)
                test_acc = evaluate(model, x, S, labels, test_mask)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch

                if epoch % 20 == 0 or epoch == epochs - 1:
                    print(f"  Epoch {epoch:3d}: loss={loss:.4f} train={train_acc:.4f} "
                          f"val={val_acc:.4f} test={test_acc:.4f} grad={grad_norm:.4f}")

                history.append({
                    'epoch': epoch, 'loss': loss, 'train_acc': train_acc,
                    'val_acc': val_acc, 'test_acc': test_acc, 'grad_norm': grad_norm,
                })

            final_train = evaluate(model, x, S, labels, train_mask)
            final_val = evaluate(model, x, S, labels, val_mask)
            final_test = evaluate(model, x, S, labels, test_mask)

            print(f"\n  Final: train={final_train:.4f} val={final_val:.4f} test={final_test:.4f}")
            print(f"  Best:  val={best_val_acc:.4f} test={best_test_acc:.4f} @ epoch {best_epoch}")
            print(f"  NaN/Inf count: {nan_count}")

            os_metrics = compute_oversmoothing_metrics(model, x, S)
            var_strs = [f"{m['centered_variance']:.3f}" for m in os_metrics]
            print(f"  Layer variance: {var_strs}")

            iso_diags = None
            if hasattr(model, 'get_iso_diagnostics'):
                iso_diags = model.get_iso_diagnostics()
                max_orth = max(d['orth_error'] for d in iso_diags)
                max_fix = max(d['fix_error'] for d in iso_diags)
                print(f"  Max orth_error: {max_orth:.2e}")
                print(f"  Max fix_error: {max_fix:.2e}")

            all_results[config_key] = {
                'model': model_name, 'num_layers': num_layers,
                'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc,
                'best_epoch': best_epoch,
                'final_train_acc': final_train, 'final_val_acc': final_val, 'final_test_acc': final_test,
                'nan_count': nan_count, 'history': history,
                'oversmoothing': os_metrics, 'iso_diagnostics': iso_diags,
            }

    return all_results


def plot_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for key, r in all_results.items():
        epochs = [h['epoch'] for h in r['history']]
        vals = [h['val_acc'] for h in r['history']]
        ax.plot(epochs, vals, label=key, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy over Training'); ax.legend(fontsize=8); ax.set_ylim(0, 1.0)

    ax = axes[1]
    for key, r in all_results.items():
        epochs = [h['epoch'] for h in r['history']]
        losses = [h['loss'] for h in r['history']]
        ax.plot(epochs, losses, label=key, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss over Epochs'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cora_training.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for key, r in all_results.items():
        os_metrics = r['oversmoothing']
        layers = [m['layer'] for m in os_metrics]
        variances = [m['centered_variance'] for m in os_metrics]
        ax.plot(layers, variances, marker='o', label=key, linewidth=2)
    ax.set_xlabel('Layer'); ax.set_ylabel('Centered Variance Ratio')
    ax.set_title('Hidden Variance Collapse by Layer'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cora_variance.png'), dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


def evaluate_pass(all_results):
    print("\n" + "=" * 70)
    print("Pass Evaluation")
    print("=" * 70)
    passed = True

    for key, r in all_results.items():
        print(f"\n{key}: val={r['best_val_acc']:.4f} test={r['best_test_acc']:.4f} nan={r['nan_count']}")
        if r['nan_count'] > 0:
            print("  [FAIL] NaN/Inf detected"); passed = False
        else:
            print("  [PASS] No NaN/Inf")

        if r['model'] == 'isostream' and r['iso_diagnostics']:
            max_orth = max(d['orth_error'] for d in r['iso_diagnostics'])
            max_fix = max(d['fix_error'] for d in r['iso_diagnostics'])
            print(f"  orth={max_orth:.2e} fix={max_fix:.2e}")
            if max_orth < 1e-5: print("  [PASS] orth_error")
            else: print("  [WARN] orth_error large")
            if max_fix < 1e-5: print("  [PASS] fix_error")
            else: print("  [WARN] fix_error large")

    return passed


def save_results(all_results, passed, output_dir='results/gnn_stage1/cora'):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'cora.json')
    compact = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in all_results.items()}
    with open(json_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'passed': passed, 'results': compact}, f, indent=2)

    md_path = os.path.join(output_dir, 'cora.md')
    with open(md_path, 'w') as f:
        f.write("# GNN Stage 1 G3: Cora Smoke Test Results\n\n")
        f.write(f"**Overall:** {'PASS' if passed else 'FAIL'}\n\n")
        f.write("| model | layers | best_val | best_test | nan | max_orth | max_fix |\n")
        f.write("|-------|-------:|---------:|----------:|----:|---------:|--------:|\n")
        for key, r in all_results.items():
            orth = 'n/a'; fix = 'n/a'
            if r['iso_diagnostics']:
                orth = f"{max(d['orth_error'] for d in r['iso_diagnostics']):.2e}"
                fix = f"{max(d['fix_error'] for d in r['iso_diagnostics']):.2e}"
            f.write(f"| {r['model']} | {r['num_layers']} | {r['best_val_acc']:.4f} | "
                   f"{r['best_test_acc']:.4f} | {r['nan_count']} | {orth} | {fix} |\n")
    print(f"\nResults saved to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default=['gcn', 'resgcn', 'isostream'])
    parser.add_argument('--layers', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--ns-steps', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results/gnn_stage1/cora')
    args = parser.parse_args()

    start = time.time()
    all_results = run_cora_experiment(
        args.models, args.layers, args.hidden_dim, args.streams,
        args.beta, args.ns_steps, args.dropout, args.epochs, args.lr, args.weight_decay,
        args.device, args.seed, args.out,
    )
    elapsed = time.time() - start
    plot_results(all_results, args.out)
    passed = evaluate_pass(all_results)
    save_results(all_results, passed, args.out)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}\nG3: {'PASSED' if passed else 'FAILED'}\n{'=' * 70}")
    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
