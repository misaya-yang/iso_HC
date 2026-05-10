"""Experiment G4: Cora IsoRes-GCN Hybrid Architecture Test.

Compare GCN, ResGCN, and IsoRes-GCN on Cora node classification.
IsoRes-GCN: X_{l+1} = Norm(X_l + β·σ(SX_lW_l) + γ·Q_lX_l)

Grid search over layers and gamma to find best hybrid configuration.
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

from gnn import (
    normalize_adjacency, compute_accuracy,
    compute_centered_variance, compute_pairwise_cosine, compute_dirichlet_energy,
    compute_v_centered_variance,
)
from gnn.models import GCN, ResGCN, IsoResGCN, IsoStreamGCN, PairNormGCN


# ──────────────────────────────────────────────────────────────────────────────
# Cora Data Loading
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Training / Evaluation
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_name, in_dim, hidden_dim, out_dim, num_layers,
                n_nodes, v, beta, gamma, dropout, use_norm, ns_steps, use_svd,
                n_streams=4):
    """Build model instance by name."""
    if model_name == 'gcn':
        return GCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_name == 'resgcn':
        return ResGCN(in_dim, hidden_dim, out_dim, num_layers, beta, dropout)
    elif model_name == 'pairnorm':
        return PairNormGCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_name == 'isores':
        return IsoResGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_nodes, v,
            beta=beta, gamma=gamma,
            dropout=dropout, use_norm=use_norm,
            ns_steps=ns_steps, use_svd=use_svd,
        )
    elif model_name == 'isostream':
        return IsoStreamGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_streams=n_streams, beta=beta, dropout=dropout, ns_steps=ns_steps,
            use_stream_embed=True, use_concat_readout=True, use_message_dropout=True,
        )
    elif model_name == 'isostream_v1':
        return IsoStreamGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_streams=n_streams, beta=beta, dropout=dropout, ns_steps=ns_steps,
            use_stream_embed=False, use_concat_readout=False, use_message_dropout=False,
        )
    elif model_name == 'isostream_v2a':
        return IsoStreamGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_streams=n_streams, beta=beta, dropout=dropout, ns_steps=ns_steps,
            use_stream_embed=True, use_concat_readout=False, use_message_dropout=True,
        )
    elif model_name == 'isostream_v2b':
        return IsoStreamGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_streams=n_streams, beta=beta, dropout=dropout, ns_steps=ns_steps,
            use_stream_embed=False, use_concat_readout=True, use_message_dropout=True,
        )
    elif model_name == 'isostream_v2c':
        return IsoStreamGCN(
            in_dim, hidden_dim, out_dim, num_layers,
            n_streams=n_streams, beta=beta, dropout=dropout, ns_steps=ns_steps,
            use_stream_embed=True, use_concat_readout=True, use_message_dropout=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_config(model_name, config, in_dim, hidden_dim, out_dim,
                      n_nodes, v, S, x, labels,
                      train_mask, val_mask, test_mask,
                      epochs, lr, weight_decay, device):
    """Train one configuration and return results dict."""
    model = build_model(
        model_name, in_dim, hidden_dim, out_dim,
        config['num_layers'], n_nodes, v,
        config['beta'], config.get('gamma', 0.0),
        config['dropout'], config.get('use_norm', False),
        config.get('ns_steps', 5), config.get('use_svd', False),
        config.get('n_streams', 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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

    return {
        'model': model_name,
        'config': config,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_train_acc': final_train,
        'final_val_acc': final_val,
        'final_test_acc': final_test,
        'nan_count': nan_count,
        'n_params': n_params,
        'history': history,
        'oversmoothing': os_metrics,
        'iso_diagnostics': iso_diags,
    }


def run_cora_isores_grid(layers_list, gammas, betas,
                         hidden_dim, dropout, use_norm,
                         ns_steps, use_svd,
                         epochs, lr, weight_decay,
                         device, seed, out_dir,
                         models=None):
    """Run grid search over IsoRes-GCN configurations on Cora."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("G4: Cora IsoRes-GCN Hybrid Architecture Test")
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

    v = torch.sqrt(d + 1.0).to(device=device, dtype=torch.float32)

    print(f"Nodes: {N}, Features: {in_dim}, Classes: {num_classes}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    print(f"v = sqrt(d+1), |v|_2 = {torch.norm(v).item():.2f}")

    # Build config list
    configs = []
    models = models or ['gcn', 'resgcn', 'isores', 'isostream']

    # P0: 2-layer sanity baselines
    if 'gcn' in models:
        configs.append(('gcn', {'num_layers': 2, 'beta': 0.0, 'dropout': dropout}))
    if 'resgcn' in models:
        configs.append(('resgcn', {'num_layers': 2, 'beta': 0.5, 'dropout': dropout, 'use_norm': use_norm}))

    # Baselines at requested depths
    if 'gcn' in models:
        for num_layers in layers_list:
            configs.append(('gcn', {'num_layers': num_layers, 'beta': 0.0, 'dropout': dropout}))
    if 'resgcn' in models:
        for num_layers in layers_list:
            for beta in betas:
                configs.append(('resgcn', {
                    'num_layers': num_layers, 'beta': beta,
                    'dropout': dropout, 'use_norm': use_norm,
                }))

    # IsoRes with varying gamma (only at requested layers, shallow by default)
    if 'isores' in models:
        for num_layers in layers_list:
            for beta in betas:
                for gamma in gammas:
                    configs.append(('isores', {
                        'num_layers': num_layers, 'beta': beta, 'gamma': gamma,
                        'dropout': dropout, 'use_norm': use_norm,
                        'ns_steps': ns_steps, 'use_svd': use_svd,
                    }))

    # PairNorm baseline
    if 'pairnorm' in models:
        for num_layers in layers_list:
            configs.append(('pairnorm', {
                'num_layers': num_layers, 'dropout': dropout,
            }))

    # IsoStreamGCN ablations
    for ablation_name in ['isostream_v1', 'isostream_v2a', 'isostream_v2b', 'isostream_v2c']:
        if ablation_name in models:
            for num_layers in layers_list:
                for beta in betas:
                    configs.append((ablation_name, {
                        'num_layers': num_layers, 'beta': beta,
                        'dropout': dropout,
                        'ns_steps': ns_steps, 'n_streams': 4,
                    }))

    # IsoStreamGCN v2 (alias for v2c)
    if 'isostream' in models:
        for num_layers in layers_list:
            for beta in betas:
                configs.append(('isostream_v2c', {
                    'num_layers': num_layers, 'beta': beta,
                    'dropout': dropout,
                    'ns_steps': ns_steps, 'n_streams': 4,
                }))

    all_results = {}
    total_start = time.time()

    for model_name, config in configs:
        config_key = f"{model_name}_L{config['num_layers']}"
        if model_name == 'resgcn':
            config_key += f"_b{config['beta']}"
        elif model_name == 'isores':
            config_key += f"_b{config['beta']}_g{config['gamma']}"
        elif model_name.startswith('isostream'):
            config_key += f"_b{config['beta']}_s{config.get('n_streams', 4)}"

        print(f"\n{'='*70}")
        print(f"Running: {config_key}")
        print(f"{'='*70}")

        start = time.time()
        result = run_single_config(
            model_name, config, in_dim, hidden_dim, num_classes,
            N, v, S, x, labels, train_mask, val_mask, test_mask,
            epochs, lr, weight_decay, device,
        )
        elapsed = time.time() - start
        result['elapsed_sec'] = elapsed
        all_results[config_key] = result
        print(f"  Elapsed: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Grid search complete. Total time: {total_elapsed:.1f}s")
    print(f"Configs tested: {len(configs)}")
    print(f"{'='*70}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Results & Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(all_results, layers_list, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Best test accuracy by model and depth
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in ['gcn', 'resgcn', 'isores']:
        depths = []
        accs = []
        for num_layers in layers_list:
            # Find best config for this model+depth
            best_acc = 0.0
            for key, r in all_results.items():
                if r['model'] == model_name and r['config']['num_layers'] == num_layers:
                    if r['best_test_acc'] > best_acc:
                        best_acc = r['best_test_acc']
            if best_acc > 0:
                depths.append(num_layers)
                accs.append(best_acc)

        if depths:
            label = {'gcn': 'GCN', 'resgcn': 'ResGCN', 'isores': 'IsoRes-GCN', 'isostream': 'IsoStream-GCN'}[model_name]
            marker = {'gcn': 'o', 'resgcn': 's', 'isores': '^', 'isostream': 'D'}[model_name]
            ax.plot(depths, accs, marker=marker, label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Cora Node Classification: Best Accuracy vs Depth')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cora_best_accuracy.png'), dpi=150)
    plt.close()

    # Plot 2: Gamma sweep for IsoRes at each depth
    fig, axes = plt.subplots(1, len(layers_list), figsize=(5 * len(layers_list), 5))
    if len(layers_list) == 1:
        axes = [axes]

    for idx, num_layers in enumerate(layers_list):
        ax = axes[idx]
        gamma_vals = []
        test_accs = []

        for key, r in all_results.items():
            if r['model'] == 'isores' and r['config']['num_layers'] == num_layers:
                gamma = r['config']['gamma']
                gamma_vals.append(gamma)
                test_accs.append(r['best_test_acc'])

        # Also add ResGCN baseline (gamma=0 equivalent)
        for key, r in all_results.items():
            if r['model'] == 'resgcn' and r['config']['num_layers'] == num_layers:
                gamma_vals.append(0.0)
                test_accs.append(r['best_test_acc'])

        if gamma_vals:
            sorted_pairs = sorted(zip(gamma_vals, test_accs))
            gamma_vals, test_accs = zip(*sorted_pairs)
            ax.plot(gamma_vals, test_accs, marker='o', linewidth=2, markersize=8)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='ResGCN baseline')
            ax.set_xlabel('Gamma (Iso path weight)')
            ax.set_ylabel('Best Test Accuracy')
            ax.set_title(f'IsoRes-GCN @ L={num_layers}')
            ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cora_gamma_sweep.png'), dpi=150)
    plt.close()

    # Plot 3: Variance collapse comparison (L=max)
    max_layers = max(layers_list)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in ['gcn', 'resgcn', 'isores', 'isostream']:
        # Find result with max layers for this model
        target_key = None
        for key, r in all_results.items():
            if r['model'] == model_name and r['config']['num_layers'] == max_layers:
                if target_key is None or r['best_test_acc'] > all_results[target_key]['best_test_acc']:
                    target_key = key

        if target_key:
            os_metrics = all_results[target_key]['oversmoothing']
            layers = [m['layer'] for m in os_metrics]
            variances = [m['centered_variance'] for m in os_metrics]
            label = {'gcn': 'GCN', 'resgcn': 'ResGCN', 'isores': 'IsoRes-GCN', 'isostream': 'IsoStream-GCN'}[model_name]
            ax.plot(layers, variances, marker='o', label=label, linewidth=2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Centered Variance Ratio')
    ax.set_title(f'Hidden Variance Collapse (L={max_layers})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cora_variance_collapse.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def save_summary(all_results, output_dir='results/gnn_stage1/cora_isores'):
    """Save results to JSON and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON (without full history)
    json_path = os.path.join(output_dir, 'cora_isores.json')
    compact = {
        k: {kk: vv for kk, vv in v.items() if kk != 'history'}
        for k, v in all_results.items()
    }
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': compact,
        }, f, indent=2)

    # Markdown summary
    md_path = os.path.join(output_dir, 'cora_isores.md')
    with open(md_path, 'w') as f:
        f.write("# GNN Stage 1 G4: Cora IsoRes-GCN Results\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

        # Find best overall
        best_key = max(all_results.keys(), key=lambda k: all_results[k]['best_test_acc'])
        best = all_results[best_key]
        f.write(f"**Best overall:** {best_key} — test_acc={best['best_test_acc']:.4f}\n\n")

        # Per-model best
        f.write("## Best by Model Type\n\n")
        f.write("| model | best config | best_test | best_val | n_params |\n")
        f.write("|-------|-------------|----------:|---------:|---------:|\n")
        for model_name in ['gcn', 'resgcn', 'isores', 'isostream']:
            best_for_model = max(
                [(k, v) for k, v in all_results.items() if v['model'] == model_name],
                key=lambda x: x[1]['best_test_acc'],
                default=(None, None)
            )
            if best_for_model[0]:
                k, r = best_for_model
                f.write(f"| {model_name} | {k} | {r['best_test_acc']:.4f} | "
                       f"{r['best_val_acc']:.4f} | {r['n_params']:,} |\n")

        # Full results table
        f.write("\n## Full Results\n\n")
        f.write("| config | model | layers | beta | gamma | streams | norm | best_test | best_val | nan |\n")
        f.write("|--------|-------|-------:|-----:|------:|--------:|:----:|----------:|---------:|----:|\n")
        for key, r in sorted(all_results.items()):
            cfg = r['config']
            gamma = cfg.get('gamma', 'n/a')
            gamma_str = f"{gamma:.2f}" if isinstance(gamma, float) else gamma
            beta = cfg.get('beta', 'n/a')
            beta_str = f"{beta:.2f}" if isinstance(beta, float) else beta
            streams = cfg.get('n_streams', 'n/a')
            norm_str = 'yes' if cfg.get('use_norm', False) else 'no'
            f.write(f"| {key} | {r['model']} | {cfg['num_layers']} | {beta_str} | {gamma_str} | {streams} | {norm_str} | "
                   f"{r['best_test_acc']:.4f} | {r['best_val_acc']:.4f} | {r['nan_count']} |\n")

        # Variance analysis
        f.write("\n## Variance Preservation\n\n")
        f.write("| config | final_variance |\n")
        f.write("|--------|---------------:|\n")
        for key, r in sorted(all_results.items()):
            os_metrics = r['oversmoothing']
            if os_metrics:
                final_var = os_metrics[-1]['centered_variance']
                f.write(f"| {key} | {final_var:.4f} |\n")

    print(f"\nResults saved to {json_path} and {md_path}")
    return md_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument('--betas', type=float, nargs='+', default=[0.5])
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-norm', action='store_true', default=False)
    parser.add_argument('--ns-steps', type=int, default=5)
    parser.add_argument('--use-svd', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--out', type=str, default='results/gnn_stage1/cora_isores')
    parser.add_argument('--models', type=str, nargs='+', default=['gcn', 'resgcn', 'isores', 'isostream'])
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    if len(seeds) == 1:
        # Single seed: original behavior
        all_results = run_cora_isores_grid(
            layers_list=args.layers,
            gammas=args.gammas,
            betas=args.betas,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_norm=not args.no_norm,
            ns_steps=args.ns_steps,
            use_svd=args.use_svd,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=seeds[0],
            out_dir=args.out,
            models=args.models,
        )
        plot_results(all_results, args.layers, args.out)
        md_path = save_summary(all_results, args.out)
    else:
        # Multi-seed: aggregate statistics
        from collections import defaultdict
        import statistics
        seed_results = defaultdict(list)

        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"Running with seed={seed}")
            print(f"{'='*70}")
            out_dir = f"{args.out}_seed{seed}"
            results = run_cora_isores_grid(
                layers_list=args.layers,
                gammas=args.gammas,
                betas=args.betas,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                use_norm=not args.no_norm,
                ns_steps=args.ns_steps,
                use_svd=args.use_svd,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                seed=seed,
                out_dir=out_dir,
                models=args.models,
            )
            for key, r in results.items():
                seed_results[key].append(r['best_test_acc'])

        # Aggregate report
        os.makedirs(args.out, exist_ok=True)
        md_path = os.path.join(args.out, 'cora_multiseed.md')
        with open(md_path, 'w') as f:
            f.write("# Cora Multi-Seed Results\n\n")
            f.write(f"**Seeds:** {seeds}\n\n")
            f.write("| config | mean | std | min | max | n |\n")
            f.write("|--------|-----:|----:|----:|----:|--:|\n")
            for key, accs in sorted(seed_results.items()):
                mean = statistics.mean(accs)
                std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                f.write(f"| {key} | {mean:.4f} | {std:.4f} | {min(accs):.4f} | {max(accs):.4f} | {len(accs)} |\n")

        print(f"\n{'='*70}")
        print("MULTI-SEED SUMMARY")
        print(f"{'='*70}")
        for key, accs in sorted(seed_results.items()):
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            print(f"{key}: {mean:.4f} ± {std:.4f} (n={len(accs)})")
        print(f"\nReport saved to {md_path}")

    return 0


if __name__ == '__main__':
    exit(main())
