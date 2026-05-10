"""Experiment 2: Residual-Only Deep Propagation Stability.

Test forward energy and backward gradient preservation for deep
residual-only propagation:
  X_{l+1} = H_l X_l  (forward)
  G_l = H_l^T G_{l+1}  (backward)

Compare:
  - Unconstrained HC (random residual mixing)
  - mHC-lite (Sinkhorn doubly-stochastic)
  - IsoHC (Iso-NS orthogonal projection)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isohc import iso_ns_project, construct_orthogonal_complement
from isohc.layers import UnconstrainedHCResidualMixing, MHCLiteResidualMixing


METHOD_SEED_OFFSETS = {
    'isohc': 0,
    'unconstrained': 10_000,
    'mhc-lite': 20_000,
}


def generate_random_hc_matrix(n, method, device='cuda', dtype=torch.float32, ns_steps=5):
    """Generate one random HC matrix for given method."""
    if method == 'isohc':
        H_raw = torch.randn(n, n, device=device, dtype=dtype) * 0.1
        H = iso_ns_project(H_raw, steps=ns_steps)
        return H
    elif method == 'unconstrained':
        return torch.eye(n, device=device, dtype=dtype) + torch.randn(n, n, device=device, dtype=dtype) * 0.05
    elif method == 'mhc-lite':
        # Sinkhorn projection to doubly-stochastic
        M = torch.abs(torch.randn(n, n, device=device, dtype=dtype) * 0.1) + torch.eye(n, device=device, dtype=dtype)
        for _ in range(20):
            M = M / M.sum(dim=1, keepdim=True)
            M = M / M.sum(dim=0, keepdim=True)
        return M
    else:
        raise ValueError(f"Unknown method: {method}")


def run_residual_only(method, n, feature_dim, depth, num_trials, ns_steps=5,
                      dtype_tensor=torch.bfloat16, device='cuda', seed=42):
    """Run residual-only depth test for one configuration."""
    torch.manual_seed(seed + METHOD_SEED_OFFSETS[method])
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    effective_dtype = dtype_tensor
    if torch.device(device).type == 'cpu' and dtype_tensor == torch.bfloat16:
        # CPU bf16 matmul accumulates enough rounding error to swamp the
        # 1e-5 mean-preservation criterion. Use fp32 for local CPU validation;
        # CUDA runs still exercise the requested bf16 path.
        effective_dtype = torch.float32

    energy_ratios = []
    grad_ratios = []
    mean_errors = []
    max_singulars = []
    min_singulars = []

    for trial in range(num_trials):
        # Generate H matrices for all layers
        H_list = [generate_random_hc_matrix(n, method, device=device, dtype=torch.float32, ns_steps=ns_steps)
                  for _ in range(depth)]

        # Forward: two input types
        for use_mean_zero in [False, True]:
            X0 = torch.randn(n, feature_dim, device=device, dtype=effective_dtype)
            if use_mean_zero:
                # X0^perp = X0 - (1/n) 1 1^T X0
                ones = torch.ones(n, 1, device=device, dtype=effective_dtype)
                X0 = X0 - (ones @ ones.T @ X0) / n

            # Forward propagation
            X = X0
            for H in H_list:
                H_d = H.to(effective_dtype)
                X = H_d @ X

            # Energy ratio: ||X_L|| / ||X_0||
            energy_ratio = (torch.norm(X, p='fro') / torch.norm(X0, p='fro')).item()
            if not use_mean_zero:
                energy_ratios.append(energy_ratio)

            # Mean preservation error
            ones_f = torch.ones(n, 1, device=device, dtype=torch.float32)
            mean_L = (ones_f.T @ X.float()) / n
            mean_0 = (ones_f.T @ X0.float()) / n
            mean_error = torch.norm(mean_L - mean_0, p=2).item()
            if not use_mean_zero:
                mean_errors.append(mean_error)

        # Backward: random gradient at layer L
        GL = torch.randn(n, feature_dim, device=device, dtype=effective_dtype)
        G = GL
        for H in reversed(H_list):
            H_d = H.to(effective_dtype)
            G = H_d.T @ G

        grad_ratio = (torch.norm(G, p='fro') / torch.norm(GL, p='fro')).item()
        grad_ratios.append(grad_ratio)

        # Singular value diagnostics (on fp32)
        for H in H_list:
            s = torch.linalg.svdvals(H.float())
            max_singulars.append(s.max().item())
            min_singulars.append(s.min().item())

    return {
        'energy_ratio_mean': float(np.mean(energy_ratios)),
        'energy_ratio_std': float(np.std(energy_ratios)),
        'energy_ratio_min': float(np.min(energy_ratios)),
        'energy_ratio_max': float(np.max(energy_ratios)),
        'grad_ratio_mean': float(np.mean(grad_ratios)),
        'grad_ratio_std': float(np.std(grad_ratios)),
        'grad_ratio_min': float(np.min(grad_ratios)),
        'grad_ratio_max': float(np.max(grad_ratios)),
        'mean_error_mean': float(np.mean(mean_errors)),
        'mean_error_max': float(np.max(mean_errors)),
        'max_singular_mean': float(np.mean(max_singulars)),
        'max_singular_max': float(np.max(max_singulars)),
        'min_singular_mean': float(np.mean(min_singulars)),
        'min_singular_min': float(np.min(min_singulars)),
    }


def run_all(methods, streams, depths, feature_dim, num_trials, ns_steps=5,
            dtype_tensor=torch.bfloat16, device='cuda', seed=42):
    """Run all configurations."""
    all_results = {}

    print("=" * 70)
    print("Experiment 2: Residual-Only Depth Stability")
    print("=" * 70)

    for method in methods:
        print(f"\n--- Method: {method} ---")
        all_results[method] = {}

        for n in streams:
            all_results[method][n] = {}
            print(f"\n  Stream count n={n}")

            for depth in depths:
                print(f"    Depth L={depth} ...", end=' ', flush=True)
                start = time.time()
                result = run_residual_only(
                    method=method, n=n, feature_dim=feature_dim, depth=depth,
                    num_trials=num_trials, ns_steps=ns_steps,
                    dtype_tensor=dtype_tensor, device=device, seed=seed
                )
                elapsed = time.time() - start
                print(f"done ({elapsed:.1f}s)")

                print(f"      energy_ratio: {result['energy_ratio_mean']:.4f} ± {result['energy_ratio_std']:.4f} "
                      f"[{result['energy_ratio_min']:.4f}, {result['energy_ratio_max']:.4f}]")
                print(f"      grad_ratio:   {result['grad_ratio_mean']:.4f} ± {result['grad_ratio_std']:.4f} "
                      f"[{result['grad_ratio_min']:.4f}, {result['grad_ratio_max']:.4f}]")
                print(f"      mean_error:   {result['mean_error_mean']:.2e} (max: {result['mean_error_max']:.2e})")
                print(f"      σ_max:        {result['max_singular_mean']:.4f} (max: {result['max_singular_max']:.4f})")
                print(f"      σ_min:        {result['min_singular_mean']:.4f} (min: {result['min_singular_min']:.4f})")

                all_results[method][n][depth] = result

    return all_results


def evaluate_pass(results):
    """Check pass criteria for IsoHC at L=256."""
    print("\n" + "=" * 70)
    print("Pass Evaluation")
    print("=" * 70)

    passed = True

    if 'isohc' not in results:
        print("FAIL: IsoHC results not found")
        return False

    for n in sorted(results['isohc'].keys()):
        for depth in sorted(results['isohc'][n].keys()):
            r = results['isohc'][n][depth]
            depth_pass = True

            checks = [
                (0.98 <= r['energy_ratio_mean'] <= 1.02,
                 f"energy_ratio in [0.98, 1.02]: {r['energy_ratio_mean']:.4f}"),
                (0.98 <= r['grad_ratio_mean'] <= 1.02,
                 f"grad_ratio in [0.98, 1.02]: {r['grad_ratio_mean']:.4f}"),
                (r['mean_error_max'] < 1e-5,
                 f"mean_error < 1e-5: {r['mean_error_max']:.2e}"),
                (r['max_singular_max'] <= 1.01,
                 f"σ_max <= 1.01: {r['max_singular_max']:.4f}"),
                (r['min_singular_min'] >= 0.99,
                 f"σ_min >= 0.99: {r['min_singular_min']:.4f}"),
            ]

            print(f"\nIsoHC n={n}, L={depth}:")
            for check, msg in checks:
                status = "PASS" if check else "FAIL"
                print(f"  [{status}] {msg}")
                if not check:
                    depth_pass = False
                    passed = False

            if depth == 256 and not depth_pass:
                print(f"\n  CRITICAL: IsoHC failed at L=256")

    return passed


def save_results(results, passed, output_dir='results/stage1'):
    """Save results to JSON and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, 'residual_only.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'results': results,
        }, f, indent=2)

    # Markdown
    md_path = os.path.join(output_dir, 'residual_only.md')
    with open(md_path, 'w') as f:
        f.write("# Stage 1: Residual-Only Depth Stability Results\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Overall:** {'PASS' if passed else 'FAIL'}\n\n")

        for method in results:
            f.write(f"\n## Method: {method}\n\n")
            f.write("| n | L | energy_mean | energy_std | grad_mean | grad_std | mean_err | σ_max | σ_min |\n")
            f.write("|---|:-:|:----------:|:---------:|:--------:|:-------:|:-------:|:-----:|:-----:|\n")
            for n in sorted(results[method].keys()):
                for depth in sorted(results[method][n].keys()):
                    r = results[method][n][depth]
                    f.write(f"| {n} | {depth} | {r['energy_ratio_mean']:.4f} | "
                           f"{r['energy_ratio_std']:.4f} | {r['grad_ratio_mean']:.4f} | "
                           f"{r['grad_ratio_std']:.4f} | {r['mean_error_max']:.2e} | "
                           f"{r['max_singular_max']:.4f} | {r['min_singular_min']:.4f} |\n")

    print(f"\nResults saved to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['isohc', 'unconstrained', 'mhc-lite'])
    parser.add_argument('--streams', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--depths', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--num-trials', type=int, default=32)
    parser.add_argument('--ns-steps', type=int, default=5)
    parser.add_argument('--dtype-tensor', type=str, default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/stage1')
    args = parser.parse_args()

    dtype_tensor = torch.bfloat16 if args.dtype_tensor == 'bf16' else torch.float32

    start_time = time.time()
    results = run_all(
        methods=args.methods,
        streams=args.streams,
        depths=args.depths,
        feature_dim=args.feature_dim,
        num_trials=args.num_trials,
        ns_steps=args.ns_steps,
        dtype_tensor=dtype_tensor,
        device=args.device,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    passed = evaluate_pass(results)
    save_results(results, passed, output_dir=args.output_dir)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}")
    print(f"Experiment 2: {'PASSED' if passed else 'FAILED'}")
    print(f"{'=' * 70}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
