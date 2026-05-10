"""Experiment 1: Iso-NS Projection Sanity Check.

Verify that Iso-NS correctly projects arbitrary matrices onto:
  M_iso = {H ∈ R^{n×n} : H^T H = I, H @ 1 = 1}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isohc import iso_ns_project, construct_orthogonal_complement, newton_schulz_polar


def run_projection_sanity(streams, ns_steps_list, num_matrices, device='cuda', seed=42):
    """Run projection sanity check for all configurations."""
    torch.manual_seed(seed)
    results = {}

    print("=" * 70)
    print("Experiment 1: Iso-NS Projection Sanity")
    print("=" * 70)

    for n in streams:
        results[n] = {}
        print(f"\n--- Stream count n={n} ---")

        # Precompute U
        U = construct_orthogonal_complement(n, device=device, dtype=torch.float32)

        # Generate random matrices
        H_raw_list = [torch.randn(n, n, device=device, dtype=torch.float32) * 0.5
                      for _ in range(num_matrices)]

        # First: test with exact SVD (reference / highest accuracy)
        print("  SVD (exact):")
        orth_errors = []
        fix_errors = []
        energy_errors = []
        for H_raw in H_raw_list:
            H = iso_ns_project(H_raw, U=U, use_svd=True)
            ones = torch.ones(n, 1, device=device, dtype=torch.float32)
            I = torch.eye(n, device=device, dtype=torch.float32)
            orth_errors.append(torch.norm(H.T @ H - I, p='fro').item())
            fix_errors.append(torch.norm(H @ ones - ones, p=2).item())
            X_test = torch.randn(n, 512, device=device, dtype=torch.float32)
            energy_ratio = torch.norm(H @ X_test, p='fro') / torch.norm(X_test, p='fro')
            energy_errors.append(abs(energy_ratio.item() - 1.0))

        results[n]['svd'] = {
            'mean_orth_error': float(torch.tensor(orth_errors).mean()),
            'max_orth_error': float(torch.tensor(orth_errors).max()),
            'mean_fix_error': float(torch.tensor(fix_errors).mean()),
            'max_fix_error': float(torch.tensor(fix_errors).max()),
            'mean_energy_error': float(torch.tensor(energy_errors).mean()),
            'max_energy_error': float(torch.tensor(energy_errors).max()),
        }
        print(f"    orth={results[n]['svd']['mean_orth_error']:.2e} max={results[n]['svd']['max_orth_error']:.2e} | "
              f"fix={results[n]['svd']['mean_fix_error']:.2e} max={results[n]['svd']['max_fix_error']:.2e}")

        # Then: test with NS iterations at various steps
        for K in ns_steps_list:
            orth_errors = []
            fix_errors = []
            energy_errors = []

            for H_raw in H_raw_list:
                H = iso_ns_project(H_raw, U=U, steps=K, use_svd=False)

                ones = torch.ones(n, 1, device=device, dtype=torch.float32)
                I = torch.eye(n, device=device, dtype=torch.float32)

                orth_err = torch.norm(H.T @ H - I, p='fro').item()
                orth_errors.append(orth_err)

                fix_err = torch.norm(H @ ones - ones, p=2).item()
                fix_errors.append(fix_err)

                X_test = torch.randn(n, 512, device=device, dtype=torch.float32)
                energy_ratio = torch.norm(H @ X_test, p='fro') / torch.norm(X_test, p='fro')
                energy_err = abs(energy_ratio.item() - 1.0)
                energy_errors.append(energy_err)

            results[n][K] = {
                'mean_orth_error': float(torch.tensor(orth_errors).mean()),
                'max_orth_error': float(torch.tensor(orth_errors).max()),
                'mean_fix_error': float(torch.tensor(fix_errors).mean()),
                'max_fix_error': float(torch.tensor(fix_errors).max()),
                'mean_energy_error': float(torch.tensor(energy_errors).mean()),
                'max_energy_error': float(torch.tensor(energy_errors).max()),
            }

            print(f"  K={K}:   "
                  f"orth={results[n][K]['mean_orth_error']:.2e} max={results[n][K]['max_orth_error']:.2e} | "
                  f"fix={results[n][K]['mean_fix_error']:.2e} max={results[n][K]['max_fix_error']:.2e} | "
                  f"energy={results[n][K]['mean_energy_error']:.2e}")

    return results


def evaluate_pass(results):
    """Check pass criteria."""
    passed = True
    print("\n" + "=" * 70)
    print("Pass Evaluation")
    print("=" * 70)

    for n in sorted(results.keys()):
        # Check SVD (exact) results
        r_svd = results[n].get('svd')
        if r_svd:
            print(f"\nn={n}, SVD (exact):")
            checks = [
                (r_svd['mean_orth_error'] < 1e-3, f"mean orth_error < 1e-3: {r_svd['mean_orth_error']:.2e}"),
                (r_svd['max_orth_error'] < 5e-3, f"max orth_error < 5e-3: {r_svd['max_orth_error']:.2e}"),
                (r_svd['mean_fix_error'] < 1e-6, f"mean fix_error < 1e-6: {r_svd['mean_fix_error']:.2e}"),
                (r_svd['max_fix_error'] < 1e-5, f"max fix_error < 1e-5: {r_svd['max_fix_error']:.2e}"),
                (r_svd['mean_energy_error'] < 1e-3, f"mean energy_error < 1e-3: {r_svd['mean_energy_error']:.2e}"),
            ]
            for check, msg in checks:
                status = "PASS" if check else "FAIL"
                print(f"  [{status}] {msg}")
                if not check:
                    passed = False

        # Also check highest NS step count
        ns_keys = [k for k in results[n].keys() if isinstance(k, int)]
        if ns_keys:
            K = max(ns_keys)
            r = results[n][K]
            print(f"\nn={n}, NS K={K}:")
            checks = [
                (r['mean_orth_error'] < 1e-3, f"mean orth_error < 1e-3: {r['mean_orth_error']:.2e}"),
                (r['max_orth_error'] < 5e-3, f"max orth_error < 5e-3: {r['max_orth_error']:.2e}"),
                (r['mean_fix_error'] < 1e-6, f"mean fix_error < 1e-6: {r['mean_fix_error']:.2e}"),
                (r['max_fix_error'] < 1e-5, f"max fix_error < 1e-5: {r['max_fix_error']:.2e}"),
                (r['mean_energy_error'] < 1e-3, f"mean energy_error < 1e-3: {r['mean_energy_error']:.2e}"),
            ]
            for check, msg in checks:
                status = "PASS" if check else "FAIL"
                print(f"  [{status}] {msg}")
                if not check:
                    passed = False

    return passed


def save_results(results, passed, output_dir='results/stage1'):
    """Save results to JSON and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, 'projection_sanity.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'results': results,
        }, f, indent=2)

    # Markdown
    md_path = os.path.join(output_dir, 'projection_sanity.md')
    with open(md_path, 'w') as f:
        f.write("# Stage 1: Projection Sanity Results\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Overall:** {'PASS' if passed else 'FAIL'}\n\n")

        f.write("## Results Table\n\n")
        f.write("| n | K | mean orth | max orth | mean fix | max fix | mean energy |\n")
        f.write("|---|:-:|----------:|---------:|---------:|--------:|------------:|\n")
        for n in sorted(results.keys()):
            ordered_keys = ['svd'] if 'svd' in results[n] else []
            ordered_keys.extend(sorted(k for k in results[n].keys() if isinstance(k, int)))
            for K in ordered_keys:
                r = results[n][K]
                f.write(f"| {n} | {K} | {r['mean_orth_error']:.2e} | "
                       f"{r['max_orth_error']:.2e} | {r['mean_fix_error']:.2e} | "
                       f"{r['max_fix_error']:.2e} | {r['mean_energy_error']:.2e} |\n")

    print(f"\nResults saved to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streams', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--ns-steps', type=int, nargs='+', default=[1, 2, 3, 5])
    parser.add_argument('--num-random-matrices', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/stage1')
    args = parser.parse_args()

    start_time = time.time()
    results = run_projection_sanity(
        streams=args.streams,
        ns_steps_list=args.ns_steps,
        num_matrices=args.num_random_matrices,
        device=args.device,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    passed = evaluate_pass(results)
    save_results(results, passed, output_dir=args.output_dir)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}")
    print(f"Experiment 1: {'PASSED' if passed else 'FAILED'}")
    print(f"{'=' * 70}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
