"""Master script to run all GNN Stage 1 experiments.

Usage:
  python run_gnn_stage1.py [--device cuda] [--quick]

If --quick is set, reduces problem sizes for faster iteration.
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and report results."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true', help='Run smaller configs')
    parser.add_argument('--skip-g1', action='store_true')
    parser.add_argument('--skip-g2', action='store_true')
    parser.add_argument('--skip-g3', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/gnn_stage1')
    args = parser.parse_args()

    print("=" * 70)
    print("GNN Stage 1 Validation Suite")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Environment check
    print("\n--- Environment Check ---")
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    success = True

    # ── G1: Synthetic Oversmoothing ─────────────────────────────────────────
    if not args.skip_g1:
        if args.quick:
            cmd = [sys.executable, 'experiments/gnn_stage1_synthetic_oversmoothing.py',
                   '--num-nodes', '256', '--depths', '8', '16', '32', '64',
                   '--methods', 'gcn', 'isonode', '--device', args.device,
                   '--out', f'{args.output_dir}/synthetic']
        else:
            cmd = [sys.executable, 'experiments/gnn_stage1_synthetic_oversmoothing.py',
                   '--num-nodes', '512', '--depths', '1', '2', '4', '8', '16', '32', '64', '128',
                   '--methods', 'gcn', 'residual', 'isonode', '--device', args.device,
                   '--out', f'{args.output_dir}/synthetic']
        success = run_command(cmd, "G1: Synthetic Oversmoothing") and success
    else:
        print("\n[SKIPPED] G1")

    # ── G2: Synthetic Probe ─────────────────────────────────────────────────
    if not args.skip_g2 and success:
        if args.quick:
            cmd = [sys.executable, 'experiments/gnn_stage1_synthetic_probe.py',
                   '--num-nodes', '256', '--depths', '8', '16', '32',
                   '--methods', 'raw', 'gcn', 'isonode', '--epochs', '50',
                   '--device', args.device, '--out', f'{args.output_dir}/synthetic_probe']
        else:
            cmd = [sys.executable, 'experiments/gnn_stage1_synthetic_probe.py',
                   '--num-nodes', '512', '--depths', '8', '16', '32', '64',
                   '--methods', 'raw', 'gcn', 'residual', 'isonode', '--epochs', '100',
                   '--device', args.device, '--out', f'{args.output_dir}/synthetic_probe']
        success = run_command(cmd, "G2: Synthetic Probe") and success
    else:
        print("\n[SKIPPED] G2" if args.skip_g2 else "\n[SKIPPED] G2 (G1 failed)")

    # ── G3: Cora IsoStream ──────────────────────────────────────────────────
    if not args.skip_g3 and success:
        if args.quick:
            cmd = [sys.executable, 'experiments/gnn_stage1_cora_isostream.py',
                   '--models', 'gcn', 'isostream', '--layers', '8', '16',
                   '--epochs', '100', '--device', args.device, '--out', f'{args.output_dir}/cora']
        else:
            cmd = [sys.executable, 'experiments/gnn_stage1_cora_isostream.py',
                   '--models', 'gcn', 'resgcn', 'isostream', '--layers', '16', '32',
                   '--epochs', '200', '--device', args.device, '--out', f'{args.output_dir}/cora']
        success = run_command(cmd, "G3: Cora IsoStream") and success
    else:
        print("\n[SKIPPED] G3" if args.skip_g3 else "\n[SKIPPED] G3 (previous failed)")

    # Summary
    print("\n" + "=" * 70)
    print("GNN Stage 1 Complete")
    print("=" * 70)
    if success:
        print("\nAll experiments completed!")
        print(f"\nResults in: {args.output_dir}/")
    else:
        print("\nSome experiments FAILED.")
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
