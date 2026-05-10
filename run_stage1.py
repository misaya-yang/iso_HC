"""Master script to run all Stage 1 experiments sequentially.

Usage:
  python run_stage1.py [--device cuda] [--quick]

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
    parser.add_argument('--quick', action='store_true', help='Run smaller configs for quick validation')
    parser.add_argument('--skip-projection', action='store_true')
    parser.add_argument('--skip-residual', action='store_true')
    parser.add_argument('--skip-smoke', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/stage1')
    args = parser.parse_args()

    print("=" * 70)
    print("IsoHC Stage 1 Validation Suite")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    success = True

    # ── Step 0: Environment Check ──────────────────────────────────────────
    print("\n--- Environment Check ---")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device_type = torch.device(args.device).type
    residual_dtype = 'fp32' if device_type == 'cpu' else 'bf16'
    smoke_precision = 'fp32' if device_type == 'cpu' else 'bf16'

    # ── Step 1: Projection Sanity ──────────────────────────────────────────
    if not args.skip_projection:
        if args.quick:
            cmd = [
                sys.executable, 'experiments/stage1_projection_sanity.py',
                '--streams', '4', '8',
                '--ns-steps', '3', '5',
                '--num-random-matrices', '64',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
        else:
            cmd = [
                sys.executable, 'experiments/stage1_projection_sanity.py',
                '--streams', '4', '8', '16',
                '--ns-steps', '1', '2', '3', '5',
                '--num-random-matrices', '128',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
        success = run_command(cmd, "Experiment 1: Projection Sanity") and success
    else:
        print("\n[SKIPPED] Experiment 1: Projection Sanity")

    # ── Step 2: Residual-Only Depth ────────────────────────────────────────
    if not args.skip_residual and success:
        if args.quick:
            cmd = [
                sys.executable, 'experiments/stage1_residual_only.py',
                '--methods', 'isohc', 'unconstrained',
                '--streams', '4', '8',
                '--depths', '32', '128',
                '--feature-dim', '256',
                '--num-trials', '16',
                '--ns-steps', '5',
                '--dtype-tensor', residual_dtype,
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
        else:
            cmd = [
                sys.executable, 'experiments/stage1_residual_only.py',
                '--methods', 'isohc', 'unconstrained', 'mhc-lite',
                '--streams', '4', '8',
                '--depths', '32', '64', '128', '256',
                '--feature-dim', '512',
                '--num-trials', '32',
                '--ns-steps', '5',
                '--dtype-tensor', residual_dtype,
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
        success = run_command(cmd, "Experiment 2: Residual-Only Depth") and success
    else:
        if not args.skip_residual:
            print("\n[SKIPPED] Experiment 2: Residual-Only Depth (previous experiment failed)")
        else:
            print("\n[SKIPPED] Experiment 2: Residual-Only Depth")

    # ── Step 3: Tiny Transformer Smoke Test ────────────────────────────────
    if not args.skip_smoke and success:
        # Baseline first
        if args.quick:
            baseline_cmd = [
                sys.executable, 'experiments/stage1_tiny_smoke.py',
                '--model', 'baseline',
                '--layers', '8',
                '--hidden-dim', '128',
                '--heads', '4',
                '--streams', '4',
                '--context-length', '128',
                '--steps', '100',
                '--batch-size', '4',
                '--grad-accum-steps', '2',
                '--precision', smoke_precision,
                '--ns-steps', '5',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
            isohc_cmd = [
                sys.executable, 'experiments/stage1_tiny_smoke.py',
                '--model', 'isohc',
                '--layers', '8',
                '--hidden-dim', '128',
                '--heads', '4',
                '--streams', '4',
                '--context-length', '128',
                '--steps', '100',
                '--batch-size', '4',
                '--grad-accum-steps', '2',
                '--precision', smoke_precision,
                '--ns-steps', '5',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
        else:
            baseline_cmd = [
                sys.executable, 'experiments/stage1_tiny_smoke.py',
                '--model', 'baseline',
                '--layers', '16',
                '--hidden-dim', '256',
                '--heads', '4',
                '--streams', '4',
                '--context-length', '256',
                '--steps', '300',
                '--batch-size', '8',
                '--grad-accum-steps', '4',
                '--precision', smoke_precision,
                '--ns-steps', '5',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]
            isohc_cmd = [
                sys.executable, 'experiments/stage1_tiny_smoke.py',
                '--model', 'isohc',
                '--layers', '16',
                '--hidden-dim', '256',
                '--heads', '4',
                '--streams', '4',
                '--context-length', '256',
                '--steps', '300',
                '--batch-size', '8',
                '--grad-accum-steps', '4',
                '--precision', smoke_precision,
                '--ns-steps', '5',
                '--device', args.device,
                '--output-dir', args.output_dir,
            ]

        success = run_command(baseline_cmd, "Experiment 3a: Baseline Smoke Test") and success
        success = run_command(isohc_cmd, "Experiment 3b: IsoHC Smoke Test") and success
    else:
        if not args.skip_smoke:
            print("\n[SKIPPED] Experiment 3: Smoke Test (previous experiment failed)")
        else:
            print("\n[SKIPPED] Experiment 3: Smoke Test")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Stage 1 Complete")
    print("=" * 70)

    if success:
        print("\nAll experiments completed successfully!")
        print(f"\nResults saved in: {args.output_dir}/")
        print("\nFiles generated:")
        for f in os.listdir(args.output_dir):
            print(f"  - {f}")
    else:
        print("\nSome experiments FAILED. Check logs above for details.")
        print("\nRecommended troubleshooting (in order):")
        print("  1. If projection failed: check e0=1/sqrt(n), U^T U = I, U^T e0 = 0")
        print("  2. If residual-only failed: check per-layer orthogonality, try fp32")
        print("  3. If smoke test failed: reduce LR, increase grad clip, fewer layers")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
