"""Experiment 3: Tiny Transformer Smoke Test.

Train a very small Transformer with IsoHC residual mixing for 300 steps
on synthetic data to verify forward/backward stability.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isohc import IsoHCTransformer, BaselineTransformer, UnconstrainedHCTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticLMDataset(Dataset):
    """Random token sequences for language modeling."""

    def __init__(self, num_samples, seq_len, vocab_size, seed=42):
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokens = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.tokens[idx]
        return seq[:-1], seq[1:]  # input, target (shifted by 1)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=0.0):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_smoke_test(model, train_loader, config, device='cuda'):
    """Run short training smoke test."""
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    # Mixed precision. bf16 does not need gradient scaling, and torch.amp works
    # for both CUDA and CPU smoke runs.
    use_amp = config['precision'] == 'bf16'
    device_type = torch.device(device).type

    # Metrics storage
    metrics_history = []
    step = 0
    accum_count = 0

    print(f"\nTraining {config['model_type']} for {config['steps']} steps...")
    print(f"  Model params: {model.count_parameters():,}")
    print(f"  Precision: {config['precision']}")
    print(f"  Batch size: {config['batch_size']}, Accum steps: {config['grad_accum_steps']}")

    model.train()
    optimizer.zero_grad()

    while step < config['steps']:
        for batch_inputs, batch_targets in train_loader:
            if step >= config['steps']:
                break

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            step_start = time.time()

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                logits, loss = model(batch_inputs, batch_targets)
                loss = loss / config['grad_accum_steps']

            loss.backward()

            accum_count += 1

            if accum_count >= config['grad_accum_steps']:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['grad_clip']
                ).item()

                # Check for NaN/Inf in gradients
                has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
                has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)

                # Update
                lr = get_cosine_lr(step, config['warmup_steps'], config['steps'], config['learning_rate'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()

                optimizer.zero_grad()
                accum_count = 0

                step_time = time.time() - step_start

                # Collect metrics every 20 steps (and first step)
                if step % 20 == 0 or step == 0 or step == config['steps'] - 1:
                    with torch.no_grad():
                        max_param = max(p.abs().max().item() for p in model.parameters())
                        max_activation = logits.detach().abs().max().float().item()

                        # IsoHC diagnostics
                        hc_diags = {}
                        if hasattr(model, 'get_diagnostics'):
                            diags = model.get_diagnostics()
                            if diags:
                                orth_errors = [d['orth_error'] for d in diags]
                                fix_errors = [d['fix_error'] for d in diags]
                                energy_ratios = [d['energy_ratio'] for d in diags]
                                hc_diags = {
                                    'mean_orth_error': float(np.mean(orth_errors)),
                                    'max_orth_error': float(np.max(orth_errors)),
                                    'mean_fix_error': float(np.mean(fix_errors)),
                                    'max_fix_error': float(np.max(fix_errors)),
                                    'mean_energy_ratio': float(np.mean(energy_ratios)),
                                    'min_energy_ratio': float(np.min(energy_ratios)),
                                    'max_energy_ratio': float(np.max(energy_ratios)),
                                }

                    mem_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

                    metrics = {
                        'step': step,
                        'steps_completed': step + 1,
                        'loss': loss.item() * config['grad_accum_steps'],
                        'grad_norm': grad_norm,
                        'lr': lr,
                        'max_param': max_param,
                        'max_activation': max_activation,
                        'gpu_memory_gb': mem_allocated,
                        'step_time_ms': step_time * 1000,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                    }
                    metrics.update(hc_diags)
                    metrics_history.append(metrics)

                    # Print
                    hc_str = ""
                    if hc_diags:
                        hc_str = (f" | orth={hc_diags['max_orth_error']:.2e} "
                                  f"fix={hc_diags['max_fix_error']:.2e} "
                                  f"energy=[{hc_diags['min_energy_ratio']:.4f}, {hc_diags['max_energy_ratio']:.4f}]")
                    print(f"  Step {step + 1:4d}/{config['steps']} | loss={metrics['loss']:.4f} | "
                          f"grad_norm={grad_norm:.4f} | lr={lr:.2e} | time={step_time*1000:.1f}ms"
                          f"{hc_str}")

                    if has_nan or has_inf:
                        print(f"  *** WARNING: NaN={has_nan}, Inf={has_inf} ***")

                step += 1

    return metrics_history


def evaluate_pass(config, metrics):
    """Check pass criteria."""
    print("\n" + "=" * 70)
    print("Pass Evaluation")
    print("=" * 70)

    passed = True

    # Check completion. Metrics are sampled every 20 optimizer steps plus the
    # final step, so len(metrics) is not the number of completed training steps.
    completed_steps = max((m.get('steps_completed', m.get('step', -1) + 1) for m in metrics), default=0)
    target_steps = config['steps']
    print(f"\nSteps completed: {completed_steps}/{target_steps}")
    if completed_steps < target_steps:
        print("  [FAIL] Training did not complete requested steps")
        passed = False

    # Check NaN/Inf
    has_any_nan = any(m.get('has_nan', False) for m in metrics)
    has_any_inf = any(m.get('has_inf', False) for m in metrics)
    losses = [m['loss'] for m in metrics]
    has_nan_loss = any(np.isnan(l) or np.isinf(l) for l in losses)

    print(f"NaN in grads: {has_any_nan}")
    print(f"Inf in grads: {has_any_inf}")
    print(f"NaN/Inf in loss: {has_nan_loss}")

    if has_any_nan or has_any_inf or has_nan_loss:
        print("  [FAIL] NaN or Inf detected")
        passed = False
    else:
        print("  [PASS] No NaN or Inf")

    # Check loss stability (not exploding)
    if len(losses) > 0:
        final_loss = losses[-1]
        max_loss = max(losses)
        print(f"Final loss: {final_loss:.4f}")
        print(f"Max loss: {max_loss:.4f}")

        if max_loss > 50:
            print("  [FAIL] Loss explosion detected")
            passed = False
        else:
            print("  [PASS] Loss numerically stable")

    # Check grad norm
    grad_norms = [m['grad_norm'] for m in metrics]
    if grad_norms:
        max_grad = max(grad_norms)
        print(f"Max grad_norm: {max_grad:.4f}")
        if max_grad > 100:
            print("  [WARN] Gradient norm very large")
        else:
            print("  [PASS] Gradient norm bounded")

    # IsoHC specific checks
    if config['model_type'] == 'isohc' and metrics:
        orth_errors = [m.get('max_orth_error', 0) for m in metrics if 'max_orth_error' in m]
        fix_errors = [m.get('max_fix_error', 0) for m in metrics if 'max_fix_error' in m]
        energy_ratios = [m.get('mean_energy_ratio', 0) for m in metrics if 'mean_energy_ratio' in m]

        if orth_errors:
            max_orth = max(orth_errors)
            print(f"Max orth_error: {max_orth:.2e}")
            if max_orth < 5e-3:
                print("  [PASS] orth_error within bounds")
            else:
                print("  [FAIL] orth_error too large")
                passed = False

        if fix_errors:
            max_fix = max(fix_errors)
            print(f"Max fix_error: {max_fix:.2e}")
            if max_fix < 1e-5:
                print("  [PASS] fix_error within bounds")
            else:
                print("  [FAIL] fix_error too large")
                passed = False

        if energy_ratios:
            min_energy = min([m.get('min_energy_ratio', 1.0) for m in metrics if 'min_energy_ratio' in m])
            max_energy = max([m.get('max_energy_ratio', 1.0) for m in metrics if 'max_energy_ratio' in m])
            print(f"Energy ratio range: [{min_energy:.4f}, {max_energy:.4f}]")
            if 0.98 <= min_energy and max_energy <= 1.02:
                print("  [PASS] energy ratios within bounds")
            else:
                print("  [WARN] some energy ratios outside [0.98, 1.02]")

    return passed


def save_results(config, metrics, passed, output_dir='results/stage1'):
    """Save results to JSON and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    model_type = config['model_type']

    # JSON
    json_path = os.path.join(output_dir, f'tiny_smoke_{model_type}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'passed': passed,
            'metrics': metrics,
        }, f, indent=2)

    # Markdown
    md_path = os.path.join(output_dir, f'tiny_smoke_{model_type}.md')
    with open(md_path, 'w') as f:
        f.write(f"# Stage 1: Tiny Transformer Smoke Test ({model_type})\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Overall:** {'PASS' if passed else 'FAIL'}\n\n")

        f.write("## Configuration\n\n")
        for k, v in config.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Metrics\n\n")
        f.write("| step | loss | grad_norm | lr | max_orth | max_fix | time_ms |\n")
        f.write("|-----:|-----:|----------:|---:|---------:|--------:|--------:|\n")
        for m in metrics:
            orth = m.get('max_orth_error', 'n/a')
            fix = m.get('max_fix_error', 'n/a')
            orth_str = f"{orth:.2e}" if isinstance(orth, float) else orth
            fix_str = f"{fix:.2e}" if isinstance(fix, float) else fix
            f.write(f"| {m['step']} | {m['loss']:.4f} | {m['grad_norm']:.4f} | "
                   f"{m['lr']:.2e} | {orth_str} | {fix_str} | {m['step_time_ms']:.1f} |\n")

    print(f"\nResults saved to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['isohc', 'baseline', 'unconstrained'])
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--context-length', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--ns-steps', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/stage1')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of synthetic samples (default: enough for 300 steps)')
    args = parser.parse_args()

    config = {
        'model_type': args.model,
        'num_layers': args.layers,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.heads,
        'n_streams': args.streams,
        'context_length': args.context_length,
        'vocab_size': args.vocab_size,
        'steps': args.steps,
        'eval_every': args.eval_every,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'precision': args.precision,
        'grad_clip': args.grad_clip,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'ns_steps': args.ns_steps,
        'dropout': args.dropout,
        'mlp_ratio': args.mlp_ratio,
    }

    print("=" * 70)
    print("Experiment 3: Tiny Transformer Smoke Test")
    print("=" * 70)
    print(f"\nModel: {args.model}")

    # Create model
    if args.model == 'isohc':
        model = IsoHCTransformer(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            n_streams=args.streams,
            context_length=args.context_length,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            ns_steps=args.ns_steps,
        )
    elif args.model == 'baseline':
        model = BaselineTransformer(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            context_length=args.context_length,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        )
    else:
        model = UnconstrainedHCTransformer(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            n_streams=args.streams,
            context_length=args.context_length,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        )

    print(f"Parameters: {model.count_parameters():,}")

    # Dataset
    # Calculate number of samples needed
    effective_batch = args.batch_size * args.grad_accum_steps
    samples_needed = args.steps * effective_batch
    num_samples = max(args.num_samples, samples_needed)

    dataset = SyntheticLMDataset(
        num_samples=num_samples,
        seq_len=args.context_length,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Train
    start_time = time.time()
    metrics = train_smoke_test(model, train_loader, config, device=args.device)
    elapsed = time.time() - start_time

    # Evaluate
    passed = evaluate_pass(config, metrics)
    save_results(config, metrics, passed, output_dir=args.output_dir)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'=' * 70}")
    print(f"Experiment 3 ({args.model}): {'PASSED' if passed else 'FAILED'}")
    print(f"{'=' * 70}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
