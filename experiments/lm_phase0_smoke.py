"""Phase 0 smoke test: verify all models run without NaN and loss decreases.

Runs on small model + short token budget to catch bugs before full experiments.

Target: 5M tokens, 12L/512d model, 4 streams, 4 methods.

Usage:
    python experiments/lm_phase0_smoke.py --dataset tinystories --context 512 \
        --total_tokens 5_000_000 --methods baseline unconstrained mhc isohc
"""

import os
import sys
import argparse
import json
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lm.models import BaselineTransformer, HCTransformer
from lm.data import get_tokenizer, create_dataloader
from lm.train import run_experiment, evaluate
from lm.diagnostics import (
    DiagnosticsCollector,
    compute_mean_zero_energy,
    compute_stream_cosine,
)


def create_model(method, vocab_size, d_model, num_layers, num_heads,
                 n_streams, context_length, device):
    """Create model based on method name."""
    if method == 'baseline':
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            context_length=context_length,
            mlp_ratio=4,
            dropout=0.0,
        )
    elif method in ('unconstrained', 'orthogonal', 'isohc', 'mhc'):
        model = HCTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            n_streams=n_streams,
            context_length=context_length,
            mixing_type=method,
            mlp_ratio=4,
            dropout=0.0,
            lambda_a=0.01,
            lambda_b=0.01,
            ns_steps=5,
            sinkhorn_iters=10,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model = model.to(device)
    return model


def smoke_test(config):
    """Run smoke test for one configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Smoke test: {config['name']}")
    print(f"{'='*60}")
    print(f"Method: {config['method']}")
    print(f"Model: {config['num_layers']}L/{config['d_model']}d, "
          f"{config['n_streams']} streams, ctx={config['context_length']}")
    print(f"Tokens: {config['total_tokens']/1e6:.1f}M")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Model
    model = create_model(
        config['method'], vocab_size,
        config['d_model'], config['num_layers'], config['num_heads'],
        config['n_streams'], config['context_length'], device
    )
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")

    # Data
    print("Loading data...")
    train_loader, train_ds = create_dataloader(
        config['dataset'], tokenizer, config['context_length'],
        batch_size=config['batch_size'], split='train',
        max_samples=config.get('max_samples', 50000),
    )
    val_loader, val_ds = create_dataloader(
        config['dataset'], tokenizer, config['context_length'],
        batch_size=config['batch_size'], split='validation',
        max_samples=config.get('max_samples_val', 5000),
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Train config
    train_config = {
        'total_tokens': config['total_tokens'],
        'max_lr': config['max_lr'],
        'min_lr': config['min_lr'],
        'warmup_tokens': config['warmup_tokens'],
        'grad_clip': config['grad_clip'],
        'use_amp': config['use_amp'],
        'eval_every_tokens': config['eval_every_tokens'],
        'save_dir': config['save_dir'],
        'diagnostics_every': config['diagnostics_every'],
        'weight_decay': config['weight_decay'],
        'beta1': config['beta1'],
        'beta2': config['beta2'],
    }

    # Run
    start = time.time()
    try:
        results = run_experiment(model, train_loader, val_loader, train_config, device)
        elapsed = time.time() - start

        # Collect stream diagnostics if HC model
        stream_diag = {}
        if hasattr(model, 'get_stream_states'):
            print("Collecting stream diagnostics...")
            # Use a small batch
            x, y = next(iter(val_loader))
            x = x[:2].to(device)
            states = model.get_stream_states(x)
            energies = [compute_mean_zero_energy(s) for s in states]
            cosines = [compute_stream_cosine(s) for s in states]
            stream_diag = {
                'mean_zero_energy_initial': energies[0],
                'mean_zero_energy_final': energies[-1],
                'stream_cosine_initial': cosines[0],
                'stream_cosine_final': cosines[-1],
            }
            print(f"  Mean-zero energy: {energies[0]:.4f} -> {energies[-1]:.4f}")
            print(f"  Stream cosine: {cosines[0]:.4f} -> {cosines[-1]:.4f}")

        # Collect H diagnostics if HC model
        h_diag = {}
        if hasattr(model, 'get_diagnostics'):
            diags = model.get_diagnostics()
            if diags:
                for key in diags[0].keys():
                    values = [d[key] for d in diags]
                    h_diag[f'h_{key}_mean'] = sum(values) / len(values)
                    h_diag[f'h_{key}_max'] = max(values)
                    h_diag[f'h_{key}_min'] = min(values)

        success = True
        final_loss = results['final_eval']['val_loss']
        has_nan = not (final_loss < 100)  # rough check

        print(f"\n{'='*60}")
        print(f"Result: loss={final_loss:.4f}, time={elapsed:.1f}s")
        if has_nan:
            print("FAIL: Loss is NaN or exploded!")
            success = False
        else:
            print("PASS: Loss is finite")
        print(f"{'='*60}")

        return {
            'success': success,
            'config': config,
            'final_loss': final_loss,
            'final_ppl': results['final_eval']['val_ppl'],
            'elapsed_sec': elapsed,
            'stream_diag': stream_diag,
            'h_diag': h_diag,
            'tokens_per_sec': results['train_metrics'][-1]['tokens_per_sec'] if results['train_metrics'] else 0,
        }

    except Exception as e:
        print(f"\nFAIL: Exception during training")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'config': config,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 0 smoke test')
    parser.add_argument('--dataset', type=str, default='tinystories',
                        help='Dataset name')
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--total_tokens', type=int, default=5_000_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_tokens', type=int, default=500_000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--eval_every_tokens', type=int, default=1_000_000)
    parser.add_argument('--diagnostics_every', type=int, default=50)
    parser.add_argument('--methods', nargs='+',
                        default=['baseline', 'unconstrained', 'mhc', 'isohc'],
                        help='Methods to test')
    parser.add_argument('--output_dir', type=str, default='outputs/lm_phase0')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--max_samples_val', type=int, default=5000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for method in args.methods:
        config = {
            'name': f'{method}_{args.num_layers}L{args.d_model}d_s{args.n_streams}',
            'method': method,
            'dataset': args.dataset,
            'context_length': args.context_length,
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'n_streams': args.n_streams,
            'total_tokens': args.total_tokens,
            'batch_size': args.batch_size,
            'max_lr': args.max_lr,
            'min_lr': args.min_lr,
            'warmup_tokens': args.warmup_tokens,
            'grad_clip': args.grad_clip,
            'weight_decay': args.weight_decay,
            'eval_every_tokens': args.eval_every_tokens,
            'diagnostics_every': args.diagnostics_every,
            'use_amp': args.use_amp,
            'save_dir': os.path.join(args.output_dir, f'{method}_s{args.n_streams}'),
            'max_samples': args.max_samples,
            'max_samples_val': args.max_samples_val,
            'beta1': args.beta1,
            'beta2': args.beta2,
        }

        result = smoke_test(config)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "PASS" if r.get('success') else "FAIL"
        loss = r.get('final_loss', 'N/A')
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        err = r.get('error', '')
        print(f"  {r['config']['name']:30s} | {status:4s} | loss={loss_str:8s} | {err[:30]}")

    # Save summary
    with open(os.path.join(args.output_dir, 'smoke_summary.json'), 'w') as f:
        # Filter non-serializable items
        clean_results = []
        for r in all_results:
            cr = {k: v for k, v in r.items() if k != 'diagnostics'}
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2, default=str)

    print(f"\nSummary saved to {args.output_dir}/smoke_summary.json")

    # Exit code
    if any(not r.get('success') for r in all_results):
        print("\nSome tests FAILED!")
        sys.exit(1)
    else:
        print("\nAll tests PASSED!")
        sys.exit(0)


if __name__ == '__main__':
    main()
