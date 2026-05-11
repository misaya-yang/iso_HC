"""Phase 1 controlled experiment: 50M tokens, 12L/24L, 4 methods, seed 0.

First real comparison of baseline vs unconstrained vs mHC vs IsoHC.

Usage:
    # 12L-768 wide model
    python experiments/lm_phase1_controlled.py --variant wide --methods baseline mhc isohc

    # 24L-512 deep model
    python experiments/lm_phase1_controlled.py --variant deep --methods baseline mhc isohc

    # All methods
    python experiments/lm_phase1_controlled.py --variant wide --methods baseline unconstrained mhc isohc
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
    compute_mean_zero_energy,
    compute_stream_cosine,
)


# Model configurations
MODEL_CONFIGS = {
    'wide': {
        'name': '125M_wide',
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'deep': {
        'name': '125M_deep',
        'd_model': 512,
        'num_layers': 24,
        'num_heads': 8,
        'mlp_ratio': 4,
    },
    'smoke': {
        'name': 'smoke_12L512',
        'd_model': 512,
        'num_layers': 12,
        'num_heads': 8,
        'mlp_ratio': 4,
    },
}


def create_model(method, vocab_size, model_cfg, n_streams, context_length, device):
    """Create model."""
    if method == 'baseline':
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg['d_model'],
            num_layers=model_cfg['num_layers'],
            num_heads=model_cfg['num_heads'],
            context_length=context_length,
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=0.0,
        )
    elif method in ('unconstrained', 'orthogonal', 'isohc', 'mhc'):
        model = HCTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg['d_model'],
            num_layers=model_cfg['num_layers'],
            num_heads=model_cfg['num_heads'],
            n_streams=n_streams,
            context_length=context_length,
            mixing_type=method,
            mlp_ratio=model_cfg['mlp_ratio'],
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


def run_single(config):
    """Run single experiment configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"Phase 1: {config['name']}")
    print(f"{'='*70}")
    print(f"  Method: {config['method']}")
    print(f"  Model: {config['model_cfg']['name']} "
          f"({config['model_cfg']['num_layers']}L/{config['model_cfg']['d_model']}d)")
    print(f"  Streams: {config['n_streams']}")
    print(f"  Context: {config['context_length']}")
    print(f"  Tokens: {config['total_tokens']/1e6:.1f}M")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Seed: {config['seed']}")

    torch.manual_seed(config['seed'])

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    model = create_model(
        config['method'], vocab_size,
        config['model_cfg'], config['n_streams'],
        config['context_length'], device
    )
    print(f"  Parameters: {model.count_parameters()/1e6:.2f}M")

    # Data
    print("  Loading data...")
    train_loader, _ = create_dataloader(
        config['dataset'], tokenizer, config['context_length'],
        batch_size=config['batch_size'], split='train',
        max_samples=config.get('max_samples', None),
    )
    val_loader, _ = create_dataloader(
        config['dataset'], tokenizer, config['context_length'],
        batch_size=config['batch_size'], split='validation',
        max_samples=config.get('max_samples_val', None),
    )

    train_cfg = {
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
        'eval_max_batches': config.get('eval_max_batches', None),
    }

    start = time.time()
    try:
        results = run_experiment(model, train_loader, val_loader, train_cfg, device)
        elapsed = time.time() - start

        # Stream diagnostics
        stream_diag = {}
        if hasattr(model, 'get_stream_states'):
            x, _ = next(iter(val_loader))
            x = x[:2].to(device)
            states = model.get_stream_states(x)
            energies = [compute_mean_zero_energy(s) for s in states]
            cosines = [compute_stream_cosine(s) for s in states]
            stream_diag = {
                'mean_zero_energy_initial': energies[0],
                'mean_zero_energy_final': energies[-1],
                'mean_zero_energy_trend': energies,
                'stream_cosine_initial': cosines[0],
                'stream_cosine_final': cosines[-1],
                'stream_cosine_trend': cosines,
            }

        # H diagnostics
        h_diag = {}
        if hasattr(model, 'get_diagnostics'):
            diags = model.get_diagnostics()
            if diags:
                for key in diags[0].keys():
                    values = [d[key] for d in diags]
                    h_diag[f'h_{key}_mean'] = sum(values) / len(values)

        result = {
            'success': True,
            'config': config,
            'final_loss': results['final_eval']['val_loss'],
            'final_ppl': results['final_eval']['val_ppl'],
            'elapsed_sec': elapsed,
            'stream_diag': stream_diag,
            'h_diag': h_diag,
            'tokens_per_sec': results['train_metrics'][-1]['tokens_per_sec'] if results['train_metrics'] else 0,
        }

        print(f"\n  Result: val_loss={result['final_loss']:.4f}, "
              f"val_ppl={result['final_ppl']:.2f}, time={elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'config': config,
            'error': str(e),
            'elapsed_sec': elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 1 controlled experiment')
    parser.add_argument('--variant', type=str, default='wide',
                        choices=['wide', 'deep', 'smoke'],
                        help='Model variant')
    parser.add_argument('--methods', nargs='+',
                        default=['baseline', 'mhc', 'isohc'],
                        help='Methods to run')
    parser.add_argument('--dataset', type=str, default='tinystories')
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--total_tokens', type=int, default=50_000_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_tokens', type=int, default=2_000_000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--eval_every_tokens', type=int, default=5_000_000)
    parser.add_argument('--diagnostics_every', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='outputs/lm_phase1')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_samples_val', type=int, default=None)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.variant]
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for method in args.methods:
        config = {
            'name': f'{method}_{model_cfg["name"]}_s{args.n_streams}_seed{args.seed}',
            'method': method,
            'model_cfg': model_cfg,
            'dataset': args.dataset,
            'context_length': args.context_length,
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
            'save_dir': os.path.join(args.output_dir, f'{method}_{args.variant}_s{args.n_streams}_seed{args.seed}'),
            'seed': args.seed,
            'max_samples': args.max_samples,
            'max_samples_val': args.max_samples_val,
            'beta1': args.beta1,
            'beta2': args.beta2,
        }

        result = run_single(config)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        status = "PASS" if r.get('success') else "FAIL"
        loss = r.get('final_loss', 'N/A')
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        print(f"  {r['config']['name']:50s} | {status:4s} | loss={loss_str}")

    # Save
    with open(os.path.join(args.output_dir, 'phase1_summary.json'), 'w') as f:
        clean_results = []
        for r in all_results:
            cr = {k: v for k, v in r.items() if k not in ('diagnostics',)}
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2, default=str)

    print(f"\nSummary saved to {args.output_dir}/phase1_summary.json")


if __name__ == '__main__':
    main()
