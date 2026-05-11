"""Unified training loop for LM experiments.

Supports baseline and HC transformers with diagnostic collection.
"""

import os
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .diagnostics import (
    DiagnosticsCollector,
    compute_mean_zero_energy,
    compute_stream_cosine,
    compute_gradient_stats_by_layer,
    compute_activation_stats,
)


def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, dataloader, optimizer, device,
                epoch, total_tokens_target, tokens_per_step,
                warmup_steps, max_lr, min_lr,
                grad_clip=1.0, use_amp=True,
                diagnostics=None, eval_every_steps=None,
                eval_fn=None, save_dir=None):
    """Train for one epoch (or until token budget exhausted).

    Args:
        model: nn.Module
        dataloader: DataLoader
        optimizer: Optimizer
        device: torch device
        epoch: current epoch number
        total_tokens_target: total tokens to train on
        tokens_per_step: tokens processed per step
        warmup_steps: warmup steps
        max_lr: peak learning rate
        min_lr: minimum learning rate
        grad_clip: gradient clipping threshold
        use_amp: use bfloat16 automatic mixed precision
        diagnostics: DiagnosticsCollector instance
        eval_every_steps: run eval every N steps
        eval_fn: function(model) -> metrics dict
        save_dir: directory to save checkpoints

    Returns:
        dict of training metrics
    """
    model.train()
    scaler = GradScaler() if use_amp else None

    total_loss = 0.0
    total_tokens = 0
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # LR schedule
        lr = cosine_lr_schedule(step, warmup_steps,
                                total_tokens_target // tokens_per_step,
                                max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        if loss is None:
            continue

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss at step {step}: {loss.item()}")
            continue

        # Backward
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Statistics
        batch_loss = loss.item()
        batch_tokens = x.numel()
        total_loss += batch_loss
        total_tokens += batch_tokens
        step += 1

        # Diagnostics
        if diagnostics is not None:
            diagnostics.step()
            diagnostics.record(
                train_loss=batch_loss,
                lr=lr,
                grad_norm=get_total_grad_norm(model),
            )

            # Activation stats (from logits)
            if diagnostics.should_collect():
                act_stats = compute_activation_stats(logits)
                diagnostics.record(**act_stats)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            avg_loss = total_loss / step
            ppl = math.exp(min(avg_loss, 20))
            print(f"  step {step:5d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                  f"lr {lr:.2e} | tok/s {tokens_per_sec:.0f} | "
                  f"{total_tokens/1e6:.1f}M tokens")

        # Eval
        if eval_every_steps and eval_fn and step % eval_every_steps == 0:
            val_metrics = eval_fn(model)
            val_loss = val_metrics.get('val_loss', float('inf'))
            print(f"  [EVAL] step {step} | val_loss {val_loss:.4f} | "
                  f"val_ppl {math.exp(min(val_loss, 20)):.2f}")

            if diagnostics is not None:
                diagnostics.record(**val_metrics)

            # Save best
            if val_loss < best_val_loss and save_dir:
                best_val_loss = val_loss
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best.pt'))

            model.train()

        # Token budget check
        if total_tokens >= total_tokens_target:
            break

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(step, 1)
    tokens_per_sec = total_tokens / elapsed

    return {
        'epoch': epoch,
        'steps': step,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'final_ppl': math.exp(min(avg_loss, 20)),
        'tokens_per_sec': tokens_per_sec,
        'elapsed_sec': elapsed,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=True, max_batches=None):
    """Evaluate model on dataloader.

    Returns dict with val_loss, val_ppl, and optionally diagnostics.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with autocast(dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            num_batches += 1

        if max_batches and batch_idx >= max_batches - 1:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {
        'val_loss': avg_loss,
        'val_ppl': ppl,
        'val_batches': num_batches,
    }


def get_total_grad_norm(model):
    """Get total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    return total_norm ** 0.5


def run_experiment(model, train_loader, val_loader, config, device):
    """Run a full experiment with given configuration.

    Args:
        model: nn.Module
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: dict with keys:
            - total_tokens
            - max_lr, min_lr
            - warmup_tokens
            - grad_clip
            - use_amp
            - eval_every_tokens
            - save_dir
            - diagnostics_every
            - weight_decay
            - beta1, beta2
        device: torch device

    Returns:
        dict of results
    """
    os.makedirs(config['save_dir'], exist_ok=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['max_lr'],
        betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
        weight_decay=config.get('weight_decay', 0.1),
    )

    # Diagnostics
    diagnostics = DiagnosticsCollector(
        collect_every=config.get('diagnostics_every', 100)
    )

    # Token budget
    total_tokens = config['total_tokens']
    batch_size = train_loader.batch_size
    context_length = train_loader.dataset.context_length
    tokens_per_step = batch_size * context_length
    warmup_steps = config.get('warmup_tokens', 0) // tokens_per_step
    eval_every_steps = config.get('eval_every_tokens', total_tokens // 10) // tokens_per_step

    print(f"Training config:")
    print(f"  Total tokens: {total_tokens/1e6:.1f}M")
    print(f"  Tokens/step: {tokens_per_step}")
    print(f"  Total steps: {total_tokens // tokens_per_step}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Eval every: {eval_every_steps} steps")
    print(f"  Model params: {model.count_parameters()/1e6:.1f}M")
    print(f"  Device: {device}")

    # Eval function
    def eval_fn(m):
        return evaluate(m, val_loader, device,
                       use_amp=config.get('use_amp', True),
                       max_batches=config.get('eval_max_batches', None))

    # Training loop
    epoch = 0
    all_metrics = []
    while diagnostics.step_count * tokens_per_step < total_tokens:
        metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch,
            total_tokens_target=total_tokens,
            tokens_per_step=tokens_per_step,
            warmup_steps=warmup_steps,
            max_lr=config['max_lr'],
            min_lr=config['min_lr'],
            grad_clip=config.get('grad_clip', 1.0),
            use_amp=config.get('use_amp', True),
            diagnostics=diagnostics,
            eval_every_steps=eval_every_steps,
            eval_fn=eval_fn,
            save_dir=config['save_dir'],
        )
        all_metrics.append(metrics)
        epoch += 1

        # Check if token budget exhausted
        if diagnostics.step_count * tokens_per_step >= total_tokens:
            break

    # Final eval
    final_eval = evaluate(model, val_loader, device,
                         use_amp=config.get('use_amp', True))
    print(f"\nFinal: val_loss={final_eval['val_loss']:.4f}, "
          f"val_ppl={final_eval['val_ppl']:.2f}")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_eval': final_eval,
        'diagnostics': dict(diagnostics.history),
        'train_metrics': all_metrics,
    }, os.path.join(config['save_dir'], 'final.pt'))

    # Save diagnostics summary
    summary = diagnostics.get_summary()
    summary.update(final_eval)
    with open(os.path.join(config['save_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return {
        'final_eval': final_eval,
        'diagnostics': diagnostics,
        'train_metrics': all_metrics,
        'summary': summary,
    }
