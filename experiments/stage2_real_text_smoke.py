"""Real-text byte-level LM smoke test for IsoHC experiments."""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isohc import BaselineTransformer, IsoHCTransformer


DEFAULT_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def load_text_bytes(data_path, data_url=DEFAULT_DATA_URL):
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {data_url} -> {path}")
        with urllib.request.urlopen(data_url, timeout=60) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if len(data) < 1024:
        raise ValueError(f"Dataset too small: {len(data)} bytes")
    return torch.tensor(list(data), dtype=torch.long)


def sample_batch(data, batch_size, context_length, device):
    max_start = data.numel() - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + context_length] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + context_length + 1] for s in starts]).to(device)
    return x, y


def make_model(args):
    common = dict(
        vocab_size=256,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        context_length=args.context_length,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    if args.model == "baseline":
        return BaselineTransformer(**common)
    if args.model == "isohc":
        return IsoHCTransformer(n_streams=args.streams, ns_steps=args.ns_steps, **common)
    raise ValueError(args.model)


def train(args):
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    data = load_text_bytes(args.data_path, args.data_url)
    train_data = data[: int(0.9 * data.numel())]
    val_data = data[int(0.9 * data.numel()) :]

    model = make_model(args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = args.precision == "bf16"
    device_type = torch.device(args.device).type
    metrics = []

    print("=" * 70)
    print("Stage 2: Real Text Byte-LM Smoke")
    print("=" * 70)
    print(f"model={args.model} layers={args.layers} params={model.count_parameters():,}")
    print(f"data_bytes={data.numel():,} train={train_data.numel():,} val={val_data.numel():,}")
    print(f"batch_size={args.batch_size} context={args.context_length} steps={args.steps}")

    model.train()
    start_all = time.time()
    for step in range(args.steps):
        step_start = time.time()
        x, y = sample_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        lr = get_cosine_lr(step, args.warmup_steps, args.steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            has_inf = any(torch.isinf(p).any().item() for p in model.parameters())
            hc_diags = {}
            if hasattr(model, "get_diagnostics"):
                diags = model.get_diagnostics()
                if diags:
                    orth = [d["orth_error"] for d in diags]
                    fix = [d["fix_error"] for d in diags]
                    energy = [d["energy_ratio"] for d in diags]
                    hc_diags = {
                        "max_orth_error": float(max(orth)),
                        "max_fix_error": float(max(fix)),
                        "min_energy_ratio": float(min(energy)),
                        "max_energy_ratio": float(max(energy)),
                    }
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm),
                "lr": float(lr),
                "step_time_ms": (time.time() - step_start) * 1000,
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
            }
            row.update(hc_diags)
            metrics.append(row)
            extra = ""
            if hc_diags:
                extra = (
                    f" orth={hc_diags['max_orth_error']:.2e}"
                    f" fix={hc_diags['max_fix_error']:.2e}"
                    f" energy=[{hc_diags['min_energy_ratio']:.4f},{hc_diags['max_energy_ratio']:.4f}]"
                )
            print(
                f"step {step + 1:5d}/{args.steps} loss={loss.item():.4f} "
                f"grad={grad_norm:.4f} lr={lr:.2e} time={row['step_time_ms']:.1f}ms{extra}",
                flush=True,
            )

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = sample_batch(val_data, args.batch_size, args.context_length, args.device)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            val_losses.append(float(loss.item()))
    val_loss = float(sum(val_losses) / len(val_losses))

    passed = (
        all(not m["has_nan"] and not m["has_inf"] for m in metrics)
        and max(m["loss"] for m in metrics) < 20
        and max(m["grad_norm"] for m in metrics) < 100
    )
    if args.model == "isohc":
        passed = passed and max(m.get("max_orth_error", 0.0) for m in metrics) < 5e-3
        passed = passed and max(m.get("max_fix_error", 0.0) for m in metrics) < 1e-5

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "params": model.count_parameters(),
        "train_bytes": int(train_data.numel()),
        "val_bytes": int(val_data.numel()),
        "metrics": metrics,
        "val_loss": val_loss,
        "passed": passed,
        "total_time_s": time.time() - start_all,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"real_text_{args.model}_L{args.layers}"
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    md = [
        f"# Real Text Smoke: {args.model} L{args.layers}",
        "",
        f"- passed: {passed}",
        f"- params: {model.count_parameters():,}",
        f"- final_train_loss: {metrics[-1]['loss']:.4f}",
        f"- val_loss: {val_loss:.4f}",
        f"- max_grad_norm: {max(m['grad_norm'] for m in metrics):.4f}",
        f"- total_time_s: {result['total_time_s']:.1f}",
    ]
    if args.model == "isohc":
        md.extend(
            [
                f"- max_orth_error: {max(m.get('max_orth_error', 0.0) for m in metrics):.2e}",
                f"- max_fix_error: {max(m.get('max_fix_error', 0.0) for m in metrics):.2e}",
            ]
        )
    (output_dir / f"{stem}.md").write_text("\n".join(md) + "\n")

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print(f"Saved to {output_dir / (stem + '.json')}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "isohc"], required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL)
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare/input.txt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/stage2_real_text")
    args = parser.parse_args()
    raise SystemExit(train(args))


if __name__ == "__main__":
    main()
