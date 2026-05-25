"""Posthoc mechanism analysis for trained LM checkpoints.

This script is meant for the next teacher-requested evidence package:
  1. cumulative/composite gain on the mean-zero stream subspace,
  2. layer-wise stream-state gradient profile,
  3. validation-time complement removal,
  4. IsoHC -> identity / random fixed-vector orthogonal replacement.

It requires checkpoints saved by experiments/lm_5090_next_runs.py with
save_checkpoints=True.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.lm_5090_next_runs import create_model
from isohc.projection import construct_orthogonal_complement
from lm.data import create_dataloader, get_tokenizer
from lm.train import amp_autocast
from lm.transport_analysis import collect_transport_report


def infer_vocab_size(state_dict):
    for key in ("token_embedding.weight", "module.token_embedding.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Could not infer vocab size from checkpoint state_dict.")


def load_model_from_run(run_dir, device):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    config = dict(ckpt.get("config", {}))
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        config.update(summary.get("config", {}))
    vocab_size = infer_vocab_size(ckpt["model_state_dict"])
    if "method" not in config or "preset" not in config:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not include method/preset. "
            "Keep run_summary.json beside final.pt or pass a checkpoint produced by lm_5090_next_runs.py."
        )
    model = create_model(config, vocab_size=vocab_size, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, vocab_size, ckpt_path


def make_val_loader(config, vocab_size, args):
    val_cache_path = args.val_cache_path or config.get("val_cache_path")
    dataset = args.dataset or config.get("dataset", "tinystories")

    if dataset == "random" or val_cache_path:
        tokenizer = SimpleNamespace(vocab_size=vocab_size)
    else:
        tokenizer = get_tokenizer()

    loader, _ = create_dataloader(
        dataset,
        tokenizer,
        config["context_length"],
        batch_size=args.batch_size or config.get("batch_size", 4),
        split="validation",
        max_samples=args.max_samples_val or config.get("max_samples_val"),
        cache_path=val_cache_path,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else 2,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def evaluate_loss(model, val_loader, device, use_amp=True, max_batches=4,
                  stream_intervention=None, mixing_overrides=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_autocast(device, enabled=use_amp):
            _, loss = model(
                x,
                y,
                stream_intervention=stream_intervention,
                mixing_overrides=mixing_overrides,
            )
        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            batches += 1
        if max_batches and batch_idx >= max_batches - 1:
            break

    val_loss = total_loss / max(total_tokens, 1)
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20)),
        "val_batches": batches,
    }


def stream_gradient_profile(model, val_loader, device, use_amp=True):
    if not hasattr(model, "get_named_mixing_matrices"):
        return None

    model.zero_grad(set_to_none=True)
    model.train(False)
    x, y = next(iter(val_loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with amp_autocast(device, enabled=use_amp):
        _, loss, states = model(
            x,
            y,
            return_stream_states=True,
            retain_stream_grads=True,
        )
    loss.backward()

    norms = []
    for index, state in enumerate(states):
        grad = state.grad
        norms.append({
            "state_index": index,
            "grad_norm": 0.0 if grad is None else grad.norm().item(),
        })
    final_norm = max(norms[-1]["grad_norm"], 1e-12)
    for row in norms:
        row["grad_ratio_to_final"] = row["grad_norm"] / final_norm
        row["log_grad_ratio_to_final"] = math.log(max(row["grad_ratio_to_final"], 1e-30))

    xs = torch.arange(len(norms), dtype=torch.float64)
    ys = torch.tensor([row["log_grad_ratio_to_final"] for row in norms], dtype=torch.float64)
    x_centered = xs - xs.mean()
    slope = ((x_centered * (ys - ys.mean())).sum() / (x_centered.square().sum() + 1e-12)).item()
    model.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "slope_log_grad_ratio": slope,
        "states": norms,
    }


def random_fixed_vector_orthogonal(n_streams, device, dtype):
    U = construct_orthogonal_complement(n_streams, device=device, dtype=torch.float32)
    v = torch.ones(n_streams, 1, device=device, dtype=torch.float32) / (n_streams ** 0.5)
    random = torch.randn(n_streams - 1, n_streams - 1, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(random)
    return (v @ v.T + U @ q @ U.T).to(dtype=dtype)


def make_random_iso_overrides(model, device, dtype):
    overrides = {}
    for item in model.get_named_mixing_matrices():
        overrides[(item["branch"], item["layer"])] = random_fixed_vector_orthogonal(
            model.n_streams,
            device=device,
            dtype=dtype,
        )
    return overrides


def complement_removal_curve(model, val_loader, device, base_loss, args):
    if not hasattr(model, "get_named_mixing_matrices"):
        return []

    num_states = 1 + 2 * model.num_layers
    indices = list(range(0, num_states, max(1, args.intervention_stride)))
    if indices[-1] != num_states - 1:
        indices.append(num_states - 1)

    curve = []
    for state_index in indices:
        metrics = evaluate_loss(
            model,
            val_loader,
            device,
            use_amp=not args.no_amp,
            max_batches=args.eval_batches,
            stream_intervention={"state_index": state_index, "mode": "mean_only"},
        )
        curve.append({
            "state_index": state_index,
            "val_loss": metrics["val_loss"],
            "delta_loss": metrics["val_loss"] - base_loss,
            "val_ppl": metrics["val_ppl"],
        })
    return curve


def analyze_run(run_dir, args, device):
    model, config, vocab_size, ckpt_path = load_model_from_run(run_dir, device)
    val_loader = make_val_loader(config, vocab_size, args)
    use_amp = not args.no_amp

    base = evaluate_loss(
        model,
        val_loader,
        device,
        use_amp=use_amp,
        max_batches=args.eval_batches,
    )
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "method": config.get("method"),
        "preset": config.get("preset"),
        "base_eval": base,
    }

    if hasattr(model, "get_named_mixing_matrices"):
        report["transport_complement"] = collect_transport_report(model)
        report["gradient_profile"] = stream_gradient_profile(
            model,
            val_loader,
            device,
            use_amp=use_amp,
        )
        if not args.skip_interventions:
            report["complement_removal"] = complement_removal_curve(
                model,
                val_loader,
                device,
                base["val_loss"],
                args,
            )
            identity_eval = evaluate_loss(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                max_batches=args.eval_batches,
                mixing_overrides="identity",
            )
            report["replace_with_identity"] = {
                **identity_eval,
                "delta_loss": identity_eval["val_loss"] - base["val_loss"],
            }
            random_eval = evaluate_loss(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                max_batches=args.eval_batches,
                mixing_overrides=make_random_iso_overrides(model, device, next(model.parameters()).dtype),
            )
            report["replace_with_random_iso"] = {
                **random_eval,
                "delta_loss": random_eval["val_loss"] - base["val_loss"],
            }

    return report


def write_markdown(report, output_path):
    lines = [
        f"# Mechanism Analysis: {report.get('method')}",
        "",
        f"- run: `{report['run_dir']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- base val loss/PPL: `{report['base_eval']['val_loss']:.4f}` / `{report['base_eval']['val_ppl']:.2f}`",
    ]
    transport = report.get("transport_complement")
    if transport and transport.get("final"):
        final = transport["final"]
        lines.extend([
            "",
            "## Composite Complement Gain",
            "",
            f"- final composite sv mean: `{final['composite_sv_mean']:.6g}`",
            f"- product of per-step sv mean: `{final['product_sv_mean']:.6g}`",
            f"- final composite sv range: `{final['composite_sv_min']:.6g}` to `{final['composite_sv_max']:.6g}`",
        ])

    grad = report.get("gradient_profile")
    if grad:
        lines.extend([
            "",
            "## Stream Gradient",
            "",
            f"- slope of log grad ratio: `{grad['slope_log_grad_ratio']:.6g}`",
        ])

    if "replace_with_identity" in report:
        ident = report["replace_with_identity"]
        rand = report["replace_with_random_iso"]
        lines.extend([
            "",
            "## Replacement Interventions",
            "",
            f"- replace with identity delta loss: `{ident['delta_loss']:.6g}`",
            f"- replace with random fixed-vector orthogonal delta loss: `{rand['delta_loss']:.6g}`",
        ])

    removal = report.get("complement_removal")
    if removal:
        strongest = max(removal, key=lambda row: row["delta_loss"])
        lines.extend([
            "",
            "## Complement Removal",
            "",
            f"- strongest delta loss: `{strongest['delta_loss']:.6g}` at state `{strongest['state_index']}`",
        ])

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze IsoHC/mHC mechanism checkpoints")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--val_cache_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples_val", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--intervention_stride", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--skip_interventions", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for run_dir in args.run_dirs:
        report = analyze_run(run_dir, args, device)
        reports.append(report)
        stem = Path(run_dir).name
        json_path = output_dir / f"{stem}_mechanism_analysis.json"
        md_path = output_dir / f"{stem}_mechanism_analysis.md"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        write_markdown(report, md_path)
        print(f"Wrote {json_path}")

    summary_path = output_dir / "mechanism_analysis_summary.json"
    summary_path.write_text(json.dumps(reports, indent=2, default=str))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
