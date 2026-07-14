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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.lm_5090_next_runs import create_model
from isohc.projection import construct_orthogonal_complement
from lm.data import create_dataloader, get_tokenizer
from lm.train import amp_autocast
from lm.transport_analysis import (
    collect_accessibility_report,
    collect_transport_report,
)


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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    batch_size = args.batch_size if args.batch_size is not None else min(2, int(config.get("batch_size", 2)))

    if dataset == "random" or val_cache_path:
        tokenizer = SimpleNamespace(vocab_size=vocab_size)
    else:
        tokenizer = get_tokenizer()

    loader, _ = create_dataloader(
        dataset,
        tokenizer,
        config["context_length"],
        batch_size=batch_size,
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


@torch.no_grad()
def evaluate_paired_intervention(
    model,
    val_loader,
    device,
    use_amp=True,
    max_batches=4,
    stream_intervention=None,
):
    """Compare baseline and intervention on the same validation batches."""
    model.eval()
    base_nll_sum = 0.0
    intervention_nll_sum = 0.0
    kl_sum = 0.0
    changed = 0
    token_count = 0
    batch_count = 0

    for batch_index, (x, y) in enumerate(val_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_autocast(device, enabled=use_amp):
            base_logits, _ = model(x)
            intervention_logits, _ = model(
                x, stream_intervention=stream_intervention
            )

        valid = y.ne(-100)
        count = int(valid.sum().item())
        if count:
            base_logits = base_logits.float()
            intervention_logits = intervention_logits.float()
            flat_y = y.reshape(-1)
            base_nll_sum += F.cross_entropy(
                base_logits.reshape(-1, base_logits.size(-1)),
                flat_y,
                ignore_index=-100,
                reduction="sum",
            ).item()
            intervention_nll_sum += F.cross_entropy(
                intervention_logits.reshape(
                    -1, intervention_logits.size(-1)
                ),
                flat_y,
                ignore_index=-100,
                reduction="sum",
            ).item()
            base_logp = F.log_softmax(base_logits, dim=-1)
            intervention_logp = F.log_softmax(
                intervention_logits, dim=-1
            )
            token_kl = (
                base_logp.exp() * (base_logp - intervention_logp)
            ).sum(dim=-1)
            kl_sum += token_kl[valid].sum().item()
            changed += (
                base_logits.argmax(dim=-1)[valid]
                != intervention_logits.argmax(dim=-1)[valid]
            ).sum().item()
            token_count += count
        batch_count += 1
        if max_batches and batch_index >= max_batches - 1:
            break

    denominator = max(token_count, 1)
    base_nll = base_nll_sum / denominator
    intervention_nll = intervention_nll_sum / denominator
    return {
        "base_nll": base_nll,
        "intervention_nll": intervention_nll,
        "delta_nll": intervention_nll - base_nll,
        "base_ppl": math.exp(min(base_nll, 20)),
        "intervention_ppl": math.exp(min(intervention_nll, 20)),
        "mean_token_kl": kl_sum / denominator,
        "top1_change_rate": changed / denominator,
        "tokens": token_count,
        "batches": batch_count,
    }


def persistent_complement_scale_curve(
    model,
    val_loader,
    device,
    start_indices,
    scales,
    use_amp=True,
    max_batches=4,
):
    if not hasattr(model, "num_layers"):
        return []
    final_state = 2 * model.num_layers
    rows = []
    for start in start_indices:
        if not 0 <= start <= final_state:
            raise ValueError(
                f"start state {start} is outside [0, {final_state}]"
            )
        active_states = list(range(start, final_state + 1))
        for scale in scales:
            if not 0.0 <= scale <= 1.0:
                raise ValueError("intervention scales must be in [0, 1]")
            metrics = evaluate_paired_intervention(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                max_batches=max_batches,
                stream_intervention={
                    "state_index": active_states,
                    "mode": "scale_perp",
                    "scale": float(scale),
                },
            )
            rows.append({
                "start_state": start,
                "scale": float(scale),
                "active_state_count": len(active_states),
                "semantics": "repeated_statewise_damping",
                **metrics,
            })
    return rows


def one_shot_complement_scale_curve(
    model,
    val_loader,
    device,
    state_indices,
    scales,
    use_amp=True,
    max_batches=4,
):
    """Scale complement once, avoiding repeated-gamma compounding."""
    if not hasattr(model, "num_layers"):
        return []
    final_state = 2 * model.num_layers
    rows = []
    for state_index in state_indices:
        if not 0 <= state_index <= final_state:
            raise ValueError(
                f"state {state_index} is outside [0, {final_state}]"
            )
        for scale in scales:
            if not 0.0 <= scale <= 1.0:
                raise ValueError("intervention scales must be in [0, 1]")
            metrics = evaluate_paired_intervention(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                max_batches=max_batches,
                stream_intervention={
                    "state_index": state_index,
                    "mode": "scale_perp",
                    "scale": float(scale),
                },
            )
            rows.append({
                "state_index": state_index,
                "scale": float(scale),
                "application_count": 1,
                "semantics": "one_shot_scaling",
                **metrics,
            })
    return rows


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
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
        report["accessibility_proxy"] = collect_accessibility_report(model)
        report["gradient_profile"] = stream_gradient_profile(
            model,
            val_loader,
            device,
            use_amp=use_amp,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not args.skip_interventions:
            report["complement_removal"] = complement_removal_curve(
                model,
                val_loader,
                device,
                base["val_loss"],
                args,
            )
            num_states = 1 + 2 * model.num_layers
            indices = list(range(
                0, num_states, max(1, args.intervention_stride)
            ))
            if indices[-1] != num_states - 1:
                indices.append(num_states - 1)
            report["one_shot_complement_scale"] = (
                one_shot_complement_scale_curve(
                    model,
                    val_loader,
                    device,
                    state_indices=indices,
                    scales=args.intervention_scales,
                    use_amp=use_amp,
                    max_batches=args.eval_batches,
                )
            )
            report["persistent_complement_scale"] = (
                persistent_complement_scale_curve(
                    model,
                    val_loader,
                    device,
                    start_indices=indices,
                    scales=args.intervention_scales,
                    use_amp=use_amp,
                    max_batches=args.eval_batches,
                )
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
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
            "## Weak Single-State Complement Removal",
            "",
            f"- strongest delta loss: `{strongest['delta_loss']:.6g}` at state `{strongest['state_index']}`",
        ])

    persistent = report.get("persistent_complement_scale")
    if persistent:
        max_delta = max(abs(row["delta_nll"]) for row in persistent)
        max_kl = max(row["mean_token_kl"] for row in persistent)
        lines.extend([
            "",
            "## Persistent Complement Scaling",
            "",
            f"- maximum absolute delta NLL: `{max_delta:.6g}`",
            f"- maximum mean token KL: `{max_kl:.6g}`",
        ])

    one_shot = report.get("one_shot_complement_scale")
    if one_shot:
        max_delta = max(abs(row["delta_nll"]) for row in one_shot)
        lines.extend([
            "",
            "## One-shot Complement Scaling",
            "",
            f"- maximum absolute delta NLL: `{max_delta:.6g}`",
        ])

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze IsoHC/mHC mechanism checkpoints")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--val_cache_path", default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Posthoc batch size. Defaults to min(2, training batch) to avoid gradient-capture OOM.",
    )
    parser.add_argument("--max_samples_val", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--intervention_stride", type=int, default=8)
    parser.add_argument(
        "--intervention_scales",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
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
