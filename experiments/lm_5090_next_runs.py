"""Next-stage LM experiment runner for a 32GB RTX 5090.

Presets:
  - run0: geometry/model correctness with tiny token budget
  - deep-stress: 24L 384d/6h/T512, baseline/unconstrained/mHC/IsoHC
  - 125m-smoke: 12L 768d/12h/T512, usually mHC vs IsoHC
  - headmix: 24L 384d/6h/T512, MHA head-output mixing ablation

The runner favors production-like settings for the server: bf16 AMP, SDPA
attention, torch.compile, and optional batch-size autotuning.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm.data import create_dataloader, get_tokenizer
from lm.diagnostics import (
    compute_centered_stream_cosine,
    compute_mean_zero_energy,
    compute_mean_zero_norm_ratio,
    compute_stream_cosine,
)
from lm.models import BaselineTransformer, TwoBranchHCTransformer
from lm.train import run_experiment
from lm.transport_analysis import collect_transport_report


PRESETS = {
    "run0": {
        "num_layers": 2,
        "d_model": 96,
        "num_heads": 4,
        "context_length": 128,
        "n_streams": 4,
        "total_tokens": 262_144,
        "batch_size": 4,
    },
    "deep-stress": {
        "num_layers": 24,
        "d_model": 384,
        "num_heads": 6,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 20_000_000,
        "batch_size": 64,
    },
    "deep-stress-512": {
        "num_layers": 24,
        "d_model": 512,
        "num_heads": 8,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 20_000_000,
        "batch_size": 32,
    },
    "fe-deep-36l-512": {
        "num_layers": 36,
        "d_model": 512,
        "num_heads": 8,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 30_000_000,
        "batch_size": 8,
    },
    "fe-deep-48l-512": {
        "num_layers": 48,
        "d_model": 512,
        "num_heads": 8,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 20_000_000,
        "batch_size": 4,
    },
    "125m-smoke": {
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 5_000_000,
        "batch_size": 16,
    },
    "headmix": {
        "num_layers": 24,
        "d_model": 384,
        "num_heads": 6,
        "context_length": 512,
        "n_streams": 4,
        "total_tokens": 5_000_000,
        "batch_size": 64,
    },
}

HEADMIX_TYPES = {
    "baseline": None,
    "headmix-unconstrained": "unconstrained",
    "headmix-birkhoff": "birkhoff",
    "headmix-iso": "isohc",
    "headmix-fixed-random-iso": "fixed-random-iso",
}

HC_METHOD_TO_MIXING = {
    "identity-hc": "identity",
    "unconstrained": "unconstrained",
    "mhc": "mhc",
    "static-birkhoff-hc": "mhc",
    "isohc": "isohc",
    "scaled-isohc": "isohc",
    "orthogonal": "orthogonal",
}


def build_preset_configs(
    preset,
    methods,
    output_dir,
    total_tokens=None,
    batch_size=None,
    seed=0,
    dataset="tinystories",
    use_compile=True,
    compile_mode="max-autotune",
):
    """Build per-method configs from a preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset}; choose from {sorted(PRESETS)}")

    base = dict(PRESETS[preset])
    if total_tokens is not None:
        base["total_tokens"] = total_tokens
    if batch_size is not None:
        base["batch_size"] = batch_size

    configs = []
    for method in methods:
        cfg = {
            **base,
            "preset": preset,
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_tokens": min(2_000_000, max(16_384, base["total_tokens"] // 10)),
            "grad_clip": 1.0,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "use_amp": True,
            "use_flash": True,
            "use_compile": use_compile,
            "compile_mode": compile_mode,
            "compile_fullgraph": False,
            "compile_dynamic": False,
            "eval_every_tokens": max(262_144, base["total_tokens"] // 5),
            "diagnostics_every": 50,
            "eval_max_batches": 20,
            "grad_accum_steps": 1,
            "save_checkpoints": True,
            "save_dir": os.path.join(output_dir, f"{preset}_{method}_seed{seed}"),
        }
        configs.append(cfg)
    return configs


def create_model(config, vocab_size, device):
    """Create a model for baseline, residual HC, or headmix ablations."""
    method = config["method"]
    head_mixing_type = HEADMIX_TYPES.get(method)

    if config["preset"] == "headmix" or method in HEADMIX_TYPES:
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            context_length=config["context_length"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
            use_flash=config["use_flash"],
            head_mixing_type=head_mixing_type,
            head_mixing_kwargs={"ns_steps": 5, "init_scale": 0.02},
        )
    elif method == "baseline":
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            context_length=config["context_length"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
            use_flash=config["use_flash"],
        )
    elif method in HC_METHOD_TO_MIXING:
        model = TwoBranchHCTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            n_streams=config["n_streams"],
            context_length=config["context_length"],
            mixing_type=HC_METHOD_TO_MIXING[method],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
            lambda_a=config.get("lambda_a", 0.01),
            lambda_b=config.get("lambda_b", 0.01),
            ns_steps=5,
            svd_fallback=(method not in {"isohc", "scaled-isohc"}),
            sinkhorn_iters=10,
            use_flash=config["use_flash"],
            mixing_kwargs=config.get("mixing_kwargs"),
        )
        if config.get("freeze_mixing", False):
            for mixings in (model.attn_mixings, model.mlp_mixings):
                mixings.requires_grad_(False)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device)


def _one_batch_memory_probe(model, batch_size, context_length, vocab_size, device):
    model.train()
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def autotune_batch_size(config, tokenizer, device, memory_target_gb=30.0):
    """Find the largest power-of-two-ish batch that fits the target memory."""
    if device.type != "cuda":
        return config["batch_size"], 0.0

    candidates = [4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 48, 64, 80, 96, 112, 128]
    candidates = [b for b in candidates if b <= max(128, config["batch_size"] * 2)]
    best_batch = None
    best_mem = 0.0

    for batch_size in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        probe_cfg = dict(config, batch_size=batch_size)
        model = create_model(probe_cfg, tokenizer.vocab_size, device)
        try:
            mem_gb = _one_batch_memory_probe(
                model,
                batch_size,
                config["context_length"],
                tokenizer.vocab_size,
                device,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            mem_gb = float("inf")
        finally:
            del model
            torch.cuda.empty_cache()

        if mem_gb <= memory_target_gb:
            best_batch = batch_size
            best_mem = mem_gb
        else:
            break

    if best_batch is None:
        return config["batch_size"], best_mem
    return best_batch, best_mem


def autotune_common_fair_batch(configs, vocab_size, device, memory_target_gb=29.0):
    """Probe every method and choose one common batch for a fair sweep."""
    tokenizer = SimpleNamespace(vocab_size=vocab_size)
    probe = {}
    selected_batches = []

    for cfg in configs:
        batch, mem_gb = autotune_batch_size(cfg, tokenizer, device, memory_target_gb)
        probe[cfg["method"]] = {
            "max_fitting_batch": batch,
            "probe_peak_gb": mem_gb,
        }
        selected_batches.append(batch)

    common_batch = min(selected_batches) if selected_batches else None
    return common_batch, probe


def collect_posthoc_diagnostics(model, val_loader, device):
    """Collect stream diagnostics after training without storing giant tensors."""
    result = {}

    x, _ = next(iter(val_loader))
    x = x[:2].to(device)
    if hasattr(model, "get_stream_states"):
        states = model.get_stream_states(x)
        norm_ratios = [compute_mean_zero_norm_ratio(s) for s in states]
        energies = [compute_mean_zero_energy(s) for s in states]
        cosines = [compute_stream_cosine(s) for s in states]
        centered_cosines = [compute_centered_stream_cosine(s) for s in states]
        result.update({
            "mean_zero_norm_ratio_initial": norm_ratios[0],
            "mean_zero_norm_ratio_final": norm_ratios[-1],
            "mean_zero_norm_ratio_curve": norm_ratios,
            "mean_zero_energy_initial": energies[0],
            "mean_zero_energy_final": energies[-1],
            "stream_cosine_initial": cosines[0],
            "stream_cosine_final": cosines[-1],
            "mean_zero_energy_curve": energies,
            "stream_cosine_curve": cosines,
            "centered_stream_cosine_initial": centered_cosines[0],
            "centered_stream_cosine_final": centered_cosines[-1],
            "centered_stream_cosine_curve": centered_cosines,
        })
    else:
        model.eval()
        with torch.no_grad():
            model(x, x)

    if hasattr(model, "get_diagnostics"):
        diags = model.get_diagnostics()
        if diags:
            result["h_diagnostics_layers"] = diags
            numeric_keys = sorted({
                key for diag in diags for key, value in diag.items()
                if isinstance(value, (int, float))
            })
            result["h_diagnostics"] = {
                f"{key}_mean": sum(float(d.get(key, 0.0)) for d in diags) / len(diags)
                for key in numeric_keys
            }
            result["h_diagnostics"].update({
                f"{key}_max": max(float(d.get(key, 0.0)) for d in diags)
                for key in numeric_keys
            })
    if hasattr(model, "get_named_mixing_matrices"):
        result["transport_complement"] = collect_transport_report(model)

    if hasattr(model, "get_headmix_diagnostics"):
        result["headmix"] = model.get_headmix_diagnostics()
    return result


def build_transport_history(diagnostics):
    return list(diagnostics.snapshots.get("transport", []))


def gate_values(model):
    values = {}
    for result_key, attribute in (
        ("readout_lambda", "readout_lambda"),
        ("injection_lambda", "injection_lambda"),
        ("final_readout_lambda", "readout_final_lambda"),
    ):
        parameter = getattr(model, attribute, None)
        if parameter is not None:
            values[result_key] = parameter.detach().float().item()
    return values


def runtime_provenance(model, config, device):
    is_iso = config["method"] in {"isohc", "scaled-isohc"}
    mixing = model.attn_mixings[0] if is_iso else None
    return {
        "parameter_dtype": str(next(model.parameters()).dtype),
        "amp_dtype": (
            "torch.bfloat16"
            if config.get("use_amp", True) and device.type == "cuda"
            else None
        ),
        "projection_internal_dtype": "torch.float64" if is_iso else None,
        "ns_steps": getattr(mixing, "ns_steps", None),
        "use_svd": getattr(mixing, "use_svd", None),
        "svd_fallback": getattr(mixing, "svd_fallback", None),
        "use_compile": bool(config.get("use_compile", False)),
        "compile_mode": config.get("compile_mode"),
    }


def write_run_summary(config, summary):
    os.makedirs(config["save_dir"], exist_ok=True)
    path = os.path.join(config["save_dir"], "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def _execute_run(config, auto_batch, memory_target_gb):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config["dataset"] == "random" or (
        config.get("train_cache_path") and config.get("val_cache_path")
    ):
        tokenizer = SimpleNamespace(vocab_size=config.get("vocab_size", 50257))
    else:
        tokenizer = get_tokenizer()

    if auto_batch:
        batch, mem_gb = autotune_batch_size(config, tokenizer, device, memory_target_gb)
        config.update(batch_size=batch, autotuned_peak_gb=mem_gb)

    train_loader, _ = create_dataloader(
        config["dataset"],
        tokenizer,
        config["context_length"],
        batch_size=config["batch_size"],
        split="train",
        max_samples=config.get("max_samples"),
        cache_path=config.get("train_cache_path"),
        num_workers=config.get("num_workers", 2),
        prefetch_factor=config.get("prefetch_factor", 4),
        persistent_workers=config.get("persistent_workers", True),
        drop_last=True,
    )
    val_loader, _ = create_dataloader(
        config["dataset"],
        tokenizer,
        config["context_length"],
        batch_size=config["batch_size"],
        split="validation",
        max_samples=config.get("max_samples_val"),
        cache_path=config.get("val_cache_path"),
        num_workers=config.get("num_workers", 2),
        prefetch_factor=config.get("prefetch_factor", 4),
        persistent_workers=config.get("persistent_workers", True),
        drop_last=False,
    )

    model = create_model(config, tokenizer.vocab_size, device)
    initial_transport = (
        collect_transport_report(model)
        if hasattr(model, "get_named_mixing_matrices")
        else None
    )
    initial_gate_values = gate_values(model)
    print("=" * 80)
    print(f"{config['preset']} / {config['method']} / seed={config['seed']}")
    print(f"params={model.count_parameters()/1e6:.2f}M batch={config['batch_size']} "
          f"T={config['context_length']} tokens={config['total_tokens']/1e6:.1f}M")
    print("=" * 80)

    train_cfg = {
        "total_tokens": config["total_tokens"],
        "max_lr": config["max_lr"],
        "min_lr": config["min_lr"],
        "warmup_tokens": config["warmup_tokens"],
        "grad_clip": config["grad_clip"],
        "use_amp": config["use_amp"],
        "eval_every_tokens": config["eval_every_tokens"],
        "save_dir": config["save_dir"],
        "diagnostics_every": config["diagnostics_every"],
        "weight_decay": config["weight_decay"],
        "beta1": config["beta1"],
        "beta2": config["beta2"],
        "eval_max_batches": config["eval_max_batches"],
        "grad_accum_steps": config.get("grad_accum_steps", 1),
        "save_checkpoints": config.get("save_checkpoints", True),
        "save_best_checkpoints": config.get("save_best_checkpoints", True),
        "save_final_checkpoint": config.get("save_final_checkpoint", config.get("save_checkpoints", True)),
        "use_compile": config["use_compile"],
        "compile_mode": config["compile_mode"],
        "compile_fullgraph": config["compile_fullgraph"],
        "compile_dynamic": config["compile_dynamic"],
        "fused_adamw": True,
    }

    results = run_experiment(model, train_loader, val_loader, train_cfg, device)
    posthoc = collect_posthoc_diagnostics(model, val_loader, device)
    return {
        "success": True,
        "config": config,
        "metric_schema_version": config.get("metric_schema_version", 1),
        "final_eval": results["final_eval"],
        "train_metrics": results["train_metrics"],
        "posthoc": posthoc,
        "initial_transport": initial_transport,
        "initial_gate_values": initial_gate_values,
        "training_transport_history": build_transport_history(
            results["diagnostics"]
        ),
        "final_transport": posthoc.get("transport_complement"),
        "final_gate_values": gate_values(model),
        "runtime_provenance": runtime_provenance(model, config, device),
    }


def run_single(config, auto_batch=False, memory_target_gb=30.0):
    config = dict(config)
    started = time.time()
    try:
        summary = _execute_run(config, auto_batch, memory_target_gb)
    except Exception as exc:
        summary = {
            "success": False,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.time() - started,
        }
        write_run_summary(config, summary)
        raise

    summary["elapsed_sec"] = time.time() - started
    write_run_summary(config, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run next-stage 5090 LM experiments")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="deep-stress")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--output_dir", default="outputs/lm_5090_next")
    parser.add_argument("--total_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", dest="use_compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="use_compile", action="store_false")
    parser.add_argument("--compile_mode", default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--fair_auto_batch", action="store_true",
                        help="Probe all methods and run one common largest-fitting batch.")
    parser.add_argument("--memory_target_gb", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--no_save_checkpoints", dest="save_checkpoints",
                        action="store_false", default=True)
    parser.add_argument("--no_save_best_checkpoints", dest="save_best_checkpoints",
                        action="store_false", default=True,
                        help="Skip best.pt writes during training; final.pt can still be saved.")
    parser.add_argument("--no_save_final_checkpoint", dest="save_final_checkpoint",
                        action="store_false", default=True)
    parser.add_argument("--eval_every_tokens", type=int, default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--require_cuda", action="store_true",
                        help="Abort instead of accidentally launching a CPU run.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_val", type=int, default=None)
    parser.add_argument("--train_cache_path", type=str, default=None)
    parser.add_argument("--val_cache_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=50257)
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require_cuda was set, but CUDA is not available.")

    default_methods = {
        "run0": ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
        "deep-stress": ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
        "deep-stress-512": ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
        "fe-deep-36l-512": ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
        "fe-deep-48l-512": ["baseline", "identity-hc", "unconstrained", "mhc", "isohc"],
        "125m-smoke": ["mhc", "isohc"],
        "headmix": [
            "baseline",
            "headmix-unconstrained",
            "headmix-birkhoff",
            "headmix-iso",
            "headmix-fixed-random-iso",
        ],
    }
    methods = args.methods or default_methods[args.preset]
    configs = build_preset_configs(
        preset=args.preset,
        methods=methods,
        output_dir=args.output_dir,
        total_tokens=args.total_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        dataset=args.dataset,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
    )
    for cfg in configs:
        cfg["num_workers"] = args.num_workers
        cfg["prefetch_factor"] = args.prefetch_factor
        cfg["grad_accum_steps"] = args.grad_accum_steps
        cfg["save_checkpoints"] = args.save_checkpoints
        cfg["save_best_checkpoints"] = args.save_best_checkpoints and args.save_checkpoints
        cfg["save_final_checkpoint"] = args.save_final_checkpoint and args.save_checkpoints
        if args.eval_every_tokens is not None:
            cfg["eval_every_tokens"] = args.eval_every_tokens
        if args.eval_max_batches is not None:
            cfg["eval_max_batches"] = args.eval_max_batches
        cfg["max_samples"] = args.max_samples
        cfg["max_samples_val"] = args.max_samples_val
        cfg["train_cache_path"] = args.train_cache_path
        cfg["val_cache_path"] = args.val_cache_path
        cfg["vocab_size"] = args.vocab_size

    if args.auto_batch and args.fair_auto_batch:
        raise ValueError("Use either --auto_batch or --fair_auto_batch, not both.")

    fair_probe = None
    if args.fair_auto_batch:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        common_batch, fair_probe = autotune_common_fair_batch(
            configs,
            args.vocab_size,
            device,
            args.memory_target_gb,
        )
        if common_batch is not None:
            print(f"Fair auto batch selected common batch={common_batch}")
            for method, stats in fair_probe.items():
                print(f"  {method}: max_batch={stats['max_fitting_batch']} "
                      f"peak={stats['probe_peak_gb']:.2f}GB")
            for cfg in configs:
                cfg["batch_size"] = common_batch
                cfg["fair_auto_batch_probe"] = fair_probe

    summaries = []
    for cfg in configs:
        summaries.append(
            run_single(
                cfg,
                auto_batch=args.auto_batch,
                memory_target_gb=args.memory_target_gb,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.preset}_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    if fair_probe is not None:
        with open(os.path.join(args.output_dir, f"{args.preset}_fair_batch_probe.json"), "w") as f:
            json.dump(fair_probe, f, indent=2, default=str)


if __name__ == "__main__":
    main()
